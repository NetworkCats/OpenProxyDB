"""Wikipedia block list crawler for OpenProxyDB."""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Set

import aiohttp

from src.block_manager import BlockManager
from src.config import (
    BACKOFF_MULTIPLIER,
    CSV_FILENAME,
    GITHUB_OWNER,
    GITHUB_REPO,
    GLOBAL_BLOCKS_PARAMS,
    LAST_CRAWL_TIME_FILE,
    LOCAL_BLOCKS_PARAMS,
    MAX_CONCURRENT_REQUESTS,
    MAX_RETRIES,
    METADATA_CSV_FILENAME,
    REDUCED_REQUESTS_PER_SECOND,
    REQUEST_TIMEOUT,
    REQUESTS_PER_SECOND,
    RETRY_DELAY,
    WIKIPEDIA_API_URL,
    get_user_agent,
)
from src.csv_handler import CSVHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Async rate limiter for API requests using token bucket algorithm."""

    def __init__(self, requests_per_second: float, max_concurrent: int):
        """Initialize the rate limiter.

        Args:
            requests_per_second: Maximum requests per second.
            max_concurrent: Maximum concurrent requests.
        """
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        await self._semaphore.acquire()
        async with self._lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = time.time()

    def release(self) -> None:
        """Release the semaphore after request completes."""
        self._semaphore.release()

    def reduce_rate(self, new_requests_per_second: float) -> None:
        """Reduce the rate limit.

        Args:
            new_requests_per_second: New maximum requests per second.
        """
        self._min_interval = 1.0 / new_requests_per_second
        logger.warning(f"Rate reduced to {new_requests_per_second} requests/second")


class AsyncWikipediaCrawler:
    """Async crawler for Wikipedia block lists."""

    def __init__(self):
        """Initialize the crawler."""
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = AsyncRateLimiter(
            REQUESTS_PER_SECOND, MAX_CONCURRENT_REQUESTS
        )
        self._block_manager = BlockManager()
        self._csv_handler = CSVHandler()
        self._rate_reduced = False

    async def __aenter__(self) -> "AsyncWikipediaCrawler":
        """Enter async context manager."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": get_user_agent()},
            timeout=timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        if self._session:
            await self._session.close()

    @property
    def block_manager(self) -> BlockManager:
        """Get the block manager."""
        return self._block_manager

    @property
    def csv_handler(self) -> CSVHandler:
        """Get the CSV handler."""
        return self._csv_handler

    async def _make_request(self, params: dict) -> Optional[dict]:
        """Make an API request with retry logic.

        Args:
            params: Query parameters for the API.

        Returns:
            JSON response as dictionary, or None on failure.
        """
        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        for attempt in range(MAX_RETRIES):
            await self._rate_limiter.acquire()
            try:
                async with self._session.get(
                    WIKIPEDIA_API_URL, params=params
                ) as response:
                    if response.status == 429:
                        # Rate limited
                        if not self._rate_reduced:
                            self._rate_limiter.reduce_rate(REDUCED_REQUESTS_PER_SECOND)
                            self._rate_reduced = True

                        backoff_time = BACKOFF_MULTIPLIER ** (attempt + 1)
                        logger.warning(
                            f"Rate limited (429). Backing off for {backoff_time}s"
                        )
                        await asyncio.sleep(backoff_time)
                        continue

                    response.raise_for_status()
                    return await response.json()

            except asyncio.TimeoutError:
                logger.warning(
                    f"Request timeout (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(RETRY_DELAY)
            except aiohttp.ClientError as e:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                )
                await asyncio.sleep(RETRY_DELAY)
            finally:
                self._rate_limiter.release()

        logger.error(f"Failed after {MAX_RETRIES} attempts")
        return None

    async def fetch_local_blocks(
        self, start_time: Optional[str] = None, end_time: Optional[str] = None
    ) -> int:
        """Fetch local blocks from Wikipedia.

        Args:
            start_time: Start timestamp for incremental crawl.
            end_time: End timestamp for incremental crawl.

        Returns:
            Number of blocks added.
        """
        logger.info("Fetching local blocks...")
        params = LOCAL_BLOCKS_PARAMS.copy()

        if start_time:
            params["bkstart"] = start_time
            params["bkdir"] = "newer"
        if end_time:
            params["bkend"] = end_time

        total_added = 0
        total_fetched = 0

        while True:
            data = await self._make_request(params)
            if not data:
                logger.error("Failed to fetch local blocks")
                break

            blocks = data.get("query", {}).get("blocks", [])
            total_fetched += len(blocks)

            for block in blocks:
                if self._block_manager.add_block(block, "local"):
                    total_added += 1

            logger.info(f"Fetched {total_fetched} local blocks, {total_added} added")

            # Check for continuation
            continue_data = data.get("continue")
            if continue_data:
                params.update(continue_data)
            else:
                break

        logger.info(
            f"Completed local blocks: {total_fetched} fetched, {total_added} added"
        )
        return total_added

    async def fetch_global_blocks(
        self, start_time: Optional[str] = None, end_time: Optional[str] = None
    ) -> int:
        """Fetch global blocks from Wikipedia.

        Args:
            start_time: Start timestamp for incremental crawl.
            end_time: End timestamp for incremental crawl.

        Returns:
            Number of blocks added.
        """
        logger.info("Fetching global blocks...")
        params = GLOBAL_BLOCKS_PARAMS.copy()

        if start_time:
            params["bgstart"] = start_time
            params["bgdir"] = "newer"
        if end_time:
            params["bgend"] = end_time

        total_added = 0
        total_fetched = 0

        while True:
            data = await self._make_request(params)
            if not data:
                logger.error("Failed to fetch global blocks")
                break

            blocks = data.get("query", {}).get("globalblocks", [])
            total_fetched += len(blocks)

            for block in blocks:
                if self._block_manager.add_block(block, "global"):
                    total_added += 1

            logger.info(f"Fetched {total_fetched} global blocks, {total_added} added")

            # Check for continuation
            continue_data = data.get("continue")
            if continue_data:
                params.update(continue_data)
            else:
                break

        logger.info(
            f"Completed global blocks: {total_fetched} fetched, {total_added} added"
        )
        return total_added

    def get_last_crawl_time(self) -> Optional[str]:
        """Read the last crawl timestamp from file.

        Returns:
            Timestamp string or None if file doesn't exist.
        """
        if not os.path.exists(LAST_CRAWL_TIME_FILE):
            return None

        try:
            with open(LAST_CRAWL_TIME_FILE, "r", encoding="utf-8") as f:
                timestamp = f.read().strip()
                if timestamp:
                    return timestamp
        except OSError as e:
            logger.warning(f"Failed to read last crawl time: {e}")

        return None

    def save_last_crawl_time(self, timestamp: str) -> None:
        """Save the last crawl timestamp to file.

        Args:
            timestamp: Timestamp string to save.
        """
        try:
            with open(LAST_CRAWL_TIME_FILE, "w", encoding="utf-8") as f:
                f.write(timestamp)
            logger.info(f"Saved last crawl time: {timestamp}")
        except OSError as e:
            logger.warning(f"Failed to save last crawl time: {e}")

    def load_metadata_csv(self) -> Optional[str]:
        """Load metadata CSV from local file if it exists.

        Returns:
            CSV content as string, or None if not found.
        """
        if not os.path.exists(METADATA_CSV_FILENAME):
            return None

        try:
            with open(METADATA_CSV_FILENAME, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.warning(f"Failed to read metadata CSV: {e}")
            return None

    async def check_release_exists(self) -> bool:
        """Check if any release exists in the repository.

        Returns:
            True if at least one release exists.
        """
        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            logger.warning("GITHUB_TOKEN not set, assuming no release exists")
            return False

        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            async with self._session.get(url, headers=headers) as response:
                response.raise_for_status()
                releases = await response.json()
                return len(releases) > 0
        except aiohttp.ClientError as e:
            logger.warning(f"Failed to check releases: {e}")
            return False

    def _get_current_ips(self) -> Set[str]:
        """Get the current set of IPs in the block manager.

        Returns:
            Set of IP strings.
        """
        return {block.ip for block in self._block_manager.get_all_blocks()}

    def _write_stats_to_github_env(
        self, added: int, removed: int, total: int
    ) -> None:
        """Write statistics to GITHUB_ENV for use in workflow.

        Args:
            added: Number of IPs added.
            removed: Number of IPs removed.
            total: Total number of IPs.
        """
        github_env = os.environ.get("GITHUB_ENV")
        if github_env:
            try:
                with open(github_env, "a", encoding="utf-8") as f:
                    f.write(f"IPS_ADDED={added}\n")
                    f.write(f"IPS_REMOVED={removed}\n")
                    f.write(f"TOTAL_IPS={total}\n")
                logger.info(
                    f"Stats written to GITHUB_ENV: added={added}, "
                    f"removed={removed}, total={total}"
                )
            except OSError as e:
                logger.warning(f"Failed to write stats to GITHUB_ENV: {e}")
        else:
            logger.info(
                f"GITHUB_ENV not set. Stats: added={added}, "
                f"removed={removed}, total={total}"
            )

    async def download_latest_csv(self) -> Optional[str]:
        """Download the CSV from the latest release.

        Returns:
            CSV content as string, or None if not found.
        """
        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            logger.warning("GITHUB_TOKEN not set")
            return None

        # Get latest release
        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            async with self._session.get(url, headers=headers) as response:
                response.raise_for_status()
                release = await response.json()

                # Find CSV asset
                for asset in release.get("assets", []):
                    if asset["name"] == CSV_FILENAME:
                        download_url = asset["browser_download_url"]
                        # browser_download_url is a public URL that doesn't require
                        # authentication, so we don't pass the Authorization header
                        async with self._session.get(download_url) as csv_response:
                            csv_response.raise_for_status()
                            return await csv_response.text()

                logger.warning("CSV file not found in latest release")
                return None

        except aiohttp.ClientError as e:
            logger.warning(f"Failed to download latest CSV: {e}")
            return None

    async def run_full_crawl(self) -> bool:
        """Run a full crawl of all blocks.

        Returns:
            True if successful.
        """
        logger.info("Starting full crawl...")

        # Full crawl starts with empty set
        initial_ips: Set[str] = set()

        # Fetch local and global blocks concurrently
        await asyncio.gather(
            self.fetch_local_blocks(),
            self.fetch_global_blocks(),
        )

        # Remove expired blocks
        expired_count = self._block_manager.remove_expired_blocks()
        logger.info(f"Removed {expired_count} expired blocks")

        # Get final IPs and calculate stats
        final_ips = self._get_current_ips()
        added_ips = len(final_ips - initial_ips)
        removed_ips = len(initial_ips - final_ips)
        total_ips = len(final_ips)

        # Write stats to GITHUB_ENV
        self._write_stats_to_github_env(added_ips, removed_ips, total_ips)

        # Export main CSV (without timestamps)
        csv_content = self._csv_handler.export_from_manager(self._block_manager)
        with open(CSV_FILENAME, "w", encoding="utf-8") as f:
            f.write(csv_content)

        # Export metadata CSV (timestamps only)
        metadata_content = self._csv_handler.export_metadata_from_manager(
            self._block_manager
        )
        with open(METADATA_CSV_FILENAME, "w", encoding="utf-8") as f:
            f.write(metadata_content)

        logger.info(
            f"Exported {len(self._block_manager)} blocks to {CSV_FILENAME} "
            f"and {METADATA_CSV_FILENAME}"
        )

        # Save crawl time
        crawl_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.save_last_crawl_time(crawl_time)

        return True

    async def run_incremental_crawl(self) -> bool:
        """Run an incremental crawl since last crawl time.

        Returns:
            True if successful.
        """
        last_crawl_time = self.get_last_crawl_time()
        if not last_crawl_time:
            logger.warning("No last crawl time found, running full crawl")
            return await self.run_full_crawl()

        logger.info(f"Starting incremental crawl since {last_crawl_time}...")

        # Download existing CSV from release
        existing_csv = await self.download_latest_csv()
        if not existing_csv:
            # Cannot proceed without existing data, fall back to full crawl
            logger.warning(
                "Failed to download existing CSV, falling back to full crawl"
            )
            # Clean up old timestamp file to avoid repeated failures
            if os.path.exists(LAST_CRAWL_TIME_FILE):
                os.remove(LAST_CRAWL_TIME_FILE)
                logger.info("Removed stale last_crawl_time file")
            return await self.run_full_crawl()

        # Load metadata CSV from local file (fetched by workflow from data branch)
        metadata_csv = self.load_metadata_csv()
        if not metadata_csv:
            logger.warning("No metadata CSV found, timestamps will be empty")

        # Load existing data into manager
        self._csv_handler.load_to_manager(
            existing_csv, self._block_manager, metadata_csv
        )
        logger.info(f"Loaded {len(self._block_manager)} existing blocks")

        # Record initial IPs before fetching new blocks
        initial_ips = self._get_current_ips()

        # Fetch new blocks since last crawl (duplicates are merged by BlockManager)
        await asyncio.gather(
            self.fetch_local_blocks(start_time=last_crawl_time),
            self.fetch_global_blocks(start_time=last_crawl_time),
        )

        # Record current time AFTER fetching completes as next starting point
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Remove expired blocks
        expired_count = self._block_manager.remove_expired_blocks()
        logger.info(f"Removed {expired_count} expired blocks")

        # Get final IPs and calculate stats
        final_ips = self._get_current_ips()
        added_ips = len(final_ips - initial_ips)
        removed_ips = len(initial_ips - final_ips)
        total_ips = len(final_ips)

        # Write stats to GITHUB_ENV
        self._write_stats_to_github_env(added_ips, removed_ips, total_ips)

        # Export main CSV (without timestamps)
        csv_content = self._csv_handler.export_from_manager(self._block_manager)
        with open(CSV_FILENAME, "w", encoding="utf-8") as f:
            f.write(csv_content)

        # Export metadata CSV (timestamps only)
        metadata_content = self._csv_handler.export_metadata_from_manager(
            self._block_manager
        )
        with open(METADATA_CSV_FILENAME, "w", encoding="utf-8") as f:
            f.write(metadata_content)

        logger.info(
            f"Exported {len(self._block_manager)} blocks to {CSV_FILENAME} "
            f"and {METADATA_CSV_FILENAME}"
        )

        # Save current_time as next crawl starting point
        self.save_last_crawl_time(current_time)

        return True

    async def run(self) -> bool:
        """Run the crawler in appropriate mode.

        Returns:
            True if successful.
        """
        # Check if release exists to determine mode
        release_exists = await self.check_release_exists()

        if release_exists:
            logger.info("Release exists, running incremental crawl")
            return await self.run_incremental_crawl()
        else:
            logger.info("No release found, running full crawl")
            return await self.run_full_crawl()


async def async_main() -> int:
    """Async main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger.info("OpenProxyDB Crawler starting...")

    async with AsyncWikipediaCrawler() as crawler:
        success = await crawler.run()

    if success:
        logger.info("Crawler completed successfully")
        return 0
    else:
        logger.error("Crawler failed")
        return 1


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
