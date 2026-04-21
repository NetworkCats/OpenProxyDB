"""Block record management for OpenProxyDB."""

import ipaddress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.config import PROXY_CLASSIFICATIONS


def _parse_timestamp(timestamp: str) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp string to datetime.

    Args:
        timestamp: ISO 8601 formatted timestamp string.

    Returns:
        datetime object or None if parsing fails.
    """
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def _to_network(ip: str) -> Optional[ipaddress._BaseNetwork]:
    """Parse an IP or CIDR string into an ip_network.

    Bare addresses are widened to their single-host form (/32 for IPv4,
    /128 for IPv6) so every stored entry can be compared via subnet
    containment.
    """
    if not ip:
        return None
    try:
        if "/" in ip:
            return ipaddress.ip_network(ip, strict=False)
        addr = ipaddress.ip_address(ip)
        return ipaddress.ip_network(f"{addr}/{addr.max_prefixlen}")
    except ValueError:
        return None


def _compare_timestamps(ts1: str, ts2: str, prefer_earlier: bool = True) -> str:
    """Compare two timestamps and return the preferred one.

    Args:
        ts1: First timestamp string.
        ts2: Second timestamp string.
        prefer_earlier: If True, return earlier timestamp; if False, return later.

    Returns:
        The preferred timestamp string based on comparison.
    """
    dt1 = _parse_timestamp(ts1)
    dt2 = _parse_timestamp(ts2)

    # If both parse successfully, compare datetime objects
    if dt1 and dt2:
        if prefer_earlier:
            return ts1 if dt1 < dt2 else ts2
        else:
            return ts1 if dt1 > dt2 else ts2

    # If only one parses, return the valid one
    if dt1:
        return ts1
    if dt2:
        return ts2

    # If neither parses, return the first one as fallback
    return ts1


@dataclass
class BlockRecord:
    """Represents a single block record."""

    ip: str
    classifications: set = field(default_factory=set)
    block_time: str = ""
    expiry_time: str = ""


class BlockManager:
    """Manages block records with filtering and deduplication."""

    def __init__(self):
        """Initialize the block manager."""
        self._blocks: dict[str, BlockRecord] = {}

    def is_proxy_related(self, reason: str) -> bool:
        """Check if a block reason contains proxy-related keywords.

        Args:
            reason: The block reason text.

        Returns:
            True if the reason contains any proxy-related keyword.
        """
        if not reason:
            return False

        reason_lower = reason.lower()
        return any(keyword in reason_lower for keyword in PROXY_CLASSIFICATIONS)

    def extract_classifications(self, reason: str) -> set[str]:
        """Extract classification types from block reason.

        Args:
            reason: The block reason text.

        Returns:
            Set of classification strings found in the reason.
        """
        classifications = set()
        if not reason:
            return classifications

        reason_lower = reason.lower()
        for keyword, classification in PROXY_CLASSIFICATIONS.items():
            if keyword in reason_lower:
                classifications.add(classification)

        return classifications

    def extract_ip_from_block(self, block: dict, source: str) -> Optional[str]:
        """Extract IP address or range from a block record.

        Args:
            block: The block record from API.
            source: Either "local" or "global".

        Returns:
            IP address/range string or None if not found/applicable.
        """
        if source == "local":
            # For local blocks, use "user" field which contains IP or username
            user = block.get("user", "")
            # Check if it's an IP address or range
            if self._is_ip_or_range(user):
                return user
            # Also check rangestart/rangeend for autoblock cases
            rangestart = block.get("rangestart", "")
            rangeend = block.get("rangeend", "")
            if rangestart and rangeend:
                cidr = self._range_to_cidr(rangestart, rangeend)
                if cidr:
                    return cidr
            elif rangestart and self._is_ip_or_range(rangestart):
                return rangestart
            return None
        else:
            # For global blocks, use "target" field directly
            # Note: Global blocks API does not return rangestart/rangeend fields
            target = block.get("target", "")
            if self._is_ip_or_range(target):
                return target
            return None

    def _range_to_cidr(self, start: str, end: str) -> Optional[str]:
        """Convert IP range (start-end) to CIDR notation.

        Args:
            start: Starting IP address.
            end: Ending IP address.

        Returns:
            CIDR notation string, or None if conversion fails.
        """
        try:
            start_ip = ipaddress.ip_address(start)
            end_ip = ipaddress.ip_address(end)

            # If same IP, just return it
            if start_ip == end_ip:
                return str(start_ip)

            # Use summarize_address_range to get CIDR blocks
            networks = list(ipaddress.summarize_address_range(start_ip, end_ip))
            if len(networks) == 1:
                # Single CIDR block
                return str(networks[0])
            else:
                # Multiple CIDR blocks needed, return the first one
                # This covers most of the range
                return str(networks[0])
        except (ValueError, TypeError):
            return None

    def _is_ip_or_range(self, value: str) -> bool:
        """Check if a value is an IP address or CIDR range.

        Uses Python's ipaddress module for robust validation.

        Args:
            value: The string to check.

        Returns:
            True if it's an IP address or range.
        """
        if not value:
            return False

        # Skip if it starts with ~ (temporary account identifier)
        if value.startswith("~"):
            return False

        try:
            # Try parsing as network (handles CIDR notation)
            if "/" in value:
                ipaddress.ip_network(value, strict=False)
            else:
                ipaddress.ip_address(value)
            return True
        except ValueError:
            return False

    def add_block(self, block: dict, source: str) -> bool:
        """Add a block record if it's proxy-related.

        Args:
            block: The block record from API.
            source: Either "local" or "global".

        Returns:
            True if the block was added, False otherwise.
        """
        reason = block.get("reason", "")
        if not self.is_proxy_related(reason):
            return False

        ip = self.extract_ip_from_block(block, source)
        if not ip:
            return False

        classifications = self.extract_classifications(reason)
        if not classifications:
            return False

        block_time = block.get("timestamp", "")
        expiry_time = block.get("expiry", "")

        if ip in self._blocks:
            # Merge with existing record
            existing = self._blocks[ip]
            existing.classifications.update(classifications)
            # Keep earliest block time using proper datetime comparison
            if block_time and existing.block_time:
                existing.block_time = _compare_timestamps(
                    existing.block_time, block_time, prefer_earlier=True
                )
            elif block_time:
                existing.block_time = block_time
            # Keep latest expiry time using proper datetime comparison
            if expiry_time:
                if existing.expiry_time == "infinity" or expiry_time == "infinity":
                    existing.expiry_time = "infinity"
                elif existing.expiry_time:
                    existing.expiry_time = _compare_timestamps(
                        existing.expiry_time, expiry_time, prefer_earlier=False
                    )
                else:
                    existing.expiry_time = expiry_time
        else:
            # Create new record
            self._blocks[ip] = BlockRecord(
                ip=ip,
                classifications=classifications,
                block_time=block_time,
                expiry_time=expiry_time,
            )

        return True

    def is_expired(self, expiry_time: str) -> bool:
        """Check if a block has expired.

        Args:
            expiry_time: The expiry timestamp string.

        Returns:
            True if the block has expired, False otherwise.
        """
        if not expiry_time or expiry_time == "infinity":
            return False

        expiry = _parse_timestamp(expiry_time)
        if not expiry:
            return False
        return expiry < datetime.now(timezone.utc)

    def remove_expired_blocks(self) -> int:
        """Remove all expired blocks from the manager.

        Returns:
            Number of blocks removed.
        """
        expired_ips = [
            ip for ip, block in self._blocks.items() if self.is_expired(block.expiry_time)
        ]
        for ip in expired_ips:
            del self._blocks[ip]
        return len(expired_ips)

    def deduplicate(self) -> int:
        """Collapse redundant CIDR subsets into their containing ranges.

        Exact-IP duplicates from multiple sources are already merged at
        insertion time by `add_block`. This method handles the remaining
        case: entries whose address range is fully contained within a
        larger range that is also tracked (e.g. 1.2.3.4 inside 1.2.3.0/24,
        or 10.0.0.0/24 inside 10.0.0.0/16). The subset's classifications
        and timestamps are merged into the container before it is removed.

        Returns:
            Number of redundant subset entries removed.
        """
        # Parse each stored key into an ip_network (bare addresses become
        # /32 or /128 single-host networks). Keep the mapping from network
        # object back to the original stored key.
        networks: dict[str, ipaddress._BaseNetwork] = {}
        lookup: dict[int, dict[ipaddress._BaseNetwork, str]] = {4: {}, 6: {}}
        for ip in self._blocks:
            net = _to_network(ip)
            if net is None:
                continue
            networks[ip] = net
            lookup[net.version][net] = ip

        # For each entry, walk up its supernet chain. Record the closest
        # containing entry (not itself). Narrower subsets are scheduled
        # for removal first so chained redundancy (A ⊃ B ⊃ C) collapses
        # upward correctly when we apply the merges in order.
        pending: list[tuple[str, str, int]] = []
        for ip, net in networks.items():
            current = net
            while current.prefixlen > 0:
                current = current.supernet(prefixlen_diff=1)
                container_ip = lookup[net.version].get(current)
                if container_ip is not None and container_ip != ip:
                    pending.append((ip, container_ip, net.prefixlen))
                    break

        # Sort narrowest-first so C→B merges before B→A (otherwise B would
        # already be gone when we try to merge C into it).
        pending.sort(key=lambda item: -item[2])

        removed = 0
        for subset_ip, container_ip, _prefixlen in pending:
            subset = self._blocks.pop(subset_ip, None)
            if subset is None:
                continue
            container = self._blocks.get(container_ip)
            if container is None:
                # Container was itself removed via a chain merge; put the
                # subset back under the broader ancestor that absorbed it.
                # Find the absorbing root by chasing pending entries.
                self._blocks[subset_ip] = subset
                continue
            self._merge_record(container, subset)
            removed += 1
        return removed

    @staticmethod
    def _merge_record(target: BlockRecord, source: BlockRecord) -> None:
        """Fold `source`'s classifications and timestamps into `target`."""
        target.classifications.update(source.classifications)
        if source.block_time and target.block_time:
            target.block_time = _compare_timestamps(
                target.block_time, source.block_time, prefer_earlier=True
            )
        elif source.block_time:
            target.block_time = source.block_time
        if source.expiry_time:
            if target.expiry_time == "infinity" or source.expiry_time == "infinity":
                target.expiry_time = "infinity"
            elif target.expiry_time:
                target.expiry_time = _compare_timestamps(
                    target.expiry_time, source.expiry_time, prefer_earlier=False
                )
            else:
                target.expiry_time = source.expiry_time

    def get_all_blocks(self) -> list[BlockRecord]:
        """Get all block records.

        Returns:
            List of all BlockRecord objects.
        """
        return list(self._blocks.values())

    def load_from_records(self, records: list[dict]) -> None:
        """Load blocks from a list of record dictionaries.

        Args:
            records: List of dicts with IP, Classifications, Block Time, Expiry Time.
        """
        for record in records:
            ip = record.get("IP", "")
            if not ip:
                continue

            classifications_str = record.get("Classifications", "")
            classifications = set(classifications_str.split()) if classifications_str else set()

            self._blocks[ip] = BlockRecord(
                ip=ip,
                classifications=classifications,
                block_time=record.get("Block Time", ""),
                expiry_time=record.get("Expiry Time", ""),
            )

    def __len__(self) -> int:
        """Return number of blocks."""
        return len(self._blocks)
