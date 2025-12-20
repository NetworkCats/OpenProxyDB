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
