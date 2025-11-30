"""CSV file handling for OpenProxyDB."""

import csv
import io
import ipaddress
from typing import Optional, Union

from src.block_manager import BlockManager, BlockRecord
from src.config import PROXY_CLASSIFICATIONS


class CSVHandler:
    """Handles CSV file operations for block records."""

    def read_csv(self, content: str) -> list[dict]:
        """Read CSV content and return list of records.

        Args:
            content: CSV file content as string.

        Returns:
            List of dictionaries representing each row.
        """
        return list(csv.DictReader(io.StringIO(content)))

    def write_csv(self, blocks: list[BlockRecord]) -> str:
        """Write block records to CSV format (without timestamps).

        Args:
            blocks: List of BlockRecord objects.

        Returns:
            CSV content as string.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Generate headers from PROXY_CLASSIFICATIONS
        classification_columns = list(PROXY_CLASSIFICATIONS.values())
        headers = ["ip"] + classification_columns
        writer.writerow(headers)

        # Sort blocks by IP address for consistent output
        sorted_blocks = sorted(blocks, key=lambda b: self._sort_key(b.ip))

        for block in sorted_blocks:
            row = [block.ip]
            # Add boolean columns for each classification type
            for col in classification_columns:
                row.append(col in block.classifications)
            writer.writerow(row)

        return output.getvalue()

    def write_metadata_csv(self, blocks: list[BlockRecord]) -> str:
        """Write block metadata (timestamps) to CSV format.

        Args:
            blocks: List of BlockRecord objects.

        Returns:
            CSV content as string with ip, block_time, expiry_time columns.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        headers = ["ip", "block_time", "expiry_time"]
        writer.writerow(headers)

        # Sort blocks by IP address for consistent output
        sorted_blocks = sorted(blocks, key=lambda b: self._sort_key(b.ip))

        for block in sorted_blocks:
            writer.writerow([block.ip, block.block_time, block.expiry_time])

        return output.getvalue()

    def _sort_key(self, ip: str) -> tuple[int, Union[int, str], int]:
        """Generate a sort key for IP addresses.

        Sorts IPv4 addresses first, then IPv6 addresses, both numerically.
        CIDR ranges are sorted by their base address, then by prefix length.

        Args:
            ip: IP address string (with optional CIDR notation).

        Returns:
            Tuple for sorting: (ip_version, numeric_address, prefix_length).
        """
        try:
            if "/" in ip:
                network = ipaddress.ip_network(ip, strict=False)
                return (network.version, int(network.network_address), network.prefixlen)
            else:
                addr = ipaddress.ip_address(ip)
                # Use max prefix length for single addresses
                max_prefix = 32 if addr.version == 4 else 128
                return (addr.version, int(addr), max_prefix)
        except ValueError:
            # Fallback for invalid addresses - sort at the end
            return (99, ip, 0)

    def load_to_manager(
        self,
        content: str,
        manager: Optional[BlockManager] = None,
        metadata_content: Optional[str] = None,
    ) -> BlockManager:
        """Load CSV content into a BlockManager.

        Args:
            content: Main CSV file content (ip + classifications).
            manager: Existing BlockManager to load into, or None to create new.
            metadata_content: Metadata CSV content (ip + timestamps), optional.

        Returns:
            BlockManager with loaded records.
        """
        if manager is None:
            manager = BlockManager()

        records = self.read_csv(content)
        # Classification columns from PROXY_CLASSIFICATIONS
        classification_columns = list(PROXY_CLASSIFICATIONS.values())

        # Load metadata if provided
        metadata_by_ip: dict[str, dict] = {}
        if metadata_content:
            metadata_records = self.read_csv(metadata_content)
            for record in metadata_records:
                ip = record.get("ip", "")
                if ip:
                    metadata_by_ip[ip] = {
                        "block_time": record.get("block_time", ""),
                        "expiry_time": record.get("expiry_time", ""),
                    }

        # Convert records to format expected by BlockManager
        converted_records = []
        for record in records:
            ip = record.get("ip", "")
            if not ip:
                continue

            # Extract classifications from boolean columns
            classifications = []
            for col in classification_columns:
                if record.get(col) == "True":
                    classifications.append(col)

            # Get timestamps from metadata if available
            metadata = metadata_by_ip.get(ip, {})

            converted_records.append({
                "IP": ip,
                "Classifications": " ".join(classifications),
                "Block Time": metadata.get("block_time", ""),
                "Expiry Time": metadata.get("expiry_time", ""),
            })

        manager.load_from_records(converted_records)
        return manager

    def export_from_manager(self, manager: BlockManager) -> str:
        """Export BlockManager contents to CSV string (without timestamps).

        Args:
            manager: BlockManager with block records.

        Returns:
            CSV content as string.
        """
        blocks = manager.get_all_blocks()
        return self.write_csv(blocks)

    def export_metadata_from_manager(self, manager: BlockManager) -> str:
        """Export BlockManager metadata to CSV string.

        Args:
            manager: BlockManager with block records.

        Returns:
            Metadata CSV content as string.
        """
        blocks = manager.get_all_blocks()
        return self.write_metadata_csv(blocks)
