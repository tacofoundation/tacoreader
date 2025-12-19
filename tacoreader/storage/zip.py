"""
ZIP backend for TACO datasets.

Reads .tacozip format files with offset-based access via TACO_HEADER.
Optimized for single-file distribution and cloud storage with range reads.

Main class:
    ZipBackend: Backend for .tacozip format
"""

import mmap
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import tacozip

from tacoreader._constants import (
    LEVEL_TABLE_SUFFIX,
    LEVEL_VIEW_PREFIX,
    METADATA_GDAL_VSI,
    METADATA_OFFSET,
    METADATA_SIZE,
    ZIP_MAX_GAP_SIZE,
)
from tacoreader._exceptions import TacoFormatError
from tacoreader._logging import get_logger
from tacoreader._remote_io import download_range
from tacoreader._vsi import to_vsi_root
from tacoreader.dataset import TacoDataset
from tacoreader.storage.base import TacoBackend

logger = get_logger(__name__)


@lru_cache(maxsize=64)
def _read_taco_header_cached(path: str) -> list[tuple[int, int]]:
    """
    Read and cache TACO_HEADER (256 bytes).

    Cached because headers are tiny and frequently accessed
    when exploring remote datasets.

    Args:
        path: Path to .tacozip file (local or remote)

    Returns:
        List of (offset, size) tuples for embedded files
    """
    if Path(path).exists():
        return tacozip.read_header(path)

    # Remote: read first 256 bytes only
    header_bytes = download_range(path, 0, 256)
    return tacozip.read_header(header_bytes)


class ZipBackend(TacoBackend):
    """
    Backend for .tacozip format.

    Uses TACO_HEADER (256-byte structure at start) for direct offset-based
    access to embedded files without full ZIP extraction.

    All metadata loaded into memory, no temp files created.
    Remote ZIPs: groups consecutive files to minimize HTTP requests.
    """

    @property
    def format_name(self) -> str:
        return "zip"

    def load(self, path: str) -> TacoDataset:
        """
        Load ZIP dataset with grouped requests.

        Remote optimization: groups files with gaps <4MB to minimize HTTP requests.
        Local: reads individually with mmap.
        """
        t_start = time.time()
        logger.debug(f"Loading ZIP from {path}")

        # Read TACO_HEADER to get offsets (uses cache)
        header = self._read_taco_header(path)
        if not header:
            raise TacoFormatError(f"Empty TACO_HEADER in {path}")
        logger.debug(f"Read TACO_HEADER with {len(header)} entries")

        # Download all files
        all_files_data = self._download_all_files(path, header)

        # Parse COLLECTION.json (last file in header)
        collection_bytes = all_files_data[len(header) - 1]
        collection = self._parse_collection_json(collection_bytes, path)
        logger.debug("Parsed COLLECTION.json")

        # Setup DuckDB and register tables
        db = self._setup_duckdb_connection()
        level_ids = self._register_parquet_tables(db, all_files_data, header)

        if not level_ids:
            raise TacoFormatError(f"No metadata levels found in ZIP: {path}")

        # Finalize dataset
        root_path = to_vsi_root(path)
        dataset = self._finalize_dataset(db, path, root_path, collection, level_ids)

        total_time = time.time() - t_start
        logger.info(f"Loaded ZIP in {total_time:.2f}s")
        return dataset

    def read_collection(self, path: str) -> dict[str, Any]:
        """
        Read COLLECTION.json from ZIP file.

        COLLECTION.json is always the last entry in TACO_HEADER.
        Uses offset-based reading to avoid extracting entire ZIP.
        """
        header = self._read_taco_header(path)

        if not header:
            raise TacoFormatError(f"Empty TACO_HEADER in {path}")

        # COLLECTION.json is always last entry
        collection_offset, collection_size = header[-1]

        if collection_size == 0:
            raise TacoFormatError(f"Empty COLLECTION.json in {path}")

        is_local = Path(path).exists()

        if is_local:
            with open(path, "rb") as f:
                f.seek(collection_offset)
                collection_bytes = f.read(collection_size)
        else:
            collection_bytes = download_range(path, collection_offset, collection_size)

        return self._parse_collection_json(collection_bytes, path)

    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        level_ids: list[int],
        root_path: str,
    ) -> None:
        """
        Create DuckDB views with GDAL VSI paths.

        Views simply expose parquet data with GDAL VSI paths for raster access.
        Path format: /vsisubfile/{offset}_{size},{zip_path}
        """
        filter_clause = self._build_view_filter()

        for level_id in level_ids:
            table_name = f"{LEVEL_VIEW_PREFIX}{level_id}{LEVEL_TABLE_SUFFIX}"
            view_name = f"{LEVEL_VIEW_PREFIX}{level_id}"
            db.execute(
                f"""
                CREATE VIEW {view_name} AS
                SELECT
                  *,
                  '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
                  "{METADATA_SIZE}" || ',{root_path}' as "{METADATA_GDAL_VSI}"
                FROM {table_name}
                WHERE {filter_clause}
            """
            )

    def _read_taco_header(self, path: str) -> list[tuple[int, int]]:
        """
        Read TACO_HEADER with caching.

        Delegates to cached function for efficiency.
        """
        return _read_taco_header_cached(path)

    def _download_all_files(
        self, path: str, header: list[tuple[int, int]]
    ) -> dict[int, bytes]:
        """Download all files from ZIP using grouped requests."""
        is_local = Path(path).exists()

        # Group files by proximity for efficient batch downloading
        if is_local:
            file_groups = [
                [(i, offset, size)] for i, (offset, size) in enumerate(header)
            ]
        else:
            file_groups = self._group_files_by_proximity(header)
            logger.debug(
                f"Grouped {len(header)} files into {len(file_groups)} request(s)"
            )

        # Download and parse all files
        all_files_data = {}

        for group in file_groups:
            if not group:
                continue

            # Calculate range for this group
            first_idx, first_offset, first_size = group[0]
            last_idx, last_offset, last_size = group[-1]
            range_start = first_offset
            range_end = last_offset + last_size
            range_size = range_end - range_start

            logger.debug(
                f"Downloading group: {len(group)} files, {range_size/1024:.1f}KB"
            )

            # Download entire group as single blob
            blob = (
                self._read_blob_local(path, range_start, range_size)
                if is_local
                else self._read_blob_remote(path, range_start, range_size)
            )

            # Extract individual files from blob
            for idx, offset, size in group:
                relative_offset = offset - range_start
                all_files_data[idx] = blob[relative_offset : relative_offset + size]

        return all_files_data

    def _register_parquet_tables(
        self,
        db: duckdb.DuckDBPyConnection,
        all_files_data: dict[int, bytes],
        header: list[tuple[int, int]],
    ) -> list[int]:
        """Register Parquet tables in DuckDB from downloaded files."""
        level_ids = []

        for i in range(len(header) - 1):  # All except last (COLLECTION.json)
            if i not in all_files_data:
                continue

            parquet_bytes = all_files_data[i]
            if len(parquet_bytes) == 0:
                continue

            # Load to PyArrow and register in DuckDB
            reader = pa.BufferReader(parquet_bytes)
            arrow_table = pq.read_table(reader)

            table_name = f"{LEVEL_VIEW_PREFIX}{i}{LEVEL_TABLE_SUFFIX}"
            db.register(table_name, arrow_table)
            level_ids.append(i)

            logger.debug(
                f"Registered {table_name} in DuckDB ({len(parquet_bytes)} bytes)"
            )

        return level_ids

    def _should_group(
        self,
        current_file: tuple[int, int, int],
        next_file: tuple[int, int, int],
        max_gap: int = ZIP_MAX_GAP_SIZE,
    ) -> bool:
        """
        Determine if two files should be grouped in same request.

        Strategy:
        1. Gap must be < max_gap (4MB default)
        2. Gap must be < 50% of useful data size

        This prevents downloading excessive unused data when files are small
        but far apart.

        Args:
            current_file: (idx, offset, size) of current file
            next_file: (idx, offset, size) of next file
            max_gap: Maximum allowed gap in bytes

        Returns:
            True if files should be grouped, False otherwise

        Examples:
            file1(1KB) -- [3.9MB gap] -- file2(1KB)
            gap=3.9MB, total_size=2KB, ratio=1950%
            -> DON'T group (gap > 50% of useful data)

            file1(5MB) -- [2MB gap] -- file2(5MB)
            gap=2MB, total_size=10MB, ratio=20%
            -> DO group (gap < 50% of useful data)
        """
        _, current_offset, current_size = current_file
        _, next_offset, next_size = next_file

        gap = next_offset - (current_offset + current_size)
        total_useful_size = current_size + next_size

        # Both conditions must be true to group files:
        # 1. Gap must be < max_gap (hard limit)
        # 2. Gap must be < 50% of useful data (efficiency)
        return gap < max_gap and gap <= total_useful_size * 0.5

    def _group_files_by_proximity(
        self, header: list[tuple[int, int]]
    ) -> list[list[tuple[int, int, int]]]:
        """
        Group files to minimize HTTP requests while avoiding excessive waste.

        Files grouped if:
        - Gap < ZIP_MAX_GAP_SIZE (4MB)
        - Gap < 50% of useful data size

        Returns:
            List of groups: [[(idx, offset, size), ...], ...]
        """
        if not header:
            return []

        groups = []
        current_group: list[tuple[int, int, int]] = []

        for i, (offset, size) in enumerate(header):
            if size == 0:
                continue

            if not current_group:
                current_group.append((i, offset, size))
            else:
                last_file = current_group[-1]
                next_file = (i, offset, size)

                if self._should_group(last_file, next_file):
                    current_group.append(next_file)
                else:
                    # Start new group
                    groups.append(current_group)
                    current_group = [next_file]

        if current_group:
            groups.append(current_group)

        return groups

    def _read_blob_local(self, path: str, offset: int, size: int) -> bytes:
        """Read blob from local ZIP with mmap."""
        with (
            open(path, "rb") as f,
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm,
        ):
            return bytes(mm[offset : offset + size])

    def _read_blob_remote(self, path: str, offset: int, size: int) -> bytes:
        """Read blob from remote ZIP via HTTP range request."""
        return download_range(path, offset, size)
