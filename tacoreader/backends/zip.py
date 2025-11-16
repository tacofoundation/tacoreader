"""
ZIP backend for TACO datasets.

Reads .tacozip format files with offset-based access via TACO_HEADER.
Optimized for single-file distribution and cloud storage with range reads.

Main class:
    ZipBackend: Backend for .tacozip format
"""

import json
import mmap
import time
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import tacozip

from tacoreader.backends.base import TacoBackend
from tacoreader.io import download_range
from tacoreader._constants import ZIP_MAX_GAP_SIZE, PADDING_PREFIX
from tacoreader._logging import get_logger
from tacoreader.dataset import TacoDataset
from tacoreader.schema import PITSchema
from tacoreader.utils.vsi import to_vsi_root

logger = get_logger(__name__)


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

        # Read TACO_HEADER to get offsets
        header = self._read_taco_header(path)

        if not header:
            raise ValueError(f"Empty TACO_HEADER in {path}")

        logger.debug(f"Read TACO_HEADER with {len(header)} entries")

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
        all_files_data = {}  # {index: bytes}

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
            if is_local:
                blob = self._read_blob_local(path, range_start, range_size)
            else:
                blob = self._read_blob_remote(path, range_start, range_size)

            # Extract individual files from blob
            for idx, offset, size in group:
                relative_offset = offset - range_start
                file_bytes = blob[relative_offset : relative_offset + size]
                all_files_data[idx] = file_bytes

        # Parse COLLECTION.json (last file in header)
        collection_bytes = all_files_data[len(header) - 1]

        try:
            collection = json.loads(collection_bytes)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid COLLECTION.json in {path}: {e.msg}", e.doc, e.pos
            )

        logger.debug("Parsed COLLECTION.json")

        # Setup DuckDB with spatial
        db = duckdb.connect(":memory:")
        try:
            db.execute("INSTALL spatial")
            db.execute("LOAD spatial")
            logger.debug("Loaded DuckDB spatial extension")
        except Exception as e:
            logger.debug(f"Spatial extension not available: {e}")

        # Register Parquet tables from parsed bytes
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

            table_name = f"level{i}_table"
            db.register(table_name, arrow_table)
            level_ids.append(i)

            logger.debug(
                f"Registered {table_name} in DuckDB ({len(parquet_bytes)} bytes)"
            )

        if not level_ids:
            raise ValueError(f"No metadata levels found in ZIP: {path}")

        # Setup views and create dataset
        root_path = to_vsi_root(path)
        self.setup_duckdb_views(db, level_ids, root_path)
        logger.debug("Created DuckDB views")

        db.execute("CREATE VIEW data AS SELECT * FROM level0")
        schema = PITSchema(collection["taco:pit_schema"])

        dataset = TacoDataset.model_construct(
            id=collection["id"],
            version=collection.get("dataset_version", "unknown"),
            description=collection.get("description", ""),
            tasks=collection.get("tasks", []),
            extent=collection.get("extent", {}),
            providers=collection.get("providers", []),
            licenses=collection.get("licenses", []),
            title=collection.get("title"),
            curators=collection.get("curators"),
            keywords=collection.get("keywords"),
            pit_schema=schema,
            _path=path,
            _format=self.format_name,
            _collection=collection,
            _duckdb=db,
            _view_name="data",
            _root_path=root_path,
        )

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
            raise ValueError(f"Empty TACO_HEADER in {path}")

        # COLLECTION.json is always last entry
        collection_offset, collection_size = header[-1]

        if collection_size == 0:
            raise ValueError(f"Empty COLLECTION.json in {path}")

        is_local = Path(path).exists()

        if is_local:
            with open(path, "rb") as f:
                f.seek(collection_offset)
                collection_bytes = f.read(collection_size)
        else:
            collection_bytes = download_range(path, collection_offset, collection_size)

        try:
            return json.loads(collection_bytes)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid COLLECTION.json in {path}: {e.msg}", e.doc, e.pos
            )

    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        level_ids: list[int],
        root_path: str,
    ) -> None:
        """
        Create DuckDB views with GDAL VSI paths for ZIP format.

        Path format: /vsisubfile/{offset}_{size},{zip_path}
        Enables GDAL to read embedded files without extraction.
        """
        for level_id in level_ids:
            table_name = f"level{level_id}_table"
            view_name = f"level{level_id}"

            db.execute(
                f"""
                CREATE VIEW {view_name} AS 
                SELECT *,
                  '/vsisubfile/' || "internal:offset" || '_' || 
                  "internal:size" || ',{root_path}' as "internal:gdal_vsi"
                FROM {table_name}
                WHERE id NOT LIKE '{PADDING_PREFIX}%'
            """
            )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _group_files_by_proximity(
        self, header: list[tuple[int, int]]
    ) -> list[list[tuple[int, int, int]]]:
        """
        Group files to minimize HTTP requests.

        Files grouped if gap < ZIP_MAX_GAP_SIZE (4MB).
        Avoids downloading large unused data while reducing request count.

        Returns:
            List of groups: [[(idx, offset, size), ...], ...]

        Example:
            header = [(1000, 50000), (51000, 30000), (5000000, 20000)]
            -> [[(0, 1000, 50000), (1, 51000, 30000)], [(2, 5000000, 20000)]]
            First two grouped (gap=1000), third separate (gap=4.9MB)
        """
        if not header:
            return []

        groups = []
        current_group = []

        for i, (offset, size) in enumerate(header):
            if size == 0:
                continue

            if not current_group:
                current_group.append((i, offset, size))
            else:
                # Check gap from last file
                last_idx, last_offset, last_size = current_group[-1]
                last_end = last_offset + last_size
                gap = offset - last_end

                if gap < ZIP_MAX_GAP_SIZE:
                    current_group.append((i, offset, size))
                else:
                    # Large gap: start new group
                    groups.append(current_group)
                    current_group = [(i, offset, size)]

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

    def _read_taco_header(self, path: str) -> list[tuple[int, int]]:
        """
        Read TACO_HEADER from beginning of ZIP.

        TACO_HEADER: 256-byte structure with array of (offset, size) pairs.
        Last entry is always COLLECTION.json.

        Returns:
            List of (offset, size) tuples for embedded files
        """
        if Path(path).exists():
            return tacozip.read_header(path)

        # Remote: read first 256 bytes
        header_bytes = download_range(path, 0, 256)
        return tacozip.read_header(header_bytes)
