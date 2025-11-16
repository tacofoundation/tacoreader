"""
ZIP backend for TACO datasets.

This module provides the ZipBackend class for reading .tacozip format files.
ZIP format is optimized for single-file distribution and supports both local
and cloud storage with efficient range-based reads.

ZIP Structure:
    dataset.tacozip
    ├── TACO_HEADER (157 bytes with offset/size array)
    ├── DATA/
    │   ├── sample_001.tif (if level 0 = FILEs)
    │   OR
    │   ├── folder_A/ (if level 0 = FOLDERs)
    │   │   ├── __meta__ (Parquet with offset/size)
    │   │   ├── nested_001.tif
    │   │   └── nested_002.tif
    ├── METADATA/
    │   ├── level0.parquet (with internal:parent_id, offset, size)
    │   └── level1.parquet (with internal:parent_id, offset, size)
    └── COLLECTION.json

Main class:
    ZipBackend: Backend implementation for .tacozip format

Example:
    >>> from tacoreader.backends.zip import ZipBackend
    >>> backend = ZipBackend()
    >>> dataset = backend.load("data.tacozip")
    >>> print(dataset.schema.root["n"])
    1000
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
from tacoreader._constants import ZIP_MAX_GAP_SIZE
from tacoreader._logging import get_logger
from tacoreader.dataset import TacoDataset
from tacoreader.schema import PITSchema
from tacoreader.utils.vsi import to_vsi_root

logger = get_logger(__name__)

# ============================================================================
# ZIP BACKEND
# ============================================================================


class ZipBackend(TacoBackend):
    """
    Backend for .tacozip format.

    Handles reading TACO datasets stored as single compressed ZIP files.
    Uses TACO_HEADER for direct offset-based access to embedded files
    without full ZIP extraction.

    The ZIP format stores all metadata as Parquet files and data files
    with byte offsets, enabling efficient random access via GDAL's
    /vsisubfile/ virtual file system.

    All metadata is loaded directly into memory without creating temporary
    files, making it efficient for typical datasets.

    Attributes:
        format_name: Returns "zip"

    Example:
        >>> backend = ZipBackend()
        >>> dataset = backend.load("s3://bucket/data.tacozip")
        >>> peru = dataset.sql("SELECT * FROM data WHERE country = 'Peru'")
        >>> df = peru.data
    """

    @property
    def format_name(self) -> str:
        """
        Format identifier for ZIP backend.

        Returns:
            "zip" string constant
        """
        return "zip"

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================

    def load(self, path: str) -> TacoDataset:
        """
        Load ZIP dataset efficiently with grouped requests.

        For remote ZIPs, groups consecutive metadata files into batches to
        minimize HTTP requests. Files separated by gaps larger than 4MB are
        fetched separately to avoid downloading unnecessary data.

        Args:
            path: Path to .tacozip file (local or remote)

        Returns:
            TacoDataset with lazy SQL interface

        Raises:
            ValueError: If ZIP is invalid or corrupted
            IOError: If file cannot be read

        Example:
            >>> backend = ZipBackend()
            >>> dataset = backend.load("s3://bucket/data.tacozip")
            >>> print(dataset.id)
            'sentinel2-l2a'
        """
        t_start = time.time()
        
        logger.debug(f"Loading ZIP from {path}")

        # Step 1: Read TACO_HEADER to get offsets
        header = self._read_taco_header(path)
        
        if not header:
            raise ValueError(f"Empty TACO_HEADER in {path}")
        
        logger.debug(f"Read TACO_HEADER with {len(header)} entries")

        is_local = Path(path).exists()

        # Step 2: Group files by proximity for efficient batch downloading
        if is_local:
            # Local files: no grouping needed, read individually
            file_groups = [[(i, offset, size)] for i, (offset, size) in enumerate(header)]
        else:
            # Remote files: group consecutive files with small gaps
            file_groups = self._group_files_by_proximity(header)
            logger.debug(f"Grouped {len(header)} files into {len(file_groups)} request(s)")

        # Step 3: Download and parse all files
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
            
            logger.debug(f"Downloading group: {len(group)} files, {range_size/1024:.1f}KB")
            
            # Download entire group as single blob
            if is_local:
                blob = self._read_blob_local(path, range_start, range_size)
            else:
                blob = self._read_blob_remote(path, range_start, range_size)
            
            # Extract individual files from blob
            for idx, offset, size in group:
                relative_offset = offset - range_start
                file_bytes = blob[relative_offset:relative_offset + size]
                all_files_data[idx] = file_bytes

        # Step 4: Parse COLLECTION.json (last file)
        collection_bytes = all_files_data[len(header) - 1]
        
        try:
            collection = json.loads(collection_bytes)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid COLLECTION.json in {path}: {e.msg}", e.doc, e.pos
            )
        
        logger.debug("Parsed COLLECTION.json")

        # Step 5: Create DuckDB connection
        db = duckdb.connect(":memory:")

        # Step 6: Load spatial extension
        try:
            db.execute("INSTALL spatial")
            db.execute("LOAD spatial")
            logger.debug("Loaded DuckDB spatial extension")
        except Exception as e:
            logger.debug(f"Spatial extension not available: {e}")

        # Step 7: Register Parquet tables from parsed bytes
        level_ids = []
        
        for i in range(len(header) - 1):  # All except last (COLLECTION.json)
            if i not in all_files_data:
                continue
            
            parquet_bytes = all_files_data[i]
            
            if len(parquet_bytes) == 0:
                continue
            
            # Load directly to PyArrow
            reader = pa.BufferReader(parquet_bytes)
            arrow_table = pq.read_table(reader)
            
            # Register in DuckDB
            table_name = f"level{i}_table"
            db.register(table_name, arrow_table)
            level_ids.append(i)
            
            logger.debug(f"Registered {table_name} in DuckDB ({len(parquet_bytes)} bytes)")

        if not level_ids:
            raise ValueError(f"No metadata levels found in ZIP: {path}")

        # Step 8: Convert path to VSI format
        root_path = to_vsi_root(path)

        # Step 9: Setup DuckDB views with VSI paths
        self.setup_duckdb_views(db, level_ids, root_path)
        logger.debug("Created DuckDB views")

        # Step 10: Create 'data' view
        db.execute("CREATE VIEW data AS SELECT * FROM level0")

        # Step 11: Extract PIT schema
        schema = PITSchema(collection["taco:pit_schema"])

        # Step 12: Construct TacoDataset
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

        Args:
            path: Path to .tacozip file (local or remote)

        Returns:
            Dictionary containing COLLECTION.json content

        Raises:
            ValueError: If TACO_HEADER cannot be read
            json.JSONDecodeError: If COLLECTION.json is invalid

        Example:
            >>> backend = ZipBackend()
            >>> collection = backend.read_collection("data.tacozip")
            >>> print(collection["id"])
            'sentinel2-l2a'
        """
        header = self._read_taco_header(path)

        if not header:
            raise ValueError(f"Empty TACO_HEADER in {path}")

        # COLLECTION.json is always the last entry
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

        Each view references a table already registered in DuckDB and adds
        a computed 'internal:gdal_vsi' column that constructs /vsisubfile/
        paths using offset and size columns:

            /vsisubfile/{offset}_{size},{zip_path}

        This enables GDAL to read embedded files without extraction.

        Args:
            db: DuckDB connection with tables already registered
            level_ids: List of level IDs that have tables registered
            root_path: VSI root path to ZIP file (e.g., /vsis3/bucket/data.tacozip)

        Example:
            >>> import duckdb
            >>> db = duckdb.connect(":memory:")
            >>> # ... register tables ...
            >>> backend = ZipBackend()
            >>> backend.setup_duckdb_views(db, [0, 1], "/vsis3/bucket/data.tacozip")
            >>> result = db.execute("SELECT * FROM level0 LIMIT 1").fetchone()
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
                WHERE id NOT LIKE '__TACOPAD__%'
            """
            )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _group_files_by_proximity(
        self, header: list[tuple[int, int]]
    ) -> list[list[tuple[int, int, int]]]:
        """
        Group files by proximity to minimize HTTP requests.

        Files are grouped together if the gap between them is smaller than
        ZIP_MAX_GAP_SIZE (4MB). This reduces the number of HTTP range requests
        for remote ZIPs while avoiding downloading large amounts of unused data.

        Args:
            header: List of (offset, size) tuples from TACO_HEADER

        Returns:
            List of groups, where each group is a list of (index, offset, size) tuples

        Example:
            >>> header = [(1000, 50000), (51000, 30000), (5000000, 20000)]
            >>> groups = backend._group_files_by_proximity(header)
            >>> # Returns: [[(0, 1000, 50000), (1, 51000, 30000)], [(2, 5000000, 20000)]]
            >>> # First two files grouped (gap=0), third separate (gap=4.9MB)
        """
        if not header:
            return []

        groups = []
        current_group = []

        for i, (offset, size) in enumerate(header):
            if size == 0:
                continue

            if not current_group:
                # Start new group
                current_group.append((i, offset, size))
            else:
                # Check gap from last file in current group
                last_idx, last_offset, last_size = current_group[-1]
                last_end = last_offset + last_size
                gap = offset - last_end

                if gap < ZIP_MAX_GAP_SIZE:
                    # Small gap: add to current group
                    current_group.append((i, offset, size))
                else:
                    # Large gap: start new group
                    groups.append(current_group)
                    current_group = [(i, offset, size)]

        # Add last group
        if current_group:
            groups.append(current_group)

        return groups

    def _read_blob_local(self, path: str, offset: int, size: int) -> bytes:
        """
        Read blob of bytes from local ZIP file.

        Uses memory mapping for efficient reading without loading entire file.

        Args:
            path: Path to local .tacozip file
            offset: Starting byte offset
            size: Number of bytes to read

        Returns:
            Blob of bytes

        Raises:
            IOError: If file cannot be read
        """
        with (
            open(path, "rb") as f,
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm,
        ):
            return bytes(mm[offset : offset + size])

    def _read_blob_remote(self, path: str, offset: int, size: int) -> bytes:
        """
        Read blob of bytes from remote ZIP file.

        Uses HTTP range request to download only the specified byte range.

        Args:
            path: Remote path to .tacozip file (s3://, gs://, http://)
            offset: Starting byte offset
            size: Number of bytes to read

        Returns:
            Blob of bytes

        Raises:
            IOError: If remote read fails
        """
        return download_range(path, offset, size)

    def _read_taco_header(self, path: str) -> list[tuple[int, int]]:
        """
        Read TACO_HEADER from beginning of ZIP file.

        TACO_HEADER is a 256-byte structure at the start of every .tacozip
        file containing an array of (offset, size) pairs for each embedded
        file.

        Args:
            path: Path to .tacozip file (local or remote)

        Returns:
            List of (offset, size) tuples for embedded files
            Last entry is always COLLECTION.json

        Raises:
            ValueError: If header cannot be parsed

        Example:
            >>> backend = ZipBackend()
            >>> header = backend._read_taco_header("data.tacozip")
            >>> print(len(header))
            3  # level0, level1, COLLECTION.json
        """
        if Path(path).exists():
            # Local file - use tacozip library directly
            return tacozip.read_header(path)

        # Remote file - read first 256 bytes
        header_bytes = download_range(path, 0, 256)
        return tacozip.read_header(header_bytes)