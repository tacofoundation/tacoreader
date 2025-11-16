"""
TacoCat backend for consolidated TACO datasets.

This module provides the TacoCatBackend class for reading consolidated
datasets that combine metadata from multiple .tacozip files into a single
high-performance file. TacoCat enables querying terabytes of data distributed
across hundreds of ZIPs without opening each file individually.

TacoCat Format Specification:
    __TACOCAT__ (fixed 128-byte header + data sections)
    ┌─────────────────────────────────────────────┐
    │ HEADER (16 bytes)                           │
    │   MAGIC: "TACOCAT\x00" (8 bytes)            │
    │   VERSION: uint32 = 1 (4 bytes)             │
    │   MAX_DEPTH: uint32 = 0-5 (4 bytes)         │
    ├─────────────────────────────────────────────┤
    │ INDEX BLOCK (112 bytes)                     │
    │   7 entries × 16 bytes:                     │
    │   - LEVEL0_OFFSET (8) + SIZE (8)            │
    │   - LEVEL1_OFFSET (8) + SIZE (8)            │
    │   - LEVEL2_OFFSET (8) + SIZE (8)            │
    │   - LEVEL3_OFFSET (8) + SIZE (8)            │
    │   - LEVEL4_OFFSET (8) + SIZE (8)            │
    │   - LEVEL5_OFFSET (8) + SIZE (8)            │
    │   - COLLECTION_OFFSET (8) + SIZE (8)        │
    ├─────────────────────────────────────────────┤
    │ DATA SECTION (starts at byte 128)           │
    │   - CONSOLIDATED_LEVEL0.parquet             │
    │   - CONSOLIDATED_LEVEL1.parquet             │
    │   - ... (only existing levels)              │
    │   - COLLECTION.json                         │
    └─────────────────────────────────────────────┘

Key features:
- Consolidates metadata from 100+ ZIPs into single file
- Queries across terabytes without opening all ZIPs
- Fixed-size header (128 bytes) for instant access
- internal:source_file column identifies origin ZIP
- All-in-memory loading (no temp files)
- DuckDB-optimized Parquet configuration

Main classes:
    TacoCatHeader: Binary header parser
    TacoCatBackend: Backend implementation for TacoCat format

Example:
    >>> from tacoreader.backends.tacocat import TacoCatBackend
    >>> backend = TacoCatBackend()
    >>> dataset = backend.load("__TACOCAT__")
    >>> # Query across all consolidated ZIPs
    >>> peru = dataset.sql("SELECT * FROM data WHERE country = 'Peru'")
    >>> df = peru.data
"""

import json
import mmap
import struct
import time
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from tacoreader.backends.base import TacoBackend
from tacoreader.io import download_bytes
from tacoreader._constants import (
    TACOCAT_HEADER_SIZE,
    TACOCAT_INDEX_ENTRY_SIZE,
    TACOCAT_MAGIC,
    TACOCAT_MAX_LEVELS,
    TACOCAT_TOTAL_HEADER_SIZE,
    TACOCAT_VERSION,
)
from tacoreader._logging import get_logger
from tacoreader.dataset import TacoDataset
from tacoreader.schema import PITSchema
from tacoreader.utils.vsi import to_vsi_root

logger = get_logger(__name__)


# ============================================================================
# TACOCAT HEADER
# ============================================================================


class TacoCatHeader:
    """
    Parsed TacoCat file header.

    Parses the fixed 128-byte header at the beginning of every TacoCat file.
    The header contains magic number validation, version info, and an index
    of (offset, size) pairs for direct access to consolidated metadata files.

    Attributes:
        version: TacoCat format version (currently 1)
        max_depth: Maximum hierarchy depth in dataset (0-5)
        level_index: Dict mapping level number to (offset, size) tuple
        collection_offset: Byte offset of COLLECTION.json
        collection_size: Size of COLLECTION.json in bytes

    Raises:
        ValueError: If magic number is invalid or version unsupported

    Example:
        >>> with open("__TACOCAT__", "rb") as f:
        ...     header_bytes = f.read(128)
        >>> header = TacoCatHeader(header_bytes)
        >>> print(header.version)
        1
        >>> print(header.max_depth)
        2
        >>> print(header.level_index)
        {0: (128, 5000), 1: (5128, 3000)}
    """

    def __init__(self, header_bytes: bytes):
        """
        Parse TacoCat header from raw bytes.

        Args:
            header_bytes: First 128 bytes of TacoCat file

        Raises:
            ValueError: If magic number is invalid
            ValueError: If version is unsupported
            struct.error: If header is malformed
        """
        if len(header_bytes) < TACOCAT_TOTAL_HEADER_SIZE:
            raise ValueError(
                f"Header too short: {len(header_bytes)} bytes\n"
                f"Expected: {TACOCAT_TOTAL_HEADER_SIZE} bytes"
            )

        # Parse magic number (8 bytes)
        magic = header_bytes[0:8]
        if magic != TACOCAT_MAGIC:
            raise ValueError(
                f"Invalid TacoCat magic number: {magic!r}\n"
                f"Expected: {TACOCAT_MAGIC!r}\n"
                f"This may not be a valid TacoCat file."
            )

        # Parse version and max_depth (4 bytes each)
        self.version = struct.unpack("<I", header_bytes[8:12])[0]
        self.max_depth = struct.unpack("<I", header_bytes[12:16])[0]

        if self.version != TACOCAT_VERSION:
            raise ValueError(
                f"Unsupported TacoCat version: {self.version}\n"
                f"This reader supports version: {TACOCAT_VERSION}\n"
                f"Please upgrade tacoreader to read this file."
            )

        if self.max_depth > 5:
            raise ValueError(
                f"Invalid max_depth: {self.max_depth}\n" f"Valid range: 0-5"
            )

        # Parse level index (6 levels + collection = 7 entries)
        self.level_index = {}

        for level in range(TACOCAT_MAX_LEVELS):
            offset_pos = TACOCAT_HEADER_SIZE + (level * TACOCAT_INDEX_ENTRY_SIZE)
            offset = struct.unpack("<Q", header_bytes[offset_pos : offset_pos + 8])[0]
            size = struct.unpack("<Q", header_bytes[offset_pos + 8 : offset_pos + 16])[
                0
            ]

            # Only store non-empty entries
            if offset > 0 and size > 0:
                self.level_index[level] = (offset, size)

        # Parse collection entry (7th entry)
        col_offset_pos = TACOCAT_HEADER_SIZE + (TACOCAT_MAX_LEVELS * TACOCAT_INDEX_ENTRY_SIZE)
        self.collection_offset = struct.unpack(
            "<Q", header_bytes[col_offset_pos : col_offset_pos + 8]
        )[0]
        self.collection_size = struct.unpack(
            "<Q", header_bytes[col_offset_pos + 8 : col_offset_pos + 16]
        )[0]

        if self.collection_offset == 0 or self.collection_size == 0:
            raise ValueError("TacoCat header missing COLLECTION.json entry")

    def __repr__(self) -> str:
        """String representation of header."""
        return (
            f"TacoCatHeader(version={self.version}, max_depth={self.max_depth}, "
            f"levels={list(self.level_index.keys())}, "
            f"collection_offset={self.collection_offset})"
        )


# ============================================================================
# TACOCAT BACKEND
# ============================================================================


class TacoCatBackend(TacoBackend):
    """
    Backend for TacoCat consolidated format.

    Handles reading TACO datasets consolidated from multiple .tacozip files.
    TacoCat consolidates metadata into a single file while preserving
    references to original ZIP files via internal:source_file column.

    This enables querying across hundreds of ZIPs without opening each file,
    dramatically improving performance for large distributed datasets.

    All data is loaded directly into memory without creating temporary files,
    making it efficient for datasets up to ~1GB.

    Attributes:
        format_name: Returns "tacocat"

    Example:
        >>> backend = TacoCatBackend()
        >>> dataset = backend.load("datasets/__TACOCAT__")
        >>>
        >>> # Query consolidated metadata
        >>> peru = dataset.sql("SELECT * FROM data WHERE country = 'Peru'")
        >>> df = peru.data
    """

    @property
    def format_name(self) -> str:
        """
        Format identifier for TacoCat backend.

        Returns:
            "tacocat" string constant
        """
        return "tacocat"

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================

    def load(self, path: str) -> TacoDataset:
        """
        Load TacoCat dataset entirely in memory.

        Unlike other backends, TacoCat loads everything into memory without
        creating temporary files. This is efficient because TacoCat files
        are typically small (metadata only, no actual raster data).

        Args:
            path: Path to __TACOCAT__ file (local or remote)

        Returns:
            TacoDataset with lazy SQL interface

        Raises:
            ValueError: If TacoCat file is invalid or corrupted
            IOError: If file cannot be read

        Example:
            >>> backend = TacoCatBackend()
            >>> dataset = backend.load("__TACOCAT__")
            >>> print(dataset.id)
            'consolidated-dataset'
        """
        t_start = time.time()
        
        logger.debug(f"Loading TacoCat from {path}")

        # Step 1: Download entire TacoCat file to memory
        t_download = time.time()
        full_bytes = self._get_full_file(path)
        download_time = time.time() - t_download
        file_size_mb = len(full_bytes) / (1024 * 1024)
        logger.debug(f"Downloaded {file_size_mb:.1f}MB in {download_time:.2f}s ({file_size_mb/download_time:.1f}MB/s)")

        # Step 2: Parse header from memory
        header = TacoCatHeader(full_bytes[0:TACOCAT_TOTAL_HEADER_SIZE])
        logger.debug(f"Parsed header: version={header.version}, max_depth={header.max_depth}, levels={list(header.level_index.keys())}")

        # Step 3: Parse COLLECTION.json from memory
        collection_bytes = full_bytes[
            header.collection_offset : header.collection_offset + header.collection_size
        ]
        try:
            collection = json.loads(collection_bytes)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid COLLECTION.json in TacoCat {path}: {e.msg}", e.doc, e.pos
            )
        
        logger.debug("Parsed COLLECTION.json")

        # Step 4: Create DuckDB connection
        db = duckdb.connect(":memory:")

        # Step 5: Load spatial extension
        try:
            db.execute("INSTALL spatial")
            db.execute("LOAD spatial")
            logger.debug("Loaded DuckDB spatial extension")
        except Exception as e:
            logger.debug(f"Spatial extension not available: {e}")

        # Step 6: Register Parquet tables directly from memory
        for level_id, (offset, size) in header.level_index.items():
            if size == 0:
                continue

            # Extract parquet bytes from memory
            parquet_bytes = full_bytes[offset : offset + size]

            # Read with PyArrow from bytes
            reader = pa.BufferReader(parquet_bytes)
            arrow_table = pq.read_table(reader)

            # Register table in DuckDB (no disk I/O)
            table_name = f"level{level_id}_table"
            db.register(table_name, arrow_table)
            
            logger.debug(f"Registered {table_name} in DuckDB ({size} bytes)")

        if not header.level_index:
            raise ValueError(
                f"No metadata levels found in TacoCat: {path}\n"
                f"TacoCat appears to be empty or malformed."
            )

        # Step 7: Convert path to VSI format
        root_path = to_vsi_root(path)

        # Step 8: Setup DuckDB views with VSI paths
        self.setup_duckdb_views(db, header.level_index.keys(), root_path)
        logger.debug("Created DuckDB views")

        # Step 9: Create 'data' view
        db.execute("CREATE VIEW data AS SELECT * FROM level0")

        # Step 10: Extract PIT schema
        schema = PITSchema(collection["taco:pit_schema"])

        # Step 11: Construct TacoDataset
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
        logger.info(f"Loaded TacoCat in {total_time:.2f}s")
        
        return dataset

    def read_collection(self, path: str) -> dict[str, Any]:
        """
        Read COLLECTION.json from TacoCat file.

        COLLECTION.json is stored at the offset specified in the TacoCat
        header's 7th index entry. This is the consolidated collection
        metadata from all source ZIPs.

        Args:
            path: Path to __TACOCAT__ file (local or remote)

        Returns:
            Dictionary containing consolidated COLLECTION.json content

        Raises:
            ValueError: If header cannot be read or parsed
            json.JSONDecodeError: If COLLECTION.json is invalid

        Example:
            >>> backend = TacoCatBackend()
            >>> collection = backend.read_collection("__TACOCAT__")
            >>> print(collection["id"])
            'consolidated-dataset'
            >>> print(collection["_tacocat"]["num_datasets"])
            42
        """
        # Get full file
        full_bytes = self._get_full_file(path)

        # Parse header
        header = TacoCatHeader(full_bytes[0:TACOCAT_TOTAL_HEADER_SIZE])

        if header.collection_size == 0:
            raise ValueError(f"Empty COLLECTION.json in TacoCat: {path}")

        # Extract collection from memory
        collection_bytes = full_bytes[
            header.collection_offset : header.collection_offset + header.collection_size
        ]

        try:
            return json.loads(collection_bytes)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid COLLECTION.json in TacoCat {path}: {e.msg}", e.doc, e.pos
            )

    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        level_ids: list[int],
        root_path: str,
    ) -> None:
        """
        Create DuckDB views with GDAL VSI paths for TacoCat format.

        Each view references a table already registered in DuckDB and adds
        a computed 'internal:gdal_vsi' column that constructs /vsisubfile/
        paths using internal:source_file column:

            /vsisubfile/{offset}_{size},{base_path}{source_file}

        This allows samples to point to their specific origin ZIP file
        while querying across all consolidated metadata.

        Args:
            db: DuckDB connection with tables already registered
            level_ids: List of level IDs that have tables registered
            root_path: VSI root path to TacoCat file (e.g., /vsis3/bucket/datasets/__TACOCAT__)

        Example:
            >>> import duckdb
            >>> db = duckdb.connect(":memory:")
            >>> # ... register tables ...
            >>> backend = TacoCatBackend()
            >>> backend.setup_duckdb_views(db, [0, 1, 2], "/vsis3/bucket/data/__TACOCAT__")
            >>>
            >>> # Query shows VSI paths to individual ZIPs
            >>> result = db.execute('''
            ...     SELECT "internal:gdal_vsi", "internal:source_file"
            ...     FROM level0 LIMIT 1
            ... ''').fetchone()
            >>> print(result)
            ('/vsisubfile/1024_5000,/vsis3/bucket/data/part0001.tacozip', 'part0001.tacozip')
        """
        base_path = self._extract_base_path(root_path)

        for level_id in level_ids:
            table_name = f"level{level_id}_table"
            view_name = f"level{level_id}"
            
            db.execute(
                f"""
                CREATE VIEW {view_name} AS 
                SELECT *,
                  '/vsisubfile/' || "internal:offset" || '_' || 
                  "internal:size" || ',{base_path}' || "internal:source_file"
                  as "internal:gdal_vsi"
                FROM {table_name}
                WHERE id NOT LIKE '__TACOPAD__%'
            """
            )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _get_full_file(self, path: str) -> bytes:
        """
        Get entire TacoCat file as bytes.
        
        Local: Read with mmap for efficiency
        Remote: Simple download with download_bytes()
        
        Args:
            path: Path to __TACOCAT__ file (local or remote)
            
        Returns:
            Complete file contents as bytes
        """
        if Path(path).exists():
            # Local file: read with mmap
            logger.debug(f"Reading local TacoCat file: {path}")
            with open(path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return bytes(mm[:])
        else:
            # Remote file: simple download
            try:
                logger.debug(f"Downloading remote TacoCat file: {path}")
                return download_bytes(path)
            except Exception as e:
                raise OSError(f"Failed to download TacoCat from {path}: {e}")

    def _extract_base_path(self, root_path: str) -> str:
        """
        Extract base directory path from __TACOCAT__ path.

        The base path is the directory containing both __TACOCAT__ and the
        source .tacozip files. This is used to construct VSI paths that
        point to individual ZIP files.

        Args:
            root_path: Full VSI path to __TACOCAT__ file

        Returns:
            VSI path to parent directory (with trailing /)

        Example:
            >>> backend = TacoCatBackend()
            >>> base = backend._extract_base_path("/vsis3/bucket/data/__TACOCAT__")
            >>> print(base)
            '/vsis3/bucket/data/'
            >>>
            >>> base = backend._extract_base_path("s3://bucket/data/")
            >>> print(base)
            's3://bucket/data/'
        """
        if root_path.endswith("__TACOCAT__"):
            # Remove __TACOCAT__ filename
            base_path = root_path[: -len("__TACOCAT__")]
        else:
            # Already a directory path
            base_path = root_path if root_path.endswith("/") else root_path + "/"

        return base_path