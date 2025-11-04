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
- Cloud-native with optimized range requests
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
    >>> # Access specific ZIP's data
    >>> df = peru.data
"""

import json
import mmap
import struct
from pathlib import Path
from typing import Any, cast

import duckdb
import obstore as obs
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from obstore.store import from_url

from tacoreader.backends.base import TacoBackend

# ============================================================================
# CONSTANTS
# ============================================================================

TACOCAT_MAGIC = b"TACOCAT\x00"
"""Magic number identifying TacoCat format (8 bytes)."""

TACOCAT_VERSION = 1
"""Current TacoCat format version."""

HEADER_SIZE = 16
"""Size of header section (magic + version + max_depth)."""

INDEX_ENTRY_SIZE = 16
"""Size of each index entry (offset + size as uint64)."""

MAX_LEVELS = 6
"""Maximum number of hierarchy levels (0-5) plus collection."""

TOTAL_HEADER_SIZE = 128
"""Total header size: 16 (header) + 112 (7 × 16 index entries)."""


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
        if len(header_bytes) < TOTAL_HEADER_SIZE:
            raise ValueError(
                f"Header too short: {len(header_bytes)} bytes\n"
                f"Expected: {TOTAL_HEADER_SIZE} bytes"
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

        for level in range(MAX_LEVELS):
            offset_pos = 16 + (level * INDEX_ENTRY_SIZE)
            offset = struct.unpack("<Q", header_bytes[offset_pos : offset_pos + 8])[0]
            size = struct.unpack("<Q", header_bytes[offset_pos + 8 : offset_pos + 16])[
                0
            ]

            # Only store non-empty entries
            if offset > 0 and size > 0:
                self.level_index[level] = (offset, size)

        # Parse collection entry (7th entry)
        col_offset_pos = 16 + (MAX_LEVELS * INDEX_ENTRY_SIZE)
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

    Attributes:
        format_name: Returns "tacocat"

    Example:
        >>> backend = TacoCatBackend()
        >>> # Load from local directory
        >>> dataset = backend.load("datasets/__TACOCAT__")
        >>>
        >>> # Load with custom base_path for ZIPs in different location
        >>> from tacoreader import load
        >>> dataset = load("__TACOCAT__", base_path="s3://other-bucket/zips/")
        >>>
        >>> # Query consolidated metadata
        >>> peru = dataset.sql("SELECT * FROM data WHERE country = 'Peru'")
        >>> # Get source ZIP for each sample
        >>> df = dataset.data.select(["id", "internal:source_file"])
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
        header = self._read_tacocat_header(path)

        if header.collection_size == 0:
            raise ValueError(f"Empty COLLECTION.json in TacoCat: {path}")

        if Path(path).exists():
            # Local file
            with open(path, "rb") as f:
                f.seek(header.collection_offset)
                collection_bytes = f.read(header.collection_size)
        else:
            # Remote file
            store = from_url(path)
            collection_bytes = obs.get_range(
                store, "", start=header.collection_offset, length=header.collection_size
            )
            collection_bytes = bytes(collection_bytes)

        try:
            return json.loads(collection_bytes)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid COLLECTION.json in TacoCat {path}: {e.msg}", e.doc, e.pos
            )

    def cache_metadata_files(self, path: str, cache_dir: Path) -> dict[str, str]:
        """
        Extract and cache consolidated Parquet metadata files from TacoCat.

        Reads consolidated Parquet files for each hierarchy level from
        TacoCat and writes them to cache directory. Each Parquet file
        contains metadata from ALL source ZIPs with internal:source_file
        column identifying the origin ZIP.

        Args:
            path: Path to __TACOCAT__ file (local or remote)
            cache_dir: Directory to write cached files

        Returns:
            Dictionary mapping level names to cached file paths
            Example: {"level0": "/tmp/level0.parquet", "level1": ...}

        Raises:
            ValueError: If TacoCat header cannot be read
            IOError: If cache directory cannot be created

        Example:
            >>> backend = TacoCatBackend()
            >>> cache_dir = Path("/tmp/cache")
            >>> files = backend.cache_metadata_files("__TACOCAT__", cache_dir)
            >>> print(files.keys())
            dict_keys(['level0', 'level1', 'level2'])
            >>>
            >>> # Check source files column
            >>> import polars as pl
            >>> df = pl.read_parquet(files["level0"])
            >>> print(df["internal:source_file"].unique())
            ['part0001.tacozip', 'part0002.tacozip', ...]
        """
        header = self._read_tacocat_header(path)
        is_local = Path(path).exists()

        consolidated_files = {}

        for level, (offset, size) in header.level_index.items():
            if size == 0:
                # Skip empty entries
                continue

            if is_local:
                df = self._read_parquet_mmap(path, offset, size)
            else:
                df = self._read_parquet_remote(path, offset, size)

            level_file = cache_dir / f"level{level}.parquet"
            df.write_parquet(level_file)
            consolidated_files[f"level{level}"] = str(level_file)

        if not consolidated_files:
            raise ValueError(
                f"No metadata levels found in TacoCat: {path}\n"
                f"TacoCat appears to be empty or malformed."
            )

        return consolidated_files

    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        consolidated_files: dict[str, str],
        root_path: str,
    ) -> None:
        """
        Create DuckDB views with GDAL VSI paths for TacoCat format.

        Each view includes a computed 'internal:gdal_vsi' column that
        constructs /vsisubfile/ paths using internal:source_file column:

            /vsisubfile/{offset}_{size},{base_path}{source_file}

        This allows samples to point to their specific origin ZIP file
        while querying across all consolidated metadata.

        Args:
            db: DuckDB connection for creating views
            consolidated_files: Cached metadata file paths
            root_path: VSI root path to TacoCat file (e.g., /vsis3/bucket/datasets/__TACOCAT__)

        Example:
            >>> import duckdb
            >>> db = duckdb.connect(":memory:")
            >>> backend = TacoCatBackend()
            >>> backend.setup_duckdb_views(db, files, "/vsis3/bucket/data/__TACOCAT__")
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

        for level_key, file_path in consolidated_files.items():
            db.execute(
                f"""
                CREATE VIEW {level_key} AS 
                SELECT *,
                  '/vsisubfile/' || "internal:offset" || '_' || 
                  "internal:size" || ',{base_path}' || "internal:source_file"
                  as "internal:gdal_vsi"
                FROM read_parquet('{file_path}')
                WHERE id NOT LIKE '__TACOPAD__%'
            """
            )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

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

    def _read_tacocat_header(self, path: str) -> TacoCatHeader:
        """
        Read and parse TacoCat header from file.

        Reads the first 128 bytes of the TacoCat file and parses them
        into a TacoCatHeader object with structured access to offsets.

        Args:
            path: Path to __TACOCAT__ file (local or remote)

        Returns:
            Parsed TacoCatHeader object

        Raises:
            ValueError: If header is invalid
            IOError: If file cannot be read

        Example:
            >>> backend = TacoCatBackend()
            >>> header = backend._read_tacocat_header("__TACOCAT__")
            >>> print(header.version)
            1
            >>> print(header.level_index)
            {0: (128, 5000), 1: (5128, 3000)}
        """
        if Path(path).exists():
            # Local file
            with open(path, "rb") as f:
                header_bytes = f.read(TOTAL_HEADER_SIZE)
        else:
            # Remote file
            try:
                store = from_url(path)
                result = obs.get_range(store, "", start=0, length=TOTAL_HEADER_SIZE)
                header_bytes = bytes(result)
            except Exception as e:
                raise OSError(f"Failed to read TacoCat header from {path}: {e}")

        if len(header_bytes) < TOTAL_HEADER_SIZE:
            raise ValueError(
                f"Incomplete TacoCat header: {len(header_bytes)} bytes\n"
                f"Expected: {TOTAL_HEADER_SIZE} bytes\n"
                f"File may be corrupted or truncated."
            )

        return TacoCatHeader(header_bytes)

    def _read_parquet_mmap(self, path: str, offset: int, size: int) -> pl.DataFrame:
        """
        Read Parquet file from local TacoCat using memory mapping.

        Uses memory-mapped I/O for efficient reading without loading
        entire TacoCat file into memory. Extracts Parquet bytes using
        offset and size from TacoCat header.

        Args:
            path: Path to local __TACOCAT__ file
            offset: Byte offset of Parquet file in TacoCat
            size: Size of Parquet file in bytes

        Returns:
            Polars DataFrame with consolidated metadata

        Raises:
            IOError: If file cannot be opened or mapped
            pyarrow.lib.ArrowInvalid: If Parquet is corrupted

        Example:
            >>> backend = TacoCatBackend()
            >>> df = backend._read_parquet_mmap("__TACOCAT__", 128, 5000)
            >>> print(df.columns)
            ['id', 'type', 'internal:source_file', ...]
        """
        with (
            open(path, "rb") as f,
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm,
        ):
            parquet_bytes = bytes(mm[offset : offset + size])
            reader = pa.BufferReader(parquet_bytes)
            table = pq.read_table(reader)
            return cast(pl.DataFrame, pl.from_arrow(table))

    def _read_parquet_remote(self, path: str, offset: int, size: int) -> pl.DataFrame:
        """
        Read Parquet file from remote TacoCat using range requests.

        Uses obstore to perform HTTP range request, fetching only the
        bytes needed for the specific Parquet file without downloading
        the entire TacoCat.

        Args:
            path: Remote path to __TACOCAT__ file (s3://, gs://, http://)
            offset: Byte offset of Parquet file in TacoCat
            size: Size of Parquet file in bytes

        Returns:
            Polars DataFrame with consolidated metadata

        Raises:
            IOError: If remote read fails
            pyarrow.lib.ArrowInvalid: If Parquet is corrupted

        Example:
            >>> backend = TacoCatBackend()
            >>> df = backend._read_parquet_remote(
            ...     "s3://bucket/__TACOCAT__",
            ...     128,
            ...     5000
            ... )
            >>> print(len(df))
            10000  # Consolidated from multiple ZIPs
        """
        try:
            store = from_url(path)
            parquet_bytes = obs.get_range(store, "", start=offset, length=size)
            reader = pa.BufferReader(bytes(parquet_bytes))
            table = pq.read_table(reader)
            return cast(pl.DataFrame, pl.from_arrow(table))
        except Exception as e:
            raise OSError(
                f"Failed to read Parquet from TacoCat {path} "
                f"at offset {offset}, size {size}: {e}"
            )