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

Key features:
- Single compressed file for easy distribution
- TACO_HEADER enables direct offset-based access
- GDAL VSI paths via /vsisubfile/ for embedded files
- Cloud-optimized with obstore range requests
- Memory-mapped reads for local files

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
from pathlib import Path
from typing import Any, cast

import duckdb
import obstore as obs
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import tacozip
from obstore.store import from_url

from tacoreader.backends.base import TacoBackend

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
            store = from_url(path)
            collection_bytes = obs.get_range(
                store, "", start=collection_offset, length=collection_size
            )
            collection_bytes = bytes(collection_bytes)

        try:
            return json.loads(collection_bytes)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid COLLECTION.json in {path}: {e.msg}", e.doc, e.pos
            )

    def cache_metadata_files(self, path: str, cache_dir: Path) -> dict[str, str]:
        """
        Extract and cache metadata Parquet files from ZIP.

        Reads level0.parquet through level5.parquet (up to 6 levels)
        from TACO_HEADER and writes them to cache directory. Uses
        memory-mapped reads for local files and range requests for
        remote files.

        Args:
            path: Path to .tacozip file (local or remote)
            cache_dir: Directory to write cached files

        Returns:
            Dictionary mapping level names to cached file paths
            Example: {"level0": "/tmp/level0.parquet", "level1": ...}

        Raises:
            ValueError: If TACO_HEADER cannot be read
            IOError: If cache directory cannot be created

        Example:
            >>> backend = ZipBackend()
            >>> cache_dir = Path("/tmp/cache")
            >>> files = backend.cache_metadata_files("data.tacozip", cache_dir)
            >>> print(files.keys())
            dict_keys(['level0', 'level1'])
        """
        header = self._read_taco_header(path)

        if not header:
            raise ValueError(f"Empty TACO_HEADER in {path}")

        is_local = Path(path).exists()
        consolidated_files = {}

        # All entries except last (which is COLLECTION.json)
        for i, (offset, size) in enumerate(header[:-1]):
            if size == 0:
                # Skip empty entries
                continue

            if is_local:
                df = self._read_parquet_mmap(path, offset, size)
            else:
                df = self._read_parquet_remote(path, offset, size)

            level_file = cache_dir / f"level{i}.parquet"
            df.write_parquet(level_file)
            consolidated_files[f"level{i}"] = str(level_file)

        return consolidated_files

    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        consolidated_files: dict[str, str],
        root_path: str,
    ) -> None:
        """
        Create DuckDB views with GDAL VSI paths for ZIP format.

        Each view includes a computed 'internal:gdal_vsi' column that
        constructs /vsisubfile/ paths using offset and size columns:

            /vsisubfile/{offset}_{size},{zip_path}

        This enables GDAL to read embedded files without extraction.

        Args:
            db: DuckDB connection for creating views
            consolidated_files: Cached metadata file paths
            root_path: VSI root path to ZIP file (e.g., /vsis3/bucket/data.tacozip)

        Example:
            >>> import duckdb
            >>> db = duckdb.connect(":memory:")
            >>> backend = ZipBackend()
            >>> backend.setup_duckdb_views(db, files, "/vsis3/bucket/data.tacozip")
            >>> result = db.execute("SELECT * FROM level0 LIMIT 1").fetchone()
        """
        for level_key, file_path in consolidated_files.items():
            db.execute(
                f"""
                CREATE VIEW {level_key} AS 
                SELECT *,
                  '/vsisubfile/' || "internal:offset" || '_' || 
                  "internal:size" || ',{root_path}' as "internal:gdal_vsi"
                FROM read_parquet('{file_path}')
                WHERE id NOT LIKE '__TACOPAD__%'
            """
            )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

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
        store = from_url(path)
        header_bytes = obs.get_range(store, "", start=0, length=256)
        return tacozip.read_header(bytes(header_bytes))

    def _read_parquet_mmap(self, path: str, offset: int, length: int) -> pl.DataFrame:
        """
        Read Parquet file from local ZIP using memory mapping.

        Uses memory-mapped I/O for efficient reading without loading
        entire ZIP into memory. Extracts Parquet bytes using offset
        and size from TACO_HEADER.

        Args:
            path: Path to local .tacozip file
            offset: Byte offset of Parquet file in ZIP
            length: Size of Parquet file in bytes

        Returns:
            Polars DataFrame with metadata

        Raises:
            IOError: If file cannot be opened or mapped
            pyarrow.lib.ArrowInvalid: If Parquet is corrupted

        Example:
            >>> backend = ZipBackend()
            >>> df = backend._read_parquet_mmap("data.tacozip", 1000, 5000)
            >>> print(df.columns)
            ['id', 'type', 'internal:offset', ...]
        """
        with (
            open(path, "rb") as f,
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm,
        ):
            parquet_bytes = bytes(mm[offset : offset + length])
            reader = pa.BufferReader(parquet_bytes)
            table = pq.read_table(reader)
            return cast(pl.DataFrame, pl.from_arrow(table))

    def _read_parquet_remote(self, path: str, offset: int, length: int) -> pl.DataFrame:
        """
        Read Parquet file from remote ZIP using range requests.

        Uses obstore to perform HTTP range request, fetching only the
        bytes needed for the specific Parquet file without downloading
        the entire ZIP.

        Args:
            path: Remote path to .tacozip file (s3://, gs://, http://)
            offset: Byte offset of Parquet file in ZIP
            length: Size of Parquet file in bytes

        Returns:
            Polars DataFrame with metadata

        Raises:
            IOError: If remote read fails
            pyarrow.lib.ArrowInvalid: If Parquet is corrupted

        Example:
            >>> backend = ZipBackend()
            >>> df = backend._read_parquet_remote(
            ...     "s3://bucket/data.tacozip",
            ...     1000,
            ...     5000
            ... )
            >>> print(len(df))
            1000
        """
        store = from_url(path)
        parquet_bytes = obs.get_range(store, "", start=offset, length=length)
        reader = pa.BufferReader(bytes(parquet_bytes))
        table = pq.read_table(reader)
        return cast(pl.DataFrame, pl.from_arrow(table))