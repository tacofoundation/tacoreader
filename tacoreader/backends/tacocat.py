"""
TacoCat backend for consolidated TACO datasets.

Consolidates metadata from 100+ .tacozip files into a single high-performance file.
Enables querying terabytes of data without opening each ZIP individually.

Format: Fixed 128-byte header + consolidated Parquet metadata + COLLECTION.json
internal:source_file column tracks origin ZIP for each sample.

Main classes:
    TacoCatHeader: Binary header parser
    TacoCatBackend: Backend for TacoCat format
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
    TACOCAT_FILENAME,
    PADDING_PREFIX,
)
from tacoreader._logging import get_logger
from tacoreader.dataset import TacoDataset
from tacoreader.schema import PITSchema
from tacoreader.utils.vsi import to_vsi_root

logger = get_logger(__name__)


class TacoCatHeader:
    """
    Parser for TacoCat binary header.

    Format (128 bytes fixed):
    ┌─────────────────────────────────────────┐
    │ MAGIC: "TACOCAT\x00" (8 bytes)          │
    │ VERSION: uint32 (4 bytes)               │
    │ MAX_DEPTH: uint32 (4 bytes)             │
    ├─────────────────────────────────────────┤
    │ INDEX: 7 entries × (offset + size)      │
    │   - LEVEL0...LEVEL5 (6 entries)         │
    │   - COLLECTION (1 entry)                │
    └─────────────────────────────────────────┘

    Attributes:
        version: TacoCat format version
        max_depth: Maximum hierarchy depth (0-5)
        level_index: {level: (offset, size)}
        collection_offset: Byte offset of COLLECTION.json
        collection_size: Size of COLLECTION.json
    """

    def __init__(self, header_bytes: bytes):
        """Parse TacoCat header from raw bytes."""
        if len(header_bytes) < TACOCAT_TOTAL_HEADER_SIZE:
            raise ValueError(
                f"Header too short: {len(header_bytes)} bytes\n"
                f"Expected: {TACOCAT_TOTAL_HEADER_SIZE} bytes"
            )

        # Validate magic number
        magic = header_bytes[0:8]
        if magic != TACOCAT_MAGIC:
            raise ValueError(
                f"Invalid TacoCat magic: {magic!r}\n" f"Expected: {TACOCAT_MAGIC!r}"
            )

        # Parse version and max_depth
        self.version = struct.unpack("<I", header_bytes[8:12])[0]
        self.max_depth = struct.unpack("<I", header_bytes[12:16])[0]

        if self.version != TACOCAT_VERSION:
            raise ValueError(
                f"Unsupported version: {self.version}\n"
                f"This reader supports: {TACOCAT_VERSION}"
            )

        if self.max_depth > 5:
            raise ValueError(f"Invalid max_depth: {self.max_depth}")

        # Parse level index (6 levels + collection = 7 entries)
        self.level_index = {}

        for level in range(TACOCAT_MAX_LEVELS):
            offset_pos = TACOCAT_HEADER_SIZE + (level * TACOCAT_INDEX_ENTRY_SIZE)
            offset = struct.unpack("<Q", header_bytes[offset_pos : offset_pos + 8])[0]
            size = struct.unpack("<Q", header_bytes[offset_pos + 8 : offset_pos + 16])[
                0
            ]

            if offset > 0 and size > 0:
                self.level_index[level] = (offset, size)

        # Parse collection entry
        col_offset_pos = TACOCAT_HEADER_SIZE + (
            TACOCAT_MAX_LEVELS * TACOCAT_INDEX_ENTRY_SIZE
        )
        self.collection_offset = struct.unpack(
            "<Q", header_bytes[col_offset_pos : col_offset_pos + 8]
        )[0]
        self.collection_size = struct.unpack(
            "<Q", header_bytes[col_offset_pos + 8 : col_offset_pos + 16]
        )[0]

        if self.collection_offset == 0 or self.collection_size == 0:
            raise ValueError("Header missing COLLECTION.json entry")

    def __repr__(self) -> str:
        return (
            f"TacoCatHeader(version={self.version}, max_depth={self.max_depth}, "
            f"levels={list(self.level_index.keys())}, "
            f"collection_offset={self.collection_offset})"
        )


class TacoCatBackend(TacoBackend):
    """
    Backend for TacoCat consolidated format.

    Consolidates metadata from multiple .tacozip files into single file.
    Queries across hundreds of ZIPs without opening each file individually.

    All data loaded to memory, no temp files created.
    internal:source_file column identifies origin ZIP for each sample.
    """

    @property
    def format_name(self) -> str:
        return "tacocat"

    def load(self, path: str) -> TacoDataset:
        """
        Load TacoCat dataset entirely in memory.

        Efficient for metadata-only files (typically <1GB).
        No temp files created during loading.
        """
        t_start = time.time()
        logger.debug(f"Loading TacoCat from {path}")

        # Download entire file to memory
        t_download = time.time()
        full_bytes = self._get_full_file(path)
        download_time = time.time() - t_download
        file_size_mb = len(full_bytes) / (1024 * 1024)
        logger.debug(
            f"Downloaded {file_size_mb:.1f}MB in {download_time:.2f}s ({file_size_mb/download_time:.1f}MB/s)"
        )

        # Parse header
        header = TacoCatHeader(full_bytes[0:TACOCAT_TOTAL_HEADER_SIZE])
        logger.debug(
            f"Parsed header: version={header.version}, max_depth={header.max_depth}, levels={list(header.level_index.keys())}"
        )

        # Parse COLLECTION.json from memory
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

        # Setup DuckDB with spatial
        db = duckdb.connect(":memory:")
        try:
            db.execute("INSTALL spatial")
            db.execute("LOAD spatial")
            logger.debug("Loaded DuckDB spatial extension")
        except Exception as e:
            logger.debug(f"Spatial extension not available: {e}")

        # Register Parquet tables directly from memory
        for level_id, (offset, size) in header.level_index.items():
            if size == 0:
                continue

            # Extract parquet bytes from memory
            parquet_bytes = full_bytes[offset : offset + size]

            # Read with PyArrow and register in DuckDB
            reader = pa.BufferReader(parquet_bytes)
            arrow_table = pq.read_table(reader)

            table_name = f"level{level_id}_table"
            db.register(table_name, arrow_table)

            logger.debug(f"Registered {table_name} in DuckDB ({size} bytes)")

        if not header.level_index:
            raise ValueError(f"No metadata levels found in TacoCat: {path}")

        # Setup views and create dataset
        root_path = to_vsi_root(path)
        self.setup_duckdb_views(db, header.level_index.keys(), root_path)
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
        logger.info(f"Loaded TacoCat in {total_time:.2f}s")

        return dataset

    def read_collection(self, path: str) -> dict[str, Any]:
        """
        Read COLLECTION.json from TacoCat file.

        COLLECTION.json stored at offset from header's 7th index entry.
        Contains consolidated metadata from all source ZIPs.
        """
        full_bytes = self._get_full_file(path)
        header = TacoCatHeader(full_bytes[0:TACOCAT_TOTAL_HEADER_SIZE])

        if header.collection_size == 0:
            raise ValueError(f"Empty COLLECTION.json in TacoCat: {path}")

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

        Path format: /vsisubfile/{offset}_{size},{base_path}{source_file}
        Allows samples to point to their specific origin ZIP file.
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
                WHERE id NOT LIKE '{PADDING_PREFIX}%'
            """
            )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _get_full_file(self, path: str) -> bytes:
        """
        Get entire TacoCat file as bytes.

        Local: mmap for efficiency
        Remote: simple download
        """
        if Path(path).exists():
            logger.debug(f"Reading local TacoCat: {path}")
            with open(path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return bytes(mm[:])
        else:
            try:
                logger.debug(f"Downloading remote TacoCat: {path}")
                return download_bytes(path)
            except Exception as e:
                raise OSError(f"Failed to download TacoCat from {path}: {e}")

    def _extract_base_path(self, root_path: str) -> str:
        """
        Extract base directory from __TACOCAT__ path.

        The base path is the directory containing both __TACOCAT__ and
        source .tacozip files. Used to construct VSI paths to individual ZIPs.

        Example:
            /vsis3/bucket/data/__TACOCAT__ -> /vsis3/bucket/data/
            s3://bucket/data/ -> s3://bucket/data/
        """
        if root_path.endswith(TACOCAT_FILENAME):
            base_path = root_path[: -len(TACOCAT_FILENAME)]
        else:
            base_path = root_path if root_path.endswith("/") else root_path + "/"

        return base_path
