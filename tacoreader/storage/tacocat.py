"""
TacoCat backend for consolidated TACO datasets.

Consolidates metadata from multiple .tacozip files into .tacocat/ folder
with unified Parquet files optimized for DuckDB queries.

Format: .tacocat/ folder with level*.parquet + COLLECTION.json
internal:source_file column tracks origin ZIP for each sample.

Main class:
    TacoCatBackend: Backend for TacoCat folder format
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from tacoreader._constants import (
    COLLECTION_JSON,
    LEVEL_TABLE_SUFFIX,
    LEVEL_VIEW_PREFIX,
    METADATA_GDAL_VSI,
    METADATA_OFFSET,
    METADATA_SIZE,
    METADATA_SOURCE_FILE,
    TACOCAT_FOLDER_NAME,
    TACOCAT_MAX_LEVELS,
)
from tacoreader._exceptions import TacoFormatError, TacoIOError
from tacoreader._format import is_local
from tacoreader._logging import get_logger
from tacoreader._remote_io import download_bytes
from tacoreader._vsi import to_vsi_root
from tacoreader.dataset import TacoDataset
from tacoreader.storage.base import TacoBackend

logger = get_logger(__name__)


@lru_cache(maxsize=32)
def _read_tacocat_collection_cached(path: str) -> dict[str, Any]:
    """
    Read and cache COLLECTION.json from .tacocat folder.

    Cached because COLLECTION.json is small (~10-50KB) and frequently accessed
    during dataset exploration and metadata queries.

    Args:
        path: Path to .tacocat folder (local or remote)

    Returns:
        Parsed COLLECTION.json as dict

    Raises:
        TacoIOError: If file cannot be read
        json.JSONDecodeError: If JSON is malformed
    """
    if is_local(path):
        collection_path = Path(path) / COLLECTION_JSON
        if not collection_path.exists():
            raise TacoIOError(f"COLLECTION.json not found in {path}")
        return json.loads(collection_path.read_bytes())
    else:
        collection_bytes = download_bytes(path, COLLECTION_JSON)
        return json.loads(collection_bytes)


def _fetch_level_file(level_id: int, base_path: str) -> tuple[int, bytes | None]:
    """
    Attempt to download single level*.parquet file.

    Used by ThreadPoolExecutor to fetch files in parallel.
    Returns None if file doesn't exist (expected for max_depth < 5).

    Args:
        level_id: Level number (0-5)
        base_path: Base path to .tacocat folder

    Returns:
        Tuple of (level_id, bytes) if found, (level_id, None) if not found
    """
    filename = f"level{level_id}.parquet"

    try:
        if is_local(base_path):
            file_path = Path(base_path) / filename
            if file_path.exists():
                return level_id, file_path.read_bytes()
            else:
                return level_id, None
        else:
            data = download_bytes(base_path, filename)
            return level_id, data
    except (FileNotFoundError, TacoIOError):
        return level_id, None
    except Exception as e:
        logger.debug(f"Failed to fetch {filename}: {e}")
        return level_id, None


class TacoCatBackend(TacoBackend):
    """
    Backend for TacoCat consolidated format (.tacocat folder).

    Consolidates metadata from multiple .tacozip files into unified folder.
    Queries across hundreds of ZIPs without opening each individually.

    Parallel loading: all level*.parquet files fetched concurrently.
    internal:source_file column identifies origin ZIP for each sample.

    Physical layout:
        .tacocat/
        ├── level0.parquet  (consolidated metadata from all ZIPs)
        ├── level1.parquet  (if max_depth >= 1)
        ├── level2.parquet  (if max_depth >= 2)
        └── COLLECTION.json (consolidated metadata)

    Each row in level*.parquet contains internal:source_file pointing to
    the original .tacozip file containing the actual raster data.
    """

    @property
    def format_name(self) -> str:
        return "tacocat"

    def load(self, path: str) -> TacoDataset:
        """
        Load TacoCat dataset (.tacocat folder).

        Strategy:
        1. Parallel fetch all level*.parquet (ThreadPoolExecutor)
        2. Parse Parquet from bytes, register in DuckDB
        3. Load COLLECTION.json metadata
        4. Create views with internal:gdal_vsi

        All data loaded into memory, no temp files created.

        Args:
            path: Path to .tacocat folder (local or remote)

        Returns:
            Fully loaded TacoDataset instance

        Raises:
            TacoFormatError: If no level files found or format invalid
            TacoIOError: If files cannot be read
        """
        t_start = time.time()
        logger.debug(f"Loading TacoCat from {path}")

        # Parallel fetch all level files
        t_fetch = time.time()
        levels_bytes = self._fetch_all_levels(path)
        fetch_time = time.time() - t_fetch

        total_mb = sum(len(b) for b in levels_bytes.values()) / (1024 * 1024)
        logger.debug(
            f"Fetched {len(levels_bytes)} levels ({total_mb:.1f}MB) "
            f"in {fetch_time:.2f}s ({total_mb/fetch_time:.1f}MB/s)"
        )

        # Setup DuckDB
        db = self._setup_duckdb_connection()

        # Register Parquet tables from bytes
        level_ids = []
        for level_id in sorted(levels_bytes.keys()):
            parquet_bytes = levels_bytes[level_id]

            reader = pa.BufferReader(parquet_bytes)
            arrow_table = pq.read_table(reader)

            table_name = f"{LEVEL_VIEW_PREFIX}{level_id}{LEVEL_TABLE_SUFFIX}"
            db.register(table_name, arrow_table)
            level_ids.append(level_id)

            logger.debug(
                f"Registered {table_name}: {arrow_table.num_rows} rows x "
                f"{arrow_table.num_columns} cols"
            )

        if not level_ids:
            raise TacoFormatError(f"No level*.parquet files found in: {path}")

        # Load COLLECTION.json
        collection = self.read_collection(path)

        # Finalize dataset
        root_path = to_vsi_root(path)
        dataset = self._finalize_dataset(db, path, root_path, collection, level_ids)

        total_time = time.time() - t_start
        logger.info(f"Loaded TacoCat in {total_time:.2f}s")

        return dataset

    def _fetch_all_levels(self, base_path: str) -> dict[int, bytes]:
        """
        Fetch all level*.parquet files in parallel using ThreadPoolExecutor.

        Attempts to download level0-5.parquet concurrently.
        Missing files are ignored (expected if max_depth < 5).

        For local paths, uses simple sequential reads (fast enough).
        For remote paths, uses parallel downloads with thread pool.

        Args:
            base_path: Path to .tacocat folder

        Returns:
            Dict of {level_id: parquet_bytes} for found levels
        """
        if is_local(base_path):
            # Local: simple sequential reads (I/O is fast, no need for threads)
            levels = {}
            base = Path(base_path)
            for i in range(TACOCAT_MAX_LEVELS):
                file = base / f"level{i}.parquet"
                if file.exists():
                    levels[i] = file.read_bytes()
            return levels

        # Remote: parallel downloads with ThreadPoolExecutor
        levels = {}
        with ThreadPoolExecutor(max_workers=TACOCAT_MAX_LEVELS) as executor:
            futures = {
                executor.submit(_fetch_level_file, i, base_path): i
                for i in range(TACOCAT_MAX_LEVELS)
            }

            for future in as_completed(futures):
                level_id, data = future.result()
                if data is not None:
                    levels[level_id] = data

        return levels

    def read_collection(self, path: str) -> dict[str, Any]:
        """
        Read COLLECTION.json from .tacocat folder with caching.

        Contains consolidated metadata from all source ZIPs including:
        - Dataset ID, version, description
        - PIT schema (hierarchy structure)
        - Field schema (column definitions)
        - Spatial/temporal extent

        Args:
            path: Path to .tacocat folder

        Returns:
            Parsed COLLECTION.json as dict
        """
        return _read_tacocat_collection_cached(path)

    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        level_ids: list[int],
        root_path: str,
    ) -> None:
        """
        Create DuckDB views with GDAL VSI paths for TacoCat format.

        Path format: /vsisubfile/{offset}_{size},{base_path}{source_file}

        Allows samples to point back to their specific origin .tacozip file
        using the internal:source_file column.

        Example VSI path:
            /vsisubfile/2048_6000,/vsis3/bucket/data/part0001.tacozip

        Args:
            db: DuckDB connection
            level_ids: List of level IDs to create views for
            root_path: VSI root path (used to extract base directory)
        """
        base_path = self._extract_base_path(root_path)
        filter_clause = self._build_view_filter()

        for level_id in level_ids:
            table_name = f"{LEVEL_VIEW_PREFIX}{level_id}{LEVEL_TABLE_SUFFIX}"
            view_name = f"{LEVEL_VIEW_PREFIX}{level_id}"
            db.execute(
                f"""
                CREATE VIEW {view_name} AS
                SELECT *,
                  '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
                  "{METADATA_SIZE}" || ',{base_path}' || "{METADATA_SOURCE_FILE}"
                  as "{METADATA_GDAL_VSI}"
                FROM {table_name}
                WHERE {filter_clause}
            """
            )

    def _extract_base_path(self, root_path: str) -> str:
        """
        Extract base directory from .tacocat/ path.

        The base path is the directory containing both .tacocat/ folder
        and source .tacozip files. Used to construct VSI paths to individual ZIPs.

        Examples:
            /vsis3/bucket/data/.tacocat/ → /vsis3/bucket/data/
            /vsis3/bucket/data/.tacocat → /vsis3/bucket/data/
            s3://bucket/data/.tacocat → s3://bucket/data/
            /local/path/.tacocat → /local/path/

        Args:
            root_path: Path to .tacocat folder (may or may not have trailing slash)

        Returns:
            Base directory path with trailing slash
        """
        # FIX: Normalize trailing slashes FIRST
        # This prevents issues where remote paths have trailing slash
        # but local paths don't (Path.resolve() removes them)
        clean_path = root_path.rstrip("/")

        # Remove .tacocat suffix
        if clean_path.endswith(f"/{TACOCAT_FOLDER_NAME}"):
            base_path = clean_path[: -(len(TACOCAT_FOLDER_NAME) + 1)]
        elif clean_path.endswith(TACOCAT_FOLDER_NAME):
            base_path = clean_path[: -len(TACOCAT_FOLDER_NAME)]
        else:
            base_path = clean_path

        # Ensure trailing slash for path concatenation
        if not base_path.endswith("/"):
            base_path += "/"

        return base_path
