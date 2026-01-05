"""TacoCat backend for consolidated TACO datasets.

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

from tacoreader._cache import (
    get_remote_metadata,
    is_cache_valid,
    load_from_cache,
    save_to_cache,
)
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
    """Read and cache COLLECTION.json from .tacocat folder.

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
    else:  # pragma: no cover
        collection_bytes = download_bytes(path, COLLECTION_JSON)
        return json.loads(collection_bytes)


def _fetch_level_file(level_id: int, base_path: str) -> tuple[int, bytes | None]:  # pragma: no cover - remote helper
    """Attempt to download single level*.parquet file.

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
    """Backend for TacoCat consolidated format (.tacocat folder).

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

    def load(self, path: str, cache: bool = True) -> TacoDataset:
        """Load TacoCat dataset (.tacocat folder).

        Strategy:
        1. Parallel fetch all level*.parquet (ThreadPoolExecutor)
        2. Parse Parquet from bytes, register in DuckDB
        3. Load COLLECTION.json metadata
        4. Create views with internal:gdal_vsi

        All data loaded into memory, no temp files created.

        Args:
            path: Path to .tacocat folder (local or remote)
            cache: Use disk cache for remote datasets (default True)

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
        levels_bytes = self._fetch_all_levels(path) if is_local(path) else self._fetch_remote_with_cache(path, cache)
        fetch_time = time.time() - t_fetch

        total_mb = sum(len(b) for b in levels_bytes.values()) / (1024 * 1024)
        logger.debug(
            f"Fetched {len(levels_bytes)} levels ({total_mb:.1f}MB) "
            f"in {fetch_time:.2f}s ({total_mb / fetch_time:.1f}MB/s)"
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

            logger.debug(f"Registered {table_name}: {arrow_table.num_rows} rows x {arrow_table.num_columns} cols")

        if not level_ids:  # pragma: no cover - defensive
            raise TacoFormatError(f"No level*.parquet files found in: {path}")

        # Load COLLECTION.json
        collection = self.read_collection(path)

        # For TacoCat: vsi_base_path is the PARENT directory (where .tacozip files are)
        # NOT the .tacocat folder itself
        root_path = to_vsi_root(path)
        vsi_base_path = self._extract_base_path(root_path)

        dataset = self._finalize_dataset(db, path, vsi_base_path, collection, level_ids)

        total_time = time.time() - t_start
        logger.info(f"Loaded TacoCat in {total_time:.2f}s")

        return dataset

    def _fetch_remote_with_cache(self, path: str, cache: bool) -> dict[int, bytes]:  # pragma: no cover
        """Fetch remote TacoCat with optional disk cache.

        Args:
            path: Remote path to .tacocat folder
            cache: Whether to use disk cache

        Returns:
            Dict of {level_id: parquet_bytes}
        """
        if not cache:
            return self._fetch_all_levels(path)

        # Get remote metadata (HEAD request for ETag/size)
        remote_meta = get_remote_metadata(path)
        remote_etag = remote_meta.get("etag") if remote_meta else None
        remote_size = remote_meta.get("size") if remote_meta else None

        # Check cache validity
        if is_cache_valid(path, remote_etag, remote_size):
            cached = load_from_cache(path)
            if cached:
                logger.info(f"Using cached TacoCat: {path}")
                return self._cached_files_to_levels(cached)

        # Cache miss or invalid - download fresh
        logger.debug(f"Cache miss, downloading: {path}")
        levels_bytes = self._fetch_all_levels(path)

        # Save to cache
        files_to_cache = {f"level{k}.parquet": v for k, v in levels_bytes.items()}
        try:
            collection_bytes = download_bytes(path, COLLECTION_JSON)
            files_to_cache[COLLECTION_JSON] = collection_bytes
        except Exception as e:
            logger.debug(f"COLLECTION.json not cached: {e}")

        save_to_cache(path, files_to_cache, remote_etag, remote_size)

        return levels_bytes

    def _cached_files_to_levels(self, cached: dict[str, bytes]) -> dict[int, bytes]:  # pragma: no cover
        """Convert cache dict {filename: bytes} to {level_id: bytes}."""
        levels = {}
        for filename, data in cached.items():
            if filename.startswith("level") and filename.endswith(".parquet"):
                level_id = int(filename[5:-8])  # "level0.parquet" -> 0
                levels[level_id] = data
        return levels

    def _fetch_all_levels(self, base_path: str) -> dict[int, bytes]:
        """Fetch all level*.parquet files in parallel using ThreadPoolExecutor.

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
        levels = {}  # pragma: no cover
        with ThreadPoolExecutor(max_workers=TACOCAT_MAX_LEVELS) as executor:  # pragma: no cover
            futures = {executor.submit(_fetch_level_file, i, base_path): i for i in range(TACOCAT_MAX_LEVELS)}

            for future in as_completed(futures):
                level_id, data = future.result()
                if data is not None:
                    levels[level_id] = data

        return levels  # pragma: no cover

    def read_collection(self, path: str) -> dict[str, Any]:
        """Read COLLECTION.json from .tacocat folder with caching.

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
        vsi_base_path: str,
    ) -> None:
        """Create DuckDB views with GDAL VSI paths for TacoCat format.

        Path format: /vsisubfile/{offset}_{size},{vsi_base_path}{source_file}

        Note: vsi_base_path is already the parent directory (where .tacozip files are),
        NOT the .tacocat folder. The _extract_base_path transformation happens in load().

        Example VSI path:
            /vsisubfile/2048_6000,/vsis3/bucket/data/part0001.tacozip

        Args:
            db: DuckDB connection
            level_ids: List of level IDs to create views for
            vsi_base_path: Base path to directory containing .tacozip files
        """
        filter_clause = self._build_view_filter()

        for level_id in level_ids:
            table_name = f"{LEVEL_VIEW_PREFIX}{level_id}{LEVEL_TABLE_SUFFIX}"
            view_name = f"{LEVEL_VIEW_PREFIX}{level_id}"
            db.execute(
                f"""
                CREATE VIEW {view_name} AS
                SELECT *,
                  '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
                  "{METADATA_SIZE}" || ',{vsi_base_path}' || "{METADATA_SOURCE_FILE}"
                  as "{METADATA_GDAL_VSI}"
                FROM {table_name}
                WHERE {filter_clause}
            """
            )

    def _extract_base_path(self, root_path: str) -> str:
        """Extract base directory from .tacocat/ path.

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
        clean_path = root_path.rstrip("/")

        if clean_path.endswith(f"/{TACOCAT_FOLDER_NAME}"):
            base_path = clean_path[: -(len(TACOCAT_FOLDER_NAME) + 1)]
        elif clean_path.endswith(TACOCAT_FOLDER_NAME):
            base_path = clean_path[: -len(TACOCAT_FOLDER_NAME)]
        else:  # pragma: no cover - defensive
            base_path = clean_path

        if not base_path.endswith("/"):
            base_path += "/"

        return base_path
