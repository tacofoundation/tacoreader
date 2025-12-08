"""
FOLDER backend for TACO datasets.

Reads datasets stored as directory structures with loose files.
Optimized for development workflows and frequent updates.

Main class:
    FolderBackend: Backend for FOLDER format
"""

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from tacoreader._constants import (
    COLUMN_ID,
    COLUMN_TYPE,
    LEVEL_TABLE_SUFFIX,
    LEVEL_VIEW_PREFIX,
    METADATA_GDAL_VSI,
    METADATA_RELATIVE_PATH,
    SAMPLE_TYPE_FILE,
    SAMPLE_TYPE_FOLDER,
)
from tacoreader._exceptions import TacoFormatError, TacoIOError
from tacoreader._logging import get_logger
from tacoreader._remote_io import download_bytes
from tacoreader._vsi import to_vsi_root
from tacoreader.dataset import TacoDataset
from tacoreader.storage.base import TacoBackend

logger = get_logger(__name__)


@lru_cache(maxsize=32)
def _read_collection_folder_cached(path: str) -> dict[str, Any]:
    """
    Read and cache COLLECTION.json for FOLDER format.

    Cached because COLLECTION.json is small (~10-50KB typically)
    and accessed frequently during dataset exploration.

    Args:
        path: Path to FOLDER dataset (local or remote)

    Returns:
        Parsed COLLECTION.json as dictionary
    """
    if Path(path).exists():
        # Local filesystem
        collection_path = Path(path) / "COLLECTION.json"
        if not collection_path.exists():
            raise TacoFormatError(
                f"COLLECTION.json not found in {path}\n" f"Expected: {collection_path}"
            )

        with open(collection_path) as f:
            collection_bytes = f.read().encode("utf-8")
    else:
        # Remote storage - download_bytes is NOT cached (generic function)
        try:
            collection_bytes = download_bytes(path, "COLLECTION.json")
        except Exception as e:
            raise TacoIOError(f"Failed to read COLLECTION.json from {path}: {e}") from e

    # Parse JSON
    try:
        return json.loads(collection_bytes)
    except json.JSONDecodeError as e:
        raise TacoFormatError(
            f"Invalid COLLECTION.json in folder format at {path}: {e.msg}"
        ) from e


class FolderBackend(TacoBackend):
    """
    Backend for FOLDER format.

    Handles datasets stored as directory structures with loose files.
    Metadata is Parquet, data files accessed via filesystem paths.

    Loading strategy:
    - Local: reads metadata directly from disk (no memory loading)
    - Remote: downloads all metadata to memory
    """

    @property
    def format_name(self) -> str:
        return "folder"

    def load(self, path: str) -> TacoDataset:
        """
        Load FOLDER dataset.

        Local: lazy access from disk
        Remote: downloads all metadata to memory
        """
        t_start = time.time()
        logger.debug(f"Loading FOLDER from {path}")

        # Read collection (uses cache)
        collection = self.read_collection(path)
        logger.debug("Parsed COLLECTION.json")

        # Setup DuckDB with spatial
        db = self._setup_duckdb_connection()

        is_local = Path(path).exists()
        level_ids = []

        if is_local:
            # LOCAL: read directly from disk (no memory loading)
            base_path = Path(path)
            metadata_dir = base_path / "METADATA"

            if not metadata_dir.exists():
                raise TacoFormatError(
                    f"METADATA directory not found in {path}\n"
                    f"Expected: {metadata_dir}"
                )

            for i in range(6):  # Max 6 levels (0-5)
                level_file = metadata_dir / f"{LEVEL_VIEW_PREFIX}{i}.parquet"

                if not level_file.exists():
                    break

                # Create table reading directly from disk
                table_name = f"{LEVEL_VIEW_PREFIX}{i}{LEVEL_TABLE_SUFFIX}"
                db.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{level_file}')"
                )

                level_ids.append(i)
                logger.debug(f"Loaded {table_name} from disk")

        else:
            # REMOTE: download all metadata and load to memory
            for i in range(6):
                try:
                    # Download parquet
                    parquet_bytes = download_bytes(
                        path, f"METADATA/{LEVEL_VIEW_PREFIX}{i}.parquet"
                    )

                    # Load to PyArrow from bytes
                    reader = pa.BufferReader(parquet_bytes)
                    arrow_table = pq.read_table(reader)

                    # Register in DuckDB (all in memory)
                    table_name = f"{LEVEL_VIEW_PREFIX}{i}{LEVEL_TABLE_SUFFIX}"
                    db.register(table_name, arrow_table)

                    level_ids.append(i)
                    logger.debug(
                        f"Registered {table_name} in DuckDB ({len(parquet_bytes)} bytes)"
                    )

                except Exception:
                    break  # Stop at first missing level

        if not level_ids:
            raise TacoFormatError(
                f"No metadata files found in {path}/METADATA/\n"
                f"Expected at least {LEVEL_VIEW_PREFIX}0.parquet"
            )

        # Finalize dataset using common method
        root_path = to_vsi_root(path)
        dataset = self._finalize_dataset(db, path, root_path, collection, level_ids)

        total_time = time.time() - t_start
        logger.info(f"Loaded FOLDER in {total_time:.2f}s")

        return dataset

    def read_collection(self, path: str) -> dict[str, Any]:
        """
        Read COLLECTION.json from FOLDER root with caching.

        Supports local filesystem and remote storage (S3/GCS/Azure).
        """
        return _read_collection_folder_cached(path)

    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        level_ids: list[int],
        root_path: str,
    ) -> None:
        """
        Create DuckDB views with direct filesystem paths.

        Path construction:
        - Level 0 FILE: {root}DATA/{id}
        - Level 0 FOLDER: {root}DATA/{id}/__meta__
        - Level 1+ FILE: {root}DATA/{internal:relative_path}
        - Level 1+ FOLDER: {root}DATA/{internal:relative_path}__meta__
        """
        # Ensure trailing slash
        root = root_path if root_path.endswith("/") else root_path + "/"

        # Get filter condition
        filter_clause = self._build_view_filter()

        for level_id in level_ids:
            table_name = f"{LEVEL_VIEW_PREFIX}{level_id}{LEVEL_TABLE_SUFFIX}"
            view_name = f"{LEVEL_VIEW_PREFIX}{level_id}"

            if level_id == 0:
                # Level 0: use id directly (no relative_path column)
                db.execute(
                    f"""
                    CREATE VIEW {view_name} AS
                    SELECT *,
                      CASE
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FOLDER}' THEN '{root}DATA/' || {COLUMN_ID} || '/__meta__'
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FILE}' THEN '{root}DATA/' || {COLUMN_ID}
                        ELSE NULL
                      END as "{METADATA_GDAL_VSI}"
                    FROM {table_name}
                    WHERE {filter_clause}
                """
                )
            else:
                # Level 1+: use internal:relative_path for nested structure
                db.execute(
                    f"""
                    CREATE VIEW {view_name} AS
                    SELECT *,
                      CASE
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FOLDER}' THEN '{root}DATA/' || "{METADATA_RELATIVE_PATH}" || '__meta__'
                        WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FILE}' THEN '{root}DATA/' || "{METADATA_RELATIVE_PATH}"
                        ELSE NULL
                      END as "{METADATA_GDAL_VSI}"
                    FROM {table_name}
                    WHERE {filter_clause}
                """
                )
