"""
FOLDER backend for TACO datasets.

Reads datasets stored as directory structures with loose files.
Optimized for development workflows and frequent updates.

Main class:
    FolderBackend: Backend for FOLDER format
"""

import json
import time
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from tacoreader.backends.base import TacoBackend
from tacoreader.io import download_bytes
from tacoreader._logging import get_logger
from tacoreader.dataset import TacoDataset
from tacoreader.schema import PITSchema
from tacoreader.utils.vsi import to_vsi_root
from tacoreader._constants import PADDING_PREFIX

logger = get_logger(__name__)


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

        # Read collection
        collection = self.read_collection(path)
        logger.debug("Parsed COLLECTION.json")

        # Setup DuckDB with spatial
        db = duckdb.connect(":memory:")
        try:
            db.execute("INSTALL spatial")
            db.execute("LOAD spatial")
            logger.debug("Loaded DuckDB spatial extension")
        except Exception as e:
            logger.debug(f"Spatial extension not available: {e}")

        is_local = Path(path).exists()
        level_ids = []

        if is_local:
            # LOCAL: read directly from disk (no memory loading)
            base_path = Path(path)
            metadata_dir = base_path / "METADATA"

            if not metadata_dir.exists():
                raise ValueError(
                    f"METADATA directory not found in {path}\n"
                    f"Expected: {metadata_dir}"
                )

            for i in range(6):  # Max 6 levels (0-5)
                level_file = metadata_dir / f"level{i}.parquet"

                if not level_file.exists():
                    break

                # Create table reading directly from disk
                table_name = f"level{i}_table"
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
                    parquet_bytes = download_bytes(path, f"METADATA/level{i}.parquet")

                    # Load to PyArrow from bytes
                    reader = pa.BufferReader(parquet_bytes)
                    arrow_table = pq.read_table(reader)

                    # Register in DuckDB (all in memory)
                    table_name = f"level{i}_table"
                    db.register(table_name, arrow_table)

                    level_ids.append(i)
                    logger.debug(
                        f"Registered {table_name} in DuckDB ({len(parquet_bytes)} bytes)"
                    )

                except Exception:
                    break  # Stop at first missing level

        if not level_ids:
            raise ValueError(
                f"No metadata files found in {path}/METADATA/\n"
                f"Expected at least level0.parquet"
            )

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
        logger.info(f"Loaded FOLDER in {total_time:.2f}s")

        return dataset

    def read_collection(self, path: str) -> dict[str, Any]:
        """
        Read COLLECTION.json from FOLDER root.

        Supports local filesystem and remote storage (S3/GCS/Azure).
        """
        if Path(path).exists():
            # Local filesystem
            collection_path = Path(path) / "COLLECTION.json"
            if not collection_path.exists():
                raise ValueError(
                    f"COLLECTION.json not found in {path}\n"
                    f"Expected: {collection_path}"
                )

            with open(collection_path) as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid COLLECTION.json in {path}: {e.msg}", e.doc, e.pos
                    )

        # Remote storage
        try:
            collection_bytes = download_bytes(path, "COLLECTION.json")
            return json.loads(collection_bytes)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid COLLECTION.json in {path}: {e.msg}", e.doc, e.pos
            )
        except Exception as e:
            raise OSError(f"Failed to read COLLECTION.json from {path}: {e}")

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

        for level_id in level_ids:
            table_name = f"level{level_id}_table"
            view_name = f"level{level_id}"

            if level_id == 0:
                # Level 0: use id directly (no relative_path column)
                db.execute(
                    f"""
                    CREATE VIEW {view_name} AS 
                    SELECT *,
                      CASE 
                        WHEN type = 'FOLDER' THEN '{root}DATA/' || id || '/__meta__'
                        WHEN type = 'FILE' THEN '{root}DATA/' || id
                        ELSE NULL
                      END as "internal:gdal_vsi"
                    FROM {table_name}
                    WHERE id NOT LIKE '{PADDING_PREFIX}%'
                """
                )
            else:
                # Level 1+: use internal:relative_path for nested structure
                db.execute(
                    f"""
                    CREATE VIEW {view_name} AS 
                    SELECT *,
                      CASE 
                        WHEN type = 'FOLDER' THEN '{root}DATA/' || "internal:relative_path" || '__meta__'
                        WHEN type = 'FILE' THEN '{root}DATA/' || "internal:relative_path"
                        ELSE NULL
                      END as "internal:gdal_vsi"
                    FROM {table_name}
                    WHERE id NOT LIKE '{PADDING_PREFIX}%'
                """
                )
