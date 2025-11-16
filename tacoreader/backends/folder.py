"""
FOLDER backend for TACO datasets.

This module provides the FolderBackend class for reading TACO datasets stored
as directory structures with loose files. FOLDER format is optimized for
development workflows and scenarios requiring frequent file updates.

FOLDER Structure:
    dataset/
    ├── DATA/
    │   ├── sample_001.tif (if level 0 = FILEs)
    │   OR
    │   ├── folder_A/ (if level 0 = FOLDERs)
    │   │   ├── __meta__ (Parquet)
    │   │   ├── nested_001.tif
    │   │   └── nested_002.tif
    ├── METADATA/
    │   ├── level0.parquet
    │   └── level1.parquet
    └── COLLECTION.json

Main class:
    FolderBackend: Backend implementation for FOLDER format

Example:
    >>> from tacoreader.backends.folder import FolderBackend
    >>> backend = FolderBackend()
    >>> dataset = backend.load("s3://bucket/dataset/")
    >>> print(dataset.schema.root["n"])
    1000
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

logger = get_logger(__name__)

# ============================================================================
# FOLDER BACKEND
# ============================================================================


class FolderBackend(TacoBackend):
    """
    Backend for FOLDER format.

    Handles reading TACO datasets stored as directory structures with
    loose files. Files are accessed directly via filesystem paths or
    cloud storage URLs without requiring decompression.

    The FOLDER format stores metadata as Parquet files and constructs 
    GDAL paths dynamically based on sample IDs and types. Navigation 
    uses __meta__ files for FOLDERs and direct paths for FILEs.

    For local datasets, metadata files are read directly from disk.
    For remote datasets, metadata is downloaded and loaded into memory.

    Attributes:
        format_name: Returns "folder"

    Example:
        >>> backend = FolderBackend()
        >>> dataset = backend.load("data/")
        >>> filtered = dataset.sql("SELECT * FROM data WHERE type = 'FILE'")
        >>> df = filtered.data
    """

    @property
    def format_name(self) -> str:
        """
        Format identifier for FOLDER backend.

        Returns:
            "folder" string constant
        """
        return "folder"

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================

    def load(self, path: str) -> TacoDataset:
        """
        Load FOLDER dataset.

        For local datasets, reads metadata directly from disk without loading
        into memory. For remote datasets, downloads all metadata files and
        loads them into memory.

        Args:
            path: Path to FOLDER dataset (local or remote)
                 Local: "data/" or "/absolute/path/to/data/"
                 Remote: "s3://bucket/data/" or "gs://bucket/data/"

        Returns:
            TacoDataset with lazy SQL interface

        Raises:
            ValueError: If FOLDER is invalid or missing metadata
            IOError: If file access fails

        Example:
            >>> backend = FolderBackend()
            >>> dataset = backend.load("data/")
            >>> print(dataset.id)
            'sentinel2-l2a'
            >>>
            >>> # Remote dataset
            >>> dataset = backend.load("s3://bucket/data/")
        """
        t_start = time.time()
        
        logger.debug(f"Loading FOLDER from {path}")

        # Step 1: Read COLLECTION.json
        collection = self.read_collection(path)
        logger.debug("Parsed COLLECTION.json")

        # Step 2: Create DuckDB connection
        db = duckdb.connect(":memory:")

        # Step 3: Load spatial extension
        try:
            db.execute("INSTALL spatial")
            db.execute("LOAD spatial")
            logger.debug("Loaded DuckDB spatial extension")
        except Exception as e:
            logger.debug(f"Spatial extension not available: {e}")

        is_local = Path(path).exists()
        level_ids = []

        if is_local:
            # LOCAL: Read directly from disk without loading to memory
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
            # REMOTE: Download all metadata and load to memory
            for i in range(6):  # Max 6 levels (0-5)
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
                    
                    logger.debug(f"Registered {table_name} in DuckDB ({len(parquet_bytes)} bytes)")
                    
                except Exception:
                    # Stop at first missing level (expected behavior)
                    break

        if not level_ids:
            raise ValueError(
                f"No metadata files found in {path}/METADATA/\n"
                f"Expected at least level0.parquet"
            )

        # Step 4: Convert path to VSI format
        root_path = to_vsi_root(path)

        # Step 5: Setup DuckDB views
        self.setup_duckdb_views(db, level_ids, root_path)
        logger.debug("Created DuckDB views")

        # Step 6: Create 'data' view
        db.execute("CREATE VIEW data AS SELECT * FROM level0")

        # Step 7: Extract PIT schema
        schema = PITSchema(collection["taco:pit_schema"])

        # Step 8: Construct TacoDataset
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

        COLLECTION.json is always at the root of the dataset directory.
        Supports both local filesystem and remote storage (S3/GCS/Azure).

        Args:
            path: Path to FOLDER dataset (local or remote)
                 Local: "data/" or "/absolute/path/to/data/"
                 Remote: "s3://bucket/data/" or "gs://bucket/data/"

        Returns:
            Dictionary containing COLLECTION.json content

        Raises:
            ValueError: If COLLECTION.json not found
            json.JSONDecodeError: If COLLECTION.json is invalid
            IOError: If remote read fails

        Example:
            >>> backend = FolderBackend()
            >>> collection = backend.read_collection("data/")
            >>> print(collection["id"])
            'sentinel2-l2a'

            >>> # Remote dataset
            >>> collection = backend.read_collection("s3://bucket/data/")
            >>> print(collection["taco_version"])
            '0.5.0'
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
        Create DuckDB views with direct filesystem paths for FOLDER format.

        Each view references a table already registered in DuckDB and adds
        a computed 'internal:gdal_vsi' column that constructs paths based
        on sample type and hierarchy level:

        Level 0:
            - FILE: {root}DATA/{id}
            - FOLDER: {root}DATA/{id}/__meta__

        Level 1+:
            - FILE: {root}DATA/{internal:relative_path}
            - FOLDER: {root}DATA/{internal:relative_path}__meta__

        Args:
            db: DuckDB connection with tables already registered
            level_ids: List of level IDs that have tables registered
            root_path: Root path to FOLDER dataset (with trailing /)

        Example:
            >>> import duckdb
            >>> db = duckdb.connect(":memory:")
            >>> # ... register tables ...
            >>> backend = FolderBackend()
            >>> backend.setup_duckdb_views(db, [0, 1], "s3://bucket/data/")
            >>> result = db.execute("SELECT * FROM level0 LIMIT 1").fetchone()
        """
        # Ensure root path has trailing slash
        root = root_path if root_path.endswith("/") else root_path + "/"

        for level_id in level_ids:
            table_name = f"level{level_id}_table"
            view_name = f"level{level_id}"
            
            if level_id == 0:
                # Level 0: Use id directly (no relative_path column)
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
                    WHERE id NOT LIKE '__TACOPAD__%'
                """
                )
            else:
                # Level 1+: Use internal:relative_path for nested structure
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
                    WHERE id NOT LIKE '__TACOPAD__%'
                """
                )