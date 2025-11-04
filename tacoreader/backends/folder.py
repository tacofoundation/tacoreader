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
    │   │   ├── __meta__ (Avro)
    │   │   ├── nested_001.tif
    │   │   └── nested_002.tif
    ├── METADATA/
    │   ├── level0.avro
    │   └── level1.avro
    └── COLLECTION.json

Key features:
- Human-readable directory structure
- Direct file access without decompression
- Efficient for development and testing
- Supports both local and remote (S3/GCS) storage
- Metadata stored as Avro for schema support

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
from pathlib import Path
from typing import Any

import duckdb
import obstore as obs
from obstore.store import from_url

from tacoreader.backends.base import TacoBackend

# ============================================================================
# FOLDER BACKEND
# ============================================================================


class FolderBackend(TacoBackend):
    """
    Backend for FOLDER format.

    Handles reading TACO datasets stored as directory structures with
    loose files. Files are accessed directly via filesystem paths or
    cloud storage URLs without requiring decompression.

    The FOLDER format stores metadata as Avro files (for schema support)
    and constructs GDAL paths dynamically based on sample IDs and types.
    Navigation uses __meta__ files for FOLDERs and direct paths for FILEs.

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
            store = from_url(path)
            collection_result = obs.get(store, "COLLECTION.json")
            collection_bytes = collection_result.bytes()
            return json.loads(bytes(collection_bytes))
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid COLLECTION.json in {path}: {e.msg}", e.doc, e.pos
            )
        except Exception as e:
            raise OSError(f"Failed to read COLLECTION.json from {path}: {e}")

    def cache_metadata_files(self, path: str, cache_dir: Path) -> dict[str, str]:
        """
        Cache Avro metadata files from FOLDER/METADATA/ directory.

        For local datasets, returns direct paths to existing files.
        For remote datasets, downloads files to cache directory.

        Attempts to read level0.avro through level5.avro (up to 6 levels),
        stopping when a level file is not found.

        Args:
            path: Path to FOLDER dataset (local or remote)
            cache_dir: Directory to write cached files (remote only)

        Returns:
            Dictionary mapping level names to file paths
            Example: {"level0": "data/METADATA/level0.avro", ...}

        Raises:
            ValueError: If no metadata files found
            IOError: If remote download fails

        Example:
            >>> backend = FolderBackend()
            >>> cache_dir = Path("/tmp/cache")
            >>> files = backend.cache_metadata_files("data/", cache_dir)
            >>> print(files.keys())
            dict_keys(['level0', 'level1'])

            >>> # Remote dataset
            >>> files = backend.cache_metadata_files("s3://bucket/data/", cache_dir)
            >>> # Files downloaded to cache_dir
        """
        is_local = Path(path).exists()
        consolidated_files = {}

        if is_local:
            # Local filesystem - return direct paths
            base_path = Path(path)
            metadata_dir = base_path / "METADATA"

            if not metadata_dir.exists():
                raise ValueError(
                    f"METADATA directory not found in {path}\n"
                    f"Expected: {metadata_dir}"
                )

            for i in range(6):  # Max 6 levels (0-5)
                level_file = metadata_dir / f"level{i}.avro"
                if level_file.exists():
                    consolidated_files[f"level{i}"] = str(level_file)
                else:
                    # Stop at first missing level
                    break
        else:
            # Remote storage - download to cache
            store = from_url(path)

            for i in range(6):  # Max 6 levels (0-5)
                try:
                    avro_result = obs.get(store, f"METADATA/level{i}.avro")
                    level_file = cache_dir / f"level{i}.avro"
                    level_file.write_bytes(bytes(avro_result.bytes()))
                    consolidated_files[f"level{i}"] = str(level_file)
                except Exception:
                    # Stop at first missing level (expected behavior)
                    break

        if not consolidated_files:
            raise ValueError(
                f"No metadata files found in {path}/METADATA/\n"
                f"Expected at least level0.avro"
            )

        return consolidated_files

    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        consolidated_files: dict[str, str],
        root_path: str,
    ) -> None:
        """
        Create DuckDB views with direct filesystem paths for FOLDER format.

        Each view includes a computed 'internal:gdal_vsi' column that
        constructs paths based on sample type and hierarchy level:

        Level 0:
            - FILE: {root}DATA/{id}
            - FOLDER: {root}DATA/{id}/__meta__

        Level 1+:
            - FILE: {root}DATA/{internal:relative_path}
            - FOLDER: {root}DATA/{internal:relative_path}__meta__

        Args:
            db: DuckDB connection for creating views
            consolidated_files: Cached metadata file paths
            root_path: Root path to FOLDER dataset (with trailing /)

        Example:
            >>> import duckdb
            >>> db = duckdb.connect(":memory:")
            >>> backend = FolderBackend()
            >>> backend.setup_duckdb_views(db, files, "s3://bucket/data/")
            >>> result = db.execute("SELECT * FROM level0 LIMIT 1").fetchone()
        """
        # Ensure root path has trailing slash
        root = root_path if root_path.endswith("/") else root_path + "/"

        for level_key, file_path in consolidated_files.items():
            if level_key == "level0":
                # Level 0: Use id directly (no relative_path column)
                db.execute(
                    f"""
                    CREATE VIEW {level_key} AS 
                    SELECT *,
                      CASE 
                        WHEN type = 'FOLDER' THEN '{root}DATA/' || id || '/__meta__'
                        WHEN type = 'FILE' THEN '{root}DATA/' || id
                        ELSE NULL
                      END as "internal:gdal_vsi"
                    FROM read_avro('{file_path}')
                    WHERE id NOT LIKE '__TACOPAD__%'
                """
                )
            else:
                # Level 1+: Use internal:relative_path for nested structure
                db.execute(
                    f"""
                    CREATE VIEW {level_key} AS 
                    SELECT *,
                      CASE 
                        WHEN type = 'FOLDER' THEN '{root}DATA/' || "internal:relative_path" || '__meta__'
                        WHEN type = 'FILE' THEN '{root}DATA/' || "internal:relative_path"
                        ELSE NULL
                      END as "internal:gdal_vsi"
                    FROM read_avro('{file_path}')
                    WHERE id NOT LIKE '__TACOPAD__%'
                """
                )