"""
Abstract base class for TACO format backends.

Defines common interface for all backends (ZIP, FOLDER, TacoCat).
Not a pure template method - FOLDER uses the default load() implementation
for lazy disk access, while ZIP/TacoCat override it completely for in-memory loading.

Main class:
    TacoBackend: Abstract base with optional template method
"""

from abc import ABC, abstractmethod
from typing import Any

import duckdb

from tacoreader.dataset import TacoDataset
from tacoreader.schema import PITSchema
from tacoreader.utils.vsi import to_vsi_root


class TacoBackend(ABC):
    """
    Base class for TACO backends.

    Two loading strategies:
    - FOLDER: uses default load() for lazy disk access (files already on disk)
    - ZIP/TacoCat: override load() for in-memory loading (need to parse headers)

    All backends must implement:
    - read_collection(): format-specific COLLECTION.json reading
    - setup_duckdb_views(): create views with internal:gdal_vsi column
    - format_name: property returning format identifier
    """

    @abstractmethod
    def read_collection(self, path: str) -> dict[str, Any]:
        """
        Read COLLECTION.json from dataset.

        Format-specific: ZIP reads from offset, FOLDER from disk, TacoCat from memory.
        Must return dict with at least: id, taco:pit_schema, taco:field_schema
        """
        pass

    @abstractmethod
    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        root_path: str,
    ) -> None:
        """
        Create DuckDB views with internal:gdal_vsi column for GDAL access.

        VSI path format varies by backend:
        - ZIP: /vsisubfile/{offset}_{size},{zip_path}
        - FOLDER: {base_path}/DATA/{id} or {base_path}/DATA/{relative_path}
        - TacoCat: /vsisubfile/{offset}_{size},{base_path}{source_file}

        Views enable lazy SQL queries without loading rasters into memory.
        """
        pass

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Format identifier: 'zip', 'folder', or 'tacocat'."""
        pass

    def load(self, path: str) -> TacoDataset:
        """
        Template method for loading datasets.

        Default implementation (used by FOLDER):
        1. Read collection metadata
        2. Create DuckDB connection with spatial extension
        3. Convert path to VSI format
        4. Setup views with internal:gdal_vsi column
        5. Create TacoDataset with lazy SQL interface

        ZIP and TacoCat override this completely for in-memory loading.
        """
        # Read collection metadata
        collection = self.read_collection(path)

        # Setup DuckDB with spatial support
        db = duckdb.connect(":memory:")
        try:
            db.execute("INSTALL spatial")
            db.execute("LOAD spatial")
        except Exception:
            pass  # Spatial methods will fail with clear error if used

        # Convert to VSI format and setup views
        root_path = to_vsi_root(path)
        self.setup_duckdb_views(db, root_path)

        # Create 'data' alias for level0
        db.execute("CREATE VIEW data AS SELECT * FROM level0")

        # Build dataset
        schema = PITSchema(collection["taco:pit_schema"])

        return TacoDataset.model_construct(
            # Public metadata
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
            # Private internals
            _path=path,
            _format=self.format_name,
            _collection=collection,
            _duckdb=db,
            _view_name="data",
            _root_path=root_path,
        )