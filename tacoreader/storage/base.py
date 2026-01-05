"""Abstract base class for TACO format backends.

Defines common interface for all backends (ZIP, FOLDER, TacoCat).
All backends must implement load() with their specific loading strategy.

Main class:
    TacoBackend: Abstract base class with common utilities
"""

import json
from abc import ABC, abstractmethod
from typing import Any

import duckdb

from tacoreader._constants import (
    COLUMN_ID,
    DEFAULT_VIEW_NAME,
    PADDING_PREFIX,
)
from tacoreader._logging import get_logger
from tacoreader.dataset import TacoDataset
from tacoreader.schema import PITSchema

logger = get_logger(__name__)


class TacoBackend(ABC):
    """Base class for TACO backends.

    All backends implement format-specific loading strategies:
    - FOLDER: lazy disk access (local) or memory loading (remote)
    - ZIP: offset-based parsing with HTTP request grouping
    - TacoCat: binary header parsing with consolidated metadata

    All backends must implement:
    - load(): format-specific dataset loading
    - read_collection(): format-specific COLLECTION.json reading
    - setup_duckdb_views(): create views with internal:gdal_vsi column
    - format_name: property returning format identifier
    """

    @abstractmethod
    def read_collection(self, path: str) -> dict[str, Any]:
        """Read COLLECTION.json from dataset.

        Format-specific: ZIP reads from offset, FOLDER from disk, TacoCat from memory.
        Must return dict with at least: id, taco:pit_schema, taco:field_schema
        """
        pass  # pragma: no cover

    @abstractmethod
    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        level_ids: list[int],
        vsi_base_path: str,
    ) -> None:
        """Create DuckDB views with internal:gdal_vsi column for GDAL access.

        Args:
            db: DuckDB connection
            level_ids: List of level IDs to create views for
            vsi_base_path: Base path for constructing GDAL VSI paths
                - ZIP: path to .tacozip file
                - FOLDER: path to dataset directory
                - TacoCat: path to directory containing .tacozip files (NOT .tacocat/)

        VSI path format varies by backend:
        - ZIP: /vsisubfile/{offset}_{size},{zip_path}
        - FOLDER: {base_path}/DATA/{id} or {base_path}/DATA/{relative_path}
        - TacoCat: /vsisubfile/{offset}_{size},{base_path}{source_file}

        Views enable lazy SQL queries without loading rasters into memory.
        """
        pass  # pragma: no cover

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Format identifier: 'zip', 'folder', or 'tacocat'."""
        pass  # pragma: no cover

    @abstractmethod
    def load(self, path: str, cache: bool = True) -> TacoDataset:
        """Load dataset from path.

        Each backend implements format-specific loading:
        - FOLDER: lazy disk access (local) or memory loading (remote)
        - ZIP: offset-based parsing with request grouping
        - TacoCat: binary header parsing and consolidated metadata

        Args:
            path: Dataset path (local filesystem or cloud URL)
            cache: Use disk cache for remote datasets (default True)
                Only applicable for TacoCat backend; ignored by others.

        Returns:
            Fully loaded TacoDataset instance
        """
        pass  # pragma: no cover

    def _setup_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        """Create DuckDB connection with spatial extension if available."""
        db = duckdb.connect(":memory:")
        try:
            db.execute("INSTALL spatial")
            db.execute("LOAD spatial")
            logger.debug("DuckDB spatial extension loaded")
        except Exception as e:  # pragma: no cover
            logger.debug(f"Spatial extension not available: {e}")

        return db

    @staticmethod
    def _build_view_filter() -> str:
        """SQL WHERE clause for filtering padding samples from views."""
        return f"{COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'"

    def _parse_collection_json(self, collection_bytes: bytes, path: str) -> dict[str, Any]:
        """Parse COLLECTION.json with consistent error handling."""
        try:
            return json.loads(collection_bytes)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid COLLECTION.json in {self.format_name} format at {path}: {e.msg}",
                e.doc,
                e.pos,
            ) from e

    def _finalize_dataset(
        self,
        db: duckdb.DuckDBPyConnection,
        path: str,
        vsi_base_path: str,
        collection: dict[str, Any],
        level_ids: list[int],
    ) -> TacoDataset:
        """Common dataset finalization after metadata loading.

        Creates views, data alias, schema, and constructs TacoDataset.

        Args:
            db: DuckDB connection with registered tables
            path: Original path (for error messages/logging)
            vsi_base_path: Base path for GDAL VSI construction
            collection: Parsed COLLECTION.json
            level_ids: List of available level IDs
        """
        # Create views with internal:gdal_vsi
        self.setup_duckdb_views(db, level_ids, vsi_base_path)
        logger.debug("Created DuckDB views")

        # Create data alias for level0
        db.execute(f"CREATE VIEW {DEFAULT_VIEW_NAME} AS SELECT * FROM level0")

        # Build PIT schema
        schema = PITSchema(collection["taco:pit_schema"])

        # Construct dataset
        dataset = TacoDataset.model_construct(
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
            _view_name=DEFAULT_VIEW_NAME,
            _vsi_base_path=vsi_base_path,
            # JOIN tracking (for export validation in tacotoolbox)
            _has_level1_joins=False,
            _joined_levels=set(),
        )

        return dataset
