"""
Abstract base class for TACO format backends.

Defines the common interface for all backend implementations (ZIP, FOLDER, TacoCat)
using the Template Method pattern. The load() method coordinates the complete
loading process while subclasses implement format-specific details.

Main class:
    TacoBackend: Abstract base class with template method load()
"""

from abc import ABC, abstractmethod
from typing import Any

import duckdb

from tacoreader.dataset import TacoDataset
from tacoreader.schema import PITSchema
from tacoreader.utils.vsi import to_vsi_root

# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================


class TacoBackend(ABC):
    """
    Abstract base class for TACO format backends.

    Implements the Template Method pattern: load() orchestrates the complete
    loading workflow while subclasses provide format-specific implementations
    for reading and setting up views.

    The template method guarantees consistent behavior across all formats:
    1. Read collection metadata
    2. Setup DuckDB connection with views
    3. Create TacoDataset with lazy SQL interface

    Subclasses must implement:
    - read_collection(): Format-specific COLLECTION.json reading
    - setup_duckdb_views(): Create views for querying metadata
    - format_name: Property returning format identifier string

    Note:
        ZIP and TacoCat backends override load() completely for in-memory
        loading. FOLDER backend uses the template method since files are
        already on disk.

    Example:
        >>> class MyBackend(TacoBackend):
        ...     @property
        ...     def format_name(self) -> str:
        ...         return "myformat"
        ...
        ...     def read_collection(self, path: str) -> dict:
        ...         # Custom implementation
        ...         pass
        ...
        ...     def setup_duckdb_views(self, db, root) -> None:
        ...         # Custom implementation
        ...         pass
    """

    # ========================================================================
    # ABSTRACT METHODS (MUST BE IMPLEMENTED BY SUBCLASSES)
    # ========================================================================

    @abstractmethod
    def read_collection(self, path: str) -> dict[str, Any]:
        """
        Read COLLECTION.json from dataset.

        Subclasses implement format-specific logic for reading collection
        metadata. This is the first step in the loading process.

        Args:
            path: Dataset path (format-specific)

        Returns:
            Dictionary containing COLLECTION.json content with keys:
            - id: Dataset identifier
            - dataset_version: Dataset version string
            - description: Dataset description
            - taco:pit_schema: PIT schema structure
            - taco:field_schema: Field definitions per level
            - (other STAC-like metadata)

        Raises:
            ValueError: If collection cannot be read or is invalid
            IOError: If file access fails

        Example:
            >>> collection = backend.read_collection("data.tacozip")
            >>> print(collection["id"])
            'sentinel2-l2a'
        """
        pass

    @abstractmethod
    def setup_duckdb_views(
        self,
        db: duckdb.DuckDBPyConnection,
        root_path: str,
    ) -> None:
        """
        Create DuckDB views with internal:gdal_vsi column.

        Each format constructs GDAL VSI paths differently based on how
        files are stored and accessed:

        - ZIP: /vsisubfile/{offset}_{size},{zip_path}
        - FOLDER: {base_path}/DATA/{id} or {base_path}/DATA/{relative_path}
        - TacoCat: /vsisubfile/{offset}_{size},{base_path}{source_file}

        The views enable lazy SQL queries over metadata without loading
        all data into memory. The internal:gdal_vsi column allows GDAL
        to open files directly from their storage location.

        Args:
            db: DuckDB in-memory connection
            root_path: VSI root path for constructing file paths

        Example:
            >>> import duckdb
            >>> db = duckdb.connect(":memory:")
            >>> backend.setup_duckdb_views(db, "/vsis3/bucket/data.tacozip")
            >>> # Views created: level0, level1, etc.
            >>> result = db.execute("SELECT * FROM level0 LIMIT 1").fetchone()
        """
        pass

    @property
    @abstractmethod
    def format_name(self) -> str:
        """
        Format identifier string.

        Returns format type as lowercase string: 'zip', 'folder', or 'tacocat'.
        Used for format detection and display.

        Returns:
            Format identifier string

        Example:
            >>> backend = ZipBackend()
            >>> print(backend.format_name)
            'zip'
        """
        pass

    # ========================================================================
    # TEMPLATE METHOD (COORDINATES LOADING PROCESS)
    # ========================================================================

    def load(self, path: str) -> TacoDataset:
        """
        Template method: coordinates complete loading process.

        Orchestrates the loading workflow using format-specific implementations
        of abstract methods. This ensures all formats follow the same process:

        1. Read collection metadata (format-specific)
        2. Create DuckDB in-memory connection
        3. Load spatial extension for WKB geometry support
        4. Convert path to VSI format for GDAL
        5. Setup DuckDB views with internal:gdal_vsi column (format-specific)
        6. Create 'data' view pointing to level0
        7. Construct TacoDataset with lazy SQL interface

        Note:
            ZIP and TacoCat backends override this method completely to handle
            in-memory loading. Only FOLDER backend uses this template method.

        Args:
            path: Dataset path (format-specific)

        Returns:
            TacoDataset with lazy SQL interface and DuckDB connection

        Raises:
            ValueError: If any step fails (collection read, setup, etc.)
            IOError: If file access fails

        Example:
            >>> backend = FolderBackend()
            >>> dataset = backend.load("data/")
            >>> print(dataset.id)
            'sentinel2-l2a'
            >>>
            >>> # Query with lazy evaluation
            >>> peru = dataset.sql("SELECT * FROM data WHERE country = 'Peru'")
            >>> df = peru.data  # Materialization happens here
        """
        # Step 1: Read collection metadata (format-specific)
        collection = self.read_collection(path)

        # Step 2: Create DuckDB in-memory connection
        db = duckdb.connect(":memory:")

        # Step 3: Try to load spatial extension for WKB geometry support
        try:
            db.execute("INSTALL spatial")
            db.execute("LOAD spatial")
        except Exception:
            # Spatial extension not available
            # Spatial filtering methods will fail with clear error if used
            pass

        # Step 4: Convert path to VSI format for GDAL
        root_path = to_vsi_root(path)

        # Step 5: Setup DuckDB views (format-specific)
        self.setup_duckdb_views(db, root_path)

        # Step 6: Create 'data' view as alias for level0
        db.execute("CREATE VIEW data AS SELECT * FROM level0")

        # Step 7: Extract PIT schema for validation and display
        schema = PITSchema(collection["taco:pit_schema"])

        # Step 8: Construct TacoDataset with all context
        return TacoDataset.model_construct(
            # Public metadata fields
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
            # Private attributes (underscored)
            _path=path,
            _format=self.format_name,
            _collection=collection,
            _consolidated_files={},
            _duckdb=db,
            _view_name="data",
            _root_path=root_path,
        )