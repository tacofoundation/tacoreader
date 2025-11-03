"""
TacoDataset - metadata container with lazy SQL interface.

Provides STAC-like metadata with DuckDB connection for lazy SQL queries.
Queries are not executed until .data is called.
"""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from tacoreader.schema import PITSchema

# ============================================================================
# TACODATASET
# ============================================================================


class TacoDataset(BaseModel):
    """
    TACO dataset with lazy SQL interface.

    Container for dataset metadata with lazy query capabilities via DuckDB.
    Queries create views without materializing data until .data is accessed.

    Public attributes (STAC-like metadata):
        id: Dataset identifier
        version: Dataset version string
        description: Dataset description
        tasks: Machine learning task types
        extent: Spatial and temporal boundaries
        providers: Dataset creators/contributors
        licenses: License identifiers
        title: Human-friendly title (optional)
        curators: Dataset curators (optional)
        keywords: Searchable tags (optional)
        schema: PIT schema for hierarchy structure

    Private attributes:
        _path: Original dataset path
        _format: Backend format type ("zip", "folder", "tacocat")
        _collection: Full COLLECTION.json content
        _consolidated_files: Cached metadata file paths
        _duckdb: DuckDB in-memory connection
        _view_name: Current view name for queries
        _root_path: VSI root path for file access

    Example:
        >>> dataset = load("data.tacozip")
        >>> print(dataset.id)
        'sentinel2-l2a'
        >>>
        >>> # Lazy queries
        >>> peru = dataset.sql("SELECT * FROM data WHERE country = 'Peru'")
        >>> low_cloud = peru.sql("SELECT * FROM data WHERE cloud_cover < 10")
        >>>
        >>> # Materialization
        >>> tdf = low_cloud.data
        >>> child = tdf.read(0)
    """

    # Public metadata fields (STAC-like)
    id: str
    version: str
    description: str
    tasks: list[str]
    extent: dict[str, Any]
    providers: list[dict[str, Any]]
    licenses: list[str]
    title: str | None = None
    curators: list[dict[str, Any]] | None = None
    keywords: list[str] | None = None
    pit_schema: PITSchema

    # Private attributes (internal state)
    _path: str = PrivateAttr()
    _format: Literal["zip", "folder", "tacocat"] = PrivateAttr()
    _collection: dict[str, Any] = PrivateAttr()
    _consolidated_files: dict[str, str] = PrivateAttr()
    _duckdb: Any = PrivateAttr(default=None)
    _view_name: str = PrivateAttr(default="data")
    _root_path: str = PrivateAttr(default="")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def data(self) -> "TacoDataFrame":
        """
        Materialize current view to TacoDataFrame.

        Executes DuckDB query and returns TacoDataFrame with navigation methods.
        This is where lazy evaluation ends and data is loaded into memory.

        Returns:
            TacoDataFrame with read() method for hierarchical navigation

        Example:
            >>> tdf = dataset.data
            >>> child = tdf.read(0)
            >>> child = tdf.read("sample_001")
            >>>
            >>> filtered = dataset.sql("SELECT * FROM data WHERE country = 'Peru'").data
            >>> filtered.read(0)
        """
        from tacoreader.dataframe import TacoDataFrame

        pdf = self._duckdb.execute(f"SELECT * FROM {self._view_name}").pl()

        return TacoDataFrame(
            data=pdf,
            format_type=self._format,
            consolidated_files=self._consolidated_files,
        )

    @property
    def field_schema(self) -> dict[str, Any]:
        """
        Field schema from collection metadata.

        Returns:
            Dictionary mapping level names to field definitions
            Example: {"level0": [["id", "string"], ["type", "string"], ...]}
        """
        return self._collection.get("taco:field_schema", {})

    # ========================================================================
    # LAZY SQL INTERFACE
    # ========================================================================

    def sql(self, query: str) -> "TacoDataset":
        """
        Execute SQL query and return new TacoDataset with lazy view.

        Query is NOT executed immediately. Creates a temporary view in DuckDB
        that references the current view. Multiple queries can be chained.

        Always use 'data' as the table name in queries, even when chaining.
        The method automatically replaces 'data' with the current view name.

        Args:
            query: SQL query string (use table name 'data')

        Returns:
            New TacoDataset with filtered view (still lazy)

        Example:
            >>> peru = dataset.sql("SELECT * FROM data WHERE country = 'Peru'")
            >>> low_cloud = peru.sql("SELECT * FROM data WHERE cloud_cover < 10")
            >>> tdf = low_cloud.data
        """
        new_view_name = f"view_{uuid.uuid4().hex[:8]}"

        # Replace 'data' with current view for chaining
        modified_query = query.replace("FROM data", f"FROM {self._view_name}")

        # Create temporary view (lazy)
        self._duckdb.execute(
            f"""
            CREATE TEMP VIEW {new_view_name} AS
            {modified_query}
        """
        )

        # Count rows for schema update
        new_n = self._duckdb.execute(
            f"SELECT COUNT(*) FROM {new_view_name}"
        ).fetchone()[0]

        # Update schema with new count
        new_schema = self.pit_schema.with_n(new_n)

        # Return new TacoDataset with same connection, different view
        return TacoDataset.model_construct(
            id=self.id,
            version=self.version,
            description=self.description,
            tasks=self.tasks,
            extent=self.extent,
            providers=self.providers,
            licenses=self.licenses,
            title=self.title,
            curators=self.curators,
            keywords=self.keywords,
            pit_schema=new_schema,
            _path=self._path,
            _format=self._format,
            _collection=self._collection,
            _consolidated_files=self._consolidated_files,
            _duckdb=self._duckdb,  # Shared connection
            _view_name=new_view_name,  # New view
            _root_path=self._root_path,
        )

    # ========================================================================
    # STAC shortcuts
    # ========================================================================

    def filter_bbox(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        geometry_col: str = "auto",
        level: int = 0,
    ) -> "TacoDataset":
        """
        Filter by bounding box (PySTAC-style).

        Returns samples whose geometry is within the specified bounding box.
        When level > 0, filters level0 samples based on their children's metadata.

        Args:
            minx: Minimum longitude/X coordinate
            miny: Minimum latitude/Y coordinate
            maxx: Maximum longitude/X coordinate
            maxy: Maximum latitude/Y coordinate
            geometry_col: Geometry column name or "auto" for auto-detection
            level: Level where geometry column exists (default: 0)
                - level=0: Filter directly on level0 (standard behavior)
                - level=1: Filter level0 samples by level1 children metadata
                - level=2: Filter level0 samples by level2 descendants metadata

        Returns:
            New TacoDataset with filtered samples (still lazy)

        Raises:
            ValueError: If level does not exist or geometry column not found

        Example:
            >>> # Standard: filter by level0 geometry
            >>> peru = dataset.filter_bbox(-81, -18, -68, 0)
            >>>
            >>> # Multi-level: filter samples by children's geometry
            >>> filtered = dataset.filter_bbox(-81, -18, -68, 0, level=1)
            >>> # Returns level0 samples that have children within bbox
            >>>
            >>> # Chain with other filters
            >>> result = dataset.filter_bbox(-81, -18, -68, 0, level=1).filter_datetime("2023/2024", level=1)
        """
        from tacoreader.stac import (
            build_bbox_sql,
            build_cascade_join_sql,
            detect_geometry_column,
            get_columns_for_level,
            validate_geometry_column,
            validate_level_exists,
        )

        # Validate level exists
        validate_level_exists(self, level)

        # Get columns for target level
        if level == 0:
            current_cols = self.data.columns
        else:
            current_cols = get_columns_for_level(self, level)

        # Detect or validate geometry column
        if geometry_col == "auto":
            geometry_col = detect_geometry_column(current_cols)
        else:
            # User provided explicit column name - just validate it exists
            if geometry_col not in current_cols:
                raise ValueError(
                    f"Requested geometry column '{geometry_col}' not found.\n"
                    f"Available columns: {current_cols}"
                )

        # Build SQL filter
        sql_filter = build_bbox_sql(minx, miny, maxx, maxy, geometry_col, level)

        # Apply filter based on level
        if level == 0:
            # Standard: filter directly on current view
            return self.sql(f"SELECT * FROM data WHERE {sql_filter}")
        else:
            # Multi-level: build cascading JOINs
            full_query = build_cascade_join_sql(
                self._view_name, level, sql_filter, self._format
            )
            return self.sql(full_query)

    def filter_intersects(
        self,
        geometry,
        geometry_col: str = "auto",
        level: int = 0,
    ) -> "TacoDataset":
        """
        Filter by geometry intersection (PySTAC-style).

        Returns samples whose geometry intersects with the specified geometry.
        When level > 0, filters level0 samples based on their children's metadata.

        Args:
            geometry: Geometry to intersect with (WKT, GeoJSON, shapely, etc.)
            geometry_col: Geometry column name or "auto" for auto-detection
            level: Level where geometry column exists (default: 0)
                - level=0: Filter directly on level0 (standard behavior)
                - level=1: Filter level0 samples by level1 children metadata
                - level=2: Filter level0 samples by level2 descendants metadata

        Returns:
            New TacoDataset with filtered samples (still lazy)

        Raises:
            ValueError: If level does not exist or geometry column not found

        Example:
            >>> from shapely.geometry import box
            >>> aoi = box(-81, -18, -68, 0)
            >>>
            >>> # Standard: filter by level0 geometry
            >>> filtered = dataset.filter_intersects(aoi)
            >>>
            >>> # Multi-level: filter samples by children's geometry
            >>> filtered = dataset.filter_intersects(aoi, level=1)
        """
        from tacoreader.stac import (
            build_cascade_join_sql,
            build_intersects_sql,
            detect_geometry_column,
            get_columns_for_level,
            validate_geometry_column,
            validate_level_exists,
        )

        # Validate level exists
        validate_level_exists(self, level)

        # Get columns for target level
        if level == 0:
            current_cols = self.data.columns
        else:
            current_cols = get_columns_for_level(self, level)

        # Detect or validate geometry column
        if geometry_col == "auto":
            geometry_col = detect_geometry_column(current_cols)
        else:
            # User provided explicit column name - just validate it exists
            if geometry_col not in current_cols:
                raise ValueError(
                    f"Requested geometry column '{geometry_col}' not found.\n"
                    f"Available columns: {current_cols}"
                )

        # Build SQL filter
        sql_filter = build_intersects_sql(geometry, geometry_col, level)

        # Apply filter based on level
        if level == 0:
            # Standard: filter directly on current view
            return self.sql(f"SELECT * FROM data WHERE {sql_filter}")
        else:
            # Multi-level: build cascading JOINs
            full_query = build_cascade_join_sql(
                self._view_name, level, sql_filter, self._format
            )
            return self.sql(full_query)

    def filter_within(
        self,
        geometry,
        geometry_col: str = "auto",
        level: int = 0,
    ) -> "TacoDataset":
        """
        Filter by geometry containment (PySTAC-style).

        Returns samples whose geometry is completely within the specified geometry.
        More conservative than filter_intersects.
        When level > 0, filters level0 samples based on their children's metadata.

        Args:
            geometry: Geometry to check containment (WKT, GeoJSON, shapely, etc.)
            geometry_col: Geometry column name or "auto" for auto-detection
            level: Level where geometry column exists (default: 0)
                - level=0: Filter directly on level0 (standard behavior)
                - level=1: Filter level0 samples by level1 children metadata
                - level=2: Filter level0 samples by level2 descendants metadata

        Returns:
            New TacoDataset with filtered samples (still lazy)

        Raises:
            ValueError: If level does not exist or geometry column not found

        Example:
            >>> # Only samples completely within Peru
            >>> from shapely.geometry import box
            >>> peru_bbox = box(-81, -18, -68, 0)
            >>>
            >>> # Standard: filter by level0 geometry
            >>> filtered = dataset.filter_within(peru_bbox)
            >>>
            >>> # Multi-level: filter samples by children's geometry
            >>> filtered = dataset.filter_within(peru_bbox, level=1)
        """
        from tacoreader.stac import (
            build_cascade_join_sql,
            build_within_sql,
            detect_geometry_column,
            get_columns_for_level,
            validate_geometry_column,
            validate_level_exists,
        )

        # Validate level exists
        validate_level_exists(self, level)

        # Get columns for target level
        if level == 0:
            current_cols = self.data.columns
        else:
            current_cols = get_columns_for_level(self, level)

        # Detect or validate geometry column
        if geometry_col == "auto":
            geometry_col = detect_geometry_column(current_cols)
        else:
            # User provided explicit column name - just validate it exists
            if geometry_col not in current_cols:
                raise ValueError(
                    f"Requested geometry column '{geometry_col}' not found.\n"
                    f"Available columns: {current_cols}"
                )

        # Build SQL filter
        sql_filter = build_within_sql(geometry, geometry_col, level)

        # Apply filter based on level
        if level == 0:
            # Standard: filter directly on current view
            return self.sql(f"SELECT * FROM data WHERE {sql_filter}")
        else:
            # Multi-level: build cascading JOINs
            full_query = build_cascade_join_sql(
                self._view_name, level, sql_filter, self._format
            )
            return self.sql(full_query)

    def filter_datetime(
        self,
        datetime_range,
        time_col: str = "auto",
        level: int = 0,
    ) -> "TacoDataset":
        """
        Filter by temporal range (PySTAC-style).

        Returns samples within the specified time range.
        Always uses time_start column.
        When level > 0, filters level0 samples based on their children's metadata.

        Automatically detects column type (INTEGER, VARCHAR, TIMESTAMP, DATE) and
        generates appropriate SQL comparison.

        Args:
            datetime_range: Temporal specification:
                - String range: "2023-01-01/2023-12-31"
                - Single datetime: datetime(2023, 1, 1)
                - Datetime tuple: (start_dt, end_dt)
            time_col: Time column name or "auto" for auto-detection
            level: Level where time column exists (default: 0)
                - level=0: Filter directly on level0 (standard behavior)
                - level=1: Filter level0 samples by level1 children metadata
                - level=2: Filter level0 samples by level2 descendants metadata

        Returns:
            New TacoDataset with filtered samples (still lazy)

        Raises:
            ValueError: If level does not exist or time column not found

        Example:
            >>> # Standard: filter by level0 time
            >>> year_2023 = dataset.filter_datetime("2023-01-01/2023-12-31")
            >>>
            >>> # Multi-level: filter samples by children's time
            >>> filtered = dataset.filter_datetime("2023-01-01/2023-12-31", level=1)
            >>>
            >>> # Chain with spatial filter
            >>> result = dataset.filter_bbox(-81, -18, -68, 0, level=1).filter_datetime("2023/2024", level=1)
        """
        from tacoreader.stac import (
            build_cascade_join_sql,
            build_datetime_sql,
            detect_time_column,
            get_columns_for_level,
            parse_datetime,
            validate_level_exists,
            validate_time_column,
        )

        # Validate level exists
        validate_level_exists(self, level)

        # Get columns for target level
        if level == 0:
            current_cols = self.data.columns
        else:
            current_cols = get_columns_for_level(self, level)

        # Detect or validate time column
        if time_col == "auto":
            time_col = detect_time_column(current_cols)
        else:
            # User provided explicit column name - just validate it exists
            if time_col not in current_cols:
                raise ValueError(
                    f"Requested time column '{time_col}' not found.\n"
                    f"Available columns: {current_cols}"
                )

        # Parse datetime input
        start, end = parse_datetime(datetime_range)

        # Build SQL filter (auto-detects column type)
        sql_filter = build_datetime_sql(start, end, time_col, level, self)

        # Apply filter based on level
        if level == 0:
            # Standard: filter directly on current view
            return self.sql(f"SELECT * FROM data WHERE {sql_filter}")
        else:
            # Multi-level: build cascading JOINs
            full_query = build_cascade_join_sql(
                self._view_name, level, sql_filter, self._format
            )
            return self.sql(full_query)

    # ========================================================================
    # REPRESENTATION
    # ========================================================================

    def __repr__(self) -> str:
        """
        Rich text representation of dataset metadata.

        Returns:
            Multi-line string with tree-like structure showing key metadata
        """
        lines = [f"<TacoDataset '{self.id}'>"]
        lines.append(f"├── Version: {self.version}")

        desc_short = (
            self.description[:80] + "..."
            if len(self.description) > 80
            else self.description
        )
        lines.append(f"├── Description: {desc_short}")
        lines.append(f"├── Tasks: {', '.join(self.tasks)}")

        if "spatial" in self.extent:
            spatial = self.extent["spatial"]
            lines.append(
                f"├── Spatial Extent: [{spatial[0]:.1f}, {spatial[1]:.1f}, "
                f"{spatial[2]:.1f}, {spatial[3]:.1f}]"
            )

        if self.extent.get("temporal"):
            temporal = self.extent["temporal"]
            lines.append(f"├── Temporal Extent: {temporal[0]} -> {temporal[1]}")

        lines.append("│")
        lines.append(f"└── Level 0: {self.pit_schema.root['n']} rows")

        return "\n".join(lines)

    def _repr_html_(self):
        """
        Rich HTML representation for Jupyter notebooks.

        Returns:
            HTML string with interactive display
        """
        from tacoreader._html import build_html_repr

        return build_html_repr(self)