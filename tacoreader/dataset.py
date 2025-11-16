"""
TacoDataset - metadata container with lazy SQL interface.

Provides STAC-like metadata with DuckDB connection for lazy SQL queries.
Queries not executed until .data is called.
"""

import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from tacoreader.schema import PITSchema


class TacoDataset(BaseModel):
    """
    TACO dataset with lazy SQL interface.

    Metadata container with lazy query via DuckDB. Queries create views
    without materializing data until .data is accessed.

    Public attributes (STAC-like):
        id, version, description, tasks, extent, providers, licenses,
        title, curators, keywords, pit_schema

    Private attributes:
        _path, _format, _collection, _duckdb, _view_name, _root_path,
        _has_level1_joins, _joined_levels

    Examples:
        ds = load("data.tacozip")

        # Lazy queries
        peru = ds.sql("SELECT * FROM data WHERE country = 'Peru'")
        low_cloud = peru.sql("SELECT * FROM data WHERE cloud_cover < 10")

        # Materialization
        tdf = low_cloud.data
        child = tdf.read(0)
    """

    # Public metadata (STAC-like)
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

    # Private attributes
    _path: str = PrivateAttr()
    _format: Literal["zip", "folder", "tacocat"] = PrivateAttr()
    _collection: dict[str, Any] = PrivateAttr()
    _duckdb: Any = PrivateAttr(default=None)
    _view_name: str = PrivateAttr(default="data")
    _root_path: str = PrivateAttr(default="")

    # JOIN tracking (for migrate() validation)
    _has_level1_joins: bool = PrivateAttr(default=False)
    _joined_levels: set[str] = PrivateAttr(default_factory=set)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def data(self) -> "TacoDataFrame":
        """
        Materialize current view to TacoDataFrame.

        Executes DuckDB query, loads data into memory.
        This is where lazy evaluation ends.
        """
        from tacoreader.dataframe import TacoDataFrame

        pdf = self._duckdb.execute(f"SELECT * FROM {self._view_name}").pl()

        return TacoDataFrame(
            data=pdf,
            format_type=self._format,
        )

    @property
    def field_schema(self) -> dict[str, Any]:
        """Field schema from collection metadata."""
        return self._collection.get("taco:field_schema", {})

    @property
    def collection(self) -> dict[str, Any]:
        """Complete COLLECTION.json content with all metadata."""
        return self._collection.copy()

    # ========================================================================
    # LAZY SQL INTERFACE
    # ========================================================================

    def sql(self, query: str) -> "TacoDataset":
        """
        Execute SQL query and return new TacoDataset with lazy view.

        Query NOT executed immediately - creates temp view.
        Always use 'data' as table name, auto-replaced with current view.

        Tracks JOINs with level1+ tables for migrate() validation.
        """
        new_view_name = f"view_{uuid.uuid4().hex[:8]}"

        # Replace 'data' with current view for chaining
        modified_query = query.replace("FROM data", f"FROM {self._view_name}")

        # Create temp view (lazy)
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

        new_schema = self.pit_schema.with_n(new_n)

        # Detect JOINs with level1+ tables
        join_pattern = r"\b(?:JOIN|FROM)\s+(level[1-5])\b"
        matches = re.findall(join_pattern, query, re.IGNORECASE)

        has_new_joins = len(matches) > 0
        new_joined_levels = set(matches) if has_new_joins else set()

        # Inherit + merge tracking
        final_has_joins = self._has_level1_joins or has_new_joins
        final_joined_levels = self._joined_levels.union(new_joined_levels)

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
            _duckdb=self._duckdb,
            _view_name=new_view_name,
            _root_path=self._root_path,
            _has_level1_joins=final_has_joins,
            _joined_levels=final_joined_levels,
        )

    # ========================================================================
    # STAC SHORTCUTS
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

        When level > 0, filters level0 samples based on children's metadata.
        """
        from tacoreader.stac import (
            build_bbox_sql,
            build_cascade_join_sql,
            detect_geometry_column,
            get_columns_for_level,
            validate_level_exists,
        )

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
            if geometry_col not in current_cols:
                raise ValueError(
                    f"Geometry column '{geometry_col}' not found.\n"
                    f"Available: {current_cols}"
                )

        sql_filter = build_bbox_sql(minx, miny, maxx, maxy, geometry_col, level)

        if level == 0:
            return self.sql(f"SELECT * FROM data WHERE {sql_filter}")
        else:
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

        When level > 0, filters level0 samples based on children's metadata.
        """
        from tacoreader.stac import (
            build_cascade_join_sql,
            build_intersects_sql,
            detect_geometry_column,
            get_columns_for_level,
            validate_level_exists,
        )

        validate_level_exists(self, level)

        if level == 0:
            current_cols = self.data.columns
        else:
            current_cols = get_columns_for_level(self, level)

        if geometry_col == "auto":
            geometry_col = detect_geometry_column(current_cols)
        else:
            if geometry_col not in current_cols:
                raise ValueError(
                    f"Geometry column '{geometry_col}' not found.\n"
                    f"Available: {current_cols}"
                )

        sql_filter = build_intersects_sql(geometry, geometry_col, level)

        if level == 0:
            return self.sql(f"SELECT * FROM data WHERE {sql_filter}")
        else:
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

        More conservative than filter_intersects.
        When level > 0, filters level0 samples based on children's metadata.
        """
        from tacoreader.stac import (
            build_cascade_join_sql,
            build_within_sql,
            detect_geometry_column,
            get_columns_for_level,
            validate_level_exists,
        )

        validate_level_exists(self, level)

        if level == 0:
            current_cols = self.data.columns
        else:
            current_cols = get_columns_for_level(self, level)

        if geometry_col == "auto":
            geometry_col = detect_geometry_column(current_cols)
        else:
            if geometry_col not in current_cols:
                raise ValueError(
                    f"Geometry column '{geometry_col}' not found.\n"
                    f"Available: {current_cols}"
                )

        sql_filter = build_within_sql(geometry, geometry_col, level)

        if level == 0:
            return self.sql(f"SELECT * FROM data WHERE {sql_filter}")
        else:
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

        Always uses time_start column.
        Auto-detects column type (INTEGER, VARCHAR, TIMESTAMP, DATE).
        When level > 0, filters level0 samples based on children's metadata.
        """
        from tacoreader.stac import (
            build_cascade_join_sql,
            build_datetime_sql,
            detect_time_column,
            get_columns_for_level,
            parse_datetime,
            validate_level_exists,
        )

        validate_level_exists(self, level)

        if level == 0:
            current_cols = self.data.columns
        else:
            current_cols = get_columns_for_level(self, level)

        if time_col == "auto":
            time_col = detect_time_column(current_cols)
        else:
            if time_col not in current_cols:
                raise ValueError(
                    f"Time column '{time_col}' not found.\n"
                    f"Available: {current_cols}"
                )

        start, end = parse_datetime(datetime_range)
        sql_filter = build_datetime_sql(start, end, time_col, level, self)

        if level == 0:
            return self.sql(f"SELECT * FROM data WHERE {sql_filter}")
        else:
            full_query = build_cascade_join_sql(
                self._view_name, level, sql_filter, self._format
            )
            return self.sql(full_query)

    # ========================================================================
    # REPRESENTATION
    # ========================================================================

    def __repr__(self) -> str:
        """Rich text representation of dataset metadata."""
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
        """Rich HTML representation for Jupyter notebooks."""
        from tacoreader._html import build_html_repr

        return build_html_repr(self)
