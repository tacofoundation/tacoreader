"""
TacoDataset - metadata container with lazy SQL interface.

Provides STAC-like metadata with DuckDB connection for lazy SQL queries.
Queries not executed until .data is called.

Backend-agnostic: Uses factory pattern to create TacoDataFrame instances
without importing specific backend implementations.
"""

import re
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from tacoreader._constants import DEFAULT_VIEW_NAME, SQL_JOIN_PATTERN
from tacoreader.schema import PITSchema

if TYPE_CHECKING:
    from tacoreader.dataframe.base import TacoDataFrame


class TacoDataset(BaseModel):
    """
    TACO dataset with lazy SQL interface.

    Metadata container with lazy query via DuckDB. Queries create views
    without materializing data until .data is accessed.

    Connection Management:
        Datasets own their DuckDB connection. Use context manager for
        automatic cleanup (recommended):

        # Automatic cleanup (recommended)
        with tacoreader.load("data.taco") as ds:
            result = ds.sql("SELECT * WHERE ...").data

        # Manual cleanup
        ds = tacoreader.load("data.taco")
        try:
            result = ds.sql("SELECT * WHERE ...").data
        finally:
            ds.close()

        Note: Child datasets from .sql() share parent's connection.
        Note: Without explicit close(), connection persists until process exit.

    Public attributes (STAC-like):
        id, version, description, tasks, extent, providers, licenses,
        title, curators, keywords, pit_schema

    Private attributes:
        _path, _format, _collection, _duckdb, _view_name, _root_path,
        _dataframe_backend, _owns_connection, _has_level1_joins, _joined_levels

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
    _view_name: str = PrivateAttr(default=DEFAULT_VIEW_NAME)
    _root_path: str = PrivateAttr(default="")
    _dataframe_backend: str = PrivateAttr(default="pyarrow")
    _owns_connection: bool = PrivateAttr(default=True)

    # JOIN tracking (for export validation in tacotoolbox)
    _has_level1_joins: bool = PrivateAttr(default=False)
    _joined_levels: set[str] = PrivateAttr(default_factory=set)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def close(self):
        """
        Close DuckDB connection and cleanup views.

        Only closes the connection if this dataset owns it.
        Datasets created with .sql() share the parent's connection.

        Note: Prefer using context manager (with statement) over manual close().
        """
        if not hasattr(self, "_duckdb") or self._duckdb is None:
            return

        # Always drop temp views
        if hasattr(self, "_view_name") and self._view_name != DEFAULT_VIEW_NAME:
            with suppress(Exception):
                self._duckdb.execute(f"DROP VIEW IF EXISTS {self._view_name}")

        # Close connection only if owner
        if hasattr(self, "_owns_connection") and self._owns_connection:
            with suppress(Exception):
                self._duckdb.close()
            self._duckdb = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False

    @property
    def data(self) -> "TacoDataFrame":
        """
        Materialize current view to TacoDataFrame.

        Executes DuckDB query, loads data into memory.
        This is where lazy evaluation ends.

        Uses backend factory to create appropriate TacoDataFrame instance.
        """
        from tacoreader.dataframe import create_dataframe

        # DuckDB always returns PyArrow Table
        arrow_table = self._duckdb.execute(
            f"SELECT * FROM {self._view_name}"
        ).fetch_arrow_table()

        # Factory creates backend-specific TacoDataFrame
        return create_dataframe(
            backend=self._dataframe_backend,
            arrow_table=arrow_table,
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

    def sql(self, query: str) -> "TacoDataset":
        """
        Execute SQL query and return new TacoDataset with lazy view.

        Query NOT executed immediately - creates temp view.
        Always use 'data' as table name, auto-replaced with current view.
        """
        new_view_name = f"view_{uuid.uuid4().hex[:8]}"

        # Replace 'data' with current view for chaining
        modified_query = query.replace(
            f"FROM {DEFAULT_VIEW_NAME}", f"FROM {self._view_name}"
        )

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

        # Detect JOINs with level1+ tables (case insensitive)
        matches = re.findall(SQL_JOIN_PATTERN, query, re.IGNORECASE)

        # Track if this query introduces level1+ JOINs
        has_new_joins = len(matches) > 0
        new_joined_levels = set(matches) if has_new_joins else set()

        # Inherit tracking from parent dataset
        inherited_has_joins = self._has_level1_joins
        inherited_joined_levels = self._joined_levels.copy()

        # Merge with current query's JOINs
        final_has_joins = inherited_has_joins or has_new_joins
        final_joined_levels = inherited_joined_levels.union(new_joined_levels)

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
            _dataframe_backend=self._dataframe_backend,
            _owns_connection=False,  # Child datasets don't own connection
            _has_level1_joins=final_has_joins,
            _joined_levels=final_joined_levels,
        )

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
        from tacoreader._stac import (
            _apply_stac_filter,
            build_bbox_sql,
            detect_geometry_column,
        )

        return _apply_stac_filter(
            dataset=self,
            level=level,
            column_name=geometry_col,
            column_auto_detect_fn=detect_geometry_column,
            sql_builder_fn=build_bbox_sql,
            sql_builder_args=(minx, miny, maxx, maxy),
        )

    def filter_datetime(
        self,
        datetime_range,
        time_col: str = "auto",
        level: int = 0,
    ) -> "TacoDataset":
        """
        Filter by temporal range (PySTAC-style).

        Always uses time_start column.
        All temporal columns are native TIMESTAMP type.
        When level > 0, filters level0 samples based on children's metadata.
        """
        from tacoreader._stac import (
            _apply_stac_filter,
            build_datetime_sql,
            detect_time_column,
            parse_datetime,
        )

        start, end = parse_datetime(datetime_range)

        return _apply_stac_filter(
            dataset=self,
            level=level,
            column_name=time_col,
            column_auto_detect_fn=detect_time_column,
            sql_builder_fn=build_datetime_sql,
            sql_builder_args=(start, end),
        )

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

        # Spatial extent (always defined by auto_calculate_extent)
        spatial = self.extent["spatial"]
        lines.append(
            f"├── Spatial Extent: [{spatial[0]:.2f}°, {spatial[1]:.2f}°, "
            f"{spatial[2]:.2f}°, {spatial[3]:.2f}°]"
        )

        # Temporal extent (can be None for atemporal datasets)
        if not self.extent.get("temporal"):
            lines.append("├── Temporal Extent: Not defined (atemporal dataset)")
        else:
            temporal = self.extent["temporal"]
            # Format dates nicely (remove time if it's midnight)
            start_str = self._format_temporal_string(temporal[0])
            end_str = self._format_temporal_string(temporal[1])
            lines.append(f"├── Temporal Extent: {start_str} → {end_str}")

        lines.append("│")
        lines.append(f"└── Level 0: {self.pit_schema.root['n']} rows")

        return "\n".join(lines)

    def _format_temporal_string(self, iso_string: str) -> str:
        """
        Format ISO 8601 datetime string for display.

        If time is midnight (00:00:00), show only date.
        Otherwise show full datetime.
        """
        # Remove 'Z' suffix for parsing
        clean_str = iso_string.replace("Z", "")

        # Check if it ends with T00:00:00 (midnight)
        if "T00:00:00" in clean_str:
            # Return only date part
            return clean_str.split("T")[0]
        else:
            # Return full datetime (YYYY-MM-DD HH:MM:SS)
            date_part, time_part = clean_str.split("T")
            # Truncate microseconds if present
            if "." in time_part:
                time_part = time_part.split(".")[0]
            return f"{date_part} {time_part}"

    def _repr_html_(self):
        """Rich HTML representation for Jupyter notebooks."""
        from tacoreader._html import build_html_repr

        return build_html_repr(self)
