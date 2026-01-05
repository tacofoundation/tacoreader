"""TacoDataset - metadata container with lazy SQL interface.

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

from tacoreader._constants import (
    DEFAULT_VIEW_NAME,
    LEVEL_VIEW_PREFIX,
    SQL_JOIN_PATTERN,
    STATS_SUPPORTED_COLUMNS,
    STATS_WEIGHT_COLUMN,
)
from tacoreader._exceptions import TacoQueryError
from tacoreader._logging import get_logger
from tacoreader.schema import PITSchema

if TYPE_CHECKING:
    import numpy as np

    from tacoreader.dataframe.base import TacoDataFrame

logger = get_logger(__name__)


class TacoDataset(BaseModel):
    """TACO dataset with lazy SQL interface.

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
        _path, _format, _collection, _duckdb, _view_name, _vsi_base_path,
        _dataframe_backend, _owns_connection, _has_level1_joins, _joined_levels

    Examples:
        ds = load("data.tacozip")

        # Lazy queries
        peru = ds.sql("SELECT * FROM data WHERE country = 'Peru'")
        low_cloud = peru.sql("SELECT * FROM data WHERE cloud_cover < 10")

        # Materialization
        tdf = low_cloud.data
        child = tdf.read(0)

        # Statistics
        mean_band2 = ds.stats_mean(band=2)
        std_bands = ds.stats_std(band=[0, 1, 2])
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
    _vsi_base_path: str = PrivateAttr(default="")
    _dataframe_backend: str = PrivateAttr(default="pyarrow")
    _owns_connection: bool = PrivateAttr(default=True)

    # JOIN tracking (for export validation in tacotoolbox)
    _has_level1_joins: bool = PrivateAttr(default=False)
    _joined_levels: set[str] = PrivateAttr(default_factory=set)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def close(self):
        """Close DuckDB connection and cleanup views.

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
        """Materialize current view to TacoDataFrame.

        Executes DuckDB query, loads data into memory.
        This is where lazy evaluation ends.

        Uses backend factory to create appropriate TacoDataFrame instance.
        """
        from tacoreader.dataframe import create_dataframe

        # DuckDB always returns PyArrow Table
        arrow_table = self._duckdb.execute(f"SELECT * FROM {self._view_name}").fetch_arrow_table()

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

    def navigation_columns(self, describe: bool = False) -> list[str] | dict[str, str]:
        """Get columns required for navigation and concat operations.

        When using .sql() with specific column selection (not SELECT *),
        include these columns to preserve .read() navigation and concat()
        compatibility.

        Args:
            describe: If True, return dict with column descriptions.
                      If False, return list of column names (default).

        Returns:
            list[str]: Column names if describe=False
            dict[str, str]: {column: description} if describe=True

        Example:
            >>> ds = tacoreader.load("data.tacozip")
            >>> ds.navigation_columns()
            ['id', 'type', 'internal:current_id', 'internal:offset', 'internal:size']

            >>> # Use with .sql() for specific column selection
            >>> user_cols = ['satellite', 'cloud_cover', 'date']
            >>> all_cols = user_cols + ds.navigation_columns()
            >>> filtered = ds.sql(f"SELECT {', '.join(all_cols)} FROM data WHERE cloud_cover < 10")

            >>> # View descriptions
            >>> ds.navigation_columns(describe=True)
            {'id': 'Unique sample identifier', 'type': 'Sample type (FILE or FOLDER)', ...}
        """
        from tacoreader._constants import (
            NAVIGATION_COLUMNS_BY_FORMAT,
            NAVIGATION_COLUMN_DESCRIPTIONS,
        )

        columns = NAVIGATION_COLUMNS_BY_FORMAT.get(self._format, frozenset())

        if describe:
            return {col: NAVIGATION_COLUMN_DESCRIPTIONS.get(col, "") for col in sorted(columns)}

        return sorted(columns)

    def sql(self, query: str) -> "TacoDataset":
        """Execute SQL query and return new TacoDataset with lazy view.

        Query NOT executed immediately - creates temp view.
        Always use 'data' as table name, auto-replaced with current view.
        """
        new_view_name = f"view_{uuid.uuid4().hex[:8]}"

        # Replace 'data' with current view for chaining
        modified_query = query.replace(f"FROM {DEFAULT_VIEW_NAME}", f"FROM {self._view_name}")

        # Create temp view (lazy)
        self._duckdb.execute(
            f"""
            CREATE TEMP VIEW {new_view_name} AS
            {modified_query}
        """
        )

        # Count rows for schema update
        new_n = self._duckdb.execute(f"SELECT COUNT(*) FROM {new_view_name}").fetchone()[0]

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
            _vsi_base_path=self._vsi_base_path,
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
        """Filter by bounding box (PySTAC-style).

        Args:
            minx: Minimum X coordinate (longitude)
            miny: Minimum Y coordinate (latitude)
            maxx: Maximum X coordinate (longitude)
            maxy: Maximum Y coordinate (latitude)
            geometry_col: Geometry column name (auto-detected if "auto")
            level: Hierarchy level to filter (0=direct, >0=cascade through children)

        Returns:
            Filtered TacoDataset

        Routing:
            - level=0: Simple filtering without JOINs (fast, stable)
            - level>0: Hierarchical filtering with multi-level JOINs (complex, experimental)
        """
        if level == 0:
            # Simple route: direct filtering on level0 without JOINs
            from tacoreader._stac import apply_simple_bbox_filter

            return apply_simple_bbox_filter(self, minx, miny, maxx, maxy, geometry_col)
        else:
            # Cascade route: hierarchical filtering with JOINs
            from tacoreader._stac_cascade import apply_cascade_bbox_filter

            return apply_cascade_bbox_filter(self, minx, miny, maxx, maxy, geometry_col, level)

    def filter_datetime(
        self,
        datetime_range,
        time_col: str = "auto",
        level: int = 0,
    ) -> "TacoDataset":
        """Filter by temporal range (PySTAC-style).

        Args:
            datetime_range: Temporal range as:
                - String range: "2023-01-01/2023-12-31"
                - Single datetime: datetime(2023, 1, 1)
                - Tuple: (start_dt, end_dt)
            time_col: Time column name (auto-detected if "auto")
            level: Hierarchy level to filter (0=direct, >0=cascade through children)

        Returns:
            Filtered TacoDataset

        Note:
            Automatically handles both TIMESTAMP and STRING date columns via TRY_CAST.

        Routing:
            - level=0: Simple filtering without JOINs (fast, stable)
            - level>0: Hierarchical filtering with multi-level JOINs (complex, experimental)
        """
        if level == 0:
            # Simple route: direct filtering on level0 without JOINs
            from tacoreader._stac import apply_simple_datetime_filter

            return apply_simple_datetime_filter(self, datetime_range, time_col)
        else:
            # Cascade route: hierarchical filtering with JOINs
            from tacoreader._stac_cascade import apply_cascade_datetime_filter

            return apply_cascade_datetime_filter(self, datetime_range, time_col, level)

    # Statistics API
    def _get_stats_column(self, level: int) -> str:
        """Find which stats column exists in the given level.

        Returns the column name if found, raises TacoQueryError if not.
        """
        level_view = f"{LEVEL_VIEW_PREFIX}{level}"

        # Get columns for this level
        result = self._duckdb.execute(f"DESCRIBE {level_view}").fetchall()
        columns = {row[0] for row in result}

        # Check for any supported stats column
        for stats_col in STATS_SUPPORTED_COLUMNS:
            if stats_col in columns:
                return stats_col

        raise TacoQueryError(
            f"Level {level} does not contain statistics.\n"
            f"Expected one of: {', '.join(STATS_SUPPORTED_COLUMNS)}\n"
            f"Available columns: {sorted(columns)}"
        )

    def _validate_stats_params(
        self,
        band: int | list[int],
        level: int,
        id: str | None,
    ) -> None:
        """Validate stats parameters according to RSUT rules."""
        # Validate level exists
        max_depth = self.pit_schema.max_depth()
        if level < 0 or level > max_depth:
            raise TacoQueryError(f"Level {level} does not exist.\nAvailable levels: 0 to {max_depth}")

        # Level > 0 requires id (RSUT: heterogeneous below level 1)
        if level > 0 and id is None:
            raise TacoQueryError(
                f"id is required for level > 0.\n"
                f"Level {level} may have heterogeneous structure across branches.\n"
                f"Specify which sample to aggregate: stats_*(band=..., level={level}, id='...')"
            )

        # Level 0: id is ignored (log for debugging)
        if level == 0 and id is not None:
            logger.debug(f"id='{id}' ignored for level=0 (aggregates all samples)")

    def _fetch_stats_table(self, level: int, id: str | None, stats_col: str):
        """Fetch PyArrow table with stats data for aggregation."""
        level_view = f"{LEVEL_VIEW_PREFIX}{level}"

        if level == 0:
            query = f'SELECT "{stats_col}", "{STATS_WEIGHT_COLUMN}" FROM {level_view}'
        else:
            # Navigate to children of specified id
            parent_level = level - 1
            parent_view = f"{LEVEL_VIEW_PREFIX}{parent_level}"

            query = f"""
                SELECT l."{stats_col}", l."{STATS_WEIGHT_COLUMN}"
                FROM {level_view} l
                INNER JOIN {parent_view} p ON l."internal:parent_id" = p."internal:current_id"
                WHERE p.id = '{id}'
            """

        return self._duckdb.execute(query).fetch_arrow_table()

    def _extract_band(
        self,
        result: "np.ndarray",
        band: int | list[int],
    ) -> "np.ndarray":
        """Extract specific band(s) from aggregated result."""
        if isinstance(band, int):
            if band < 0 or band >= result.shape[0]:
                raise TacoQueryError(f"Band {band} out of range.\nAvailable bands: 0 to {result.shape[0] - 1}")
            return result[band]
        else:
            for b in band:
                if b < 0 or b >= result.shape[0]:
                    raise TacoQueryError(f"Band {b} out of range.\nAvailable bands: 0 to {result.shape[0] - 1}")
            return result[list(band)]

    def stats_mean(
        self,
        band: int | list[int],
        level: int = 0,
        id: str | None = None,
    ) -> "np.ndarray":
        """Aggregate mean values across samples.

        Args:
            band: Band index or list of band indices (required)
            level: Hierarchy level (default 0)
            id: Sample ID (required if level > 0, ignored if level == 0)

        Returns:
            np.ndarray: Aggregated mean(s) for specified band(s)

        Examples:
            ds.stats_mean(band=2)                    # Mean of band 2, all level 0
            ds.stats_mean(band=[0, 1, 2], level=0)  # Bands 0-2, all level 0
            ds.stats_mean(band=0, level=1, id="tile_1")  # Band 0, children of tile_1
        """
        self._validate_stats_params(band, level, id)
        stats_col = self._get_stats_column(level)
        table = self._fetch_stats_table(level, id, stats_col)

        from tacoreader.dataframe._stats import _aggregate_continuous

        result = _aggregate_continuous(table, stats_col, "mean")
        return self._extract_band(result, band)

    def stats_std(
        self,
        band: int | list[int],
        level: int = 0,
        id: str | None = None,
    ) -> "np.ndarray":
        """Aggregate standard deviation using pooled variance formula.

        Args:
            band: Band index or list of band indices (required)
            level: Hierarchy level (default 0)
            id: Sample ID (required if level > 0, ignored if level == 0)

        Returns:
            np.ndarray: Aggregated std(s) for specified band(s)
        """
        self._validate_stats_params(band, level, id)
        stats_col = self._get_stats_column(level)
        table = self._fetch_stats_table(level, id, stats_col)

        from tacoreader.dataframe._stats import _aggregate_std

        result = _aggregate_std(table, stats_col)
        return self._extract_band(result, band)

    def stats_min(
        self,
        band: int | list[int],
        level: int = 0,
        id: str | None = None,
    ) -> "np.ndarray":
        """Get global minimum across samples.

        Args:
            band: Band index or list of band indices (required)
            level: Hierarchy level (default 0)
            id: Sample ID (required if level > 0, ignored if level == 0)

        Returns:
            np.ndarray: Global min(s) for specified band(s)
        """
        self._validate_stats_params(band, level, id)
        stats_col = self._get_stats_column(level)
        table = self._fetch_stats_table(level, id, stats_col)

        from tacoreader.dataframe._stats import _aggregate_continuous

        result = _aggregate_continuous(table, stats_col, "min")
        return self._extract_band(result, band)

    def stats_max(
        self,
        band: int | list[int],
        level: int = 0,
        id: str | None = None,
    ) -> "np.ndarray":
        """Get global maximum across samples.

        Args:
            band: Band index or list of band indices (required)
            level: Hierarchy level (default 0)
            id: Sample ID (required if level > 0, ignored if level == 0)

        Returns:
            np.ndarray: Global max(s) for specified band(s)
        """
        self._validate_stats_params(band, level, id)
        stats_col = self._get_stats_column(level)
        table = self._fetch_stats_table(level, id, stats_col)

        from tacoreader.dataframe._stats import _aggregate_continuous

        result = _aggregate_continuous(table, stats_col, "max")
        return self._extract_band(result, band)

    def stats_p25(
        self,
        band: int | list[int],
        level: int = 0,
        id: str | None = None,
    ) -> "np.ndarray":
        """Aggregate 25th percentile (approximation via averaging).

        Args:
            band: Band index or list of band indices (required)
            level: Hierarchy level (default 0)
            id: Sample ID (required if level > 0, ignored if level == 0)

        Returns:
            np.ndarray: Approximated p25 for specified band(s)
        """
        self._validate_stats_params(band, level, id)
        stats_col = self._get_stats_column(level)
        table = self._fetch_stats_table(level, id, stats_col)

        from tacoreader.dataframe._stats import _aggregate_continuous

        result = _aggregate_continuous(table, stats_col, "p25")
        return self._extract_band(result, band)

    def stats_p50(
        self,
        band: int | list[int],
        level: int = 0,
        id: str | None = None,
    ) -> "np.ndarray":
        """Aggregate 50th percentile / median (approximation via averaging).

        Args:
            band: Band index or list of band indices (required)
            level: Hierarchy level (default 0)
            id: Sample ID (required if level > 0, ignored if level == 0)

        Returns:
            np.ndarray: Approximated median for specified band(s)
        """
        self._validate_stats_params(band, level, id)
        stats_col = self._get_stats_column(level)
        table = self._fetch_stats_table(level, id, stats_col)

        from tacoreader.dataframe._stats import _aggregate_continuous

        result = _aggregate_continuous(table, stats_col, "p50")
        return self._extract_band(result, band)

    def stats_median(
        self,
        band: int | list[int],
        level: int = 0,
        id: str | None = None,
    ) -> "np.ndarray":
        """Alias for stats_p50()."""
        return self.stats_p50(band, level, id)

    def stats_p75(
        self,
        band: int | list[int],
        level: int = 0,
        id: str | None = None,
    ) -> "np.ndarray":
        """Aggregate 75th percentile (approximation via averaging).

        Args:
            band: Band index or list of band indices (required)
            level: Hierarchy level (default 0)
            id: Sample ID (required if level > 0, ignored if level == 0)

        Returns:
            np.ndarray: Approximated p75 for specified band(s)
        """
        self._validate_stats_params(band, level, id)
        stats_col = self._get_stats_column(level)
        table = self._fetch_stats_table(level, id, stats_col)

        from tacoreader.dataframe._stats import _aggregate_continuous

        result = _aggregate_continuous(table, stats_col, "p75")
        return self._extract_band(result, band)

    def stats_p95(
        self,
        band: int | list[int],
        level: int = 0,
        id: str | None = None,
    ) -> "np.ndarray":
        """Aggregate 95th percentile (approximation via averaging).

        Args:
            band: Band index or list of band indices (required)
            level: Hierarchy level (default 0)
            id: Sample ID (required if level > 0, ignored if level == 0)

        Returns:
            np.ndarray: Approximated p95 for specified band(s)
        """
        self._validate_stats_params(band, level, id)
        stats_col = self._get_stats_column(level)
        table = self._fetch_stats_table(level, id, stats_col)

        from tacoreader.dataframe._stats import _aggregate_continuous

        result = _aggregate_continuous(table, stats_col, "p95")
        return self._extract_band(result, band)

    def stats_categorical(
        self,
        band: int | list[int],
        level: int = 0,
        id: str | None = None,
    ) -> "np.ndarray":
        """Aggregate categorical probabilities using weighted average.

        Args:
            band: Band index or list of band indices (required)
            level: Hierarchy level (default 0)
            id: Sample ID (required if level > 0, ignored if level == 0)

        Returns:
            np.ndarray: Aggregated class probabilities for specified band(s)
        """
        self._validate_stats_params(band, level, id)
        stats_col = self._get_stats_column(level)
        table = self._fetch_stats_table(level, id, stats_col)

        from tacoreader.dataframe._stats import _aggregate_categorical

        result = _aggregate_categorical(table, stats_col)
        return self._extract_band(result, band)

    # Repr
    def __repr__(self) -> str:
        """Rich text representation of dataset metadata."""
        lines = [f"<TacoDataset '{self.id}'>"]
        lines.append(f"├── Version: {self.version}")

        desc_short = self.description[:80] + "..." if len(self.description) > 80 else self.description
        lines.append(f"├── Description: {desc_short}")
        lines.append(f"├── Tasks: {', '.join(self.tasks)}")

        # Spatial extent (always defined by auto_calculate_extent)
        spatial = self.extent["spatial"]
        lines.append(
            f"├── Spatial Extent: [{spatial[0]:.2f}°, {spatial[1]:.2f}°, {spatial[2]:.2f}°, {spatial[3]:.2f}°]"
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
        """Format ISO 8601 datetime string for display.

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