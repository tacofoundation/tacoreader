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
    NAVIGATION_COLUMNS_BY_FORMAT,
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
        _dataframe_backend, _owns_connection, __joined_levels,
        _rsut_compliant, _filtered_level_views, _extent_modified

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

        # RSUT compliance check
        ds.is_rsut()  # True if structurally homogeneous
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

    # Original path to dataset (local or cloud URL). Used for path reconstruction.
    _path: str = PrivateAttr()

    # Storage format. Determines VSI path construction and navigation behavior.
    _format: Literal["zip", "folder", "tacocat"] = PrivateAttr()

    # Complete parsed COLLECTION.json. Contains field_schema, pit_schema, extent, STAC metadata.
    _collection: dict[str, Any] = PrivateAttr()

    # In-memory DuckDB connection with metadata tables (level0_table, level1_table, ...)
    # and views (level0, level1, ..., data). Views include internal:gdal_vsi column.
    _duckdb: Any = PrivateAttr(default=None)

    # Current view name for SQL queries. Starts as "data", changes to "view_<uuid>" after .sql() calls.
    # Enables query chaining: ds.sql(...).sql(...) creates nested temp views.
    _view_name: str = PrivateAttr(default=DEFAULT_VIEW_NAME)

    # Base path for GDAL VSI construction. Format-specific:
    # - ZIP: path to .tacozip file :: /vsisubfile/{offset}_{size},{vsi_base_path}
    # - FOLDER: path to dataset dir :: {vsi_base_path}/DATA/{id}
    # - TacoCat: path to parent dir :: /vsisubfile/...{vsi_base_path}{source_file}
    _vsi_base_path: str = PrivateAttr(default="")

    # DataFrame backend for .data property. One of: "pyarrow", "polars", "pandas".
    _dataframe_backend: str = PrivateAttr(default="pyarrow")

    # True if this dataset owns the DuckDB connection and should close it on cleanup.
    # False for child datasets created via .sql() (share parent's connection).
    _owns_connection: bool = PrivateAttr(default=True)

    # Set of level table names joined in query chain (e.g., {"level1", "level2"}).
    # Tracks which levels were touched for debugging. Not used for validation.
    _joined_levels: set[str] = PrivateAttr(default_factory=set)

    # True if dataset maintains RSUT Invariant 3 (structural homogeneity).
    # False after: cascade filters that modify internal structure or sql() that removes
    # navigation columns. Checked by concat() and export operations.
    _rsut_compliant: bool = PrivateAttr(default=True)

    # Dict of filtered views by level from cascade filters.
    # When cascade filtering at level>0, child views are stored here so
    # TacoDataFrame.read() can query DuckDB instead of physical __meta__.
    # Example: {1: "filtered_1_abc123", 2: "filtered_2_abc123"}
    # Empty dict means no cascade filtering applied (use physical __meta__).
    _filtered_level_views: dict[int, str] = PrivateAttr(default_factory=dict)

    # True if extent from COLLECTION.json no longer reflects current data.
    # Set to True after any filtering operation (sql, filter_bbox, filter_datetime).
    # Used by __repr__ to indicate extent is approximate.
    _extent_modified: bool = PrivateAttr(default=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _check_rsut_compliance(self) -> bool:
        """Verify Invariant 3: structural homogeneity at level-1.

        Executes query to verify if all level-0 FOLDERs have the same
        child signature (same ids in same order).

        Returns:
            True if RSUT compliant, False otherwise
        """
        if self.pit_schema.root["type"] == "FILE":
            return True
        if self.pit_schema.max_depth() == 0:
            return True

        query = """
            WITH parent_children AS (
                SELECT
                    l0."internal:current_id" as parent_id,
                    STRING_AGG(l1.id, '|' ORDER BY l1.id) as children_sig
                FROM level0 l0
                LEFT JOIN level1 l1 ON l1."internal:parent_id" = l0."internal:current_id"
                GROUP BY l0."internal:current_id"
            )
            SELECT COUNT(DISTINCT children_sig) <= 1 as is_homogeneous
            FROM parent_children
        """
        try:
            result = self._duckdb.execute(query).fetchone()
            return bool(result[0]) if result else True
        except Exception:
            return True

    def is_rsut(self) -> bool:
        """Check if dataset is RSUT compliant.

        Returns:
            True if dataset maintains structural homogeneity,
            False if structure is heterogeneous across samples.
        """
        return self._rsut_compliant

    def verify_rsut(self) -> bool:
        """Re-verify RSUT compliance.

        Useful after complex operations where you want to confirm current state.

        Returns:
            True if RSUT compliant, False otherwise.
            Also updates internal state.
        """
        self._rsut_compliant = self._check_rsut_compliance()
        return self._rsut_compliant

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
        """Materialize current view to TacoDataFrame (clean output).

        Executes DuckDB query, loads data into memory.
        Removes internal:* columns except those needed for navigation.

        Use .data_raw if you need all internal columns for debugging.

        Uses backend factory to create appropriate TacoDataFrame instance.
        Passes DuckDB connection and filtered views for cascade navigation.
        """
        from tacoreader._constants import (
            METADATA_CURRENT_ID,
            METADATA_PARENT_ID,
            METADATA_SOURCE_FILE,
        )
        from tacoreader.dataframe import create_dataframe

        # DuckDB always returns PyArrow Table
        arrow_table = self._duckdb.execute(f"SELECT * FROM {self._view_name}").fetch_arrow_table()

        # Columns always needed for navigation
        navigation_internals = {"internal:gdal_vsi"}

        # When cascade filters are active, we need additional columns for .read()
        if self._filtered_level_views:
            navigation_internals.add(METADATA_CURRENT_ID)  # internal:current_id
            navigation_internals.add(METADATA_PARENT_ID)  # internal:parent_id
            if self._format == "tacocat":
                navigation_internals.add(METADATA_SOURCE_FILE)  # internal:source_file

        # Remove internal:* columns except those needed for navigation
        columns_to_keep = [
            col for col in arrow_table.column_names if not col.startswith("internal:") or col in navigation_internals
        ]
        arrow_table = arrow_table.select(columns_to_keep)

        # Factory creates backend-specific TacoDataFrame
        # Pass duckdb and filtered_views for cascade filter navigation
        return create_dataframe(
            backend=self._dataframe_backend,
            arrow_table=arrow_table,
            format_type=self._format,
            duckdb=self._duckdb,
            filtered_level_views=self._filtered_level_views,
            current_level=0,
        )

    @property
    def data_raw(self) -> "TacoDataFrame":
        """Materialize current view to TacoDataFrame (all columns).

        Like .data but keeps all internal:* columns for debugging
        and advanced use cases.

        Passes DuckDB connection and filtered views for cascade navigation.
        """
        from tacoreader.dataframe import create_dataframe

        # DuckDB always returns PyArrow Table
        arrow_table = self._duckdb.execute(f"SELECT * FROM {self._view_name}").fetch_arrow_table()

        # Factory creates backend-specific TacoDataFrame
        # Pass duckdb and filtered_views for cascade filter navigation
        return create_dataframe(
            backend=self._dataframe_backend,
            arrow_table=arrow_table,
            format_type=self._format,
            duckdb=self._duckdb,
            filtered_level_views=self._filtered_level_views,
            current_level=0,
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
            NAVIGATION_COLUMN_DESCRIPTIONS,
            NAVIGATION_COLUMNS_BY_FORMAT,
        )

        columns = NAVIGATION_COLUMNS_BY_FORMAT.get(self._format, frozenset())

        if describe:
            return {col: NAVIGATION_COLUMN_DESCRIPTIONS.get(col, "") for col in sorted(columns)}

        return sorted(columns)

    def sql(self, query: str) -> "TacoDataset":
        """Execute SQL query and return new TacoDataset with lazy view.

        Query NOT executed immediately - creates temp view.
        Always use 'data' as table name, auto-replaced with current view.

        RSUT compliance is evaluated based on:
        - Missing navigation columns → _rsut_compliant = False
        - JOINs with level1+ tables → _rsut_compliant = False
        - Otherwise → inherits from parent

        Note: sql() clears _filtered_level_views since custom SQL may
        invalidate the cascade filter structure.
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

        # Evaluate RSUT compliance with DESCRIBE (O(columns), not O(rows))
        result_cols = {row[0] for row in self._duckdb.execute(f"DESCRIBE {new_view_name}").fetchall()}
        required = NAVIGATION_COLUMNS_BY_FORMAT.get(self._format, frozenset())
        missing = required - result_cols

        # Detect JOINs with level1+ tables (case insensitive)
        matches = re.findall(SQL_JOIN_PATTERN, query, re.IGNORECASE)
        has_new_joins = len(matches) > 0

        # Evaluate RSUT compliance - NO ERROR, just evaluate
        if missing:
            new_rsut_compliant = False
            logger.debug(f"Missing navigation columns {sorted(missing)}. Result is not RSUT compliant.")
        elif has_new_joins:
            new_rsut_compliant = False
            logger.debug(f"Query references {matches}. Result is not RSUT compliant.")
        else:
            new_rsut_compliant = self._rsut_compliant

        # Count rows for schema update
        new_n = self._duckdb.execute(f"SELECT COUNT(*) FROM {new_view_name}").fetchone()[0]

        new_schema = self.pit_schema.with_n(new_n)

        # Track joined levels for debugging
        final_joined_levels = self._joined_levels.union(set(matches))

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
            _joined_levels=final_joined_levels,
            _rsut_compliant=new_rsut_compliant,
            _filtered_level_views={},  # Clear: custom SQL invalidates cascade structure
            _extent_modified=True,  # Any filter invalidates extent
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

        # Spatial extent (only if defined)
        spatial = self.extent.get("spatial")
        if spatial:
            extent_str = f"[{spatial[0]:.2f}°, {spatial[1]:.2f}°, {spatial[2]:.2f}°, {spatial[3]:.2f}°]"
            if self._extent_modified:
                extent_str += " (filtered)"
            lines.append(f"├── Spatial Extent: {extent_str}")

        # Temporal extent (only if defined)
        temporal = self.extent.get("temporal")
        if temporal:
            start_str = self._format_temporal_string(temporal[0])
            end_str = self._format_temporal_string(temporal[1])
            extent_str = f"{start_str} → {end_str}"
            if self._extent_modified:
                extent_str += " (filtered)"
            lines.append(f"├── Temporal Extent: {extent_str}")

        lines.append("│")
        rsut_status = "True" if self._rsut_compliant else "False"
        lines.append(f"└── Level 0: {self.pit_schema.root['n']} rows (RSUT: {rsut_status})")

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
