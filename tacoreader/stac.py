"""
STAC-style spatiotemporal filtering helpers for TacoDataset.

Provides PySTAC-like API for filtering datasets by spatial and temporal criteria.
All spatial operations use DuckDB Spatial extension with WKB columns from TACO.

Main functions:
    - detect_geometry_column: Auto-detect best geometry column
    - detect_time_column: Auto-detect time column
    - build_bbox_sql: Generate bounding box filter SQL
    - build_intersects_sql: Generate intersection filter SQL
    - build_within_sql: Generate within filter SQL
    - build_datetime_sql: Generate temporal filter SQL (with auto type detection)
    - validate_level_exists: Validate that level exists in dataset
    - get_columns_for_level: Get available columns for a specific level
    - build_cascade_join_sql: Build SQL with cascading JOINs for multi-level filtering
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset


# ============================================================================
# LEVEL HELPERS (Multi-level filtering support)
# ============================================================================


def validate_level_exists(dataset: "TacoDataset", level: int) -> None:
    """
    Validate that a specific level exists in the dataset.

    Checks if the levelN view exists in consolidated_files.
    Level 0 always exists by definition.

    Args:
        dataset: TacoDataset instance
        level: Level number to validate (0, 1, 2, ...)

    Raises:
        ValueError: If level does not exist in dataset

    Examples:
        >>> validate_level_exists(dataset, 1)  # OK if level1 exists
        >>> validate_level_exists(dataset, 5)  # ValueError if max level is 2
    """
    # Level 0 always exists
    if level == 0:
        return

    level_key = f"level{level}"

    if level_key not in dataset._consolidated_files:
        max_level = dataset.pit_schema.max_depth()
        raise ValueError(
            f"Level {level} does not exist in dataset.\n"
            f"Available levels: 0 to {max_level}\n"
            f"Available views: {list(dataset._consolidated_files.keys())}"
        )


def get_columns_for_level(dataset: "TacoDataset", level: int) -> list[str]:
    """
    Get available column names for a specific level.

    Queries DuckDB to get actual columns from the levelN view.
    Includes both metadata columns and internal columns.

    Args:
        dataset: TacoDataset instance
        level: Level number (0, 1, 2, ...)

    Returns:
        List of column names available at that level

    Raises:
        ValueError: If level does not exist

    Examples:
        >>> cols = get_columns_for_level(dataset, 1)
        >>> print(cols)
        ['id', 'type', 'stac:bbox', 'stac:time_start', 'internal:parent_id', ...]
    """
    validate_level_exists(dataset, level)

    level_view = f"level{level}" if level > 0 else "data"

    # Query DuckDB for column names
    result = dataset._duckdb.execute(f"DESCRIBE {level_view}").fetchall()

    # Result is list of tuples: (column_name, column_type, null, key, default, extra)
    return [row[0] for row in result]


def build_cascade_join_sql(
    current_view: str, target_level: int, where_clause: str, format_type: str = "zip"
) -> str:
    """
    Build SQL with cascading JOINs from level0 to target level.

    Creates INNER JOINs connecting level0 → level1 → level2 → ... → target_level
    using internal:parent_id foreign keys. Always returns DISTINCT level0 samples.

    JOIN strategy varies by format:
    - ZIP/FOLDER: parent_id references parent's ID (string)
    - TacoCat: parent_id is local index + source_file for disambiguation

    Hierarchy structure:
        Level0 (id: "sample_001")
          └── Level1 (id: "s2_l1c", parent_id: "sample_001")
               └── Level2 (id: "band_B04", parent_id: "s2_l1c")

    Args:
        current_view: Current view name (e.g., "data" or "view_abc123")
        target_level: Level where filter will be applied (1, 2, 3, ...)
        where_clause: SQL WHERE clause to apply (without "WHERE" keyword)
        format_type: Dataset format ("zip", "folder", or "tacocat")

    Returns:
        Complete SQL query string with JOINs and WHERE clause

    Examples:
        >>> # Filter by level1 metadata (ZIP format)
        >>> sql = build_cascade_join_sql(
        ...     "data",
        ...     1,
        ...     'ST_Within(ST_GeomFromWKB(l1."stac:bbox"), ST_MakeEnvelope(...))',
        ...     "zip"
        ... )
        >>> print(sql)
        SELECT DISTINCT l0.*
        FROM data l0
        INNER JOIN level1 l1 ON l1."internal:parent_id" = l0.id
        WHERE ST_Within(ST_GeomFromWKB(l1."stac:bbox"), ...)

        >>> # Filter by level1 metadata (TacoCat format)
        >>> sql = build_cascade_join_sql(
        ...     "data",
        ...     1,
        ...     'l1.id LIKE "l3_swot%"',
        ...     "tacocat"
        ... )
        >>> print(sql)
        SELECT DISTINCT l0.*
        FROM data l0
        INNER JOIN level1 l1
          ON l1."internal:parent_id" = l0."internal:parent_id"
         AND l1."internal:source_file" = l0."internal:source_file"
        WHERE l1.id LIKE "l3_swot%"
    """
    if target_level == 0:
        # No JOINs needed - filter directly on level0
        return f"""
            SELECT *
            FROM {current_view}
            WHERE {where_clause}
        """

    # Build cascading JOINs
    joins = []

    # First JOIN: current_view (level0) → level1
    if format_type == "tacocat":
        # TacoCat: parent_id is local index, need source_file for disambiguation
        joins.append(
            "INNER JOIN level1 l1\n"
            '      ON l1."internal:parent_id" = l0."internal:parent_id"\n'
            '     AND l1."internal:source_file" = l0."internal:source_file"'
        )
    else:
        # ZIP/FOLDER: parent_id references parent's ID string
        joins.append('INNER JOIN level1 l1 ON l1."internal:parent_id" = l0.id')

    # Subsequent JOINs: level1 → level2 → level3 ...
    for level in range(2, target_level + 1):
        prev_level = level - 1
        joins.append(
            f"INNER JOIN level{level} l{level} "
            f'ON l{level}."internal:parent_id" = l{prev_level}.id'
        )

    join_clause = "\n    ".join(joins)

    # Build complete query
    return f"""
        SELECT DISTINCT l0.*
        FROM {current_view} l0
        {join_clause}
        WHERE {where_clause}
    """


# ============================================================================
# AUTO-DETECTION
# ============================================================================


def detect_geometry_column(columns: list[str]) -> str:
    """
    Auto-detect best geometry column from available columns.

    Priority order:
    1. istac:geometry (most precise - full geometry)
    2. stac:centroid (point representation for STAC)
    3. istac:centroid (point representation for ISTAC)

    Args:
        columns: List of column names from dataset

    Returns:
        Name of detected geometry column

    Raises:
        ValueError: If no geometry column found

    Examples:
        >>> detect_geometry_column(["id", "istac:geometry", "istac:centroid"])
        'istac:geometry'
        >>> detect_geometry_column(["id", "stac:centroid"])
        'stac:centroid'
    """
    # Priority: full geometry > STAC centroid > ISTAC centroid
    if "istac:geometry" in columns:
        return "istac:geometry"
    elif "stac:centroid" in columns:
        return "stac:centroid"
    elif "istac:centroid" in columns:
        return "istac:centroid"
    else:
        raise ValueError(
            "No geometry column found in dataset.\n"
            "Expected one of: istac:geometry, stac:centroid, istac:centroid\n"
            f"Available columns: {columns}"
        )


def detect_time_column(columns: list[str]) -> str:
    """
    Auto-detect time column from available columns.

    Always uses time_start (not time_middle or time_end).
    Priority order:
    1. istac:time_start
    2. stac:time_start

    Args:
        columns: List of column names from dataset

    Returns:
        Name of detected time column

    Raises:
        ValueError: If no time column found

    Examples:
        >>> detect_time_column(["id", "istac:time_start"])
        'istac:time_start'
        >>> detect_time_column(["id", "stac:time_start"])
        'stac:time_start'
    """
    # Priority: ISTAC > STAC
    if "istac:time_start" in columns:
        return "istac:time_start"
    elif "stac:time_start" in columns:
        return "stac:time_start"
    else:
        raise ValueError(
            "No time column found in dataset.\n"
            "Expected one of: istac:time_start, stac:time_start\n"
            f"Available columns: {columns}"
        )


# ============================================================================
# VALIDATION
# ============================================================================


def validate_geometry_column(columns: list[str], requested: str, detected: str) -> str:
    """
    Validate that requested geometry column exists.

    If user requested "auto", returns detected column.
    If user requested specific column, validates it exists and returns it.

    Args:
        columns: List of available column names
        requested: User-requested column name or "auto"
        detected: Auto-detected column name (from detect_geometry_column)

    Returns:
        Column name to use

    Raises:
        ValueError: If requested column does not exist

    Examples:
        >>> validate_geometry_column(
        ...     ["id", "istac:geometry"],
        ...     "auto",
        ...     "istac:geometry"
        ... )
        'istac:geometry'
        >>> validate_geometry_column(
        ...     ["id", "istac:geometry"],
        ...     "stac:centroid",
        ...     "istac:geometry"
        ... )
        ValueError: Requested geometry column 'stac:centroid' not found
    """
    if requested == "auto":
        return detected

    if requested not in columns:
        raise ValueError(
            f"Requested geometry column '{requested}' not found.\n"
            f"Auto-detected column: {detected}\n"
            f"Available columns: {columns}\n"
            f"Use geometry_col='auto' to use auto-detection."
        )

    return requested


def validate_time_column(columns: list[str], requested: str, detected: str) -> str:
    """
    Validate that requested time column exists.

    If user requested "auto", returns detected column.
    If user requested specific column, validates it exists and returns it.

    Args:
        columns: List of available column names
        requested: User-requested column name or "auto"
        detected: Auto-detected column name (from detect_time_column)

    Returns:
        Column name to use

    Raises:
        ValueError: If requested column does not exist

    Examples:
        >>> validate_time_column(
        ...     ["id", "istac:time_start"],
        ...     "auto",
        ...     "istac:time_start"
        ... )
        'istac:time_start'
    """
    if requested == "auto":
        return detected

    if requested not in columns:
        raise ValueError(
            f"Requested time column '{requested}' not found.\n"
            f"Auto-detected column: {detected}\n"
            f"Available columns: {columns}\n"
            f"Use time_col='auto' to use auto-detection."
        )

    return requested


# ============================================================================
# INPUT CONVERSION
# ============================================================================


def parse_datetime(
    dt_input: str | datetime | tuple[datetime, datetime]
) -> tuple[int, int | None]:
    """
    Parse datetime input to (start_timestamp, end_timestamp).

    Accepts multiple formats:
    - String range: "2023-01-01/2023-12-31"
    - Single datetime: datetime(2023, 1, 1)
    - Datetime tuple: (start_dt, end_dt)

    Args:
        dt_input: Datetime specification in various formats

    Returns:
        Tuple of (start_timestamp, end_timestamp)
        end_timestamp is None for single datetime

    Raises:
        ValueError: If format is invalid or start > end

    Examples:
        >>> parse_datetime("2023-01-01/2023-12-31")
        (1672531200, 1704067199)
        >>> parse_datetime(datetime(2023, 1, 1))
        (1672531200, None)
        >>> parse_datetime((datetime(2023, 1, 1), datetime(2023, 12, 31)))
        (1672531200, 1704067199)
    """
    # String range format: "2023-01-01/2023-12-31"
    if isinstance(dt_input, str):
        if "/" in dt_input:
            start_str, end_str = dt_input.split("/", 1)
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())

            if start_ts > end_ts:
                raise ValueError(
                    f"Invalid datetime range: start ({start_str}) > end ({end_str})"
                )

            return start_ts, end_ts
        else:
            # Single date string
            start_dt = datetime.fromisoformat(dt_input.replace("Z", "+00:00"))
            return int(start_dt.timestamp()), None

    # Single datetime object
    elif isinstance(dt_input, datetime):
        return int(dt_input.timestamp()), None

    # Tuple of datetime objects
    elif isinstance(dt_input, tuple) and len(dt_input) == 2:
        start_dt, end_dt = dt_input
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        if start_ts > end_ts:
            raise ValueError("Invalid datetime range: start > end")

        return start_ts, end_ts

    else:
        raise ValueError(
            f"Invalid datetime input: {dt_input}\n"
            f"Expected: string range ('2023-01-01/2023-12-31'), "
            f"datetime object, or tuple of datetime objects"
        )


def geometry_to_wkt(geometry: Any) -> str:
    """
    Convert geometry to WKT string for DuckDB SQL.

    Accepts multiple formats:
    - WKT string: "POLYGON((-81 -18, ...))"
    - GeoJSON dict: {"type": "Polygon", "coordinates": [...]}
    - Shapely object: Polygon(...) (if shapely installed)
    - Any object with __geo_interface__

    Args:
        geometry: Geometry in various formats

    Returns:
        WKT string ready for ST_GeomFromText

    Raises:
        ValueError: If geometry format is unsupported

    Examples:
        >>> geometry_to_wkt("POLYGON((-81 -18, -68 -18, -68 0, -81 0, -81 -18))")
        'POLYGON((-81 -18, -68 -18, -68 0, -81 0, -81 -18))'
        >>> geometry_to_wkt({"type": "Point", "coordinates": [-77, -12]})
        'POINT(-77 -12)'
    """
    # Already WKT string
    if isinstance(geometry, str):
        return geometry

    # GeoJSON dict
    elif isinstance(geometry, dict):
        return geojson_to_wkt(geometry)

    # Shapely object (has .wkt attribute)
    elif hasattr(geometry, "wkt"):
        return geometry.wkt

    # Any object implementing __geo_interface__ protocol
    elif hasattr(geometry, "__geo_interface__"):
        return geojson_to_wkt(geometry.__geo_interface__)

    else:
        raise ValueError(
            f"Unsupported geometry type: {type(geometry)}\n"
            f"Expected: WKT string, GeoJSON dict, shapely object, "
            f"or object with __geo_interface__"
        )


def geojson_to_wkt(geojson: dict) -> str:
    """
    Convert GeoJSON dictionary to WKT string.

    Pure Python implementation, no dependencies required.
    Supports: Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon.

    Args:
        geojson: GeoJSON geometry dictionary with "type" and "coordinates"

    Returns:
        WKT string representation

    Raises:
        ValueError: If geometry type is unsupported

    Examples:
        >>> geojson_to_wkt({"type": "Point", "coordinates": [-77, -12]})
        'POINT(-77 -12)'
        >>> geojson_to_wkt({
        ...     "type": "Polygon",
        ...     "coordinates": [[(-81, -18), (-68, -18), (-68, 0), (-81, 0), (-81, -18)]]
        ... })
        'POLYGON((-81 -18, -68 -18, -68 0, -81 0, -81 -18))'
    """
    geom_type = geojson["type"]
    coords = geojson["coordinates"]

    if geom_type == "Point":
        return f"POINT({coords[0]} {coords[1]})"

    elif geom_type == "LineString":
        points = ", ".join(f"{x} {y}" for x, y in coords)
        return f"LINESTRING({points})"

    elif geom_type == "Polygon":
        rings = []
        for ring in coords:
            points = ", ".join(f"{x} {y}" for x, y in ring)
            rings.append(f"({points})")
        return f"POLYGON({', '.join(rings)})"

    elif geom_type == "MultiPoint":
        points = ", ".join(f"({x} {y})" for x, y in coords)
        return f"MULTIPOINT({points})"

    elif geom_type == "MultiLineString":
        lines = []
        for line in coords:
            points = ", ".join(f"{x} {y}" for x, y in line)
            lines.append(f"({points})")
        return f"MULTILINESTRING({', '.join(lines)})"

    elif geom_type == "MultiPolygon":
        polygons = []
        for polygon in coords:
            rings = []
            for ring in polygon:
                points = ", ".join(f"{x} {y}" for x, y in ring)
                rings.append(f"({points})")
            polygons.append(f"({', '.join(rings)})")
        return f"MULTIPOLYGON({', '.join(polygons)})"

    else:
        raise ValueError(
            f"Unsupported GeoJSON geometry type: {geom_type}\n"
            f"Supported types: Point, LineString, Polygon, "
            f"MultiPoint, MultiLineString, MultiPolygon"
        )


# ============================================================================
# TYPE DETECTION (for datetime columns)
# ============================================================================


def _get_column_type(dataset: "TacoDataset", col: str, level: int) -> str:
    """
    Detect column type from DuckDB schema.

    Queries DuckDB DESCRIBE to get actual column type.
    This enables automatic type-aware SQL generation.

    Args:
        dataset: TacoDataset instance with DuckDB connection
        col: Column name to check
        level: Level where column exists (0, 1, 2, ...)

    Returns:
        DuckDB type name (e.g., 'VARCHAR', 'BIGINT', 'TIMESTAMP', 'DATE')

    Raises:
        ValueError: If column does not exist in view

    Examples:
        >>> _get_column_type(dataset, "istac:time_start", 0)
        'BIGINT'
        >>> _get_column_type(dataset, "stac:time_start", 1)
        'VARCHAR'
    """
    level_view = f"level{level}" if level > 0 else "data"

    # Query DuckDB for column type
    # Note: DuckDB returns 'column_type' not 'data_type'
    result = dataset._duckdb.execute(
        f"SELECT column_type FROM (DESCRIBE {level_view}) WHERE column_name = '{col}'"
    ).fetchone()

    if result:
        return result[0].upper()

    raise ValueError(
        f"Column '{col}' not found in {level_view}\n"
        f"Available columns: {get_columns_for_level(dataset, level)}"
    )


def _timestamp_to_iso_string(timestamp: int) -> str:
    """
    Convert Unix timestamp to ISO 8601 string.

    Args:
        timestamp: Seconds since epoch

    Returns:
        ISO 8601 datetime string (e.g., '2024-04-01T00:00:00')

    Examples:
        >>> _timestamp_to_iso_string(1711929600)
        '2024-04-01T00:00:00'
    """
    return datetime.fromtimestamp(timestamp).isoformat()


def _timestamp_to_date_string(timestamp: int) -> str:
    """
    Convert Unix timestamp to DATE string.

    Args:
        timestamp: Seconds since epoch

    Returns:
        DATE string (e.g., '2024-04-01')

    Examples:
        >>> _timestamp_to_date_string(1711929600)
        '2024-04-01'
    """
    return datetime.fromtimestamp(timestamp).date().isoformat()


# ============================================================================
# SQL BUILDERS
# ============================================================================


def build_bbox_sql(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    geometry_col: str,
    level: int = 0,
) -> str:
    """
    Build SQL for bounding box filter.

    Uses ST_Within to check if geometry is completely within bbox.
    Geometry column is WKB format, converted with ST_GeomFromWKB.

    Note: DuckDB's ST_MakeEnvelope only accepts 4 coordinates (no SRID parameter).
    The crs parameter is kept for API compatibility but currently not used.
    TACO geometries are stored in EPSG:4326 (WGS84) by default.

    Args:
        minx: Minimum longitude/X coordinate
        miny: Minimum latitude/Y coordinate
        maxx: Maximum longitude/X coordinate
        maxy: Maximum latitude/Y coordinate
        geometry_col: Name of WKB geometry column
        level: Level where the geometry column exists (0, 1, 2, ...)

    Returns:
        SQL WHERE clause (without "WHERE" keyword)

    Examples:
        >>> build_bbox_sql(-81, -18, -68, 0, "istac:geometry", level=0)
        'ST_Within(ST_GeomFromWKB("istac:geometry"), ST_MakeEnvelope(-81.0, -18.0, -68.0, 0.0))'
        >>> build_bbox_sql(-81, -18, -68, 0, "stac:bbox", level=1)
        'ST_Within(ST_GeomFromWKB(l1."stac:bbox"), ST_MakeEnvelope(-81.0, -18.0, -68.0, 0.0))'
    """
    # Prefix column with level alias if level > 0
    if level > 0:
        col_ref = f'l{level}."{geometry_col}"'
    else:
        col_ref = f'"{geometry_col}"'

    return (
        f"ST_Within("
        f"ST_GeomFromWKB({col_ref}), "
        f"ST_MakeEnvelope({float(minx)}, {float(miny)}, {float(maxx)}, {float(maxy)})"
        f")"
    )


def build_intersects_sql(geometry: Any, geometry_col: str, level: int = 0) -> str:
    """
    Build SQL for geometry intersection filter.

    Uses ST_Intersects to check if geometries overlap.
    Geometry column is WKB format, converted with ST_GeomFromWKB.

    Args:
        geometry: User geometry (WKT string, GeoJSON dict, or shapely object)
        geometry_col: Name of WKB geometry column
        level: Level where the geometry column exists (0, 1, 2, ...)

    Returns:
        SQL WHERE clause (without "WHERE" keyword)

    Examples:
        >>> polygon = {"type": "Polygon", "coordinates": [[...]]}
        >>> build_intersects_sql(polygon, "istac:geometry", level=0)
        'ST_Intersects(ST_GeomFromWKB("istac:geometry"), ST_GeomFromText(...))'
        >>> build_intersects_sql(polygon, "stac:bbox", level=1)
        'ST_Intersects(ST_GeomFromWKB(l1."stac:bbox"), ST_GeomFromText(...))'
    """
    wkt = geometry_to_wkt(geometry)
    # Escape single quotes in WKT
    wkt_escaped = wkt.replace("'", "''")

    # Prefix column with level alias if level > 0
    if level > 0:
        col_ref = f'l{level}."{geometry_col}"'
    else:
        col_ref = f'"{geometry_col}"'

    return (
        f"ST_Intersects("
        f"ST_GeomFromWKB({col_ref}), "
        f"ST_GeomFromText('{wkt_escaped}')"
        f")"
    )


def build_within_sql(geometry: Any, geometry_col: str, level: int = 0) -> str:
    """
    Build SQL for within filter.

    Uses ST_Within to check if geometry is completely within user polygon.
    Geometry column is WKB format, converted with ST_GeomFromWKB.

    Args:
        geometry: User geometry (WKT string, GeoJSON dict, or shapely object)
        geometry_col: Name of WKB geometry column
        level: Level where the geometry column exists (0, 1, 2, ...)

    Returns:
        SQL WHERE clause (without "WHERE" keyword)

    Examples:
        >>> polygon = "POLYGON((-81 -18, -68 -18, -68 0, -81 0, -81 -18))"
        >>> build_within_sql(polygon, "istac:geometry", level=0)
        'ST_Within(ST_GeomFromWKB("istac:geometry"), ST_GeomFromText(...))'
        >>> build_within_sql(polygon, "stac:bbox", level=1)
        'ST_Within(ST_GeomFromWKB(l1."stac:bbox"), ST_GeomFromText(...))'
    """
    wkt = geometry_to_wkt(geometry)
    # Escape single quotes in WKT
    wkt_escaped = wkt.replace("'", "''")

    # Prefix column with level alias if level > 0
    if level > 0:
        col_ref = f'l{level}."{geometry_col}"'
    else:
        col_ref = f'"{geometry_col}"'

    return (
        f"ST_Within("
        f"ST_GeomFromWKB({col_ref}), "
        f"ST_GeomFromText('{wkt_escaped}')"
        f")"
    )


def build_datetime_sql(
    start: int,
    end: int | None,
    time_col: str,
    level: int,
    dataset: "TacoDataset",
) -> str:
    """
    Build SQL for temporal filter with automatic type detection.

    Detects column type from DuckDB schema and generates appropriate SQL.
    Supports multiple column types:
    - INTEGER/BIGINT: Direct timestamp comparison (current behavior)
    - VARCHAR: ISO string comparison
    - TIMESTAMP: Timestamp comparison with conversion
    - DATE: Date comparison

    Args:
        start: Start timestamp (seconds since epoch)
        end: End timestamp (seconds since epoch) or None
        time_col: Name of time column
        level: Level where the time column exists (0, 1, 2, ...)
        dataset: TacoDataset instance (for automatic type detection)

    Returns:
        SQL WHERE clause (without "WHERE" keyword)

    Examples:
        >>> # INTEGER column
        >>> build_datetime_sql(1672531200, 1704067199, "istac:time_start", 0, ds)
        '("istac:time_start" BETWEEN 1672531200 AND 1704067199)'
        
        >>> # VARCHAR column
        >>> build_datetime_sql(1672531200, 1704067199, "time_str", 0, ds)
        '("time_str" BETWEEN '2023-01-01T00:00:00' AND '2023-12-31T23:59:59')'
        
        >>> # TIMESTAMP column
        >>> build_datetime_sql(1672531200, None, "timestamp_col", 1, ds)
        '(l1."timestamp_col" = to_timestamp(1672531200))'
    """
    # Detect column type automatically
    col_type = _get_column_type(dataset, time_col, level)

    # Prefix column with level alias if level > 0
    if level > 0:
        col_ref = f'l{level}."{time_col}"'
    else:
        col_ref = f'"{time_col}"'

    # Generate SQL based on detected type
    if col_type in ("BIGINT", "INTEGER", "HUGEINT"):
        # INTEGER types: Direct timestamp comparison (current behavior)
        if end is None:
            return f"({col_ref} = {start})"
        else:
            return f"({col_ref} BETWEEN {start} AND {end})"

    elif col_type == "VARCHAR":
        # STRING type: ISO string comparison
        start_str = _timestamp_to_iso_string(start)
        if end is None:
            return f"({col_ref} = '{start_str}')"
        else:
            end_str = _timestamp_to_iso_string(end)
            return f"({col_ref} BETWEEN '{start_str}' AND '{end_str}')"

    elif col_type in ("TIMESTAMP", "TIMESTAMP WITH TIME ZONE"):
        # TIMESTAMP type: Convert INT to TIMESTAMP for comparison
        if end is None:
            return f"({col_ref} = to_timestamp({start}))"
        else:
            return f"({col_ref} BETWEEN to_timestamp({start}) AND to_timestamp({end}))"

    elif col_type == "DATE":
        # DATE type: Convert INT to DATE for comparison
        start_date = _timestamp_to_date_string(start)
        if end is None:
            return f"({col_ref} = DATE '{start_date}')"
        else:
            end_date = _timestamp_to_date_string(end)
            return f"({col_ref} BETWEEN DATE '{start_date}' AND DATE '{end_date}')"

    else:
        raise ValueError(
            f"Unsupported time column type: {col_type}\n"
            f"Supported types: INTEGER, BIGINT, VARCHAR, TIMESTAMP, DATE\n"
            f"Column '{time_col}' has type '{col_type}'"
        )