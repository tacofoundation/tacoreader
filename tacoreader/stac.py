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
    - build_datetime_sql: Generate temporal filter SQL
"""

from datetime import datetime
from typing import Any


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
            raise ValueError(f"Invalid datetime range: start > end")

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
# SQL BUILDERS
# ============================================================================


def build_bbox_sql(
    minx: float, miny: float, maxx: float, maxy: float, geometry_col: str
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

    Returns:
        SQL WHERE clause (without "WHERE" keyword)

    Examples:
        >>> build_bbox_sql(-81, -18, -68, 0, "istac:geometry")
        'ST_Within(ST_GeomFromWKB("istac:geometry"), ST_MakeEnvelope(-81.0, -18.0, -68.0, 0.0))'
    """
    return (
        f"ST_Within("
        f'ST_GeomFromWKB("{geometry_col}"), '
        f"ST_MakeEnvelope({float(minx)}, {float(miny)}, {float(maxx)}, {float(maxy)})"
        f")"
    )


def build_intersects_sql(geometry: Any, geometry_col: str) -> str:
    """
    Build SQL for geometry intersection filter.

    Uses ST_Intersects to check if geometries overlap.
    Geometry column is WKB format, converted with ST_GeomFromWKB.

    Args:
        geometry: User geometry (WKT string, GeoJSON dict, or shapely object)
        geometry_col: Name of WKB geometry column

    Returns:
        SQL WHERE clause (without "WHERE" keyword)

    Examples:
        >>> polygon = {"type": "Polygon", "coordinates": [[...]]}
        >>> build_intersects_sql(polygon, "istac:geometry")
        'ST_Intersects(ST_GeomFromWKB("istac:geometry"), ST_GeomFromText(...))'
    """
    wkt = geometry_to_wkt(geometry)
    # Escape single quotes in WKT
    wkt_escaped = wkt.replace("'", "''")

    return (
        f"ST_Intersects("
        f'ST_GeomFromWKB("{geometry_col}"), '
        f"ST_GeomFromText('{wkt_escaped}')"
        f")"
    )


def build_within_sql(geometry: Any, geometry_col: str) -> str:
    """
    Build SQL for within filter.

    Uses ST_Within to check if geometry is completely within user polygon.
    Geometry column is WKB format, converted with ST_GeomFromWKB.

    Args:
        geometry: User geometry (WKT string, GeoJSON dict, or shapely object)
        geometry_col: Name of WKB geometry column

    Returns:
        SQL WHERE clause (without "WHERE" keyword)

    Examples:
        >>> polygon = "POLYGON((-81 -18, -68 -18, -68 0, -81 0, -81 -18))"
        >>> build_within_sql(polygon, "istac:geometry")
        'ST_Within(ST_GeomFromWKB("istac:geometry"), ST_GeomFromText(...))'
    """
    wkt = geometry_to_wkt(geometry)
    # Escape single quotes in WKT
    wkt_escaped = wkt.replace("'", "''")

    return (
        f"ST_Within("
        f'ST_GeomFromWKB("{geometry_col}"), '
        f"ST_GeomFromText('{wkt_escaped}')"
        f")"
    )


def build_datetime_sql(start: int, end: int | None, time_col: str) -> str:
    """
    Build SQL for temporal filter.

    If end is None, filters for exact timestamp match.
    If end is provided, filters for time range (start <= time <= end).

    Args:
        start: Start timestamp (seconds since epoch)
        end: End timestamp (seconds since epoch) or None
        time_col: Name of time column

    Returns:
        SQL WHERE clause (without "WHERE" keyword)

    Examples:
        >>> build_datetime_sql(1672531200, 1704067199, "istac:time_start")
        '("istac:time_start" BETWEEN 1672531200 AND 1704067199)'
        >>> build_datetime_sql(1672531200, None, "stac:time_start")
        '("stac:time_start" = 1672531200)'
    """
    if end is None:
        # Single timestamp
        return f'("{time_col}" = {start})'
    else:
        # Time range
        return f'("{time_col}" BETWEEN {start} AND {end})'
