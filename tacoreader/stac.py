"""
STAC-style spatiotemporal filtering for TacoDataset.

PySTAC-like API for filtering by spatial and temporal criteria.
All spatial ops use DuckDB Spatial extension with WKB columns.

Main functions:
    - detect_geometry_column / detect_time_column: Auto-detection
    - build_bbox_sql / build_intersects_sql / build_within_sql: Spatial filters
    - build_datetime_sql: Temporal filter with auto type detection
    - build_cascade_join_sql: Multi-level filtering with JOINs
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from tacoreader._constants import (
    STAC_GEOMETRY_COLUMN_PRIORITY,
    STAC_TIME_COLUMN_PRIORITY,
)

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset


# ============================================================================
# LEVEL HELPERS
# ============================================================================


def validate_level_exists(dataset: "TacoDataset", level: int) -> None:
    """Validate level exists in dataset using pit_schema.max_depth()."""
    max_level = dataset.pit_schema.max_depth()

    if level < 0 or level > max_level:
        raise ValueError(
            f"Level {level} does not exist.\n" f"Available levels: 0 to {max_level}"
        )


def get_columns_for_level(dataset: "TacoDataset", level: int) -> list[str]:
    """Get available columns for a specific level by querying DuckDB."""
    validate_level_exists(dataset, level)

    level_view = f"level{level}" if level > 0 else "data"
    result = dataset._duckdb.execute(f"DESCRIBE {level_view}").fetchall()

    # Result: [(column_name, column_type, null, key, default, extra), ...]
    return [row[0] for row in result]


def build_cascade_join_sql(
    current_view: str, target_level: int, where_clause: str, format_type: str = "zip"
) -> str:
    """
    Build SQL with cascading JOINs from level0 to target level.

    Creates INNER JOINs: level0 → level1 → level2 → ... → target_level
    using internal:parent_id foreign keys. Returns DISTINCT level0 samples.

    JOIN strategy by format:
    - ZIP/FOLDER: parent_id references parent's ID string
    - TacoCat: parent_id is local index + source_file for disambiguation

    Example hierarchy:
        Level0 (id: "sample_001")
          └── Level1 (id: "s2_l1c", parent_id: "sample_001")
               └── Level2 (id: "band_B04", parent_id: "s2_l1c")
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
        # TacoCat: parent_id is local index, need source_file
        joins.append(
            "INNER JOIN level1 l1\n"
            '      ON l1."internal:parent_id" = l0."internal:parent_id"\n'
            '     AND l1."internal:source_file" = l0."internal:source_file"'
        )
    else:
        # ZIP/FOLDER: parent_id references parent's ID
        joins.append('INNER JOIN level1 l1 ON l1."internal:parent_id" = l0.id')

    # Subsequent JOINs: level1 → level2 → level3 ...
    for level in range(2, target_level + 1):
        prev_level = level - 1
        joins.append(
            f"INNER JOIN level{level} l{level} "
            f'ON l{level}."internal:parent_id" = l{prev_level}.id'
        )

    join_clause = "\n    ".join(joins)

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
    Auto-detect best geometry column.

    Priority: istac:geometry > stac:centroid > istac:centroid
    """
    for col in STAC_GEOMETRY_COLUMN_PRIORITY:
        if col in columns:
            return col

    raise ValueError(
        "No geometry column found.\n"
        f"Expected one of: {', '.join(STAC_GEOMETRY_COLUMN_PRIORITY)}\n"
        f"Available: {columns}"
    )


def detect_time_column(columns: list[str]) -> str:
    """
    Auto-detect time column (always time_start, not middle/end).

    Priority: istac:time_start > stac:time_start
    """
    for col in STAC_TIME_COLUMN_PRIORITY:
        if col in columns:
            return col

    raise ValueError(
        "No time column found.\n"
        f"Expected one of: {', '.join(STAC_TIME_COLUMN_PRIORITY)}\n"
        f"Available: {columns}"
    )


# ============================================================================
# VALIDATION
# ============================================================================


def validate_geometry_column(columns: list[str], requested: str, detected: str) -> str:
    """Validate requested geometry column exists, or use auto-detected."""
    if requested == "auto":
        return detected

    if requested not in columns:
        raise ValueError(
            f"Requested geometry column '{requested}' not found.\n"
            f"Auto-detected: {detected}\n"
            f"Available: {columns}\n"
            f"Use geometry_col='auto' for auto-detection."
        )

    return requested


def validate_time_column(columns: list[str], requested: str, detected: str) -> str:
    """Validate requested time column exists, or use auto-detected."""
    if requested == "auto":
        return detected

    if requested not in columns:
        raise ValueError(
            f"Requested time column '{requested}' not found.\n"
            f"Auto-detected: {detected}\n"
            f"Available: {columns}\n"
            f"Use time_col='auto' for auto-detection."
        )

    return requested


# ============================================================================
# INPUT CONVERSION
# ============================================================================


def parse_datetime(
    dt_input: str | datetime | tuple[datetime, datetime]
) -> tuple[int, int | None]:
    """
    Parse datetime to (start_timestamp, end_timestamp).

    Formats:
    - String range: "2023-01-01/2023-12-31"
    - Single datetime: datetime(2023, 1, 1)
    - Tuple: (start_dt, end_dt)

    Returns end_timestamp=None for single datetime.
    """
    # String range: "2023-01-01/2023-12-31"
    if isinstance(dt_input, str):
        if "/" in dt_input:
            start_str, end_str = dt_input.split("/", 1)
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())

            if start_ts > end_ts:
                raise ValueError(
                    f"Invalid range: start ({start_str}) > end ({end_str})"
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
            raise ValueError("Invalid range: start > end")

        return start_ts, end_ts

    else:
        raise ValueError(
            f"Invalid datetime: {dt_input}\n"
            f"Expected: string range ('2023-01-01/2023-12-31'), "
            f"datetime object, or tuple"
        )


def geometry_to_wkt(geometry: Any) -> str:
    """
    Convert geometry to WKT string for DuckDB.

    Accepts:
    - WKT string: "POLYGON((-81 -18, ...))"
    - GeoJSON dict: {"type": "Polygon", "coordinates": [...]}
    - Shapely object: Polygon(...)
    - Any object with __geo_interface__
    """
    # Already WKT
    if isinstance(geometry, str):
        return geometry

    # GeoJSON dict
    elif isinstance(geometry, dict):
        return geojson_to_wkt(geometry)

    # Shapely object
    elif hasattr(geometry, "wkt"):
        return geometry.wkt

    # __geo_interface__ protocol
    elif hasattr(geometry, "__geo_interface__"):
        return geojson_to_wkt(geometry.__geo_interface__)

    else:
        raise ValueError(
            f"Unsupported geometry type: {type(geometry)}\n"
            f"Expected: WKT string, GeoJSON dict, shapely, or __geo_interface__"
        )


def geojson_to_wkt(geojson: dict) -> str:
    """
    Convert GeoJSON to WKT (pure Python, no dependencies).

    Supports: Point, LineString, Polygon, Multi*.
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
            f"Unsupported GeoJSON type: {geom_type}\n"
            f"Supported: Point, LineString, Polygon, Multi*"
        )


# ============================================================================
# TYPE DETECTION
# ============================================================================


def _get_column_type(dataset: "TacoDataset", col: str, level: int) -> str:
    """
    Detect column type from DuckDB schema for type-aware SQL generation.

    Returns DuckDB type: 'VARCHAR', 'BIGINT', 'TIMESTAMP', etc.
    """
    # SQL injection protection
    if "'" in col or '"' in col:
        raise ValueError(f"Invalid column name: {col} (contains quotes)")

    suspicious = [";", "--", "/*", "*/", "DROP", "DELETE", "INSERT", "UPDATE"]
    if any(s in col.upper() for s in suspicious):
        raise ValueError(f"Invalid column name: {col} (suspicious SQL)")

    level_view = f"level{level}" if level > 0 else "data"
    query = f"SELECT column_type FROM (DESCRIBE {level_view}) WHERE column_name = ?"

    try:
        result = dataset._duckdb.execute(query, [col]).fetchone()
    except Exception as e:
        raise ValueError(f"Failed to get type for '{col}' in {level_view}: {e}")

    if result:
        return result[0].upper()

    raise ValueError(
        f"Column '{col}' not found in {level_view}\n"
        f"Available: {get_columns_for_level(dataset, level)}"
    )


def _timestamp_to_iso_string(timestamp: int) -> str:
    """Convert Unix timestamp to ISO 8601 string."""
    return datetime.fromtimestamp(timestamp).isoformat()


def _timestamp_to_date_string(timestamp: int) -> str:
    """Convert Unix timestamp to DATE string."""
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
    Build bounding box filter SQL.

    Uses ST_Within + ST_MakeEnvelope for bbox check.
    Prefixes column with level alias if level > 0.
    """
    col_ref = f'l{level}."{geometry_col}"' if level > 0 else f'"{geometry_col}"'

    return (
        f"ST_Within("
        f"ST_GeomFromWKB({col_ref}), "
        f"ST_MakeEnvelope({float(minx)}, {float(miny)}, {float(maxx)}, {float(maxy)})"
        f")"
    )


def build_intersects_sql(geometry: Any, geometry_col: str, level: int = 0) -> str:
    """
    Build intersection filter SQL.

    Uses ST_Intersects to check if geometries overlap.
    """
    wkt = geometry_to_wkt(geometry)
    wkt_escaped = wkt.replace("'", "''")  # SQL escape

    col_ref = f'l{level}."{geometry_col}"' if level > 0 else f'"{geometry_col}"'

    return (
        f"ST_Intersects("
        f"ST_GeomFromWKB({col_ref}), "
        f"ST_GeomFromText('{wkt_escaped}')"
        f")"
    )


def build_within_sql(geometry: Any, geometry_col: str, level: int = 0) -> str:
    """
    Build within filter SQL.

    Uses ST_Within to check if dataset geometry is inside user geometry.
    """
    wkt = geometry_to_wkt(geometry)
    wkt_escaped = wkt.replace("'", "''")

    col_ref = f'l{level}."{geometry_col}"' if level > 0 else f'"{geometry_col}"'

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
    Build temporal filter SQL with automatic type detection.

    Detects column type from DuckDB schema and generates appropriate SQL:
    - INTEGER/BIGINT: Direct timestamp comparison
    - VARCHAR: ISO string comparison
    - TIMESTAMP: Timestamp comparison with to_timestamp()
    - DATE: Date comparison with DATE cast
    """
    # Auto-detect column type
    col_type = _get_column_type(dataset, time_col, level)

    col_ref = f'l{level}."{time_col}"' if level > 0 else f'"{time_col}"'

    # Generate SQL by detected type
    if col_type in ("BIGINT", "INTEGER", "HUGEINT"):
        # Direct timestamp comparison
        if end is None:
            return f"({col_ref} = {start})"
        else:
            return f"({col_ref} BETWEEN {start} AND {end})"

    elif col_type == "VARCHAR":
        # ISO string comparison
        start_str = _timestamp_to_iso_string(start)
        if end is None:
            return f"({col_ref} = '{start_str}')"
        else:
            end_str = _timestamp_to_iso_string(end)
            return f"({col_ref} BETWEEN '{start_str}' AND '{end_str}')"

    elif col_type in ("TIMESTAMP", "TIMESTAMP WITH TIME ZONE"):
        # Timestamp comparison
        if end is None:
            return f"({col_ref} = to_timestamp({start}))"
        else:
            return f"({col_ref} BETWEEN to_timestamp({start}) AND to_timestamp({end}))"

    elif col_type == "DATE":
        # Date comparison
        start_date = _timestamp_to_date_string(start)
        if end is None:
            return f"({col_ref} = DATE '{start_date}')"
        else:
            end_date = _timestamp_to_date_string(end)
            return f"({col_ref} BETWEEN DATE '{start_date}' AND DATE '{end_date}')"

    else:
        raise ValueError(
            f"Unsupported time column type: {col_type}\n"
            f"Supported: INTEGER, BIGINT, VARCHAR, TIMESTAMP, DATE\n"
            f"Column '{time_col}' has type '{col_type}'"
        )
