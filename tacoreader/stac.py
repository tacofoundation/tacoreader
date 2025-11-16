"""
STAC-style spatiotemporal filtering for TacoDataset.

Simple API for filtering by spatial and temporal criteria.
All spatial ops use DuckDB Spatial extension with WKB columns.
All temporal ops use native Parquet TIMESTAMP columns.

Main functions:
    - detect_geometry_column / detect_time_column: Auto-detection
    - build_bbox_sql: Spatial bounding box filter
    - build_datetime_sql: Temporal filter (TIMESTAMP native)
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
# INPUT CONVERSION
# ============================================================================


def parse_datetime(
    dt_input: str | datetime | tuple[datetime, datetime]
) -> tuple[int, int | None]:
    """
    Parse datetime to (start_epoch, end_epoch).

    Formats:
    - String range: "2023-01-01/2023-12-31"
    - Single datetime: datetime(2023, 1, 1)
    - Tuple: (start_dt, end_dt)

    Returns end_epoch=None for single datetime.
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


def build_datetime_sql(
    start: int,
    end: int | None,
    time_col: str,
    level: int,
) -> str:
    """
    Build temporal filter SQL with native TIMESTAMP comparison.

    All temporal columns are stored as Parquet TIMESTAMP type.
    DuckDB automatically parses ISO 8601 strings when comparing with TIMESTAMP columns.

    Args:
        start: Unix epoch seconds
        end: Unix epoch seconds (None for point-in-time query)
        time_col: Column name (e.g., "stac:time_start")
        level: Level number for multi-level filtering

    Returns:
        SQL WHERE clause fragment
    """
    col_ref = f'l{level}."{time_col}"' if level > 0 else f'"{time_col}"'

    # Convert epoch to ISO string - DuckDB parses it automatically
    start_str = datetime.fromtimestamp(start).isoformat()

    if end is None:
        # Point-in-time query
        return f"({col_ref} = '{start_str}')"
    else:
        # Range query
        end_str = datetime.fromtimestamp(end).isoformat()
        return f"({col_ref} BETWEEN '{start_str}' AND '{end_str}')"