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

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tacoreader._constants import (
    COLUMN_ID,
    DEFAULT_VIEW_NAME,
    LEVEL_VIEW_PREFIX,
    METADATA_PARENT_ID,
    METADATA_SOURCE_FILE,
    STAC_GEOMETRY_COLUMN_PRIORITY,
    STAC_TIME_COLUMN_PRIORITY,
)
from tacoreader._exceptions import TacoQueryError

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset


def validate_level_exists(dataset: "TacoDataset", level: int) -> None:
    """Validate level exists in dataset using pit_schema.max_depth()."""
    max_level = dataset.pit_schema.max_depth()

    if level < 0 or level > max_level:
        raise TacoQueryError(
            f"Level {level} does not exist.\n" f"Available levels: 0 to {max_level}"
        )


def get_columns_for_level(dataset: "TacoDataset", level: int) -> list[str]:
    """Get available columns for a specific level by querying DuckDB."""
    validate_level_exists(dataset, level)

    level_view = f"{LEVEL_VIEW_PREFIX}{level}" if level > 0 else DEFAULT_VIEW_NAME
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
            f"INNER JOIN {LEVEL_VIEW_PREFIX}1 l1\n"
            f'      ON l1."{METADATA_PARENT_ID}" = l0."{METADATA_PARENT_ID}"\n'
            f'     AND l1."{METADATA_SOURCE_FILE}" = l0."{METADATA_SOURCE_FILE}"'
        )
    else:
        # ZIP/FOLDER: parent_id references parent's ID
        joins.append(
            f'INNER JOIN {LEVEL_VIEW_PREFIX}1 l1 ON l1."{METADATA_PARENT_ID}" = l0.{COLUMN_ID}'
        )

    # Subsequent JOINs: level1 → level2 → level3 ...
    for level in range(2, target_level + 1):
        prev_level = level - 1
        joins.append(
            f"INNER JOIN {LEVEL_VIEW_PREFIX}{level} l{level} "
            f'ON l{level}."{METADATA_PARENT_ID}" = l{prev_level}.{COLUMN_ID}'
        )

    join_clause = "\n    ".join(joins)

    return f"""
        SELECT DISTINCT l0.*
        FROM {current_view} l0
        {join_clause}
        WHERE {where_clause}
    """


def detect_geometry_column(columns: list[str]) -> str:
    """
    Auto-detect best geometry column.

    Priority: istac:geometry > stac:centroid > istac:centroid
    """
    for col in STAC_GEOMETRY_COLUMN_PRIORITY:
        if col in columns:
            return col

    raise TacoQueryError(
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

    raise TacoQueryError(
        "No time column found.\n"
        f"Expected one of: {', '.join(STAC_TIME_COLUMN_PRIORITY)}\n"
        f"Available: {columns}"
    )


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
            start_dt = _parse_iso_string(start_str)
            end_dt = _parse_iso_string(end_str)

            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())

            if start_ts > end_ts:
                raise TacoQueryError(
                    f"Invalid range: start ({start_str}) > end ({end_str})"
                )

            return start_ts, end_ts
        else:
            # Single date string
            start_dt = _parse_iso_string(dt_input)
            return int(start_dt.timestamp()), None

    # Single datetime object
    elif isinstance(dt_input, datetime):
        dt_utc = _ensure_utc(dt_input)
        return int(dt_utc.timestamp()), None

    # Tuple of datetime objects
    elif isinstance(dt_input, tuple) and len(dt_input) == 2:
        start_dt, end_dt = dt_input
        start_utc = _ensure_utc(start_dt)
        end_utc = _ensure_utc(end_dt)

        start_ts = int(start_utc.timestamp())
        end_ts = int(end_utc.timestamp())

        if start_ts > end_ts:
            raise TacoQueryError("Invalid range: start > end")

        return start_ts, end_ts

    else:
        raise TacoQueryError(
            f"Invalid datetime: {dt_input}\n"
            f"Expected: string range ('2023-01-01/2023-12-31'), "
            f"datetime object, or tuple"
        )


def _parse_iso_string(dt_str: str) -> datetime:
    """Parse ISO 8601 string to UTC datetime."""
    # Replace 'Z' with UTC offset for fromisoformat()
    dt_str_normalized = dt_str.replace("Z", "+00:00")

    dt = datetime.fromisoformat(dt_str_normalized)

    # If naive (no timezone), assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to UTC if needed
    dt_utc = dt.astimezone(timezone.utc)

    return dt_utc


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    else:
        return dt.astimezone(timezone.utc)


def _apply_stac_filter(
    dataset: "TacoDataset",
    level: int,
    column_name: str,
    column_auto_detect_fn,
    sql_builder_fn,
    sql_builder_args: tuple,
) -> "TacoDataset":
    """
    Base method for STAC-like filtering operations.

    Abstracts common pattern:
    1. Validate level exists
    2. Get columns for target level
    3. Auto-detect or validate column
    4. Build SQL filter
    5. Apply filter (level 0 direct, level N with JOINs)
    """
    validate_level_exists(dataset, level)

    # Get columns for target level
    current_cols = (
        dataset.data.columns if level == 0 else get_columns_for_level(dataset, level)
    )

    # Auto-detect or validate column
    if column_name == "auto":
        column_name = column_auto_detect_fn(current_cols)
    else:
        if column_name not in current_cols:
            raise TacoQueryError(
                f"Column '{column_name}' not found.\nAvailable: {current_cols}"
            )

    # Build SQL filter
    sql_filter = sql_builder_fn(*sql_builder_args, column_name, level)

    # Apply filter
    if level == 0:
        return dataset.sql(f"SELECT * FROM {DEFAULT_VIEW_NAME} WHERE {sql_filter}")
    else:
        full_query = build_cascade_join_sql(
            dataset._view_name, level, sql_filter, dataset._format
        )
        return dataset.sql(full_query)


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

    # Convert epoch to UTC datetime, then to ISO string
    start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
    start_str = start_dt.isoformat()

    if end is None:
        # Point-in-time query
        return f"({col_ref} = '{start_str}')"
    else:
        # Range query
        end_dt = datetime.fromtimestamp(end, tz=timezone.utc)
        end_str = end_dt.isoformat()
        return f"({col_ref} BETWEEN '{start_str}' AND '{end_str}')"
