"""
Hierarchical STAC filtering with multi-level JOINs (level>0).

Enables filtering level0 samples based on children's metadata at deeper levels.
Example: Find all dates (level0) with files (level2) in a specific region.

Uses internal:current_id column for hierarchical JOINs. The internal:current_id
column contains 0-indexed row positions enabling correct parent-child
relationships via internal:parent_id.

Functions:
    - apply_cascade_bbox_filter: Spatial filtering with hierarchical JOINs
    - apply_cascade_datetime_filter: Temporal filtering with hierarchical JOINs
    - build_cascade_join_sql: Generate multi-level JOIN queries
    - validate_level_exists: Validate level exists in dataset
    - get_columns_for_level: Get columns for specific level
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tacoreader._constants import (
    LEVEL_VIEW_PREFIX,
    METADATA_PARENT_ID,
    METADATA_SOURCE_FILE,
)
from tacoreader._exceptions import TacoQueryError
from tacoreader._stac import (
    detect_geometry_column,
    detect_time_column,
    parse_datetime,
)

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset


def validate_level_exists(dataset: "TacoDataset", level: int) -> None:
    """
    Validate that requested level exists in dataset.

    Args:
        dataset: TacoDataset to validate
        level: Level number to check

    Raises:
        TacoQueryError: If level does not exist
    """
    max_level = dataset.pit_schema.max_depth()

    if level < 0 or level > max_level:
        raise TacoQueryError(
            f"Level {level} does not exist in dataset.\n"
            f"Available levels: 0 to {max_level}"
        )


def get_columns_for_level(dataset: "TacoDataset", level: int) -> list[str]:
    """
    Get available columns for a specific level.

    Queries DuckDB DESCRIBE to get column names from level view.

    Args:
        dataset: TacoDataset to query
        level: Level number (must be > 0)

    Returns:
        List of column names available at that level

    Raises:
        TacoQueryError: If level does not exist
    """
    validate_level_exists(dataset, level)

    level_view = f"{LEVEL_VIEW_PREFIX}{level}"
    result = dataset._duckdb.execute(f"DESCRIBE {level_view}").fetchall()

    # Result format: [(column_name, column_type, null, key, default, extra), ...]
    return [row[0] for row in result]


def apply_cascade_bbox_filter(
    dataset: "TacoDataset",
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    geometry_col: str,
    level: int,
) -> "TacoDataset":
    """
    Filter level0 based on children's geometry at target level.

    Uses hierarchical JOINs to find level0 samples whose descendants
    at the target level intersect with the bounding box.

    Example:
        Find all dates (level0) that have files (level2) in Pacific region:

        dataset.filter_bbox(
            minx=120, miny=20, maxx=170, maxy=55,
            geometry_col="istac:geometry",
            level=2
        )

    Args:
        dataset: TacoDataset to filter
        minx: Minimum X coordinate (longitude)
        miny: Minimum Y coordinate (latitude)
        maxx: Maximum X coordinate (longitude)
        maxy: Maximum Y coordinate (latitude)
        geometry_col: Geometry column name at target level
        level: Target level for filtering (must be > 0)

    Returns:
        Filtered TacoDataset with level0 samples
    """
    validate_level_exists(dataset, level)

    # Get columns for target level
    target_cols = get_columns_for_level(dataset, level)

    # Auto-detect or validate geometry column
    if geometry_col == "auto":
        geometry_col = detect_geometry_column(target_cols)
    else:
        if geometry_col not in target_cols:
            raise TacoQueryError(
                f"Column '{geometry_col}' not found in level {level}.\n"
                f"Available columns: {target_cols}"
            )

    # Build spatial WHERE clause for target level
    # Prefixed with level alias (e.g., l2."istac:geometry")
    where_clause = (
        f"ST_Intersects("
        f'ST_GeomFromWKB(l{level}."{geometry_col}"), '
        f"ST_MakeEnvelope({float(minx)}, {float(miny)}, {float(maxx)}, {float(maxy)})"
        f")"
    )

    # Build full query with cascading JOINs
    full_query = build_cascade_join_sql(
        current_view=dataset._view_name,
        target_level=level,
        where_clause=where_clause,
        format_type=dataset._format,
    )

    return dataset.sql(full_query)


def apply_cascade_datetime_filter(
    dataset: "TacoDataset",
    datetime_range: str | datetime | tuple[datetime, datetime],
    time_col: str,
    level: int,
) -> "TacoDataset":
    """
    Filter level0 based on children's timestamps at target level.

    Uses hierarchical JOINs to find level0 samples whose descendants
    at the target level fall within the datetime range.

    Example:
        Find all regions (level0) with files (level2) from April 2023:

        dataset.filter_datetime(
            "2023-04-01/2023-04-30",
            time_col="istac:time_start",
            level=2
        )

    Args:
        dataset: TacoDataset to filter
        datetime_range: Temporal range (string, datetime, or tuple)
        time_col: Time column name at target level
        level: Target level for filtering (must be > 0)

    Returns:
        Filtered TacoDataset with level0 samples
    """
    validate_level_exists(dataset, level)

    # Get columns for target level
    target_cols = get_columns_for_level(dataset, level)

    # Auto-detect or validate time column
    if time_col == "auto":
        time_col = detect_time_column(target_cols)
    else:
        if time_col not in target_cols:
            raise TacoQueryError(
                f"Column '{time_col}' not found in level {level}.\n"
                f"Available columns: {target_cols}"
            )

    # Parse datetime range
    start, end = parse_datetime(datetime_range)

    # Build temporal WHERE clause with TRY_CAST
    # Prefixed with level alias (e.g., l2."istac:time_start")
    col_cast = f'TRY_CAST(l{level}."{time_col}" AS DATE)'

    start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
    start_str = start_dt.strftime("%Y-%m-%d")

    if end is None:
        # Point-in-time query
        where_clause = f"({col_cast} = DATE '{start_str}')"
    else:
        # Range query
        end_dt = datetime.fromtimestamp(end, tz=timezone.utc)
        end_str = end_dt.strftime("%Y-%m-%d")
        where_clause = f"({col_cast} BETWEEN DATE '{start_str}' AND DATE '{end_str}')"

    # Build full query with cascading JOINs
    full_query = build_cascade_join_sql(
        current_view=dataset._view_name,
        target_level=level,
        where_clause=where_clause,
        format_type=dataset._format,
    )

    return dataset.sql(full_query)


def build_cascade_join_sql(
    current_view: str,
    target_level: int,
    where_clause: str,
    format_type: str,
) -> str:
    """
    Build SQL with cascading JOINs from level0 to target level.

    Creates INNER JOINs: level0 → level1 → level2 → ... → target_level
    Returns DISTINCT level0 samples that match the WHERE clause at target level.

    Uses internal:current_id for JOIN relationships:
        l1."internal:parent_id" = l0."internal:current_id"
        ^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^
        INT64 (0, 1, 2...)         INT64 (0, 1, 2...)

    The internal:current_id column is stored in parquets (TACO v2.1.0+) during
    dataset creation by tacotoolbox, eliminating need for runtime calculations.

    Format-Specific Behavior:
        - ZIP: parent_id references parent internal:current_id
        - FOLDER: parent_id references parent internal:current_id
        - TacoCat: parent_id + source_file for disambiguation

    Args:
        current_view: Name of level0 view (may be temp view from prior query)
        target_level: Level to filter on (must be > 0)
        where_clause: SQL WHERE condition for target level
        format_type: Dataset format ("zip", "folder", or "tacocat")

    Returns:
        Complete SQL query with JOINs

    Example Generated SQL:
        SELECT DISTINCT l0.*
        FROM data l0
        INNER JOIN level1 l1 ON l1."internal:parent_id" = l0."internal:current_id"
        INNER JOIN level2 l2 ON l2."internal:parent_id" = l1."internal:current_id"
        WHERE ST_Within(l2."istac:geometry", ST_MakeEnvelope(...))
    """
    # Build cascading JOINs using internal:current_id
    joins = []

    # First JOIN: level0 → level1
    if format_type == "tacocat":
        # TacoCat: Uses parent_id + source_file for disambiguation
        joins.append(
            f"INNER JOIN {LEVEL_VIEW_PREFIX}1 l1 "
            f'ON l1."{METADATA_PARENT_ID}" = l0."internal:current_id" '
            f'AND l1."{METADATA_SOURCE_FILE}" = l0."{METADATA_SOURCE_FILE}"'
        )
    else:
        # ZIP/FOLDER: Use internal:current_id column
        joins.append(
            f"INNER JOIN {LEVEL_VIEW_PREFIX}1 l1 "
            f'ON l1."{METADATA_PARENT_ID}" = l0."internal:current_id"'
        )

    # Subsequent JOINs: level1 → level2 → level3 ...
    # All levelN parquets have internal:current_id
    for level in range(2, target_level + 1):
        prev_level = level - 1
        joins.append(
            f"INNER JOIN {LEVEL_VIEW_PREFIX}{level} l{level} "
            f'ON l{level}."{METADATA_PARENT_ID}" = l{prev_level}."internal:current_id"'
        )

    join_clause = "\n        ".join(joins)

    return f"""
        SELECT DISTINCT l0.*
        FROM {current_view} l0
        {join_clause}
        WHERE {where_clause}
    """
