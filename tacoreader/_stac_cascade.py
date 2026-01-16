"""Hierarchical STAC filtering with multi-level JOINs (level>0).

Enables filtering level0 samples based on children's metadata at deeper levels.
Example: Find all dates (level0) with files (level2) in a specific region.

When filtering at level>0, children that don't match are EXCLUDED from results,
which may break RSUT compliance (structural homogeneity).

Uses internal:current_id column for hierarchical JOINs. The internal:current_id
column contains 0-indexed row positions enabling correct parent-child
relationships via internal:parent_id.

The filtered views are stored in _filtered_level_views dict and propagated
to TacoDataFrame so .read() can query DuckDB instead of physical __meta__ files.

Functions:
    - apply_cascade_bbox_filter: Spatial filtering with hierarchical JOINs
    - apply_cascade_datetime_filter: Temporal filtering with hierarchical JOINs
    - validate_level_exists: Validate level exists in dataset
    - get_columns_for_level: Get columns for specific level
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tacoreader._constants import (
    LEVEL_VIEW_PREFIX,
    METADATA_CURRENT_ID,
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
    """Validate that requested level exists in dataset.

    Args:
        dataset: TacoDataset to validate
        level: Level number to check

    Raises:
        TacoQueryError: If level does not exist
    """
    max_level = dataset.pit_schema.max_depth()

    if level < 0 or level > max_level:
        raise TacoQueryError(f"Level {level} does not exist in dataset.\nAvailable levels: 0 to {max_level}")


def get_columns_for_level(dataset: "TacoDataset", level: int) -> list[str]:
    """Get available columns for a specific level.

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
    """Filter level0 based on children's geometry at target level.

    Uses hierarchical JOINs to find level0 samples whose descendants
    at the target level intersect with the bounding box.

    Children that don't match the bbox are EXCLUDED from results.
    This may break RSUT compliance if different parents end up with
    different surviving children.

    The filtered views are stored in _filtered_level_views so
    TacoDataFrame.read() can query DuckDB instead of physical __meta__.

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
        Filtered TacoDataset with level0 samples and _filtered_level_views
    """
    validate_level_exists(dataset, level)

    target_cols = get_columns_for_level(dataset, level)

    if geometry_col == "auto":
        geometry_col = detect_geometry_column(target_cols)
    elif geometry_col not in target_cols:
        raise TacoQueryError(f"Column '{geometry_col}' not found in level {level}.\nAvailable columns: {target_cols}")

    where_clause = (
        f'ST_Intersects(ST_GeomFromWKB("{geometry_col}"), '
        f"ST_MakeEnvelope({float(minx)}, {float(miny)}, {float(maxx)}, {float(maxy)}))"
    )

    return _apply_cascade_filter(dataset, level, where_clause)


def apply_cascade_datetime_filter(
    dataset: "TacoDataset",
    datetime_range: str | datetime | tuple[datetime, datetime],
    time_col: str,
    level: int,
) -> "TacoDataset":
    """Filter level0 based on children's timestamps at target level.

    Uses hierarchical JOINs to find level0 samples whose descendants
    at the target level fall within the datetime range.

    Children that don't match the datetime range are EXCLUDED from results.
    This may break RSUT compliance if different parents end up with
    different surviving children.

    The filtered views are stored in _filtered_level_views so
    TacoDataFrame.read() can query DuckDB instead of physical __meta__.

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
        Filtered TacoDataset with level0 samples and _filtered_level_views
    """
    validate_level_exists(dataset, level)

    target_cols = get_columns_for_level(dataset, level)

    if time_col == "auto":
        time_col = detect_time_column(target_cols)
    elif time_col not in target_cols:
        raise TacoQueryError(f"Column '{time_col}' not found in level {level}.\nAvailable columns: {target_cols}")

    start, end = parse_datetime(datetime_range)

    col_cast = f'TRY_CAST("{time_col}" AS DATE)'

    start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
    start_str = start_dt.strftime("%Y-%m-%d")

    if end is None:
        where_clause = f"({col_cast} = DATE '{start_str}')"
    else:
        end_dt = datetime.fromtimestamp(end, tz=timezone.utc)
        end_str = end_dt.strftime("%Y-%m-%d")
        where_clause = f"({col_cast} BETWEEN DATE '{start_str}' AND DATE '{end_str}')"

    return _apply_cascade_filter(dataset, level, where_clause)


def _apply_cascade_filter(
    dataset: "TacoDataset",
    target_level: int,
    where_clause: str,
) -> "TacoDataset":
    """Apply filter at target level and propagate upward.

    Filters children at target level, then propagates upward keeping only
    parents that have surviving children. This excludes non-matching children
    from the result.

    The filtered views dict is propagated to the resulting TacoDataset so
    TacoDataFrame.read() can query DuckDB instead of physical __meta__ files.

    Strategy:
        1. Filter target level by where_clause → filtered_level{N}
        2. For each level from target-1 down to 0:
           - Keep only rows whose children exist in filtered level below
           - For level0, intersect with dataset._view_name to respect prior filters
        3. Create new level0 view as the final result
        4. Check RSUT compliance
        5. Store all filtered_views in _filtered_level_views

    Args:
        dataset: TacoDataset to filter
        target_level: Level to apply where_clause
        where_clause: SQL WHERE condition

    Returns:
        Filtered TacoDataset with _filtered_level_views for navigation
    """
    db = dataset._duckdb
    suffix = uuid.uuid4().hex[:8]
    is_tacocat = dataset._format == "tacocat"

    # Dict to store all filtered views by level
    filtered_views: dict[int, str] = {}

    # Step 1: Filter target level
    target_view = f"{LEVEL_VIEW_PREFIX}{target_level}"
    filtered_target = f"filtered_{target_level}_{suffix}"

    db.execute(
        f"""
        CREATE TEMP VIEW {filtered_target} AS
        SELECT * FROM {target_view}
        WHERE {where_clause}
    """
    )
    filtered_views[target_level] = filtered_target

    # Step 2: Propagate upward from target-1 to 0
    for lvl in range(target_level - 1, -1, -1):
        child_filtered = filtered_views[lvl + 1]
        filtered_current = f"filtered_{lvl}_{suffix}"

        # Level 0 uses dataset._view_name to respect prior filters
        current_view = dataset._view_name if lvl == 0 else f"{LEVEL_VIEW_PREFIX}{lvl}"

        # TacoCat needs source_file in JOIN condition
        if is_tacocat:
            join_condition = (
                f'child."{METADATA_PARENT_ID}" = parent."{METADATA_CURRENT_ID}" '
                f'AND child."{METADATA_SOURCE_FILE}" = parent."{METADATA_SOURCE_FILE}"'
            )
        else:
            join_condition = f'child."{METADATA_PARENT_ID}" = parent."{METADATA_CURRENT_ID}"'

        db.execute(
            f"""
            CREATE TEMP VIEW {filtered_current} AS
            SELECT DISTINCT parent.*
            FROM {current_view} parent
            INNER JOIN {child_filtered} child ON {join_condition}
        """
        )
        filtered_views[lvl] = filtered_current

    # Step 3: Create final view for level0
    final_view = f"view_{suffix}"
    db.execute(
        f"""
        CREATE TEMP VIEW {final_view} AS
        SELECT * FROM {filtered_views[0]}
    """
    )

    # Count rows and update schema
    new_n = db.execute(f"SELECT COUNT(*) FROM {final_view}").fetchone()[0]
    new_schema = dataset.pit_schema.with_n(new_n)

    # Step 4: Check RSUT compliance using filtered views
    new_rsut_compliant = _check_cascade_rsut(db, filtered_views, is_tacocat)

    # Track joined levels for debugging
    joined_levels = dataset._joined_levels.copy()
    for lvl in range(1, target_level + 1):
        joined_levels.add(f"{LEVEL_VIEW_PREFIX}{lvl}")

    from tacoreader.dataset import TacoDataset as TDS

    return TDS.model_construct(
        id=dataset.id,
        version=dataset.version,
        description=dataset.description,
        tasks=dataset.tasks,
        extent=dataset.extent,
        providers=dataset.providers,
        licenses=dataset.licenses,
        title=dataset.title,
        curators=dataset.curators,
        keywords=dataset.keywords,
        pit_schema=new_schema,
        _path=dataset._path,
        _format=dataset._format,
        _collection=dataset._collection,
        _duckdb=db,
        _view_name=final_view,
        _vsi_base_path=dataset._vsi_base_path,
        _dataframe_backend=dataset._dataframe_backend,
        _owns_connection=False,
        _joined_levels=joined_levels,
        _rsut_compliant=new_rsut_compliant,
        _filtered_level_views=filtered_views,
        _extent_modified=True,
    )


def _check_cascade_rsut(
    db,
    filtered_views: dict[int, str],
    is_tacocat: bool,
) -> bool:
    """Check if filtered result maintains RSUT (structural homogeneity).

    RSUT Invariant 3: All level0 FOLDERs must have identical child signatures.
    After cascade filtering, different parents may have different surviving children.

    Args:
        db: DuckDB connection
        filtered_views: Dict mapping level -> filtered view name
        is_tacocat: True if TacoCat format (needs source_file in JOIN)

    Returns:
        True if all level0 samples have same child signature, False otherwise
    """
    if 0 not in filtered_views or 1 not in filtered_views:
        return True

    level0_view = filtered_views[0]
    level1_view = filtered_views[1]

    # TacoCat needs source_file in JOIN condition
    if is_tacocat:
        join_condition = (
            f'l1."{METADATA_PARENT_ID}" = l0."{METADATA_CURRENT_ID}" '
            f'AND l1."{METADATA_SOURCE_FILE}" = l0."{METADATA_SOURCE_FILE}"'
        )
    else:
        join_condition = f'l1."{METADATA_PARENT_ID}" = l0."{METADATA_CURRENT_ID}"'

    query = f"""
        WITH parent_children AS (
            SELECT
                l0."{METADATA_CURRENT_ID}" as parent_id,
                COALESCE(STRING_AGG(l1.id, '|' ORDER BY l1.id), '') as children_sig
            FROM {level0_view} l0
            LEFT JOIN {level1_view} l1 ON {join_condition}
            GROUP BY l0."{METADATA_CURRENT_ID}"
        )
        SELECT COUNT(DISTINCT children_sig) = 1 as is_homogeneous
        FROM parent_children
    """

    try:
        result = db.execute(query).fetchone()
        return bool(result[0]) if result else True
    except Exception:
        return False
