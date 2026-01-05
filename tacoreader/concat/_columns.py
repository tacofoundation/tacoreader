"""Column compatibility validation and resolution.

Handles three column modes:
- intersection: Keep only common columns (DEFAULT, safest)
- fill_missing: Keep all columns, fill missing with NULL
- strict: Fail if columns differ
"""

from typing import TYPE_CHECKING

from tacoreader._constants import (
    COLUMN_ID,
    COLUMN_TYPE,
    DEFAULT_VIEW_NAME,
    LEVEL_TABLE_SUFFIX,
    LEVEL_VIEW_PREFIX,
    METADATA_OFFSET,
    METADATA_PARENT_ID,
    METADATA_RELATIVE_PATH,
    METADATA_SIZE,
)
from tacoreader._exceptions import TacoQueryError, TacoSchemaError
from tacoreader._logging import get_logger

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset

logger = get_logger(__name__)


# Critical columns by format type
# These are REQUIRED for navigation to work
CRITICAL_COLUMNS_ZIP = frozenset(
    {
        COLUMN_ID,
        COLUMN_TYPE,
        METADATA_PARENT_ID,
        METADATA_OFFSET,
        METADATA_SIZE,
    }
)

CRITICAL_COLUMNS_FOLDER = frozenset(
    {
        COLUMN_ID,
        COLUMN_TYPE,
        METADATA_PARENT_ID,
    }
)

# Level 0 doesn't need parent_id (root level)
CRITICAL_COLUMNS_ZIP_LEVEL0 = CRITICAL_COLUMNS_ZIP - {METADATA_PARENT_ID}
CRITICAL_COLUMNS_FOLDER_LEVEL0 = CRITICAL_COLUMNS_FOLDER - {METADATA_PARENT_ID}


def _get_critical_columns(format_type: str, level_key: str) -> frozenset[str]:
    """Get critical columns for format and level.

    ZIP/TacoCat: need offset/size for /vsisubfile/ paths
    FOLDER: need relative_path for direct filesystem paths (level1+)
    """
    is_level0 = level_key == f"{LEVEL_VIEW_PREFIX}0"

    if format_type == "folder":
        base = CRITICAL_COLUMNS_FOLDER_LEVEL0 if is_level0 else CRITICAL_COLUMNS_FOLDER
        # Level 1+ needs relative_path for path construction
        if not is_level0:
            return base | {METADATA_RELATIVE_PATH}
        return base
    else:
        # zip, tacocat
        return CRITICAL_COLUMNS_ZIP_LEVEL0 if is_level0 else CRITICAL_COLUMNS_ZIP


def validate_column_compatibility(datasets: list["TacoDataset"], mode: str = "intersection") -> dict[str, list[str]]:
    """Validate column compatibility between datasets.

    Args:
        datasets: List of datasets to validate
        mode: Column handling strategy:
            - "intersection": Keep only common columns (DEFAULT)
            - "fill_missing": Fill missing columns with NULL
            - "strict": Fail if columns differ

    Returns:
        Dict mapping level_key -> list of final columns for that level

    Raises:
        TacoQueryError: If invalid column mode specified
        TacoSchemaError: If critical columns missing or strict mode violations
    """
    if mode not in ("intersection", "fill_missing", "strict"):
        raise TacoQueryError(
            f"Invalid column_mode: '{mode}'\nValid options: 'intersection' (default), 'fill_missing', 'strict'"
        )

    # Get format type (already validated as uniform by _validation.py)
    format_type = datasets[0]._format

    # Collect all available levels
    all_levels: set[str] = set()
    for ds in datasets:
        max_depth = ds.pit_schema.max_depth()
        all_levels.update(f"{LEVEL_VIEW_PREFIX}{i}" for i in range(max_depth + 1))

    # Collect column sets per level
    level_columns = _collect_level_columns(datasets, all_levels)

    # Process each level
    final_columns: dict[str, list[str]] = {}

    for level_key, column_sets in level_columns.items():
        common_cols_list, all_cols_list = _compute_column_lists(column_sets)

        # Validate critical columns (format-aware)
        _validate_critical_columns(level_key, column_sets, common_cols_list, format_type)

        # Handle mode-specific logic
        if mode == "strict":
            _handle_strict_mode(level_key, column_sets, common_cols_list, all_cols_list)
            final_columns[level_key] = common_cols_list
        elif mode == "intersection":
            final_columns[level_key] = common_cols_list
            _log_intersection_mode(level_key, common_cols_list, all_cols_list)
        else:  # fill_missing
            final_columns[level_key] = all_cols_list
            _log_fill_missing_mode(level_key, common_cols_list, all_cols_list)

    return final_columns


def _collect_level_columns(datasets: list["TacoDataset"], all_levels: set[str]) -> dict[str, list[set[str]]]:
    """Collect column sets for each level across all datasets.

    Level0: Reads from ds._view_name to respect .sql() column filters.
    Level1+: Reads from raw tables (_table suffix) since .sql() only affects level0.

    Generated columns (internal:gdal_vsi, internal:source_path) are regenerated
    by ViewBuilder, so their presence/absence in source doesn't matter.
    """
    level_columns: dict[str, list[set[str]]] = {}

    for level_key in sorted(all_levels):
        level_columns[level_key] = []

        for ds in datasets:
            max_depth = ds.pit_schema.max_depth()
            available_levels = [f"{LEVEL_VIEW_PREFIX}{i}" for i in range(max_depth + 1)]

            if level_key in available_levels:
                if level_key == f"{LEVEL_VIEW_PREFIX}0":
                    # Level0: Read from current view to respect .sql() filters
                    # This ensures column selection via .sql() is honored by concat()
                    arrow_table = ds._duckdb.execute(f"SELECT * FROM {ds._view_name}").fetch_arrow_table()
                else:
                    # Level1+: Read from raw table
                    # .sql() on level0 doesn't affect level1+ column selection
                    # Tables have actual columns (offset, size) needed for VSI construction
                    arrow_table = ds._duckdb.execute(
                        f"SELECT * FROM {level_key}{LEVEL_TABLE_SUFFIX}"
                    ).fetch_arrow_table()

                level_columns[level_key].append(set(arrow_table.column_names))

    return level_columns


def _compute_column_lists(
    column_sets: list[set[str]],
) -> tuple[list[str], list[str]]:
    """Compute common and all column lists from column sets."""
    column_lists = [list(cs) for cs in column_sets]

    # Compute intersection (common columns)
    common_cols_list = column_lists[0][:]
    for col_list in column_lists[1:]:
        common_cols_list = [c for c in common_cols_list if c in col_list]

    # Compute union (all columns)
    all_cols_list = []
    for col_list in column_lists:
        for col in col_list:
            if col not in all_cols_list:
                all_cols_list.append(col)

    return common_cols_list, all_cols_list


def _validate_critical_columns(
    level_key: str,
    column_sets: list[set[str]],
    common_cols: list[str],
    format_type: str,
) -> None:
    """Validate that critical columns are present in all datasets.

    This catches cases where .sql() removed columns needed for navigation.
    """
    critical_cols = _get_critical_columns(format_type, level_key)

    missing_critical = critical_cols - set(common_cols)
    if missing_critical:
        problematic = []
        for i, cols in enumerate(column_sets):
            missing = critical_cols - cols
            if missing:
                problematic.append(f"  - Dataset {i}: missing {sorted(missing)}")

        # Check if this might be due to .sql() filtering
        sql_hint = ""
        if level_key == f"{LEVEL_VIEW_PREFIX}0":
            sql_hint = (
                "\n\nIf you used .sql() to select columns before concat(), ensure you include:\n"
                f"  {sorted(critical_cols)}\n"
                "These columns are required for navigation even after filtering."
            )

        raise TacoSchemaError(
            f"Cannot concat: Critical columns missing in {level_key}\n"
            f"\n"
            f"Required columns for {format_type} format navigation:\n"
            f"  {sorted(critical_cols)}\n"
            f"\n"
            f"Missing in some datasets:\n"
            f"  {sorted(missing_critical)}\n"
            f"\n"
            f"Problems found:\n" + "\n".join(problematic) + "\n"
            f"These columns are required for .read() and hierarchical navigation."
            f"{sql_hint}"
        )


def _handle_strict_mode(
    level_key: str,
    column_sets: list[set[str]],
    common_cols: list[str],
    all_cols: list[str],
) -> None:
    """Validate columns in strict mode and raise error if mismatched."""
    if len(set(map(frozenset, column_sets))) > 1:
        all_unique_cols = []
        for i, cols in enumerate(column_sets):
            all_unique_cols.append(f"  Dataset {i}: {sorted(cols)}")

        all_cols_set = set(all_cols)
        common_cols_set = set(common_cols)

        error_msg = (
            f"Cannot concat in strict mode: Column mismatch in {level_key}\n"
            f"\n"
            f"Columns per dataset:\n" + "\n".join(all_unique_cols) + f"\n"
            f"\n"
            f"Only in some datasets: {sorted(all_cols_set - common_cols_set)}\n"
            f"Common to all: {sorted(common_cols_set)}\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Use column_mode='intersection' (default) to keep only common columns\n"
            f"  2. Use column_mode='fill_missing' to fill missing columns with NULL\n"
            f"  3. Align columns with SQL before concat:\n"
            f"     ds1 = ds1.sql('SELECT col1, col2, ... FROM {DEFAULT_VIEW_NAME}')"
        )
        raise TacoSchemaError(error_msg)


def _log_intersection_mode(
    level_key: str,
    common_cols: list[str],
    all_cols: list[str],
) -> None:
    """Log dropped columns in intersection mode."""
    dropped = set(all_cols) - set(common_cols)
    if dropped:
        logger.debug(f"{level_key}: dropped {len(dropped)} columns {sorted(dropped)}")


def _log_fill_missing_mode(
    level_key: str,
    common_cols: list[str],
    all_cols: list[str],
) -> None:
    """Log filled columns in fill_missing mode."""
    filled = set(all_cols) - set(common_cols)
    if filled:
        logger.debug(f"{level_key}: filled {len(filled)} columns with NULL {sorted(filled)}")
