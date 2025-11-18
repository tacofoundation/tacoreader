"""
Concatenate multiple TACO datasets into single dataset.

Creates in-memory DuckDB with consolidated metadata using UNION ALL views.
All operations remain lazy - no temp files or disk materialization.
"""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from tacoreader._constants import (
    COLUMN_ID,
    COLUMN_TYPE,
    DEFAULT_VIEW_NAME,
    LEVEL_TABLE_SUFFIX,
    LEVEL_VIEW_PREFIX,
    METADATA_GDAL_VSI,
    METADATA_OFFSET,
    METADATA_RELATIVE_PATH,
    METADATA_SIZE,
    METADATA_SOURCE_FILE,
    PADDING_PREFIX,
    PROTECTED_COLUMNS,
    SAMPLE_TYPE_FILE,
    SAMPLE_TYPE_FOLDER,
    UNION_VIEW_SUFFIX,
)
from tacoreader._logging import get_logger

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset
    from tacoreader.schema import PITSchema

logger = get_logger(__name__)


def _get_available_levels(dataset: "TacoDataset") -> list[str]:
    """Get list of available level views from pit_schema.max_depth()."""
    max_depth = dataset.pit_schema.max_depth()
    return [f"{LEVEL_VIEW_PREFIX}{i}" for i in range(max_depth + 1)]


def _collect_level_columns(
    datasets: list["TacoDataset"], all_levels: set[str]
) -> dict[str, list[set[str]]]:
    """Collect column sets for each level across all datasets."""
    level_columns: dict[str, list[set[str]]] = {}

    for level_key in sorted(all_levels):
        level_columns[level_key] = []

        for ds in datasets:
            available_levels = _get_available_levels(ds)
            if level_key in available_levels:
                df = ds._duckdb.execute(f"SELECT * FROM {level_key}").pl()
                level_columns[level_key].append(set(df.columns))

    return level_columns


def _compute_column_lists(
    column_sets: list[set[str]],
) -> tuple[list[str], list[str]]:
    """Compute common and all column lists from column sets."""
    column_lists = [list(cs) for cs in column_sets]

    if len(column_lists) == 1:
        return column_lists[0][:], column_lists[0][:]

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
    level_key: str, column_sets: list[set[str]], common_cols: list[str]
) -> None:
    """Validate that critical columns are present in all datasets."""
    if level_key == f"{LEVEL_VIEW_PREFIX}0":
        critical_cols = PROTECTED_COLUMNS - {
            METADATA_SOURCE_FILE,
            METADATA_RELATIVE_PATH,
        }
    else:
        critical_cols = PROTECTED_COLUMNS - {METADATA_SOURCE_FILE}

    missing_critical = critical_cols - set(common_cols)
    if missing_critical:
        problematic = []
        for i, cols in enumerate(column_sets):
            missing = critical_cols - cols
            if missing:
                problematic.append(f"  - Dataset {i}: missing {sorted(missing)}")

        raise ValueError(
            f"Cannot concat: Critical columns missing in {level_key}\n"
            f"\n"
            f"Required columns for navigation:\n"
            f"  {sorted(critical_cols)}\n"
            f"\n"
            f"Missing in some datasets:\n"
            f"  {sorted(missing_critical)}\n"
            f"\n"
            f"Problems found:\n" + "\n".join(problematic) + "\n"
            "These columns are required for .read() and hierarchical navigation."
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
        raise ValueError(error_msg)


def _handle_intersection_mode(
    level_key: str,
    column_sets: list[set[str]],
    common_cols: list[str],
    all_cols: list[str],
) -> None:
    """Warn about dropped columns in intersection mode."""
    if set(all_cols) == set(common_cols):
        return

    dropped_set = set(all_cols) - set(common_cols)
    column_sources = {}
    for col in dropped_set:
        sources = []
        for i, cols in enumerate(column_sets):
            if col in cols:
                sources.append(i)
        column_sources[col] = sources

    details = []
    for col, sources in sorted(column_sources.items()):
        if len(sources) < len(column_sets):
            details.append(f"  - '{col}' (only in dataset(s) {sources})")

    warnings.warn(
        f"\n"
        f"concat() dropped {len(dropped_set)} column(s) from {level_key}\n"
        f"\n"
        f"Reason: Using column_mode='intersection' (default behavior)\n"
        f"        Only columns present in ALL datasets are kept.\n"
        f"\n"
        f"Dropped columns:\n" + "\n".join(details) + f"\n"
        f"\n"
        f"Kept columns ({len(common_cols)}): {sorted(common_cols)}\n"
        f"\n"
        f"To keep all columns (fill missing with NULL):\n"
        f"   concat([ds1, ds2], column_mode='fill_missing')",
        UserWarning,
        stacklevel=4,
    )


def _handle_fill_missing_mode(
    level_key: str,
    column_sets: list[set[str]],
    common_cols: list[str],
    all_cols: list[str],
) -> None:
    """Warn about filled columns in fill_missing mode."""
    if set(all_cols) == set(common_cols):
        return

    missing_set = set(all_cols) - set(common_cols)
    column_gaps = {}
    for col in missing_set:
        gaps = []
        for i, cols in enumerate(column_sets):
            if col not in cols:
                gaps.append(i)
        column_gaps[col] = gaps

    details = []
    for col, gaps in sorted(column_gaps.items()):
        details.append(
            f"  - '{col}' (missing in dataset(s) {gaps}, will fill with NULL)"
        )

    warnings.warn(
        f"\n"
        f"concat() filling missing columns in {level_key} with NULL\n"
        f"\n"
        f"Reason: Using column_mode='fill_missing'\n"
        f"        All columns from all datasets are kept.\n"
        f"\n"
        f"Columns being filled:\n" + "\n".join(details) + f"\n"
        f"\n"
        f"Total columns: {len(all_cols)} (common: {len(common_cols)}, filled: {len(missing_set)})\n"
        f"\n"
        f"To avoid NULLs, use column_mode='intersection' (drops columns not in all datasets)",
        UserWarning,
        stacklevel=4,
    )


def _validate_column_compatibility(
    datasets: list["TacoDataset"], mode: str = "intersection"
) -> dict[str, list[str]]:
    """
    Validate column compatibility between datasets.

    Args:
        mode: Column handling strategy:
            - "intersection": Keep only common columns (DEFAULT)
            - "fill_missing": Fill missing columns with NULL
            - "strict": Fail if columns differ

    Returns:
        Dict mapping level_key to set of final columns
    """
    if mode not in ("intersection", "fill_missing", "strict"):
        raise ValueError(
            f"Invalid column_mode: '{mode}'\n"
            f"Valid options: 'intersection' (default), 'fill_missing', 'strict'"
        )

    all_levels = set()
    for ds in datasets:
        all_levels.update(_get_available_levels(ds))

    level_columns = _collect_level_columns(datasets, all_levels)
    final_columns: dict[str, list[str]] = {}

    for level_key, column_sets in level_columns.items():
        common_cols_list, all_cols_list = _compute_column_lists(column_sets)

        # Validate critical columns
        _validate_critical_columns(level_key, column_sets, common_cols_list)

        # Handle mode-specific logic
        if mode == "strict":
            _handle_strict_mode(level_key, column_sets, common_cols_list, all_cols_list)
            final_columns[level_key] = common_cols_list
        elif mode == "intersection":
            final_columns[level_key] = common_cols_list
            _handle_intersection_mode(
                level_key, column_sets, common_cols_list, all_cols_list
            )
        else:  # fill_missing
            final_columns[level_key] = all_cols_list
            _handle_fill_missing_mode(
                level_key, column_sets, common_cols_list, all_cols_list
            )

    return final_columns


def _create_union_views(
    db: duckdb.DuckDBPyConnection,
    datasets: list["TacoDataset"],
    all_levels: set[str],
    target_columns_by_level: dict[str, list[str]],
) -> None:
    """Create UNION ALL views for each level."""
    for level_key in sorted(all_levels):
        logger.debug(f"  Consolidating {level_key}...")

        target_cols = target_columns_by_level[level_key]
        union_parts = []

        for ds_idx, ds in enumerate(datasets):
            available_levels = _get_available_levels(ds)
            if level_key not in available_levels:
                continue

            # Extract PyArrow table from original dataset
            arrow_table = ds._duckdb.execute(
                f"SELECT * FROM {level_key}{LEVEL_TABLE_SUFFIX}"
            ).fetch_arrow_table()

            # Register in new connection with unique name
            table_name = f"ds{ds_idx}_{level_key}{LEVEL_TABLE_SUFFIX}"
            db.register(table_name, arrow_table)

            # Build SELECT with aligned columns + internal:source_file
            source_file = Path(ds._path).name
            current_cols = set(arrow_table.column_names)

            # Build column list with alignment
            select_parts = []
            for col in sorted(target_cols):
                if col in current_cols:
                    escaped_col = f'"{col}"' if ":" in col or " " in col else col
                    select_parts.append(escaped_col)
                else:
                    select_parts.append(f'NULL AS "{col}"')

            # Add internal:source_file
            select_parts.append(f"'{source_file}' AS \"{METADATA_SOURCE_FILE}\"")

            union_parts.append(f"SELECT {', '.join(select_parts)} FROM {table_name}")

        if not union_parts:
            continue

        # Create consolidated view with UNION ALL
        union_query = " UNION ALL ".join(union_parts)
        db.execute(f"CREATE VIEW {level_key}{UNION_VIEW_SUFFIX} AS {union_query}")

        logger.debug(
            f"    {level_key}: {len(union_parts)} dataset(s), {len(target_cols)} columns"
        )


def _create_zip_views(
    db: duckdb.DuckDBPyConnection, all_levels: set[str], root_path: str
) -> None:
    """Create final views for ZIP format."""
    for level_key in sorted(all_levels):
        if not db.execute(
            f"SELECT 1 FROM information_schema.tables WHERE table_name = '{level_key}{UNION_VIEW_SUFFIX}'"
        ).fetchone():
            continue

        db.execute(
            f"""
            CREATE VIEW {level_key} AS
            SELECT *,
              '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
              "{METADATA_SIZE}" || ',{root_path}' as "{METADATA_GDAL_VSI}"
            FROM {level_key}{UNION_VIEW_SUFFIX}
            WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
        """
        )


def _create_folder_views(
    db: duckdb.DuckDBPyConnection, all_levels: set[str], root_path: str
) -> None:
    """Create final views for FOLDER format."""
    root = root_path if root_path.endswith("/") else root_path + "/"

    for level_key in sorted(all_levels):
        if not db.execute(
            f"SELECT 1 FROM information_schema.tables WHERE table_name = '{level_key}{UNION_VIEW_SUFFIX}'"
        ).fetchone():
            continue

        if level_key == f"{LEVEL_VIEW_PREFIX}0":
            db.execute(
                f"""
                CREATE VIEW {level_key} AS
                SELECT *,
                  CASE
                    WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FOLDER}' THEN '{root}DATA/' || {COLUMN_ID} || '/__meta__'
                    WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FILE}' THEN '{root}DATA/' || {COLUMN_ID}
                    ELSE NULL
                  END as "{METADATA_GDAL_VSI}"
                FROM {level_key}{UNION_VIEW_SUFFIX}
                WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
            """
            )
        else:
            db.execute(
                f"""
                CREATE VIEW {level_key} AS
                SELECT *,
                  CASE
                    WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FOLDER}' THEN '{root}DATA/' || "{METADATA_RELATIVE_PATH}" || '__meta__'
                    WHEN {COLUMN_TYPE} = '{SAMPLE_TYPE_FILE}' THEN '{root}DATA/' || "{METADATA_RELATIVE_PATH}"
                    ELSE NULL
                  END as "{METADATA_GDAL_VSI}"
                FROM {level_key}{UNION_VIEW_SUFFIX}
                WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
            """
            )


def _create_tacocat_views(
    db: duckdb.DuckDBPyConnection, all_levels: set[str], base_path: str
) -> None:
    """Create final views for TacoCat format."""
    for level_key in sorted(all_levels):
        if not db.execute(
            f"SELECT 1 FROM information_schema.tables WHERE table_name = '{level_key}{UNION_VIEW_SUFFIX}'"
        ).fetchone():
            continue

        db.execute(
            f"""
            CREATE VIEW {level_key} AS
            SELECT *,
              '/vsisubfile/' || "{METADATA_OFFSET}" || '_' ||
              "{METADATA_SIZE}" || ',{base_path}' || "{METADATA_SOURCE_FILE}"
              as "{METADATA_GDAL_VSI}"
            FROM {level_key}{UNION_VIEW_SUFFIX}
            WHERE {COLUMN_ID} NOT LIKE '{PADDING_PREFIX}%'
        """
        )


def concat(
    datasets: list["TacoDataset"], column_mode: str = "intersection"
) -> "TacoDataset":
    """
    Concatenate multiple datasets into single dataset with lazy SQL.

    Creates in-memory DuckDB with consolidated metadata using UNION ALL views.
    By default, only keeps columns present in ALL datasets (intersection mode).

    Args:
        datasets: List of TacoDataset instances (minimum 2)
        column_mode: Column handling strategy (default "intersection")
            - "intersection": Keep only common columns (DEFAULT, safest)
            - "fill_missing": Keep all columns, fill missing with NULL
            - "strict": Fail if columns differ

    Returns:
        TacoDataset with consolidated data and lazy SQL
    """
    if len(datasets) < 2:
        raise ValueError(f"Need at least 2 datasets to concat, got {len(datasets)}")

    logger.info(f"Concatenating {len(datasets)} datasets...")

    # Validate PIT schemas
    reference_schema = datasets[0].pit_schema
    for i, ds in enumerate(datasets[1:], 1):
        if not reference_schema.is_compatible(ds.pit_schema):
            raise ValueError(
                f"Dataset {i} has incompatible schema. "
                f"All datasets must share same hierarchy structure.\n"
                f"Reference: {reference_schema}\n"
                f"Dataset {i}: {ds.pit_schema}"
            )

    logger.debug("All schemas compatible")

    # Validate columns and get target columns
    logger.debug(f"Validating columns (mode={column_mode})...")
    target_columns_by_level = _validate_column_compatibility(datasets, mode=column_mode)

    for level_key, cols in target_columns_by_level.items():
        logger.debug(f"  {level_key}: {len(cols)} columns")

    # Consolidate schemas
    consolidated_schema = _merge_schemas([ds.pit_schema for ds in datasets])
    logger.debug("Consolidating levels in-memory...")

    # Create new DuckDB connection
    db = duckdb.connect(":memory:")

    # Load spatial extension
    try:
        db.execute("INSTALL spatial")
        db.execute("LOAD spatial")
        logger.debug("Loaded DuckDB spatial extension")
    except Exception as e:
        logger.debug(f"Spatial extension not available: {e}")

    # Get all available levels
    all_levels = set()
    for ds in datasets:
        all_levels.update(_get_available_levels(ds))

    # Create UNION ALL views for each level
    _create_union_views(db, datasets, all_levels, target_columns_by_level)

    # Create final views with internal:gdal_vsi by format
    first_format = datasets[0]._format
    if first_format == "zip":
        _create_zip_views(db, all_levels, datasets[0]._root_path)
    elif first_format == "folder":
        _create_folder_views(db, all_levels, datasets[0]._root_path)
    elif first_format == "tacocat":
        _create_tacocat_views(db, all_levels, datasets[0]._root_path)

    # Create 'data' view
    db.execute(f"CREATE VIEW {DEFAULT_VIEW_NAME} AS SELECT * FROM {LEVEL_VIEW_PREFIX}0")

    total_samples = consolidated_schema.root["n"]
    logger.info(
        f"Concatenated {len(datasets)} datasets ({total_samples:,} total samples)"
    )

    # Build TacoDataset
    from tacoreader.dataset import TacoDataset

    dataset = TacoDataset.model_construct(
        id=datasets[0].id,
        version=datasets[0].version,
        description=datasets[0].description,
        tasks=datasets[0].tasks,
        extent=datasets[0].extent,
        providers=datasets[0].providers,
        licenses=datasets[0].licenses,
        title=datasets[0].title,
        curators=datasets[0].curators,
        keywords=datasets[0].keywords,
        pit_schema=consolidated_schema,
        _path="<concatenated>",
        _format=first_format,
        _collection=datasets[0]._collection,
        _duckdb=db,
        _view_name=DEFAULT_VIEW_NAME,
        _root_path=datasets[0]._root_path,
    )

    return dataset


def _merge_schemas(schemas: list["PITSchema"]) -> "PITSchema":
    """Merge compatible schemas by summing n values."""
    if not schemas:
        raise ValueError("Need at least one schema to merge")

    reference = schemas[0]
    merged_dict = reference.to_dict()

    merged_dict["root"]["n"] = sum(s.root["n"] for s in schemas)

    for depth_str in merged_dict["hierarchy"]:
        for pattern_idx in range(len(merged_dict["hierarchy"][depth_str])):
            total_n = sum(s.hierarchy[depth_str][pattern_idx]["n"] for s in schemas)
            merged_dict["hierarchy"][depth_str][pattern_idx]["n"] = total_n

    from tacoreader.schema import PITSchema

    return PITSchema(merged_dict)
