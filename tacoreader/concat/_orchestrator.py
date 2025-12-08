"""
Main concatenation orchestrator.

Coordinates the 4-phase concat process:
1. Validation: Check dataset compatibility
2. Preparation: Resolve columns and merge schemas
3. Construction: Build DuckDB views with UNION ALL
4. Finalization: Create TacoDataset instance
"""

from typing import TYPE_CHECKING

import duckdb

from tacoreader._constants import DEFAULT_VIEW_NAME, LEVEL_VIEW_PREFIX
from tacoreader._exceptions import TacoQueryError
from tacoreader._logging import get_logger
from tacoreader.concat._validation import validate_datasets
from tacoreader.concat._view_builder import ViewBuilder
from tacoreader.dataset import TacoDataset
from tacoreader.schema import PITSchema

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


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

    Raises:
        TacoQueryError: If less than 2 datasets provided
        TacoBackendError: If datasets have incompatible backends
        TacoSchemaError: If datasets have incompatible schemas or formats
    """
    if len(datasets) < 2:
        raise TacoQueryError(f"Need at least 2 datasets to concat, got {len(datasets)}")

    logger.info(f"Concatenating {len(datasets)} datasets...")

    validate_datasets(datasets)

    # Use backend from first dataset (all are the same after validation)
    backend = datasets[0]._dataframe_backend
    logger.debug(f"Using DataFrame backend: {backend}")

    # Validate columns and get target columns
    logger.debug(f"Validating columns (mode={column_mode})...")
    from tacoreader.concat._columns import validate_column_compatibility

    target_columns_by_level = validate_column_compatibility(datasets, mode=column_mode)

    for level_key, cols in target_columns_by_level.items():
        logger.debug(f"  {level_key}: {len(cols)} columns")

    # Merge schemas
    consolidated_schema = _merge_schemas(datasets)
    logger.debug("Merged schemas")

    logger.debug("Building DuckDB views...")

    # Create new DuckDB connection with spatial extension
    db = _setup_duckdb_connection()

    # Get all available levels
    all_levels = _get_all_levels(datasets)

    # Build all views using ViewBuilder
    view_builder = ViewBuilder(
        db=db,
        datasets=datasets,
        all_levels=all_levels,
        target_columns_by_level=target_columns_by_level,
    )
    view_builder.build_all_views()

    # Create 'data' view pointing to level0
    db.execute(f"CREATE VIEW {DEFAULT_VIEW_NAME} AS SELECT * FROM {LEVEL_VIEW_PREFIX}0")

    total_samples = consolidated_schema.root["n"]
    logger.info(
        f"Concatenated {len(datasets)} datasets ({total_samples:,} total samples)"
    )

    # Build TacoDataset
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
        _format=datasets[0]._format,
        _collection=datasets[0]._collection,
        _duckdb=db,
        _view_name=DEFAULT_VIEW_NAME,
        _root_path=datasets[0]._root_path,
        _dataframe_backend=backend,
    )

    return dataset


def _setup_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create DuckDB connection with spatial extension if available."""
    db = duckdb.connect(":memory:")

    try:
        db.execute("INSTALL spatial")
        db.execute("LOAD spatial")
        logger.debug("Loaded DuckDB spatial extension")
    except Exception as e:
        logger.debug(f"Spatial extension not available: {e}")

    return db


def _get_all_levels(datasets: list["TacoDataset"]) -> set[str]:
    """Get union of all available level views across datasets."""
    all_levels: set[str] = set()
    for ds in datasets:
        max_depth = ds.pit_schema.max_depth()
        all_levels.update(f"{LEVEL_VIEW_PREFIX}{i}" for i in range(max_depth + 1))
    return all_levels


def _merge_schemas(datasets: list["TacoDataset"]) -> PITSchema:
    """Merge compatible schemas by summing n values."""
    if not datasets:
        raise TacoQueryError("Need at least one dataset to merge")

    reference = datasets[0].pit_schema
    merged_dict = reference.to_dict()

    merged_dict["root"]["n"] = sum(ds.pit_schema.root["n"] for ds in datasets)

    for depth_str in merged_dict["hierarchy"]:
        for pattern_idx in range(len(merged_dict["hierarchy"][depth_str])):
            total_n = sum(
                ds.pit_schema.hierarchy[depth_str][pattern_idx]["n"] for ds in datasets
            )
            merged_dict["hierarchy"][depth_str][pattern_idx]["n"] = total_n

    return PITSchema(merged_dict)
