"""
Load TACO datasets from any format.

Main entry point. Auto-detects format and dispatches to backend.
Supports single files or lists (automatically concatenated).
"""

from tacoreader._constants import DEFAULT_VIEW_NAME, LEVEL_VIEW_PREFIX
from tacoreader._exceptions import TacoQueryError
from tacoreader._format import detect_and_resolve_format
from tacoreader._legacy import is_legacy_format, raise_legacy_error
from tacoreader._vsi import to_vsi_root
from tacoreader.dataset import TacoDataset
from tacoreader.storage import create_backend, load_dataset


def load(
    path: str | list[str],
    base_path: str | None = None,
    backend: str | None = None,
) -> TacoDataset:
    """
    Load TACO dataset(s) with auto format detection.

    Returns TacoDataset with lazy SQL interface. Multiple files are
    automatically concatenated if schemas are compatible.

    Args:
        path: Single path or list of paths to load
            Formats: .tacozip (ZIP), directory (FOLDER), __TACOCAT__
            Examples: "data.tacozip", ["part1.tacozip", "part2.tacozip"]
        base_path: Override base path for TacoCat ZIP files (TacoCat only)
            Use when __TACOCAT__ and .tacozip files are in different locations
        backend: DataFrame backend to use ('pyarrow', 'polars', 'pandas')
            If None, uses global backend set by tacoreader.use()
            Default: 'pyarrow'

    Returns:
        TacoDataset with lazy SQL interface

    Examples:
        # Use default backend (pyarrow)
        reader = tacoreader.load('data.taco')

        # Override backend for this load
        reader = tacoreader.load('data.taco', backend='polars')

        # Set global backend
        tacoreader.use('polars')
        reader = tacoreader.load('data.taco')  # Uses polars
    """
    # Resolve backend: use specified or fall back to global
    from tacoreader import _DATAFRAME_BACKEND

    backend = backend or _DATAFRAME_BACKEND

    # Handle list of paths
    if isinstance(path, list):
        if len(path) == 0:
            raise TacoQueryError("Cannot load empty list of paths")

        if len(path) == 1:
            return load(path[0], base_path, backend)

        # Multiple files - load and concatenate
        datasets = [load(p, base_path, backend) for p in path]

        from tacoreader.concat import concat

        return concat(datasets)

    # Check for legacy format (fail fast with helpful message)
    if is_legacy_format(path):
        raise_legacy_error(path)

    # Detect and resolve format (handles TacoCat fallback)
    format_type, resolved_path = detect_and_resolve_format(path)

    # TacoCat with base_path override: rebuild views with new base
    if format_type == "tacocat" and base_path is not None:
        backend_obj = create_backend("tacocat")
        dataset = backend_obj.load(resolved_path)
        dataset._dataframe_backend = backend

        base_vsi = to_vsi_root(base_path)
        max_depth = dataset.pit_schema.max_depth()

        # Drop existing views
        dataset._duckdb.execute(f"DROP VIEW IF EXISTS {DEFAULT_VIEW_NAME}")
        for i in range(max_depth + 1):
            dataset._duckdb.execute(f"DROP VIEW IF EXISTS {LEVEL_VIEW_PREFIX}{i}")

        # Recreate views with new base_path using backend method
        level_ids = list(range(max_depth + 1))
        backend_obj.setup_duckdb_views(dataset._duckdb, level_ids, base_vsi)

        # Recreate 'data' view
        dataset._duckdb.execute(
            f"CREATE VIEW {DEFAULT_VIEW_NAME} AS SELECT * FROM {LEVEL_VIEW_PREFIX}0"
        )
        dataset._root_path = base_vsi

        return dataset

    # Standard loading with resolved path
    backend_obj = create_backend(format_type)

    # Load dataset and set backend
    if format_type == "tacocat":
        dataset = backend_obj.load(resolved_path)
        dataset._dataframe_backend = backend
        return dataset
    else:
        dataset = load_dataset(resolved_path, format_type)
        dataset._dataframe_backend = backend
        return dataset
