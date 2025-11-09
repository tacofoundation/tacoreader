"""
Load TACO datasets from any format.

Main entry point for loading datasets. Automatically detects format
and dispatches to appropriate backend. Supports loading single files
or lists of files (automatically concatenated with parallel loading).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tacoreader.backends import create_backend, load_dataset
from tacoreader.constants import (
    PARALLEL_DEFAULT_MAX_WORKERS,
    PARALLEL_MIN_WORKERS,
)
from tacoreader.dataset import TacoDataset
from tacoreader.utils.format import detect_format
from tacoreader.utils.vsi import to_vsi_root

# ============================================================================
# PUBLIC API
# ============================================================================


def load(
    path: str | list[str],
    cache_dir: Path | None = None,
    base_path: str | None = None,
    max_workers: int | None = None,
    debug: bool = False,
) -> TacoDataset:
    """
    Load TACO dataset(s) from any format.

    Automatically detects format (ZIP, FOLDER, or TacoCat) and loads
    dataset with lazy SQL interface. Returns TacoDataset with DuckDB
    connection for efficient queries.

    Supports loading multiple files at once by passing a list of paths.
    Files are loaded in parallel using ThreadPoolExecutor and automatically
    concatenated if schemas are compatible.

    Args:
        path: Path to TACO dataset OR list of paths to concatenate
            - Single path: "data.tacozip" or "s3://bucket/data.tacozip"
            - List of paths: ["data1.tacozip", "data2.tacozip", ...]
            Formats: ZIP (.tacozip), FOLDER (directory), TacoCat (__TACOCAT__)
        cache_dir: Cache directory for metadata files (optional)
        base_path: Override base path for TacoCat ZIP files (optional, TacoCat only)
            Used when __TACOCAT__ and .tacozip files are in different locations
        max_workers: Max parallel workers for loading multiple files (optional)
            - None (default): Auto-detect as min(len(paths), 8)
            - 1: Sequential loading
            - N: Use N threads
        debug: Print timing information for debugging (optional, default: False)

    Returns:
        TacoDataset with lazy SQL interface

    Raises:
        ValueError: If format cannot be detected, path invalid, or empty list
        ValueError: If multiple files have incompatible schemas
        IOError: If files cannot be read

    Example:
        >>> # Load single file
        >>> dataset = load("data.tacozip")
        >>> print(dataset.id)
        'sentinel2-l2a'

        >>> # Load from S3
        >>> dataset = load("s3://bucket/data.tacozip")

        >>> # Load multiple files (parallel by default)
        >>> dataset = load([
        ...     "part001.tacozip",
        ...     "part002.tacozip",
        ...     "part003.tacozip"
        ... ])
        >>> print(dataset.pit_schema.root['n'])
        3000  # Combined count

        >>> # Control parallelism
        >>> dataset = load(files, max_workers=16)  # More aggressive
        >>> dataset = load(files, max_workers=1)   # Sequential (debug)

        >>> # Load with glob patterns
        >>> from pathlib import Path
        >>> files = list(Path("data/").glob("*.tacozip"))
        >>> dataset = load(files)

        >>> # Lazy SQL queries
        >>> peru = dataset.sql("SELECT * FROM data WHERE country = 'Peru'")
        >>> low_cloud = peru.sql("SELECT * FROM data WHERE cloud_cover < 10")

        >>> # Materialization
        >>> df = low_cloud.data
        >>> tdf = low_cloud.collect()

        >>> # TacoCat with ZIPs in different location
        >>> cat = load(
        ...     "__TACOCAT__",
        ...     base_path="s3://other-bucket/zips/"
        ... )
    """
    # Handle list of paths
    if isinstance(path, list):
        if len(path) == 0:
            raise ValueError("Cannot load empty list of paths")

        if len(path) == 1:
            # Single file in list - unwrap and load normally
            return load(path[0], cache_dir, base_path)

        # Multiple files - load in parallel and concatenate
        # Auto-detect workers if not specified
        if max_workers is None:
            max_workers = min(len(path), PARALLEL_DEFAULT_MAX_WORKERS)

        # Sequential loading if max_workers=1
        if max_workers == PARALLEL_MIN_WORKERS:
            datasets = []
            for p in path:
                datasets.append(load(p, cache_dir, base_path))
        else:
            # Parallel loading with ThreadPoolExecutor
            def _load_single(p: str) -> TacoDataset:
                return load(p, cache_dir, base_path)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                datasets = list(executor.map(_load_single, path))

        from tacoreader.concat import concat

        return concat(datasets)

    # Handle single path
    format_type = detect_format(path)

    # Special handling for TacoCat with base_path override
    if format_type == "tacocat" and base_path is not None:
        backend = create_backend("tacocat")
        
        # Check if we're in async context
        try:
            asyncio.get_running_loop()
            # Async context - return coroutine
            return _load_tacocat_with_base_path_async(
                backend, path, cache_dir, base_path, debug
            )
        except RuntimeError:
            # Sync context - run internally
            return asyncio.run(
                _load_tacocat_with_base_path_async(
                    backend, path, cache_dir, base_path, debug
                )
            )

    # Standard loading path
    backend = create_backend(format_type)

    # Check if backend supports optimized loading
    if hasattr(backend, "load_async") and format_type == "tacocat":
        try:
            # Check if we're in async context
            asyncio.get_running_loop()
            return backend.load_async(path, cache_dir, debug=debug)
        except RuntimeError:
            # Run internally
            return asyncio.run(backend.load_async(path, cache_dir, debug=debug))

    # Standard loading for ZIP/FOLDER
    return load_dataset(path, format_type, cache_dir)


async def _load_tacocat_with_base_path_async(
    backend, path: str, cache_dir: Path | None, base_path: str, debug: bool = False
) -> TacoDataset:
    """
    Load TacoCat with base_path override.
    
    Used when __TACOCAT__ and .tacozip files are in different locations.
    """
    # Load dataset
    dataset = await backend.load_async(path, cache_dir, debug=debug)

    # Rebuild views with new base_path
    base_vsi = to_vsi_root(base_path)

    # Drop existing views
    for level_key in dataset._consolidated_files.keys():
        dataset._duckdb.execute(f"DROP VIEW IF EXISTS {level_key}")

    # Recreate views with new base_path
    for level_key, file_path in dataset._consolidated_files.items():
        dataset._duckdb.execute(
            f"""
            CREATE VIEW {level_key} AS 
            SELECT *,
              '/vsisubfile/' || "internal:offset" || '_' || 
              "internal:size" || ',{base_vsi}' || "internal:source_file"
              as "internal:gdal_vsi"
            FROM read_parquet('{file_path}')
        """
        )

    # Recreate 'data' view
    dataset._duckdb.execute("DROP VIEW IF EXISTS data")
    dataset._duckdb.execute("CREATE VIEW data AS SELECT * FROM level0")
    dataset._root_path = base_vsi

    return dataset