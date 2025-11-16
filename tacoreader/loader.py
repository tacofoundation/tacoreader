"""
Load TACO datasets from any format.

Main entry point for loading datasets. Automatically detects format
and dispatches to appropriate backend. Supports loading single files
or lists of files (automatically concatenated).
"""

import re
from pathlib import Path

from tacoreader.backends import create_backend, load_dataset
from tacoreader.dataset import TacoDataset
from tacoreader.utils.format import detect_format
from tacoreader.utils.vsi import to_vsi_root

# ============================================================================
# PUBLIC API
# ============================================================================


def load(
    path: str | list[str],
    base_path: str | None = None,
) -> TacoDataset:
    """
    Load TACO dataset(s) from any format.

    Automatically detects format (ZIP, FOLDER, or TacoCat) and loads
    dataset with lazy SQL interface. Returns TacoDataset with DuckDB
    connection for efficient queries.

    Supports loading multiple files at once by passing a list of paths.
    Files are loaded sequentially and automatically concatenated if 
    schemas are compatible.

    Args:
        path: Path to TACO dataset OR list of paths to concatenate
            - Single path: "data.tacozip" or "s3://bucket/data.tacozip"
            - List of paths: ["data1.tacozip", "data2.tacozip", ...]
            Formats: ZIP (.tacozip), FOLDER (directory), TacoCat (__TACOCAT__)
        base_path: Override base path for TacoCat ZIP files (optional, TacoCat only)
            Used when __TACOCAT__ and .tacozip files are in different locations

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

        >>> # Load multiple files
        >>> dataset = load([
        ...     "part001.tacozip",
        ...     "part002.tacozip",
        ...     "part003.tacozip"
        ... ])
        >>> print(dataset.pit_schema.root['n'])
        3000  # Combined count

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
        
        >>> # Enable logging to see load progress
        >>> import tacoreader
        >>> tacoreader.verbose()
        >>> dataset = load("data.tacozip")
        INFO [tacoreader.backends.tacocat] Loaded TacoCat in 1.23s
    """
    # Handle list of paths
    if isinstance(path, list):
        if len(path) == 0:
            raise ValueError("Cannot load empty list of paths")

        if len(path) == 1:
            # Single file in list - unwrap and load normally
            return load(path[0], base_path)

        # Multiple files - load sequentially and concatenate
        datasets = [load(p, base_path) for p in path]

        from tacoreader.concat import concat

        return concat(datasets)

    # Handle single path
    format_type = detect_format(path)

    # Special handling for TacoCat with base_path override
    if format_type == "tacocat" and base_path is not None:
        backend = create_backend("tacocat")
        dataset = backend.load(path)
        
        # Rebuild views with new base_path
        base_vsi = to_vsi_root(base_path)

        # Get available levels from pit_schema
        max_depth = dataset.pit_schema.max_depth()

        # Drop 'data' view FIRST (depends on level0)
        dataset._duckdb.execute("DROP VIEW IF EXISTS data")

        # Drop and recreate level views with new base_path
        for i in range(max_depth + 1):
            level_key = f"level{i}"
            
            # Drop old view
            dataset._duckdb.execute(f"DROP VIEW IF EXISTS {level_key}")
            
            # Recreate view using existing TABLE with new base_path
            dataset._duckdb.execute(
                f"""
                CREATE VIEW {level_key} AS 
                SELECT *,
                  '/vsisubfile/' || "internal:offset" || '_' || 
                  "internal:size" || ',{base_vsi}' || "internal:source_file"
                  as "internal:gdal_vsi"
                FROM {level_key}_table
                WHERE id NOT LIKE '__TACOPAD__%'
            """
            )

        # Finally recreate 'data' view
        dataset._duckdb.execute("CREATE VIEW data AS SELECT * FROM level0")
        dataset._root_path = base_vsi

        return dataset

    # Standard loading path
    backend = create_backend(format_type)
    return backend.load(path) if format_type == "tacocat" else load_dataset(path, format_type)