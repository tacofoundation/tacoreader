"""
Load TACO datasets from any format.

Main entry point. Auto-detects format and dispatches to backend.
Supports single files or lists (automatically concatenated).
"""

from tacoreader.backends import create_backend, load_dataset
from tacoreader.dataset import TacoDataset
from tacoreader.utils.format import detect_format
from tacoreader.utils.vsi import to_vsi_root
from tacoreader._legacy import is_legacy_format, raise_legacy_error
from tacoreader._constants import PADDING_PREFIX


def load(
    path: str | list[str],
    base_path: str | None = None,
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

    Returns:
        TacoDataset with lazy SQL interface

    Examples:
        # Single file
        ds = load("data.tacozip")

        # Multiple files (auto-concatenated)
        ds = load(["part1.tacozip", "part2.tacozip"])

        # S3
        ds = load("s3://bucket/data.tacozip")

        # TacoCat with ZIPs elsewhere
        ds = load("__TACOCAT__", base_path="s3://other-bucket/zips/")

        # Lazy queries
        peru = ds.sql("SELECT * FROM data WHERE country = 'Peru'")
        df = peru.data  # Materialization
    """
    # Handle list of paths
    if isinstance(path, list):
        if len(path) == 0:
            raise ValueError("Cannot load empty list of paths")

        if len(path) == 1:
            return load(path[0], base_path)

        # Multiple files - load and concatenate
        datasets = [load(p, base_path) for p in path]

        from tacoreader.concat import concat

        return concat(datasets)

    # Check for legacy format (fail fast with helpful message)
    if is_legacy_format(path):
        raise_legacy_error(path)

    # Single path
    format_type = detect_format(path)

    # TacoCat with base_path override: rebuild views with new base
    if format_type == "tacocat" and base_path is not None:
        backend = create_backend("tacocat")
        dataset = backend.load(path)

        base_vsi = to_vsi_root(base_path)
        max_depth = dataset.pit_schema.max_depth()

        # Drop 'data' view first (depends on level0)
        dataset._duckdb.execute("DROP VIEW IF EXISTS data")

        # Recreate level views with new base_path
        for i in range(max_depth + 1):
            level_key = f"level{i}"

            dataset._duckdb.execute(f"DROP VIEW IF EXISTS {level_key}")

            dataset._duckdb.execute(
                f"""
                CREATE VIEW {level_key} AS 
                SELECT *,
                  '/vsisubfile/' || "internal:offset" || '_' || 
                  "internal:size" || ',{base_vsi}' || "internal:source_file"
                  as "internal:gdal_vsi"
                FROM {level_key}_table
                WHERE id NOT LIKE '{PADDING_PREFIX}%'
            """
            )

        # Recreate 'data' view
        dataset._duckdb.execute("CREATE VIEW data AS SELECT * FROM level0")
        dataset._root_path = base_vsi

        return dataset

    # Standard loading
    backend = create_backend(format_type)
    return (
        backend.load(path)
        if format_type == "tacocat"
        else load_dataset(path, format_type)
    )
