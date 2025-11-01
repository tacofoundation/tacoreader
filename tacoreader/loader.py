"""
Load TACO datasets from any format.

Main entry point for loading datasets. Automatically detects format
and dispatches to appropriate backend.
"""

from pathlib import Path

from tacoreader.backends import load_dataset, create_backend
from tacoreader.dataset import TacoDataset
from tacoreader.utils.format import detect_format
from tacoreader.utils.vsi import to_vsi_root


# ============================================================================
# PUBLIC API
# ============================================================================


def load(
    path: str,
    cache_dir: Path | None = None,
    base_path: str | None = None,
) -> TacoDataset:
    """
    Load TACO dataset from any format.

    Automatically detects format (ZIP, FOLDER, or TacoCat) and loads
    dataset with lazy SQL interface. Returns TacoDataset with DuckDB
    connection for efficient queries.

    Args:
        path: Path to TACO dataset
            - ZIP: "data.tacozip" or "s3://bucket/data.tacozip"
            - FOLDER: "data/" or "s3://bucket/data/"
            - TacoCat: "__TACOCAT__" or "s3://bucket/__TACOCAT__"
        cache_dir: Cache directory for metadata files (optional)
        base_path: Override base path for TacoCat ZIP files (optional, TacoCat only)
            Used when __TACOCAT__ and .tacozip files are in different locations

    Returns:
        TacoDataset with lazy SQL interface

    Raises:
        ValueError: If format cannot be detected or path invalid
        IOError: If files cannot be read

    Example:
        >>> # Load local ZIP
        >>> dataset = load("data.tacozip")
        >>> print(dataset.id)
        'sentinel2-l2a'

        >>> # Load from S3
        >>> dataset = load("s3://bucket/data.tacozip")

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
    format_type = detect_format(path)

    # Special handling for TacoCat with base_path override
    if format_type == "tacocat" and base_path is not None:
        backend = create_backend("tacocat")
        dataset = backend.load(path, cache_dir)

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

    # Standard loading path
    return load_dataset(path, format_type, cache_dir)
