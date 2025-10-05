import polars as pl


def build_vsi_paths(
    df: pl.DataFrame,
    root_path: str,
    level: int,
) -> pl.DataFrame:
    """
    Build internal:gdal_vsi column for FOLDER format.

    TORTILLA nodes point to their level's metadata Avro file.
    SAMPLE nodes point to their file using internal:relative_path.

    Args:
        df: DataFrame with type and internal:relative_path columns
        root_path: VSI root path (e.g., "/vsis3/bucket/dataset/")
        level: Current level number (0, 1, 2, ...)

    Returns:
        DataFrame with internal:gdal_vsi column added

    Raises:
        ValueError: If required columns are missing

    Examples:
        >>> df = build_vsi_paths(df, "/vsis3/bucket/data/", 1)
        >>> df["internal:gdal_vsi"][0]
        '/vsis3/bucket/data/METADATA/level1.avro'
    """
    if "type" not in df.columns:
        raise ValueError("Missing required 'type' column")

    # Check if SAMPLE nodes have relative_path
    has_samples = (df["type"] != "TORTILLA").any()
    if has_samples and "internal:relative_path" not in df.columns:
        raise ValueError("FOLDER format SAMPLEs require internal:relative_path column")

    # Ensure root ends with /
    root = root_path if root_path.endswith("/") else root_path + "/"

    # Build VSI paths:
    # TORTILLA: root + "METADATA/levelN.avro"
    # SAMPLE: root + relative_path
    vsi_paths = (
        pl.when(pl.col("type") == "TORTILLA")
        .then(
            pl.lit(root)
            + pl.lit("METADATA/level")
            + pl.lit(str(level))
            + pl.lit(".avro")
        )
        .otherwise(pl.lit(root) + pl.col("internal:relative_path"))
    )

    # Add gdal_vsi column
    df = df.with_columns(vsi_paths.alias("internal:gdal_vsi"))

    # Drop relative_path if it exists (no longer needed)
    if "internal:relative_path" in df.columns:
        df = df.drop("internal:relative_path")

    return df


def enrich_all_levels(
    dataframes: list[pl.DataFrame],
    root_path: str,
) -> list[pl.DataFrame]:
    """
    Enrich all levels with internal:gdal_vsi paths.

    Applies build_vsi_paths to each level's DataFrame.

    Args:
        dataframes: List of DataFrames [level0, level1, ...]
        root_path: VSI root path

    Returns:
        List of enriched DataFrames with internal:gdal_vsi column

    Examples:
        >>> enriched = enrich_all_levels(dataframes, "/vsis3/bucket/data/")
        >>> enriched[0]["internal:gdal_vsi"][0]
        '/vsis3/bucket/data/METADATA/level0.avro'
    """
    enriched = []

    for level, df in enumerate(dataframes):
        enriched_df = build_vsi_paths(df, root_path, level)
        enriched.append(enriched_df)

    return enriched
