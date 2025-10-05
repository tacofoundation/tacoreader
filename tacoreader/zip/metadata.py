import polars as pl


def build_vsi_paths(
    df: pl.DataFrame,
    root_path: str,
    metadata_offset: int,
    metadata_size: int,
) -> pl.DataFrame:
    """
    Build internal:gdal_vsi column for ZIP format.

    TORTILLA nodes point to their level's metadata parquet.
    SAMPLE nodes point to their file using internal:offset and internal:size.

    Args:
        df: DataFrame with type, internal:offset, internal:size columns
        root_path: VSI root (e.g., "/vsis3/bucket/data.tacozip")
        metadata_offset: Offset of this level's parquet in ZIP
        metadata_size: Size of this level's parquet in ZIP

    Returns:
        DataFrame with internal:gdal_vsi column added

    Raises:
        ValueError: If required columns are missing

    Examples:
        >>> df = build_vsi_paths(df, "/vsis3/bucket/data.tacozip", 1024, 5000)
        >>> df["internal:gdal_vsi"][0]
        '/vsisubfile/1024_5000,/vsis3/bucket/data.tacozip'
    """
    if "type" not in df.columns:
        raise ValueError("Missing required 'type' column")

    if "internal:offset" not in df.columns or "internal:size" not in df.columns:
        raise ValueError(
            "ZIP format requires internal:offset and internal:size columns"
        )

    vsi_paths = (
        pl.when(pl.col("type") == "TORTILLA")
        .then(
            pl.lit("/vsisubfile/")
            + pl.lit(str(metadata_offset))
            + pl.lit("_")
            + pl.lit(str(metadata_size))
            + pl.lit(",")
            + pl.lit(root_path)
        )
        .otherwise(
            pl.lit("/vsisubfile/")
            + pl.col("internal:offset").cast(pl.Utf8)
            + pl.lit("_")
            + pl.col("internal:size").cast(pl.Utf8)
            + pl.lit(",")
            + pl.lit(root_path)
        )
    )

    df = df.with_columns(vsi_paths.alias("internal:gdal_vsi"))
    df = df.drop(["internal:offset", "internal:size"])

    return df


def enrich_all_levels(
    dataframes: list[pl.DataFrame],
    root_path: str,
    metadata_offsets: list[tuple[int, int]],
) -> list[pl.DataFrame]:
    """
    Enrich all levels with internal:gdal_vsi paths.

    Applies build_vsi_paths to each level's DataFrame.

    Args:
        dataframes: List of DataFrames [level0, level1, ...]
        root_path: VSI root path
        metadata_offsets: List of (offset, size) for each level's metadata

    Returns:
        List of enriched DataFrames with internal:gdal_vsi column

    Raises:
        ValueError: If dataframes and offsets length mismatch

    Examples:
        >>> offsets = [(1024, 5000), (6024, 8000)]
        >>> enriched = enrich_all_levels(dataframes, "/vsis3/bucket/data.tacozip", offsets)
    """
    if len(dataframes) != len(metadata_offsets):
        raise ValueError(
            f"Dataframes count ({len(dataframes)}) != "
            f"metadata offsets count ({len(metadata_offsets)})"
        )

    enriched = []

    for df, (offset, size) in zip(dataframes, metadata_offsets, strict=False):
        enriched_df = build_vsi_paths(df, root_path, offset, size)
        enriched.append(enriched_df)

    return enriched
