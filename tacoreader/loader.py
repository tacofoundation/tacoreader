import asyncio

from tacoreader._format import detect_format
from tacoreader._legacy import is_legacy_format, raise_legacy_error
from tacoreader._tree import TacoDataFrame
from tacoreader._vsi import to_vsi_root
from tacoreader.folder.metadata import enrich_all_levels as enrich_folder_levels
from tacoreader.folder.multi import load_multiple_folders, load_single_folder
from tacoreader.folder.multi import merge_dataframes_by_level as merge_folder_dataframes
from tacoreader.zip.metadata import enrich_all_levels as enrich_zip_levels
from tacoreader.zip.multi import load_multiple_zips, load_single_zip
from tacoreader.zip.multi import merge_dataframes_by_level as merge_zip_dataframes
from tacoreader.zip.reader import get_metadata_offsets


def load(paths: str | list[str]) -> TacoDataFrame:
    """
    Load TACO dataset(s) into TacoDataFrame.

    Automatically detects format (ZIP vs FOLDER) and handles both local
    and remote storage. Supports multiple files with schema validation.

    Args:
        paths: Single path or list of paths to TACO datasets

    Returns:
        TacoDataFrame (level 0 with navigation capabilities)

    Raises:
        ValueError: If formats differ or schemas incompatible

    Examples:
        >>> df = load("dataset.tacozip")
        >>> df = load("s3://bucket/dataset/")
        >>> df = load(["part1.tacozip", "part2.tacozip"])
    """
    path_list: list[str] = [paths] if isinstance(paths, str) else paths

    if not path_list:
        raise ValueError("At least one path must be provided")

    # Check for legacy v1 format
    for path in path_list:
        if is_legacy_format(path):
            raise_legacy_error(path)

    format_type = detect_format(path_list[0])

    for path in path_list[1:]:
        if detect_format(path) != format_type:
            raise ValueError(
                f"All paths must be same format. "
                f"First path is {format_type}, but found different format in: {path}"
            )

    if format_type == "zip":
        return _load_zip(path_list)
    return _load_folder(path_list)


def _load_zip(paths: list[str]) -> TacoDataFrame:
    """
    Load ZIP format dataset(s).

    Process:
    1. Read COLLECTION.json for schema + read all levelN.parquet metadata
    2. If multiple files: validate schemas identical, merge them (sum 'n' values)
    3. Concatenate DataFrames by level (level0 + level0, level1 + level1, etc)
    4. Get metadata offsets from ZIP header for VSI path construction
    5. Enrich DataFrames with internal:gdal_vsi using /vsisubfile/offset_size,root
    6. Convert from Polars to Pandas
    7. Build TacoDataFrame with all levels

    Args:
        paths: List of ZIP paths

    Returns:
        TacoDataFrame instance (level 0)
    """
    if len(paths) == 1:
        schema, dataframes = load_single_zip(paths[0])
        root_path = to_vsi_root(paths[0])

        metadata_offsets = asyncio.run(get_metadata_offsets(paths[0]))

        enriched = enrich_zip_levels(dataframes, root_path, metadata_offsets)

        # Convert Polars to Pandas
        pandas_levels = [df.to_pandas() for df in enriched]

        return TacoDataFrame(
            pandas_levels[0],
            all_levels=pandas_levels,
            schema=schema,
            current_depth=0,
            root_path=root_path,
        )

    merged_schema, all_dataframes = load_multiple_zips(paths)

    merged_dataframes = merge_zip_dataframes(all_dataframes)

    root_path = to_vsi_root(paths[0])

    metadata_offsets = asyncio.run(get_metadata_offsets(paths[0]))

    enriched = enrich_zip_levels(merged_dataframes, root_path, metadata_offsets)

    # Convert Polars to Pandas
    pandas_levels = [df.to_pandas() for df in enriched]

    return TacoDataFrame(
        pandas_levels[0],
        all_levels=pandas_levels,
        schema=merged_schema,
        current_depth=0,
        root_path=root_path,
    )


def _load_folder(paths: list[str]) -> TacoDataFrame:
    """
    Load FOLDER format dataset(s).

    Process:
    1. Read COLLECTION.json for schema + read all METADATA/levelN.avro files
    2. If multiple files: validate schemas identical, merge them (sum 'n' values)
    3. Concatenate DataFrames by level (level0 + level0, level1 + level1, etc)
    4. Enrich DataFrames with internal:gdal_vsi paths
       - TORTILLA nodes: root/METADATA/levelN.avro
       - SAMPLE nodes: root/DATA/path/to/file.tif (from internal:relative_path)
    5. Convert from Polars to Pandas
    6. Build TacoDataFrame with all levels

    Args:
        paths: List of FOLDER paths

    Returns:
        TacoDataFrame instance (level 0)
    """
    if len(paths) == 1:
        schema, dataframes = load_single_folder(paths[0])
        root_path = to_vsi_root(paths[0])

        enriched = enrich_folder_levels(dataframes, root_path)

        # Convert Polars to Pandas
        pandas_levels = [df.to_pandas() for df in enriched]

        return TacoDataFrame(
            pandas_levels[0],
            all_levels=pandas_levels,
            schema=schema,
            current_depth=0,
            root_path=root_path,
        )

    merged_schema, all_dataframes = load_multiple_folders(paths)

    merged_dataframes = merge_folder_dataframes(all_dataframes)

    root_path = to_vsi_root(paths[0])

    enriched = enrich_folder_levels(merged_dataframes, root_path)

    # Convert Polars to Pandas
    pandas_levels = [df.to_pandas() for df in enriched]

    return TacoDataFrame(
        pandas_levels[0],
        all_levels=pandas_levels,
        schema=merged_schema,
        current_depth=0,
        root_path=root_path,
    )
