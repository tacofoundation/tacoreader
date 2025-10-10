import asyncio

import polars as pl
from shapely import wkt
from shapely.wkb import loads as wkb_loads

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
    # Normalize input to list format
    path_list: list[str] = [paths] if isinstance(paths, str) else paths

    # Validate at least one path provided
    if not path_list:
        raise ValueError("At least one path must be provided")

    # Check for legacy v1 format and raise helpful migration error
    for path in path_list:
        if is_legacy_format(path):
            raise_legacy_error(path)

    # Detect format from first path (ZIP or FOLDER)
    format_type = detect_format(path_list[0])

    # Validate all paths have same format
    for path in path_list[1:]:
        if detect_format(path) != format_type:
            raise ValueError(
                f"All paths must be same format. "
                f"First path is {format_type}, but found different format in: {path}"
            )

    # Route to appropriate loader
    if format_type == "zip":
        return _load_zip(path_list)
    return _load_folder(path_list)


def _convert_stored_types(dataframes: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """
    Convert stored binary/epoch formats back to user-friendly types.

    Performs rollback conversions from storage-optimized formats to
    human-readable formats for better user experience:

    - WKB geometries → WKT strings (shapely/geopandas compatible)
    - Unix timestamps → datetime objects (pandas datetime64)

    Args:
        dataframes: List of Polars DataFrames (one per level)

    Returns:
        List of DataFrames with converted types
    """
    converted = []

    for df in dataframes:
        # Start with original DataFrame
        result = df

        # Iterate through all columns looking for known stored types
        for col_name in df.columns:
            # Convert WKB binary geometries to WKT strings
            if col_name in ["istac:geometry", "istac:centroid", "stac:centroid"]:
                result = result.with_columns(
                    pl.col(col_name).map_elements(
                        lambda wkb: wkt.dumps(wkb_loads(wkb)) if wkb else None,
                        return_dtype=pl.Utf8,  # Result is string (WKT format)
                    )
                )

            # Convert Unix epoch timestamps to datetime objects
            elif col_name in [
                "stac:time_start",
                "stac:time_end",
                "istac:time_start",
                "istac:time_end",
            ]:
                result = result.with_columns(
                    pl.from_epoch(col_name, time_unit="s")  # Seconds since 1970-01-01
                )

        converted.append(result)

    return converted


def _load_zip(paths: list[str]) -> TacoDataFrame:
    """
    Load ZIP format dataset(s).

    Process:
    1. Read COLLECTION.json for schema + read all levelN.parquet metadata
    2. If multiple files: validate schemas identical, merge them (sum 'n' values)
    3. Concatenate DataFrames by level (level0 + level0, level1 + level1, etc)
    4. Get metadata offsets from ZIP header for VSI path construction
    5. Enrich DataFrames with internal:gdal_vsi using /vsisubfile/offset_size,root
    6. Convert stored types (WKB→WKT, epoch→datetime)
    7. Convert from Polars to Pandas
    8. Build TacoDataFrame with all levels

    Args:
        paths: List of ZIP paths

    Returns:
        TacoDataFrame instance (level 0)
    """
    if len(paths) == 1:
        # Single file loading path
        schema, dataframes = load_single_zip(paths[0])

        # Convert to GDAL VSI path (handles S3, GCS, HTTP, etc.)
        root_path = to_vsi_root(paths[0])

        # Get file offsets for /vsisubfile/ construction
        metadata_offsets = asyncio.run(get_metadata_offsets(paths[0]))

        # Add internal:gdal_vsi column with /vsisubfile paths
        enriched = enrich_zip_levels(dataframes, root_path, metadata_offsets)

        # Convert stored types to user-friendly formats
        converted = _convert_stored_types(enriched)

        # Convert from Polars to Pandas for TacoDataFrame compatibility
        pandas_levels = [df.to_pandas() for df in converted]

        # Build navigable tree structure
        return TacoDataFrame(
            pandas_levels[0],  # Root level (level 0)
            all_levels=pandas_levels,  # All levels for navigation
            schema=schema,  # PIT schema for arithmetic navigation
            current_depth=0,  # Starting at root
            root_path=root_path,  # VSI root for path construction
        )

    # Multiple file loading path
    merged_schema, all_dataframes = load_multiple_zips(paths)

    # Concatenate DataFrames by level (level0s together, level1s together, etc.)
    merged_dataframes = merge_zip_dataframes(all_dataframes)

    # Use first path as representative root (all should have same storage)
    root_path = to_vsi_root(paths[0])

    # Get file offsets from first file (structure should be identical)
    metadata_offsets = asyncio.run(get_metadata_offsets(paths[0]))

    # Add internal:gdal_vsi column with /vsisubfile paths
    enriched = enrich_zip_levels(merged_dataframes, root_path, metadata_offsets)

    # Convert stored types to user-friendly formats
    converted = _convert_stored_types(enriched)

    # Convert from Polars to Pandas for TacoDataFrame compatibility
    pandas_levels = [df.to_pandas() for df in converted]

    # Build navigable tree structure
    return TacoDataFrame(
        pandas_levels[0],  # Root level (level 0)
        all_levels=pandas_levels,  # All levels for navigation
        schema=merged_schema,  # Merged PIT schema
        current_depth=0,  # Starting at root
        root_path=root_path,  # VSI root for path construction
    )


def _load_folder(paths: list[str]) -> TacoDataFrame:
    """
    Load FOLDER format dataset(s).

    Process:
    1. Read COLLECTION.json for schema + read all METADATA/levelN.avro files
    2. If multiple files: validate schemas identical, merge them (sum 'n' values)
    3. Concatenate DataFrames by level (level0 + level0, level1 + level1, etc)
    4. Enrich DataFrames with internal:gdal_vsi paths
       - FOLDER nodes: root/METADATA/levelN.avro
       - FILE nodes: root/DATA/path/to/file.tif (from internal:relative_path)
    5. Convert stored types (WKB→WKT, epoch→datetime)
    6. Convert from Polars to Pandas
    7. Build TacoDataFrame with all levels

    Args:
        paths: List of FOLDER paths

    Returns:
        TacoDataFrame instance (level 0)
    """
    if len(paths) == 1:
        # Single folder loading path
        schema, dataframes = load_single_folder(paths[0])

        # Convert to GDAL VSI path (handles S3, GCS, HTTP, etc.)
        root_path = to_vsi_root(paths[0])

        # Add internal:gdal_vsi column pointing to DATA/ or METADATA/
        enriched = enrich_folder_levels(dataframes, root_path)

        # Convert stored types to user-friendly formats
        converted = _convert_stored_types(enriched)

        # Convert from Polars to Pandas for TacoDataFrame compatibility
        pandas_levels = [df.to_pandas() for df in converted]

        # Build navigable tree structure
        return TacoDataFrame(
            pandas_levels[0],  # Root level (level 0)
            all_levels=pandas_levels,  # All levels for navigation
            schema=schema,  # PIT schema for arithmetic navigation
            current_depth=0,  # Starting at root
            root_path=root_path,  # VSI root for path construction
        )

    # Multiple folder loading path
    merged_schema, all_dataframes = load_multiple_folders(paths)

    # Concatenate DataFrames by level (level0s together, level1s together, etc.)
    merged_dataframes = merge_folder_dataframes(all_dataframes)

    # Use first path as representative root (all should have same storage)
    root_path = to_vsi_root(paths[0])

    # Add internal:gdal_vsi column pointing to DATA/ or METADATA/
    enriched = enrich_folder_levels(merged_dataframes, root_path)

    # Convert stored types to user-friendly formats
    converted = _convert_stored_types(enriched)

    # Convert from Polars to Pandas for TacoDataFrame compatibility
    pandas_levels = [df.to_pandas() for df in converted]

    # Build navigable tree structure
    return TacoDataFrame(
        pandas_levels[0],  # Root level (level 0)
        all_levels=pandas_levels,  # All levels for navigation
        schema=merged_schema,  # Merged PIT schema
        current_depth=0,  # Starting at root
        root_path=root_path,  # VSI root for path construction
    )
