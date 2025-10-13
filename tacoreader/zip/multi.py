import asyncio
import json
from pathlib import Path
from typing import Any

import obstore as obs
import polars as pl
from obstore.store import from_url

from tacoreader._schema import PITSchema, extract_schema_from_collection, merge_schemas
from tacoreader.zip.reader import read_all_levels, read_taco_header


async def read_collection_async(path: str) -> dict[str, Any]:
    """
    Read COLLECTION.json from ZIP (async).

    Args:
        path: ZIP path (local or remote)

    Returns:
        COLLECTION.json as dictionary

    Raises:
        ValueError: If COLLECTION.json not found
    """
    header = await read_taco_header(path)
    collection_offset, collection_size = header[-1]

    is_local = Path(path).exists()

    if is_local:
        with open(path, "rb") as f:
            f.seek(collection_offset)
            collection_bytes = f.read(collection_size)
    else:
        store = from_url(path)
        result = await obs.get_range_async(
            store, "", start=collection_offset, length=collection_size
        )
        collection_bytes = bytes(result)

    return json.loads(collection_bytes)


def read_collection(path: str) -> dict[str, Any]:
    """
    Read COLLECTION.json from ZIP (local or remote).

    Args:
        path: ZIP path

    Returns:
        COLLECTION.json as dictionary
    """
    return asyncio.run(read_collection_async(path))


def load_single_zip(path: str) -> tuple[PITSchema, list[pl.DataFrame]]:
    """
    Load single ZIP dataset.

    Reads COLLECTION.json for schema and all metadata levels.

    Args:
        path: ZIP path (local or remote)

    Returns:
        Tuple of (PITSchema, list of DataFrames)

    Examples:
        >>> schema, dataframes = load_single_zip("dataset.tacozip")
    """
    collection = read_collection(path)
    schema = extract_schema_from_collection(collection)

    dataframes = read_all_levels(path)

    return schema, dataframes


def _validate_columns_match(all_dataframes: list[list[pl.DataFrame]]) -> None:
    """
    Validate that all DataFrames at the same level have identical columns.

    Raises:
        ValueError: If columns don't match, with detailed message suggesting safe_mode

    Examples:
        >>> _validate_columns_match(all_dataframes)  # Raises if mismatch
    """
    for level_idx in range(len(all_dataframes[0])):
        reference_cols = set(all_dataframes[0][level_idx].columns)

        for file_idx, dfs in enumerate(all_dataframes[1:], 1):
            current_cols = set(dfs[level_idx].columns)

            if current_cols != reference_cols:
                missing = reference_cols - current_cols
                extra = current_cols - reference_cols

                raise ValueError(
                    f"Column mismatch at level {level_idx}:\n"
                    f"  File 0 has: {sorted(reference_cols)}\n"
                    f"  File {file_idx} has: {sorted(current_cols)}\n"
                    f"  Missing in file {file_idx}: {sorted(missing) if missing else 'none'}\n"
                    f"  Extra in file {file_idx}: {sorted(extra) if extra else 'none'}\n\n"
                    f"Solution: Use safe_mode to load only common columns:\n"
                    f"  load(paths, safe_mode=True)"
                )


def _get_common_columns(dfs: list[pl.DataFrame]) -> list[str]:
    """
    Get intersection of columns across DataFrames.

    Always includes obligatory columns: id, type, internal:offset, internal:size

    Args:
        dfs: List of DataFrames to compare

    Returns:
        Sorted list of common column names (+ obligatory columns)

    Examples:
        >>> common = _get_common_columns([df1, df2, df3])
        >>> # Returns intersection + ['id', 'type', 'internal:offset', 'internal:size']
    """
    if not dfs:
        return []

    # Intersection of columns
    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)

    # Ensure obligatory ZIP columns are included
    required = {"id", "type", "internal:offset", "internal:size"}

    return sorted(common | required)


def load_multiple_zips(
    paths: list[str],
    safe_mode: bool = False,
) -> tuple[PITSchema, list[list[pl.DataFrame]]]:
    """
    Load multiple ZIP datasets and validate compatibility.

    All ZIPs must have identical schemas (structure).

    Args:
        paths: List of ZIP paths
        safe_mode: If True, skip column validation (will filter to common columns later)

    Returns:
        Tuple of:
        - Merged PITSchema (with summed 'n' values)
        - List of dataframe lists, one per ZIP

    Raises:
        ValueError: If schemas are incompatible (structure)
        ValueError: If columns don't match at any level (unless safe_mode=True)

    Examples:
        >>> paths = ["dataset1.tacozip", "dataset2.tacozip"]
        >>> merged_schema, all_dataframes = load_multiple_zips(paths)
        >>> # With different columns
        >>> merged_schema, all_dataframes = load_multiple_zips(paths, safe_mode=True)
    """
    if not paths:
        raise ValueError("At least one path must be provided")

    schemas = []
    all_dataframes = []

    for path in paths:
        schema, dataframes = load_single_zip(path)
        schemas.append(schema)
        all_dataframes.append(dataframes)

    # 1. Validate PIT schemas are compatible (structure)
    merged_schema = merge_schemas(schemas)

    # 2. Validate columns match at each level (only if NOT safe_mode)
    if not safe_mode:
        _validate_columns_match(all_dataframes)

    return merged_schema, all_dataframes


def merge_dataframes_by_level(
    all_dataframes: list[list[pl.DataFrame]],
    safe_mode: bool = False,
) -> list[pl.DataFrame]:
    """
    Concatenate DataFrames by level across all ZIPs.

    If safe_mode=True, filters to common columns at each level.

    Args:
        all_dataframes: List of dataframe lists, one per ZIP
        safe_mode: If True, only use common columns (+ obligatory columns)

    Returns:
        List of merged DataFrames [level0, level1, ...]

    Examples:
        >>> merged = merge_dataframes_by_level([
        ...     [df0_file1, df1_file1],
        ...     [df0_file2, df1_file2]
        ... ])
        >>> # With safe_mode
        >>> merged = merge_dataframes_by_level(all_dataframes, safe_mode=True)
    """
    if not all_dataframes:
        raise ValueError("No dataframes to merge")

    max_levels = max(len(dfs) for dfs in all_dataframes)

    merged = []

    for level_idx in range(max_levels):
        level_dfs = [dfs[level_idx] for dfs in all_dataframes if level_idx < len(dfs)]

        if level_dfs:
            # Safe mode: filter to common columns
            if safe_mode:
                common_cols = _get_common_columns(level_dfs)
                level_dfs = [df.select(common_cols) for df in level_dfs]

            merged_df = pl.concat(level_dfs)
            merged.append(merged_df)

    return merged
