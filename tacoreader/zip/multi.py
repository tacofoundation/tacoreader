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


def load_multiple_zips(paths: list[str]) -> tuple[PITSchema, list[list[pl.DataFrame]]]:
    """
    Load multiple ZIP datasets and validate compatibility.

    All ZIPs must have identical schemas (structure).

    Args:
        paths: List of ZIP paths

    Returns:
        Tuple of:
        - Merged PITSchema (with summed 'n' values)
        - List of dataframe lists, one per ZIP

    Raises:
        ValueError: If schemas are incompatible

    Examples:
        >>> paths = ["dataset1.tacozip", "dataset2.tacozip"]
        >>> merged_schema, all_dataframes = load_multiple_zips(paths)
    """
    if not paths:
        raise ValueError("At least one path must be provided")

    schemas = []
    all_dataframes = []

    for path in paths:
        schema, dataframes = load_single_zip(path)
        schemas.append(schema)
        all_dataframes.append(dataframes)

    merged_schema = merge_schemas(schemas)

    return merged_schema, all_dataframes


def merge_dataframes_by_level(
    all_dataframes: list[list[pl.DataFrame]],
) -> list[pl.DataFrame]:
    """
    Concatenate DataFrames by level across all ZIPs.

    Args:
        all_dataframes: List of dataframe lists, one per ZIP

    Returns:
        List of merged DataFrames [level0, level1, ...]

    Examples:
        >>> merged = merge_dataframes_by_level([
        ...     [df0_file1, df1_file1],
        ...     [df0_file2, df1_file2]
        ... ])
    """
    if not all_dataframes:
        raise ValueError("No dataframes to merge")

    max_levels = max(len(dfs) for dfs in all_dataframes)

    merged = []

    for level_idx in range(max_levels):
        level_dfs = [dfs[level_idx] for dfs in all_dataframes if level_idx < len(dfs)]

        if level_dfs:
            merged_df = pl.concat(level_dfs)
            merged.append(merged_df)

    return merged
