import asyncio
import json
from pathlib import Path
from typing import Any

import obstore as obs
import polars as pl
from obstore.store import from_url

from tacoreader._schema import PITSchema, extract_schema_from_collection, merge_schemas
from tacoreader.folder.reader import read_all_levels


def read_collection_local(path: str) -> dict[str, Any]:
    """
    Read COLLECTION.json from local FOLDER.

    Args:
        path: Local FOLDER path

    Returns:
        COLLECTION.json as dictionary

    Raises:
        ValueError: If COLLECTION.json not found or invalid
    """
    collection_path = Path(path) / "COLLECTION.json"

    if not collection_path.exists():
        raise ValueError(f"COLLECTION.json not found in {path}")

    with open(collection_path) as f:
        return json.load(f)


def read_collection_remote(path: str) -> dict[str, Any]:
    """
    Read COLLECTION.json from remote FOLDER.

    Args:
        path: Remote FOLDER URL

    Returns:
        COLLECTION.json as dictionary

    Raises:
        ValueError: If COLLECTION.json not found or invalid
    """

    async def _read() -> dict[str, Any]:
        store = from_url(path)
        collection_result = await obs.get_async(store, "COLLECTION.json")
        collection_bytes = await collection_result.bytes_async()
        return json.loads(bytes(collection_bytes))

    return asyncio.run(_read())


def read_collection(path: str) -> dict[str, Any]:
    """
    Read COLLECTION.json from FOLDER (local or remote).

    Args:
        path: FOLDER path

    Returns:
        COLLECTION.json as dictionary
    """
    if Path(path).exists():
        return read_collection_local(path)
    return read_collection_remote(path)


def load_single_folder(path: str) -> tuple[PITSchema, list[pl.DataFrame]]:
    """
    Load single FOLDER dataset.

    Reads COLLECTION.json for schema and all metadata levels.

    Args:
        path: FOLDER path (local or remote)

    Returns:
        Tuple of (PITSchema, list of DataFrames)

    Examples:
        >>> schema, dataframes = load_single_folder("dataset/")
    """
    # Read COLLECTION.json to get schema
    collection = read_collection(path)
    schema = extract_schema_from_collection(collection)

    # Read all metadata levels
    dataframes = read_all_levels(path)

    return schema, dataframes


def load_multiple_folders(
    paths: list[str],
) -> tuple[PITSchema, list[list[pl.DataFrame]]]:
    """
    Load multiple FOLDER datasets and validate compatibility.

    All folders must have identical schemas (structure).

    Args:
        paths: List of FOLDER paths

    Returns:
        Tuple of:
        - Merged PITSchema (with summed 'n' values)
        - List of dataframe lists, one per folder

    Raises:
        ValueError: If schemas are incompatible

    Examples:
        >>> paths = ["dataset1/", "dataset2/"]
        >>> merged_schema, all_dataframes = load_multiple_folders(paths)
    """
    if not paths:
        raise ValueError("At least one path must be provided")

    # Load all folders
    schemas = []
    all_dataframes = []

    for path in paths:
        schema, dataframes = load_single_folder(path)
        schemas.append(schema)
        all_dataframes.append(dataframes)

    # Validate schemas are compatible and merge
    merged_schema = merge_schemas(schemas)

    return merged_schema, all_dataframes


def merge_dataframes_by_level(
    all_dataframes: list[list[pl.DataFrame]],
) -> list[pl.DataFrame]:
    """
    Concatenate DataFrames by level across all folders.

    Args:
        all_dataframes: List of dataframe lists, one per folder

    Returns:
        List of merged DataFrames [level0, level1, ...]

    Examples:
        >>> merged = merge_dataframes_by_level([
        ...     [df0_file1, df1_file1],
        ...     [df0_file2, df1_file2]
        ... ])
        >>> # Returns [concat(df0_file1, df0_file2), concat(df1_file1, df1_file2)]
    """
    if not all_dataframes:
        raise ValueError("No dataframes to merge")

    # Determine maximum number of levels
    max_levels = max(len(dfs) for dfs in all_dataframes)

    merged = []

    for level_idx in range(max_levels):
        # Collect all DataFrames for this level
        level_dfs = [dfs[level_idx] for dfs in all_dataframes if level_idx < len(dfs)]

        if level_dfs:
            # Concatenate DataFrames for this level
            merged_df = pl.concat(level_dfs)
            merged.append(merged_df)

    return merged
