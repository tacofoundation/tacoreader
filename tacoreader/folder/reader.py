import asyncio
import io
from pathlib import Path

import obstore as obs
import polars as pl
from obstore.store import ObjectStore, from_url


def detect_max_level_local(path: str) -> int:
    """
    Detect maximum level in local FOLDER by counting files.

    Args:
        path: Local FOLDER path

    Returns:
        Maximum level index (e.g., 2 if level0, level1, level2 exist)

    Raises:
        ValueError: If METADATA directory not found or no level files exist

    Examples:
        >>> detect_max_level_local("dataset/")
        2  # Has level0.avro, level1.avro, level2.avro
    """
    metadata_dir = Path(path) / "METADATA"

    if not metadata_dir.exists():
        raise ValueError(f"METADATA directory not found in {path}")

    level_files = sorted(metadata_dir.glob("level*.avro"))

    if not level_files:
        raise ValueError(f"No level*.avro files found in {metadata_dir}")

    # Extract level numbers and find max
    level_numbers = []
    for file_path in level_files:
        stem = file_path.stem  # "levelN"
        if stem.startswith("level"):
            try:
                level_num = int(stem[5:])  # Extract N from "levelN"
                level_numbers.append(level_num)
            except ValueError:
                continue

    if not level_numbers:
        raise ValueError(f"No valid level files found in {metadata_dir}")

    return max(level_numbers)


async def detect_max_level_remote(store: ObjectStore, prefix: str = "") -> int:
    max_level = -1
    for level in range(6):
        try:
            metadata_path = (
                f"{prefix}METADATA/level{level}.avro"
                if prefix
                else f"METADATA/level{level}.avro"
            )
            await obs.get_range_async(store, metadata_path, start=0, length=100)
            max_level = level
        except Exception:
            # First missing level found, stop
            break

    if max_level == -1:
        raise ValueError(f"No metadata files found at {prefix}METADATA/")

    return max_level


def read_avro_local(path: str, level: int) -> pl.DataFrame:
    """
    Read Avro file from local FOLDER.

    Args:
        path: Local FOLDER path
        level: Level number to read

    Returns:
        Polars DataFrame

    Examples:
        >>> df = read_avro_local("dataset/", 1)
    """
    avro_path = Path(path) / "METADATA" / f"level{level}.avro"
    return pl.read_avro(str(avro_path))


async def read_avro_remote(
    store: ObjectStore, level: int, prefix: str = ""
) -> pl.DataFrame:
    """
    Read Avro file from remote FOLDER.

    Args:
        store: obstore ObjectStore instance
        level: Level number to read
        prefix: Path prefix within the store

    Returns:
        Polars DataFrame

    Examples:
        >>> store = obs.store.from_url("s3://bucket/dataset/")
        >>> df = await read_avro_remote(store, 1)
    """
    metadata_path = (
        f"{prefix}METADATA/level{level}.avro"
        if prefix
        else f"METADATA/level{level}.avro"
    )

    # Read entire Avro file
    avro_result = await obs.get_async(store, metadata_path)
    avro_bytes = await avro_result.bytes_async()

    # Polars can read Avro from bytes
    return pl.read_avro(io.BytesIO(bytes(avro_bytes)))


def read_all_levels_local(path: str) -> list[pl.DataFrame]:
    """
    Read all metadata levels from local FOLDER (eager).

    Args:
        path: Local FOLDER path

    Returns:
        List of DataFrames [level0, level1, level2, ...]

    Examples:
        >>> dataframes = read_all_levels_local("dataset/")
        >>> len(dataframes)
        3
    """
    max_level = detect_max_level_local(path)

    dataframes = []
    for level in range(max_level + 1):
        df = read_avro_local(path, level)
        dataframes.append(df)

    return dataframes


async def read_all_levels_remote(
    store: ObjectStore, prefix: str = ""
) -> list[pl.DataFrame]:
    """
    Read all metadata levels from remote FOLDER (eager).

    Args:
        store: obstore ObjectStore instance
        prefix: Path prefix within the store

    Returns:
        List of DataFrames [level0, level1, level2, ...]

    Examples:
        >>> store = obs.store.from_url("s3://bucket/dataset/")
        >>> dataframes = await read_all_levels_remote(store)
    """
    max_level = await detect_max_level_remote(store, prefix)

    # Read all levels concurrently
    async def read_level(level: int) -> pl.DataFrame:
        return await read_avro_remote(store, level, prefix)

    dataframes = await asyncio.gather(
        *[read_level(lvl) for lvl in range(max_level + 1)]
    )

    return list(dataframes)


def read_all_levels(path: str) -> list[pl.DataFrame]:
    """
    Read all metadata levels from FOLDER (local or remote).

    Automatically detects if path is local or remote and uses
    appropriate reading method.

    Args:
        path: FOLDER path (local directory or remote URL)

    Returns:
        List of DataFrames [level0, level1, level2, ...]

    Examples:
        >>> # Local
        >>> dataframes = read_all_levels("dataset/")

        >>> # Remote
        >>> dataframes = read_all_levels("s3://bucket/dataset/")
    """
    # Check if local
    if Path(path).exists():
        return read_all_levels_local(path)

    # Remote - use obstore
    store = from_url(path)

    # Run async function in sync context
    return asyncio.run(read_all_levels_remote(store))
