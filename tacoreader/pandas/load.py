import asyncio
import struct
from io import BytesIO
from pathlib import Path
import logging

import obstore as obs
import polars as pl
from obstore.store import from_url
from tacoreader.pandas.multidataframe import MultiDataFrame

logger = logging.getLogger(__name__)


def _generate_vsi_path(file_path: str) -> str:
    """Convert file_path to proper GDAL VSI path."""
    if Path(file_path).exists():
        return file_path  # Local file, no prefix needed
    elif file_path.startswith(('http://', 'https://')):
        return f"/vsicurl/{file_path}"
    elif file_path.startswith('s3://'):
        return f"/vsis3/{file_path}"
    elif file_path.startswith('gs://'):
        return f"/vsigs/{file_path}"
    elif file_path.startswith('az://'):
        return f"/vsiaz/{file_path}"
    elif file_path.startswith('oss://'):
        return f"/vsioss/{file_path}"
    elif file_path.startswith('swift://'):
        return f"/vsiswift/{file_path}"
    else:
        raise ValueError(f"Unsupported file protocol: {file_path}")


def _parse_taco_header(data: bytes) -> list[tuple[int, int]]:
    """Parse TACO ZIP header to extract parquet entry offsets and lengths."""
    if len(data) < 200:
        raise ValueError("Invalid TACO file: insufficient header data")
    
    # Parse ZIP local file header
    filename_len = data[26] | (data[27] << 8)
    extra_len = data[28] | (data[29] << 8)
    payload_start = 30 + filename_len + extra_len
    
    if payload_start + 116 > len(data):
        raise ValueError("Invalid TACO file: header extends beyond available data")
    
    payload = data[payload_start:payload_start + 116]
    count = min(payload[0], 7)  # Max 7 levels
    
    entries = []
    for i in range(count):
        start = 4 + (i * 16)
        if start + 16 > len(payload):
            break
        offset, length = struct.unpack('<QQ', payload[start:start + 16])
        entries.append((offset, length))
    
    return entries


def _validate_unique_tortilla_ids(df: pl.DataFrame) -> None:
    """Validate that TORTILLA IDs are unique across merged files."""
    if 'id' not in df.columns:
        return
    
    unique_count = df.select(pl.col('id')).n_unique()
    total_count = len(df)
    
    if unique_count != total_count:
        # Extract duplicate IDs for error message
        duplicates = df.filter(pl.col('id').is_duplicated())['id'].unique().to_list()
        duplicate_count = total_count - unique_count
        raise ValueError(
            f"Duplicate TORTILLA IDs found in level 0: {duplicates[:10]}... "
            f"({duplicate_count} total duplicates). TACO files cannot have duplicate TORTILLA IDs."
        )


async def _read_parquet_entries_local(file_path: str, entries: list[tuple[int, int]]) -> list[pl.DataFrame]:
    """Read parquet entries from local file."""
    dataframes = []
    
    with open(file_path, 'rb') as f:
        for offset, length in entries:
            f.seek(offset)
            parquet_bytes = f.read(length)
            df = pl.read_parquet(BytesIO(parquet_bytes))
            dataframes.append(df)
    
    return dataframes


async def _read_parquet_entries_remote(store, entries: list[tuple[int, int]]) -> list[pl.DataFrame]:
    """Read parquet entries from remote store."""
    # Create async tasks for all parquet files
    parquet_tasks = [
        obs.get_range_async(store, "", start=offset, length=length)
        for offset, length in entries
    ]
    
    # Execute all downloads concurrently
    parquet_bytes_list = await asyncio.gather(*parquet_tasks)
    
    # Parse parquet data
    dataframes = []
    for parquet_bytes in parquet_bytes_list:
        df = pl.read_parquet(BytesIO(parquet_bytes))
        dataframes.append(df)
    
    return dataframes


def _build_vsi_paths(df: pl.DataFrame, base_vsi_path: str) -> pl.DataFrame:
    """Build VSI paths for all file types (anything that's not TORTILLA)."""
    if not all(col in df.columns for col in ["internal:offset", "internal:size"]):
        return df
    
    # Generate VSI paths for all files (anything that's not TORTILLA)
    df_with_vsi = df.with_columns(
        pl.when(pl.col("type") != "TORTILLA")
        .then(
            pl.format("/vsisubfile/{}_{}," + base_vsi_path, 
                     pl.col("internal:offset"), 
                     pl.col("internal:size"))
        )
        .otherwise(None)
        .alias("internal:gdal_vsi")
    )
    
    # Clean up temporary columns
    return df_with_vsi.drop(["internal:offset", "internal:size"])


async def _read_single_taco(file_path: str) -> list[pl.DataFrame]:
    """Read a single TACO ZIP file and extract all Parquet levels."""
    try:
        # Read header (common for both local and remote)
        if Path(file_path).exists():
            # LOCAL FILE
            with open(file_path, 'rb') as f:
                header_data = f.read(200)
            
            entries = _parse_taco_header(header_data)
            dataframes = await _read_parquet_entries_local(file_path, entries)
            
        else:
            # REMOTE FILE
            store = from_url(file_path)
            header_data = await obs.get_range_async(store, "", start=0, length=200)
            
            entries = _parse_taco_header(header_data)
            dataframes = await _read_parquet_entries_remote(store, entries)
        
        return dataframes
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise RuntimeError(f"Failed to read TACO file {file_path}: {e}") from e


async def read_taco_data_async(
    file_paths: str | list[str], 
    max_concurrent: int = 20
) -> list[pl.DataFrame]:
    """
    Asynchronously read TACO ZIP files and return level-wise merged DataFrames.
    
    Args:
        file_paths: Single file path or list of file paths
        max_concurrent: Maximum concurrent file operations
        
    Returns:
        List of DataFrames, one per hierarchy level
        
    Raises:
        RuntimeError: If file reading fails
        ValueError: If no valid files found or duplicate TORTILLA IDs
    """
    
    if isinstance(file_paths, str):
        # Single file
        dataframes = await _read_single_taco(file_paths)
        base_vsi_path = _generate_vsi_path(file_paths)
        
    else:
        # Multiple files with concurrency control
        if not file_paths:
            raise ValueError("No file paths provided")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_read(file_path: str) -> list[pl.DataFrame]:
            async with semaphore:
                return await _read_single_taco(file_path)
        
        # Process all files
        tasks = [limited_read(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)
        
        # Merge by level
        if not results or not results[0]:
            raise ValueError("No valid data found in any files")
        
        num_levels = len(results[0])
        dataframes = []
        
        for level_idx in range(num_levels):
            level_dfs = [file_dfs[level_idx] for file_dfs in results if level_idx < len(file_dfs)]
            if level_dfs:
                merged_level = pl.concat(level_dfs)
                
                # Validate unique TORTILLA IDs only for level 0
                if level_idx == 0:
                    _validate_unique_tortilla_ids(merged_level)
                
                dataframes.append(merged_level)
        
        base_vsi_path = _generate_vsi_path(file_paths[0])
    
    # Post-process: Add VSI paths for all levels consistently
    processed_dataframes = []
    
    for df in dataframes:
        processed_df = _build_vsi_paths(df, base_vsi_path)
        processed_dataframes.append(processed_df)
    
    return processed_dataframes


def load(file_paths: str | list[str], max_concurrent: int = 20) -> MultiDataFrame:
    """
    Load TACO files as MultiDataFrame with hierarchical organization.
    
    Args:
        file_paths: Single file path or list of file paths to load
        max_concurrent: Maximum concurrent file operations for multiple files
        
    Returns:
        MultiDataFrame with level 0 as primary and rest as auxiliary levels
        
    Raises:
        RuntimeError: If file loading fails
        ValueError: If no valid files, invalid parameters, or duplicate TORTILLA IDs
    """
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")
    
    # Read TACO data asynchronously
    dataframes = asyncio.run(read_taco_data_async(file_paths, max_concurrent))
    
    if not dataframes:
        raise ValueError("No data loaded from files")
    
    # Convert to pandas DataFrames
    pandas_dfs = [df.to_pandas() for df in dataframes]
    
    # Create MultiDataFrame: level 0 as primary, rest as auxiliaries
    return MultiDataFrame(pandas_dfs[0], auxiliary_dataframes=pandas_dfs[1:])