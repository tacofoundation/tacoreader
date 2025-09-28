import asyncio
import struct
import concurrent.futures
from io import BytesIO
from pathlib import Path
import logging
import re
import pandas as pd

import obstore as obs
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import tacozip
from obstore.store import from_url
from tacoreader.pandas.treedataframe import TreeDataFrame

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


def _parse_taco_header(file_path: str) -> list[tuple[int, int]]:
    """Parse TACO ZIP header using tacozip."""
    d1, d2, j1 = tacozip.read_header(file_path)
    entries = []
    if d1:
        entries.append(d1)
    if d2:
        entries.append(d2)
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
            table = pq.read_table(pa.BufferReader(parquet_bytes))
            df = pl.from_arrow(table)
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
        table = pq.read_table(pa.BufferReader(parquet_bytes))
        df = pl.from_arrow(table)
        dataframes.append(df)
    
    return dataframes


def _build_vsi_paths_eager(df: pl.DataFrame, base_vsi_path: str) -> pl.DataFrame:
    """Build VSI paths for eager mode - terminal files only."""
    if not all(col in df.columns for col in ["internal:offset", "internal:size"]):
        return df
    
    # Generate VSI paths only for non-TORTILLA files
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
    
    # Remove offset/size columns
    return df_with_vsi.drop(["internal:offset", "internal:size"])


def _build_vsi_paths_lazy(df: pl.DataFrame, base_vsi_path: str, tortilla_id_to_hierarchy_offset: dict) -> pl.DataFrame:
    """
    Build VSI paths for lazy mode.
    
    Args:
        df: Level 0 DataFrame with TORTILLA and terminal files
        base_vsi_path: Base VSI path to TACO file
        tortilla_id_to_hierarchy_offset: Map from TORTILLA id to its HIERARCHY metadata.parquet offset
    """
    if not all(col in df.columns for col in ["internal:offset", "internal:size"]):
        return df
    
    # Build VSI paths
    vsi_paths = []
    for row in df.iter_rows(named=True):
        if row["type"] == "TORTILLA":
            # Use HIERARCHY offset for TORTILLA
            tortilla_id = row["id"]
            if tortilla_id in tortilla_id_to_hierarchy_offset:
                hierarchy_offset, hierarchy_size = tortilla_id_to_hierarchy_offset[tortilla_id]
                vsi_path = f"/vsimetadata/{hierarchy_offset}_{hierarchy_size},{base_vsi_path}"
            else:
                vsi_path = None  # Fallback to eager mode
        else:
            # Terminal files use DATA/ path
            vsi_path = f"/vsisubfile/{row['internal:offset']}_{row['internal:size']},{base_vsi_path}"
        
        vsi_paths.append(vsi_path)
    
    # Add VSI paths and remove offset/size
    df_with_vsi = df.with_columns(pl.Series("internal:gdal_vsi", vsi_paths))
    return df_with_vsi.drop(["internal:offset", "internal:size"])


def _scan_hierarchy_metadata(file_path: str) -> dict:
    """
    Scan ZIP file to find METADATA/HIERARCHY/ files and their offsets.
    Returns mapping from tortilla_id to (offset, size).
    """
    import zipfile
    
    tortilla_to_offset = {}
    
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        with open(file_path, 'rb') as f:
            for info in zip_file.infolist():
                # Look for METADATA/HIERARCHY/tortilla_id/metadata.parquet
                if (info.filename.startswith('METADATA/HIERARCHY/') and 
                    info.filename.endswith('/metadata.parquet')):
                    
                    # Extract tortilla_id from path
                    # METADATA/HIERARCHY/tortilla_a/metadata.parquet -> tortilla_a
                    path_parts = info.filename.split('/')
                    if len(path_parts) >= 3:
                        tortilla_id = path_parts[2]
                        
                        # Calculate actual data offset
                        f.seek(info.header_offset)
                        header = f.read(30)
                        
                        if len(header) < 30:
                            continue
                        
                        # Validate local file header signature
                        signature = struct.unpack('<I', header[0:4])[0]
                        if signature != 0x04034b50:
                            continue
                        
                        filename_len = struct.unpack('<H', header[26:28])[0]
                        extra_len = struct.unpack('<H', header[28:30])[0]
                        
                        actual_offset = info.header_offset + 30 + filename_len + extra_len
                        actual_size = info.compress_size
                        
                        tortilla_to_offset[tortilla_id] = (actual_offset, actual_size)
    
    return tortilla_to_offset


async def _read_single_taco(file_path: str, lazy: bool = False) -> list[pl.DataFrame]:
    """Read a single TACO ZIP file and extract Parquet levels."""
    try:
        # Read header (common for both local and remote)
        if Path(file_path).exists():
            # LOCAL FILE - use tacozip
            entries = _parse_taco_header(file_path)
            
            if lazy:
                # Only read first entry (level 0)
                entries = entries[:1]
            
            dataframes = await _read_parquet_entries_local(file_path, entries)
            
        else:
            # REMOTE FILE - lazy mode not supported for remote yet
            if lazy:
                raise NotImplementedError("Lazy mode not supported for remote files yet")
                
            store = from_url(file_path)
            header_data = await obs.get_range_async(store, "", start=0, length=200)
            
            # Use original manual parsing for remote
            entries = _parse_taco_header_manual(header_data)
            dataframes = await _read_parquet_entries_remote(store, entries)
        
        return dataframes
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise RuntimeError(f"Failed to read TACO file {file_path}: {e}") from e


def _parse_taco_header_manual(data: bytes) -> list[tuple[int, int]]:
    """Parse TACO ZIP header manually for remote files."""
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


async def read_taco_data_async(
    file_paths: str | list[str], 
    max_concurrent: int = 20,
    lazy: bool = False
) -> list[pl.DataFrame]:
    """
    Asynchronously read TACO ZIP files and return level-wise merged DataFrames.
    """
    
    if isinstance(file_paths, str):
        # Single file
        dataframes = await _read_single_taco(file_paths, lazy=lazy)
        base_vsi_path = _generate_vsi_path(file_paths)
        
        # Handle lazy mode for single file
        if lazy and Path(file_paths).exists():
            # Scan for HIERARCHY metadata
            tortilla_hierarchy_map = _scan_hierarchy_metadata(file_paths)
            processed_df = _build_vsi_paths_lazy(dataframes[0], base_vsi_path, tortilla_hierarchy_map)
            return [processed_df]
        
    else:
        # Multiple files with concurrency control
        if not file_paths:
            raise ValueError("No file paths provided")
        
        if lazy:
            raise NotImplementedError("Lazy mode not supported for multiple files yet")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_read(file_path: str) -> list[pl.DataFrame]:
            async with semaphore:
                return await _read_single_taco(file_path, lazy=lazy)
        
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
    
    # Post-process: Add VSI paths for all levels
    processed_dataframes = []
    
    for df in dataframes:
        if lazy:
            # This shouldn't happen for multi-file, but handle gracefully
            processed_df = df
        else:
            # Eager mode: build VSI paths for terminal files only
            processed_df = _build_vsi_paths_eager(df, base_vsi_path)
        processed_dataframes.append(processed_df)
    
    return processed_dataframes


def load(file_paths: str | list[str], lazy: bool = False, max_concurrent: int = 20) -> TreeDataFrame:
    """
    Load TACO files as TreeDataFrame with optional lazy loading.
    
    Args:
        file_paths: Single file path or list of file paths to load
        lazy: If True, only load level 0 and read HIERARCHY metadata on demand  
        max_concurrent: Maximum concurrent file operations for multiple files
        
    Returns:
        TreeDataFrame with level 0 as primary and rest as auxiliary levels (if not lazy)
    """
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")
    
    # Handle async execution based on environment
    try:
        # Check if there's a running event loop
        asyncio.get_running_loop()
        # We're in Jupyter/Colab - run in separate thread with new loop
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(
                    read_taco_data_async(file_paths, max_concurrent, lazy=lazy)
                )
                return result
            finally:
                new_loop.close()
        
        # Run in thread to avoid loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            dataframes = future.result()
            
    except RuntimeError:
        # No running loop - normal Python environment
        dataframes = asyncio.run(read_taco_data_async(file_paths, max_concurrent, lazy=lazy))
    
    if not dataframes:
        raise ValueError("No data loaded from files")
    
    # Convert to pandas DataFrames
    pandas_dfs = [df.to_pandas() for df in dataframes]
    
    # Store source file in TreeDataFrame for lazy loading
    source_file = file_paths if isinstance(file_paths, str) else file_paths[0]
    
    if lazy:
        # Only level 0, no auxiliaries
        mdf = TreeDataFrame(pandas_dfs[0])
        mdf._source_file = source_file
        return mdf
    else:
        # Level 0 as primary, rest as auxiliaries
        mdf = TreeDataFrame(pandas_dfs[0], auxiliary_dataframes=pandas_dfs[1:])
        mdf._source_file = source_file
        return mdf


def _parse_vsi_metadata_path(vsi_path: str) -> tuple[int, int, str]:
    """Parse /vsimetadata/offset_length,path format to extract offset, length, and path."""
    pattern = re.compile(r"/vsimetadata/(\d+)_(\d+),(.+)")
    match = pattern.match(vsi_path)
    
    if not match:
        raise ValueError(f"Invalid /vsimetadata/ format: {vsi_path}")
    
    offset, length, path = match.groups()
    return int(offset), int(length), path


def _read_hierarchy_metadata_parquet(source_file: str, vsi_path: str) -> TreeDataFrame:
    """
    Read HIERARCHY metadata.parquet using obstore from /vsimetadata/ VSI path.
    The metadata.parquet already contains correct internal:offset/size.
    """
    # Parse VSI path to get offset, length, and path
    offset, size, _ = _parse_vsi_metadata_path(vsi_path)
    
    # Create store based on file type
    if Path(source_file).exists():
        # Local file
        from obstore.store import LocalStore
        store = LocalStore()
        path_in_store = source_file
    else:
        # Remote file
        store = from_url(source_file)
        path_in_store = ""  # Empty string for remote stores
    
    # Read bytes using obstore
    parquet_bytes = obs.get_range(store, path_in_store, start=offset, length=size)
    
    # Parse parquet bytes
    table = pq.read_table(pa.BufferReader(parquet_bytes))
    df_pandas = table.to_pandas()
    
    # The metadata.parquet from HIERARCHY already has correct internal:offset/size
    # Build VSI paths for files - they should point to DATA/
    base_vsi_path = _generate_vsi_path(source_file)
    
    if all(col in df_pandas.columns for col in ["internal:offset", "internal:size"]):
        # All files in HIERARCHY metadata should be terminal files (no nested TORTILLAS)
        # They all point to DATA/ locations
        vsi_paths = []
        for _, row in df_pandas.iterrows():
            vsi_path = f"/vsisubfile/{row['internal:offset']}_{row['internal:size']},{base_vsi_path}"
            vsi_paths.append(vsi_path)
        
        df_pandas['internal:gdal_vsi'] = vsi_paths
        
        # Remove offset/size columns - clean API
        df_pandas = df_pandas.drop(["internal:offset", "internal:size"], axis=1, errors='ignore')
    
    # Create new TreeDataFrame
    result = TreeDataFrame(df_pandas)
    result._source_file = source_file  # Preserve source file for potential nested reads
    return result


def _enhanced_read(self, identifier):
    """Enhanced read method with lazy loading support for new HIERARCHY architecture."""
    # Get row and position
    if isinstance(identifier, int):
        if identifier >= len(self):
            raise IndexError(f"Index {identifier} out of bounds")
        row = self.iloc[identifier]
        position = identifier
    else:
        # Find by ID
        matches = self[self['id'] == identifier]
        if matches.empty:
            raise ValueError(f"ID '{identifier}' not found")
        row = matches.iloc[0]
        position = matches.index[0]
    
    if row['type'] == 'TORTILLA':
        # Check if we have /vsimetadata/ VSI path (lazy mode)
        if 'internal:gdal_vsi' in row and pd.notna(row['internal:gdal_vsi']):
            vsi_path = str(row['internal:gdal_vsi'])
            if vsi_path.startswith('/vsimetadata/'):
                # Lazy mode: read HIERARCHY metadata.parquet
                return _read_hierarchy_metadata_parquet(self._source_file, vsi_path)
        
        # Eager mode: navigate to auxiliary (when internal:gdal_vsi is None)
        if hasattr(self, '_auxiliary_levels') and self._auxiliary_levels:
            next_level = self._auxiliary_levels[0]
            
            # Filter by position
            if 'internal:position' in next_level.columns:
                filtered_df = next_level[next_level['internal:position'] == position].copy()
                
                if filtered_df.empty:
                    raise ValueError(f"No data for position {position}")
                
                # Create new TreeDataFrame with filtered data
                result = TreeDataFrame(filtered_df.reset_index(drop=True))
                
                # Copy auxiliary chain and source file
                result._auxiliary_levels = self._auxiliary_levels[1:] if len(self._auxiliary_levels) > 1 else []
                if hasattr(self, '_source_file'):
                    result._source_file = self._source_file
                
                return result
            else:
                return next_level
        
        raise ValueError("TORTILLA has no /vsimetadata/ VSI path for lazy loading and no auxiliaries for eager mode")
    
    else:
        # Terminal file - return VSI path
        if 'internal:gdal_vsi' in row and pd.notna(row['internal:gdal_vsi']):
            return str(row['internal:gdal_vsi'])
        
        raise ValueError(f"No VSI path for row with id '{identifier}'")


# Apply the enhanced read method to TreeDataFrame
TreeDataFrame.read = _enhanced_read