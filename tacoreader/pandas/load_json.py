import asyncio
import json
import struct
import concurrent.futures
from pathlib import Path
import logging

import obstore as obs
from obstore.store import from_url

logger = logging.getLogger(__name__)


class CollectionLoadError(Exception):
    """Exception for collection loading errors."""
    pass


def _parse_taco_header_for_collection(data: bytes) -> tuple[int, int]:
    """
    Parse TACO ZIP header to extract COLLECTION.json offset and length.
    COLLECTION.json is the LAST entry in the header after all metadata parquet files.
    """
    if len(data) < 200:
        raise CollectionLoadError("Invalid TACO file: insufficient header data")
    
    # Parse ZIP local file header
    filename_len = data[26] | (data[27] << 8)
    extra_len = data[28] | (data[29] << 8)
    payload_start = 30 + filename_len + extra_len
    
    if payload_start + 116 > len(data):
        raise CollectionLoadError("Invalid TACO file: header extends beyond available data")
    
    payload = data[payload_start:payload_start + 116]
    count = min(payload[0], 7)  # Max 7 levels
    
    if count == 0:
        raise CollectionLoadError("No entries found in TACO header")
    
    # COLLECTION.json is the LAST entry
    last_entry_index = count - 1
    start = 4 + (last_entry_index * 16)
    
    if start + 16 > len(payload):
        raise CollectionLoadError("Invalid TACO header: collection entry extends beyond payload")
    
    offset, length = struct.unpack('<QQ', payload[start:start + 16])
    return offset, length


async def _load_collection_local(file_path: str) -> dict:
    """Load COLLECTION.json from local TACO file."""
    try:
        with open(file_path, 'rb') as f:
            # Read header
            header_data = f.read(200)
            offset, length = _parse_taco_header_for_collection(header_data)
            
            # Read COLLECTION.json bytes
            f.seek(offset)
            collection_bytes = f.read(length)
            
            # Parse JSON
            collection_json = json.loads(collection_bytes.decode('utf-8'))
            return collection_json
            
    except (OSError, IOError) as e:
        raise CollectionLoadError(f"Failed to read local file {file_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise CollectionLoadError(f"Invalid JSON in COLLECTION.json: {e}") from e


async def _load_collection_remote(file_path: str) -> dict:
    """Load COLLECTION.json from remote TACO file."""
    try:
        store = from_url(file_path)
        
        # Read header
        header_data = await obs.get_range_async(store, "", start=0, length=200)
        offset, length = _parse_taco_header_for_collection(header_data)
        
        # Read COLLECTION.json bytes
        collection_bytes = await obs.get_range_async(store, "", start=offset, length=length)
        
        # Parse JSON
        collection_json = json.loads(collection_bytes.decode('utf-8'))
        return collection_json
        
    except Exception as e:
        raise CollectionLoadError(f"Failed to read remote file {file_path}: {e}") from e


async def load_collection_async(file_path: str) -> dict:
    """
    Asynchronously load COLLECTION.json from a TACO file.
    
    Args:
        file_path: Path to TACO file (local or remote URL)
        
    Returns:
        dict: Parsed COLLECTION.json content
        
    Raises:
        CollectionLoadError: If loading fails
    """
    try:
        if Path(file_path).exists():
            # Local file
            return await _load_collection_local(file_path)
        else:
            # Remote file
            return await _load_collection_remote(file_path)
            
    except CollectionLoadError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading collection from {file_path}: {e}")
        raise CollectionLoadError(f"Unexpected error loading collection: {e}") from e


def load_collection(file_path: str) -> dict:
    """
    Load COLLECTION.json from a TACO file (local or remote).
    
    This function handles both local files and remote URLs (S3, GCS, HTTP, etc.).
    
    Args:
        file_path: Path to TACO file or remote URL
        
    Returns:
        dict: Parsed COLLECTION.json content
        
    Raises:
        CollectionLoadError: If loading fails
        
    Examples:
        >>> # Local file
        >>> collection = load_collection("my_dataset.taco")
        >>> print(collection["id"])
        
        >>> # Remote file
        >>> collection = load_collection("s3://bucket/dataset.taco")
        >>> print(collection["description"])
    """
    # Handle async execution based on environment
    try:
        # Check if there's a running event loop
        asyncio.get_running_loop()
        # We're in Jupyter/Colab - run in separate thread with new loop
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(load_collection_async(file_path))
                return result
            finally:
                new_loop.close()
        
        # Run in thread to avoid loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()
            
    except RuntimeError:
        # No running loop - normal Python environment
        return asyncio.run(load_collection_async(file_path))