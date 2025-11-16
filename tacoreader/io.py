"""
Remote I/O operations for TACO datasets.

Centralized module for all remote file operations (HTTP, S3, GCS, Azure).
Currently uses obstore as backend, but designed to be easily replaceable.

Main functions:
    download_bytes: Download complete file
    download_range: Download byte range
    get_file_size: Get remote file size
"""

import obstore as obs


def _create_store(url: str):
    """
    Create obstore ObjectStore from URL.
    
    Internal helper for obstore store creation.
    
    Args:
        url: Cloud storage URL (s3://, gs://, az://, https://)
        
    Returns:
        obstore ObjectStore instance
        
    Raises:
        ValueError: If URL scheme is not supported
    """
    if url.startswith("s3://"):
        return obs.store.S3Store.from_url(url)
    
    elif url.startswith("gs://"):
        return obs.store.GCSStore.from_url(url)
    
    elif url.startswith("az://") or url.startswith("azure://"):
        return obs.store.AzureStore.from_url(url)
    
    elif url.startswith("http://") or url.startswith("https://"):
        return obs.store.HTTPStore.from_url(url)
    
    else:
        raise ValueError(
            f"Unsupported URL scheme: {url}\n"
            f"Supported schemes: s3://, gs://, az://, http://, https://"
        )


def download_bytes(url: str, subpath: str = "") -> bytes:
    """
    Download complete file from remote URL.
    
    Supports HTTP/HTTPS, S3, GCS, and Azure storage.
    
    Args:
        url: Base URL (e.g., "s3://bucket/", "https://example.com/data/")
        subpath: Optional subpath within base URL (e.g., "file.zip", "metadata/level0.parquet")
        
    Returns:
        Complete file contents as bytes
        
    Raises:
        OSError: If download fails
        
    Example:
        >>> # Download from HTTP
        >>> data = download_bytes("https://example.com/data.zip")
        >>> 
        >>> # Download with subpath
        >>> data = download_bytes("s3://bucket/", "data/file.zip")
        >>> 
        >>> # Download specific file from directory
        >>> data = download_bytes("https://example.com/dataset/", "METADATA/level0.parquet")
    """
    try:
        store = _create_store(url)
        result = obs.get(store, subpath)
        return bytes(result.bytes())
    except Exception as e:
        raise OSError(f"Failed to download {url}{subpath}: {e}")


def download_range(url: str, offset: int, size: int, subpath: str = "") -> bytes:
    """
    Download byte range from remote file.
    
    Efficient for reading specific portions of large files without
    downloading the entire file. Uses HTTP Range requests or equivalent
    for cloud storage.
    
    Args:
        url: Base URL
        offset: Starting byte offset (inclusive)
        size: Number of bytes to read
        subpath: Optional subpath within base URL
        
    Returns:
        Requested byte range as bytes
        
    Raises:
        OSError: If download fails
        
    Example:
        >>> # Read first 1000 bytes of file
        >>> header = download_range("https://example.com/data.zip", 0, 1000)
        >>> 
        >>> # Read bytes 5000-6000 from S3
        >>> chunk = download_range("s3://bucket/data.zip", 5000, 1000)
        >>> 
        >>> # Read from subdirectory
        >>> data = download_range("s3://bucket/", 1024, 5000, "data/file.zip")
    """
    try:
        store = _create_store(url)
        result = obs.get_range(store, subpath, start=offset, length=size)
        return bytes(result)
    except Exception as e:
        raise OSError(
            f"Failed to download range [{offset}:{offset+size}] from {url}{subpath}: {e}"
        )


def get_file_size(url: str, subpath: str = "") -> int | None:
    """
    Get size of remote file in bytes.
    
    Args:
        url: Base URL
        subpath: Optional subpath within base URL
        
    Returns:
        File size in bytes, or None if unavailable
        
    Example:
        >>> size = get_file_size("https://example.com/data.zip")
        >>> print(f"File is {size / 1024 / 1024:.1f} MB")
    """
    try:
        store = _create_store(url)
        head = obs.head(store, subpath)
        return head.size
    except Exception:
        return None