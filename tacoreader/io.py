"""
Remote I/O operations for TACO datasets.

Centralized remote file ops (HTTP, S3, GCS, Azure).
Uses obstore backend but designed for easy replacement.

Main functions:
    download_bytes: Download complete file
    download_range: Download byte range
    get_file_size: Get remote file size
"""

import obstore as obs


def _create_store(url: str):
    """
    Create obstore ObjectStore from URL.

    Supports: s3://, gs://, az://, http://, https://
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
            f"Supported: s3://, gs://, az://, http://, https://"
        )


def download_bytes(url: str, subpath: str = "") -> bytes:
    """
    Download complete file from remote URL.

    Args:
        url: Base URL (e.g., "s3://bucket/", "https://example.com/")
        subpath: Optional subpath (e.g., "file.zip", "metadata/level0.parquet")

    Returns:
        Complete file as bytes
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

    Efficient for reading portions of large files without full download.
    Uses HTTP Range requests or cloud storage equivalent.
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
    Get remote file size in bytes.

    Returns None if unavailable.
    """
    try:
        store = _create_store(url)
        head = obs.head(store, subpath)
        return head.size
    except Exception:
        return None
