"""
Remote I/O operations for TACO datasets.

Centralized remote file ops (HTTP, S3, GCS, Azure).
Uses obstore backend but designed for easy replacement.
TODO: The use of obstore is temporary until get time to review OpenDAL properly.
"""

import obstore as obs

from tacoreader._constants import PROTOCOL_MAPPINGS
from tacoreader._exceptions import TacoIOError


def _create_store(url: str):
    """
    Create obstore ObjectStore from URL.

    Supports all protocols defined in PROTOCOL_MAPPINGS:
    - s3:// → S3Store
    - gs:// → GCSStore
    - az://, azure:// → AzureStore
    - http://, https:// → HTTPStore
    """
    # Build mapping from standard protocol to store class
    protocol_handlers = {
        PROTOCOL_MAPPINGS["s3"]["standard"]: obs.store.S3Store,
        PROTOCOL_MAPPINGS["gcs"]["standard"]: obs.store.GCSStore,
        PROTOCOL_MAPPINGS["azure"]["standard"]: obs.store.AzureStore,
        PROTOCOL_MAPPINGS["azure"]["alt"]: obs.store.AzureStore,  # azure:// alias
        PROTOCOL_MAPPINGS["http"]["standard"]: obs.store.HTTPStore,
        PROTOCOL_MAPPINGS["https"]["standard"]: obs.store.HTTPStore,
    }

    # Find matching protocol
    for protocol, store_class in protocol_handlers.items():
        if url.startswith(protocol):
            return store_class.from_url(url)  # type: ignore[attr-defined]

    # Build error message with all supported protocols
    supported = sorted({PROTOCOL_MAPPINGS[p]["standard"] for p in PROTOCOL_MAPPINGS})
    raise TacoIOError(
        f"Unsupported URL scheme: {url}\n" f"Supported: {', '.join(supported)}"
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
        raise TacoIOError(f"Failed to download {url}{subpath}: {e}") from e


def download_range(url: str, offset: int, size: int, subpath: str = "") -> bytes:
    """
    Download byte range from remote file.

    Efficient for reading portions of large files without full download.
    Uses HTTP Range requests or cloud storage equivalent.

    Args:
        url: Base URL
        offset: Starting byte position
        size: Number of bytes to read
        subpath: Optional subpath within URL

    Returns:
        Requested byte range
    """
    try:
        store = _create_store(url)
        result = obs.get_range(store, subpath, start=offset, length=size)
        return bytes(result)
    except Exception as e:
        raise TacoIOError(
            f"Failed to download range [{offset}:{offset+size}] from {url}{subpath}: {e}"
        ) from e


def get_file_size(url: str, subpath: str = "") -> int | None:
    """
    Get remote file size in bytes.

    Returns None if size unavailable (e.g., streamed HTTP responses).

    Args:
        url: Base URL
        subpath: Optional subpath within URL

    Returns:
        File size in bytes, or None if unavailable
    """
    try:
        store = _create_store(url)
        head = obs.head(store, subpath)
    except Exception:
        return None
    else:
        return head.size  # type: ignore[attr-defined]
