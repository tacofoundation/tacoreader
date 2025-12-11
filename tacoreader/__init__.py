import importlib.metadata as _metadata
import logging

from tacoreader._constants import DEFAULT_DATAFRAME_BACKEND, DataFrameBackend
from tacoreader._exceptions import TacoBackendError
from tacoreader._logging import disable_logging, setup_basic_logging
from tacoreader.concat import concat
from tacoreader.dataframe.base import TacoDataFrame
from tacoreader.dataset import TacoDataset
from tacoreader.loader import load

__version__ = _metadata.version("tacoreader")

# Global DataFrame backend configuration
_DATAFRAME_BACKEND: DataFrameBackend = DEFAULT_DATAFRAME_BACKEND


def use(backend: str):
    """
    Set the global DataFrame backend for all future load() operations.

    Available backends:
        - 'pyarrow': Default, no extra dependencies
        - 'polars': Requires polars package (future)
        - 'pandas': Requires pandas package (future)

    Args:
        backend: Backend name to use

    Raises:
        TacoBackendError: If backend is unknown

    Warning - Thread/Fork Safety:
        This function modifies a global variable and is NOT thread-safe or fork-safe.

        Multiprocessing issues (PyTorch DataLoader, concurrent.futures):
        - Workers inherit the backend value at fork/spawn time
        - Changes in main process after fork are NOT visible to workers
        - This causes inconsistent behavior if load() is called inside workers

        Safe pattern (RECOMMENDED):
            class MyDataset(torch.utils.data.Dataset):
                def __init__(self, path):
                    tacoreader.use('pyarrow')  # Set BEFORE fork
                    self.ds = tacoreader.load(path)  # Load in __init__

                def __getitem__(self, idx):
                    return self.ds.data[idx]  # Workers only read, no load()

        Unsafe pattern (AVOID):
            class BadDataset(torch.utils.data.Dataset):
                def __getitem__(self, idx):
                    # BAD: load() in worker sees stale backend!
                    ds = tacoreader.load('data.taco')
                    return ds.data[idx]

        Thread-safe alternative:
            Pass backend explicitly to bypass global:
                tacoreader.load('data.taco', backend='pyarrow')

    Examples:
        >>> import tacoreader
        >>>
        >>> # Use PyArrow (default)
        >>> tacoreader.use('pyarrow')
        >>> reader = tacoreader.load('data.taco')
        >>> df = reader.data  # Returns PyArrow Table
        >>>
        >>> # Use Pandas (when available)
        >>> tacoreader.use('pandas')
        >>> reader = tacoreader.load('data.taco')
        >>> df = reader.data  # Returns Pandas DataFrame
    """
    global _DATAFRAME_BACKEND

    from tacoreader.dataframe import get_available_backends

    available = get_available_backends()

    if backend not in available:
        raise TacoBackendError(
            f"Unknown backend: '{backend}'\n"
            f"Available backends: {available}\n"
            f"\n"
            f"To use additional backends, install required packages:\n"
            f"  pip install polars  # For Polars backend\n"
            f"  pip install pandas  # For Pandas backend"
        )

    _DATAFRAME_BACKEND = backend  # type: ignore[assignment]


def get_backend() -> DataFrameBackend:
    """
    Get the current global DataFrame backend.

    Returns:
        Current backend name

    Example:
        >>> import tacoreader
        >>> tacoreader.get_backend()
        'pyarrow'
    """
    return _DATAFRAME_BACKEND


def verbose(level=True):
    """
    Enable/disable verbose logging for tacoreader operations.

    Args:
        level: Logging level to enable:
            - True or "info": Show INFO and above (default)
            - "debug": Show DEBUG and above (very detailed)
            - False: Disable all logging

    Example:
        >>> import tacoreader
        >>>
        >>> # Enable standard logging
        >>> tacoreader.verbose()
        >>>
        >>> # Enable debug logging (very detailed)
        >>> tacoreader.verbose("debug")
        >>>
        >>> # Disable logging
        >>> tacoreader.verbose(False)
    """
    if level is False:
        disable_logging()
    elif level is True or level == "info":
        setup_basic_logging(level=logging.INFO)
    elif level == "debug":
        setup_basic_logging(level=logging.DEBUG)
    else:
        raise ValueError(
            f"Invalid verbose level: {level}. " "Use True, 'info', 'debug', or False."
        )


def clear_cache():
    """
    Clear all metadata caches (headers, COLLECTION.json).

    Cached metadata includes:
    - ZIP headers (256 bytes)
    - COLLECTION.json files (~10-50 KB)
    - TacoCat COLLECTION.json

    Use this when remote files have changed and you want to force
    re-download on next load().

    Note: This does NOT clear any data caches (Parquet tables, rasters).
    Only small metadata files are cached.

    Examples:
        >>> import tacoreader
        >>>
        >>> # Load dataset (caches metadata)
        >>> ds1 = tacoreader.load("s3://bucket/data.tacozip")
        >>>
        >>> # Load again - uses cache (fast!)
        >>> ds2 = tacoreader.load("s3://bucket/data.tacozip")
        >>>
        >>> # File changed remotely, need fresh data
        >>> tacoreader.clear_cache()
        >>> ds3 = tacoreader.load("s3://bucket/data.tacozip")  # Re-downloads
    """
    from tacoreader.storage.folder import _read_collection_folder_cached
    from tacoreader.storage.tacocat import _read_tacocat_collection_cached
    from tacoreader.storage.zip import _read_taco_header_cached

    _read_taco_header_cached.cache_clear()
    _read_collection_folder_cached.cache_clear()
    _read_tacocat_collection_cached.cache_clear()


__all__ = [
    "TacoDataFrame",
    "TacoDataset",
    "clear_cache",
    "concat",
    "get_backend",
    "load",
    "use",
    "verbose",
]
