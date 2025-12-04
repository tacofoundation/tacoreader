import importlib.metadata as _metadata
import logging

from tacoreader._constants import DEFAULT_DATAFRAME_BACKEND, DataFrameBackend
from tacoreader._logging import disable_logging, setup_basic_logging
from tacoreader.backends.dataframe.base import TacoDataFrame
from tacoreader.concat import concat
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
        ValueError: If backend is unknown

    Examples:
        >>> import tacoreader
        >>>
        >>> # Use PyArrow (default)
        >>> tacoreader.use('pyarrow')
        >>> reader = tacoreader.load('data.taco')
        >>> df = reader.data  # Returns PyArrow Table
        >>>
        >>> # Use Polars (when available)
        >>> tacoreader.use('polars')
        >>> reader = tacoreader.load('data.taco')
        >>> df = reader.data  # Returns Polars DataFrame
    """
    global _DATAFRAME_BACKEND

    # Import registry to check available backends
    from tacoreader.backends.dataframe import get_available_backends

    available = get_available_backends()

    if backend not in available:
        raise ValueError(
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


__all__ = [
    "TacoDataFrame",
    "TacoDataset",
    "concat",
    "get_backend",
    "load",
    "use",
    "verbose",
]
