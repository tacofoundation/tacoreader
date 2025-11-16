import importlib.metadata as _metadata
import logging

from tacoreader.concat import concat
from tacoreader.dataframe import TacoDataFrame
from tacoreader.dataset import TacoDataset
from tacoreader.loader import load
from tacoreader._logging import setup_basic_logging, disable_logging

__version__ = _metadata.version("tacoreader")


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
    "load",
    "verbose",
]
