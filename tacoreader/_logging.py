"""
Logging configuration for tacoreader.

Provides centralized logging setup with appropriate levels and formats.
Users can configure logging behavior externally via standard logging config.

Usage:
    from tacoreader._logging import get_logger
    
    logger = get_logger(__name__)
    logger.debug("Detailed info for debugging")
    logger.info("General information")
    logger.warning("Warning message")
    logger.error("Error occurred")

Configuration (by user):
    import logging
    
    # Set level for all tacoreader
    logging.getLogger("tacoreader").setLevel(logging.DEBUG)
    
    # Set level for specific module
    logging.getLogger("tacoreader.loader").setLevel(logging.INFO)
    
    # Add custom handler
    handler = logging.FileHandler("taco.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger("tacoreader").addHandler(handler)
"""

import logging
from typing import Optional


# Default format for tacoreader logs
DEFAULT_FORMAT = "%(levelname)s [%(name)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for tacoreader module.

    Creates logger with tacoreader namespace for easy filtering.
    By default, loggers inherit from root logger configuration.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Loading dataset")
        INFO [tacoreader.loader] Loading dataset
    """
    # Ensure name is within tacoreader namespace
    if not name.startswith("tacoreader"):
        if name == "__main__":
            name = "tacoreader"
        else:
            name = f"tacoreader.{name}"

    return logging.getLogger(name)


def setup_basic_logging(
    level: int = logging.INFO, format: Optional[str] = None
) -> None:
    """
    Setup basic logging configuration for tacoreader.

    This is a convenience function for quick setup. Advanced users
    should configure logging directly via logging.basicConfig() or
    logging configuration files.

    Args:
        level: Logging level (default: INFO)
        format: Log message format (default: DEFAULT_FORMAT)

    Example:
        >>> from tacoreader._logging import setup_basic_logging
        >>> import logging
        >>>
        >>> # Enable debug logging
        >>> setup_basic_logging(level=logging.DEBUG)
        >>>
        >>> # Custom format
        >>> setup_basic_logging(format="%(asctime)s - %(message)s")
    """
    if format is None:
        format = DEFAULT_FORMAT

    # Configure root tacoreader logger
    logger = logging.getLogger("tacoreader")
    logger.setLevel(level)

    # Add console handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format))
        logger.addHandler(handler)

    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False


def enable_debug_logging() -> None:
    """
    Enable debug logging for all tacoreader modules.

    Convenience function equivalent to:
        logging.getLogger("tacoreader").setLevel(logging.DEBUG)

    Example:
        >>> from tacoreader._logging import enable_debug_logging
        >>> enable_debug_logging()
        >>> # Now all tacoreader modules log debug messages
    """
    setup_basic_logging(level=logging.DEBUG)


def disable_logging() -> None:
    """
    Disable all tacoreader logging.

    Sets level to CRITICAL+1, effectively silencing all logs.

    Example:
        >>> from tacoreader._logging import disable_logging
        >>> disable_logging()
        >>> # No tacoreader logs will be shown
    """
    logger = logging.getLogger("tacoreader")
    logger.setLevel(logging.CRITICAL + 1)