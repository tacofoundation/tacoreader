"""
Logging configuration for tacoreader.

Centralized logging setup. Users configure externally via standard logging config.

Usage:
    from tacoreader._logging import get_logger
    
    logger = get_logger(__name__)
    logger.debug("Debug info")
    logger.info("General info")
"""

import logging
from typing import Optional

DEFAULT_FORMAT = "%(levelname)s [%(name)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Get logger for tacoreader module."""
    # Ensure tacoreader namespace
    if not name.startswith("tacoreader"):
        name = "tacoreader" if name == "__main__" else f"tacoreader.{name}"

    return logging.getLogger(name)


def setup_basic_logging(
    level: int = logging.INFO, format: Optional[str] = None
) -> None:
    """
    Setup basic logging for tacoreader.

    Convenience function - advanced users should configure via logging.basicConfig().
    """
    if format is None:
        format = DEFAULT_FORMAT

    logger = logging.getLogger("tacoreader")
    logger.setLevel(level)

    # Add console handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format))
        logger.addHandler(handler)

    # Prevent propagation (avoid duplicate logs)
    logger.propagate = False


def enable_debug_logging() -> None:
    """Enable debug logging for all tacoreader modules."""
    setup_basic_logging(level=logging.DEBUG)


def disable_logging() -> None:
    """Disable all tacoreader logging."""
    logging.getLogger("tacoreader").setLevel(logging.CRITICAL + 1)
