"""
Format detection and path type utilities.

Low-level utilities with no tacoreader dependencies (avoids circular imports).
Pure string operations - no I/O, no side effects.
"""

from typing import Literal

from tacoreader._constants import (
    ALL_VSI_PREFIXES,
    CLOUD_PROTOCOLS,
    TACOCAT_FILENAME,
    TACOZIP_EXTENSIONS,
)


def detect_format(path: str) -> Literal["zip", "folder", "tacocat"]:
    """Detect TACO format from path"""

    clean_path = path.rstrip("/")
    filename = clean_path.split("/")[-1]

    # TacoCat has fixed name
    if filename == TACOCAT_FILENAME:
        return "tacocat"

    # ZIP format - check all valid extensions
    if filename.endswith(TACOZIP_EXTENSIONS):
        return "zip"

    # FOLDER format (default)
    return "folder"


def is_remote(path: str) -> bool:
    """Check if path requires network access"""

    # Combine cloud protocols and VSI prefixes, filtering out None values
    remote_prefixes = tuple(
        p for p in CLOUD_PROTOCOLS + ALL_VSI_PREFIXES if p is not None
    )
    return path.startswith(remote_prefixes)


def is_local(path: str) -> bool:
    """Check if path is local filesystem"""
    return not is_remote(path)
