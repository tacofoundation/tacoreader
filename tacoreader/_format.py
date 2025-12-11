"""
Format detection and path type utilities.
"""

from typing import Literal

from tacoreader._constants import (
    ALL_VSI_PREFIXES,
    CLOUD_PROTOCOLS,
    TACOCAT_FOLDER_NAME,
    TACOZIP_EXTENSIONS,
)


def detect_format(path: str) -> Literal["zip", "folder", "tacocat"]:
    """
    Detect TACO format from path.

    Rules:
    - .tacocat folder → TacoCat consolidated format
    - .tacozip or .zip → ZIP format
    - Everything else → FOLDER format

    Examples:
        detect_format("data.tacozip") → "zip"
        detect_format("/path/to/.tacocat") → "tacocat"
        detect_format("/path/to/.tacocat/") → "tacocat"
        detect_format("s3://bucket/.tacocat") → "tacocat"
        detect_format("dataset/") → "folder"
    """
    clean_path = path.rstrip("/")

    # Normalize path separators for cross-platform support
    normalized = clean_path.replace("\\", "/")
    parts = normalized.split("/")
    last_part = parts[-1] if parts else ""

    # TacoCat: folder named .tacocat (exact match)
    if last_part == TACOCAT_FOLDER_NAME:
        return "tacocat"

    # ZIP format: check all valid extensions
    if last_part.endswith(TACOZIP_EXTENSIONS):
        return "zip"

    # FOLDER format (default)
    return "folder"


def is_remote(path: str) -> bool:
    """
    Check if path requires network access.

    Returns True for cloud storage (s3://, gs://, etc.) and VSI paths.
    Returns False for local filesystem paths.
    """
    remote_prefixes = tuple(
        p for p in CLOUD_PROTOCOLS + ALL_VSI_PREFIXES if p is not None
    )
    return path.startswith(remote_prefixes)


def is_local(path: str) -> bool:
    """Check if path is local filesystem."""
    return not is_remote(path)
