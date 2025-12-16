"""
Format detection and path type utilities.

Main functions:
    detect_format: Basic format detection from path extension/structure
    detect_and_resolve_format: Advanced detection with TacoCat fallback logic
    is_remote: Check if path requires network access
    is_local: Check if path is local filesystem
"""

from typing import Literal

from tacoreader._constants import (
    ALL_VSI_PREFIXES,
    CLOUD_PROTOCOLS,
    COLLECTION_JSON,
    TACOCAT_FOLDER_NAME,
    TACOZIP_EXTENSIONS,
)
from tacoreader._logging import get_logger

logger = get_logger(__name__)


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


def detect_and_resolve_format(
    path: str,
) -> tuple[Literal["zip", "folder", "tacocat"], str]:
    """
    Detect format and resolve final path with automatic fallbacks.

    Resolution logic:
    1. Explicit formats are used as-is:
       - .tacozip → zip
       - .tacocat folder → tacocat

    2. Ambiguous folder paths check for TacoCat fallback:
       - If path has no COLLECTION.json in root
       - BUT has .tacocat/ subfolder with COLLECTION.json
       - → Redirect to .tacocat/ as tacocat format

    3. Otherwise use initial detection from detect_format()

    Returns:
        (format_type, resolved_path)

    Examples:
        detect_and_resolve_format("data.tacozip")
        → ("zip", "data.tacozip")

        detect_and_resolve_format("s2/.tacocat")
        → ("tacocat", "s2/.tacocat")

        detect_and_resolve_format("s2/")  # no COLLECTION.json, but has .tacocat/
        → ("tacocat", "s2/.tacocat")

        detect_and_resolve_format("dataset/")  # has COLLECTION.json
        → ("folder", "dataset/")
    """
    # Initial detection
    initial_format = detect_format(path)

    # Explicit formats: use as-is (no ambiguity)
    if initial_format in ("zip", "tacocat"):
        return initial_format, path

    # FOLDER format: check for TacoCat fallback
    if initial_format == "folder":
        resolved_format, resolved_path = _resolve_folder_format(path)
        return resolved_format, resolved_path

    return initial_format, path


def _resolve_folder_format(path: str) -> tuple[Literal["folder", "tacocat"], str]:
    """
    Resolve FOLDER format with TacoCat fallback.

    Strategy:
    1. Check if COLLECTION.json exists in root
    2. If not found, check for .tacocat/COLLECTION.json
    3. If .tacocat/ exists → return ("tacocat", "{path}/.tacocat")
    4. Otherwise → return ("folder", path) and let backend fail

    Args:
        path: Path to potential FOLDER dataset

    Returns:
        (format_type, resolved_path)
    """
    clean_path = path.rstrip("/")

    # Check root COLLECTION.json
    if _file_exists(clean_path, COLLECTION_JSON):
        logger.debug(f"Found COLLECTION.json in root: {clean_path}")
        return "folder", path

    # Check .tacocat/ subfolder
    tacocat_path = f"{clean_path}/{TACOCAT_FOLDER_NAME}"
    if _file_exists(tacocat_path, COLLECTION_JSON):
        logger.info("COLLECTION.json not in root, using .tacocat/ subfolder")
        return "tacocat", tacocat_path

    # Neither found - return folder (backend will fail with clear error)
    return "folder", path


def _file_exists(base_path: str, filename: str) -> bool:
    """
    Check if file exists (local or remote).

    Returns True if accessible, False otherwise.
    Does NOT raise exceptions.

    Args:
        base_path: Base directory path
        filename: Filename to check

    Returns:
        True if file exists and is accessible
    """
    try:
        if is_local(base_path):
            from pathlib import Path

            return (Path(base_path) / filename).exists()
        else:
            from tacoreader._remote_io import download_bytes

            download_bytes(base_path, filename)
            return True
    except Exception:
        return False


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
