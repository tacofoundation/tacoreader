"""
Format detection and path type utilities.

Low-level utilities with no tacoreader dependencies (avoids circular imports).
"""

from typing import Literal

from tacoreader._constants import TACOCAT_FILENAME


def detect_format(path: str) -> Literal["zip", "folder", "tacocat"]:
    """
    Detect TACO format from path.

    Examples:
        detect_format("data.tacozip") -> "zip"
        detect_format("s3://bucket/__TACOCAT__") -> "tacocat"
        detect_format("data/") -> "folder"
    """
    clean_path = path.rstrip("/")
    filename = clean_path.split("/")[-1]

    # TacoCat has fixed name
    if filename == TACOCAT_FILENAME:
        return "tacocat"

    # ZIP format
    if filename.endswith(".tacozip") or filename.endswith(".zip"):
        return "zip"

    # FOLDER format (default)
    return "folder"


def is_remote(path: str) -> bool:
    """
    Check if path requires network access.

    Includes HTTP/HTTPS, cloud storage (S3/GCS/Azure), and GDAL VSI paths.
    """
    remote_prefixes = (
        # Cloud storage
        "s3://",
        "gs://",
        "az://",
        "azure://",
        "oss://",
        "swift://",
        # HTTP
        "http://",
        "https://",
        # GDAL VSI
        "/vsis3/",
        "/vsigs/",
        "/vsiaz/",
        "/vsioss/",
        "/vsiswift/",
        "/vsicurl/",
    )
    return path.startswith(remote_prefixes)


def is_local(path: str) -> bool:
    """Check if path is local filesystem."""
    return not is_remote(path)
