"""
Format detection and path type utilities.

Low-level utilities for detecting TACO dataset formats and path types.
No dependencies on other tacoreader modules to avoid circular imports.
"""

from typing import Literal


def detect_format(path: str) -> Literal["zip", "folder", "tacocat"]:
    """
    Detect TACO dataset format from path.

    Args:
        path: Path to dataset (local or remote)

    Returns:
        Format type: "zip", "folder", or "tacocat"

    Examples:
        >>> detect_format("data.tacozip")
        'zip'
        >>> detect_format("s3://bucket/__TACOCAT__")
        'tacocat'
        >>> detect_format("data/")
        'folder'
        >>> detect_format("https://example.com/data.tacozip")
        'zip'
    """
    clean_path = path.rstrip("/")
    filename = clean_path.split("/")[-1]

    # TacoCat has fixed name
    if filename == "__TACOCAT__":
        return "tacocat"

    # ZIP format
    if filename.endswith(".tacozip") or filename.endswith(".zip"):
        return "zip"

    # FOLDER format (default)
    return "folder"


def is_remote(path: str) -> bool:
    """
    Check if path is remote storage (requires network access).

    Remote paths include HTTP/HTTPS, cloud storage URLs (S3, GCS, Azure),
    and GDAL VSI paths.

    Args:
        path: Path to check

    Returns:
        True if remote, False if local

    Examples:
        >>> is_remote("data.tacozip")
        False
        >>> is_remote("/absolute/path/data.tacozip")
        False
        >>> is_remote("https://example.com/data.tacozip")
        True
        >>> is_remote("s3://bucket/data.tacozip")
        True
        >>> is_remote("gs://bucket/data.tacozip")
        True
        >>> is_remote("az://container/data.tacozip")
        True
        >>> is_remote("/vsis3/bucket/data.tacozip")
        True
    """
    remote_prefixes = (
        # Cloud storage protocols
        "s3://",
        "gs://",
        "az://",
        "azure://",
        "oss://",
        "swift://",
        # HTTP protocols
        "http://",
        "https://",
        # GDAL VSI paths
        "/vsis3/",
        "/vsigs/",
        "/vsiaz/",
        "/vsioss/",
        "/vsiswift/",
        "/vsicurl/",
    )
    return path.startswith(remote_prefixes)


def is_local(path: str) -> bool:
    """
    Check if path is local filesystem.

    Args:
        path: Path to check

    Returns:
        True if local, False if remote

    Examples:
        >>> is_local("data.tacozip")
        True
        >>> is_local("s3://bucket/data.tacozip")
        False
    """
    return not is_remote(path)