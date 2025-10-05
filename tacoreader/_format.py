from typing import Literal


def detect_format(path: str) -> Literal["zip", "folder"]:
    """
    Detect if path points to ZIP or FOLDER format TACO dataset.

    ZIP format has .tacozip or .zip extension.
    FOLDER format is a directory structure.

    Args:
        path: File path (with or without VSI prefix)

    Returns:
        "zip" if .tacozip/.zip extension, "folder" otherwise

    Examples:
        >>> detect_format("dataset.tacozip")
        'zip'
        >>> detect_format("/vsis3/bucket/data.tacozip")
        'zip'
        >>> detect_format("dataset/")
        'folder'
        >>> detect_format("s3://bucket/data/")
        'folder'
    """
    clean_path = path.rstrip("/")
    filename = clean_path.split("/")[-1]

    if filename.endswith(".tacozip") or filename.endswith(".zip"):
        return "zip"
    return "folder"


def is_local(path: str) -> bool:
    """
    Check if path points to local filesystem.

    Returns True if path doesn't start with a remote protocol prefix.

    Args:
        path: File path to check

    Returns:
        True if local, False if remote

    Examples:
        >>> is_local("dataset.tacozip")
        True
        >>> is_local("/home/user/data/")
        True
        >>> is_local("s3://bucket/data.tacozip")
        False
    """
    return not is_remote(path)


def is_remote(path: str) -> bool:
    """
    Check if path points to remote storage (S3, GCS, HTTP, etc).

    Detects both original protocols (s3://, gs://, http://) and
    GDAL VSI prefixes (/vsis3/, /vsigs/, /vsicurl/).

    Args:
        path: File path to check

    Returns:
        True if remote, False if local

    Examples:
        >>> is_remote("s3://bucket/data.tacozip")
        True
        >>> is_remote("/vsis3/bucket/data/")
        True
        >>> is_remote("dataset.tacozip")
        False
    """
    remote_prefixes = (
        "s3://",
        "gs://",
        "http://",
        "https://",
        "/vsis3/",
        "/vsigs/",
        "/vsicurl/",
        "/vsiaz/",
        "/vsioss/",
        "/vsiswift/",
    )
    return path.startswith(remote_prefixes)
