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
        >>> detect_format("s3://bucket/__tacocat__")
        'tacocat'
        >>> detect_format("data/")
        'folder'
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


def is_local(path: str) -> bool:
    """Check if path is local filesystem."""
    return not is_remote(path)


def is_remote(path: str) -> bool:
    """
    Check if path is remote storage (S3, GCS, HTTP, etc).

    Args:
        path: Path to check

    Returns:
        True if remote, False if local
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
