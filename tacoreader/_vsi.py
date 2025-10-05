"""VSI path conversion utilities for GDAL compatibility.

This module converts various storage protocols (S3, GCS, HTTP, etc.)
to GDAL Virtual File System (VSI) paths.
"""


def to_vsi_root(path: str) -> str:
    """
    Convert any storage path to GDAL VSI-compatible root path.

    Transforms cloud storage URLs and HTTP paths into GDAL's VSI format.
    Local paths are returned unchanged.

    Args:
        path: Original path (local, S3, GCS, HTTP, etc.)

    Returns:
        VSI-compatible root path ready for GDAL operations

    Examples:
        Local paths (unchanged):
        >>> to_vsi_root("dataset.tacozip")
        'dataset.tacozip'
        >>> to_vsi_root("/home/user/data/")
        '/home/user/data/'

        S3 paths:
        >>> to_vsi_root("s3://bucket/data.tacozip")
        '/vsis3/bucket/data.tacozip'

        GCS paths:
        >>> to_vsi_root("gs://bucket/data.tacozip")
        '/vsigs/bucket/data.tacozip'

        Azure paths:
        >>> to_vsi_root("az://container/data.tacozip")
        '/vsiaz/container/data.tacozip'

        HTTP paths:
        >>> to_vsi_root("https://example.com/data.tacozip")
        '/vsicurl/https://example.com/data.tacozip'

        Already VSI (idempotent):
        >>> to_vsi_root("/vsis3/bucket/data.tacozip")
        '/vsis3/bucket/data.tacozip'
    """
    # S3 - Amazon Web Services
    if path.startswith("s3://"):
        return path.replace("s3://", "/vsis3/", 1)

    # GCS - Google Cloud Storage
    if path.startswith("gs://"):
        return path.replace("gs://", "/vsigs/", 1)

    # Azure Blob Storage
    if path.startswith("az://"):
        return path.replace("az://", "/vsiaz/", 1)

    # Alibaba Cloud OSS
    if path.startswith("oss://"):
        return path.replace("oss://", "/vsioss/", 1)

    # OpenStack Swift
    if path.startswith("swift://"):
        return path.replace("swift://", "/vsiswift/", 1)

    # HTTP/HTTPS - wrap with /vsicurl/
    if path.startswith("http://") or path.startswith("https://"):
        return f"/vsicurl/{path}"

    # Already VSI format or local path - return as-is
    return path


def is_vsi_path(path: str) -> bool:
    """
    Check if path is already in VSI format.

    Args:
        path: Path to check

    Returns:
        True if path starts with /vsi prefix

    Examples:
        >>> is_vsi_path("/vsis3/bucket/data.tacozip")
        True
        >>> is_vsi_path("/vsigs/bucket/dataset/")
        True
        >>> is_vsi_path("s3://bucket/data.tacozip")
        False
        >>> is_vsi_path("dataset.tacozip")
        False
    """
    vsi_prefixes = (
        "/vsis3/",
        "/vsigs/",
        "/vsiaz/",
        "/vsioss/",
        "/vsiswift/",
        "/vsicurl/",
        "/vsizip/",
        "/vsisubfile/",
    )
    return path.startswith(vsi_prefixes)


def strip_vsi_prefix(path: str) -> str:
    """
    Remove VSI prefix from path, returning original protocol.

    Useful for converting back to original URLs for obstore or other tools.

    Args:
        path: VSI path

    Returns:
        Path with original protocol restored

    Examples:
        >>> strip_vsi_prefix("/vsis3/bucket/data.tacozip")
        's3://bucket/data.tacozip'
        >>> strip_vsi_prefix("/vsigs/bucket/dataset/")
        'gs://bucket/dataset/'
        >>> strip_vsi_prefix("/vsicurl/https://example.com/data.zip")
        'https://example.com/data.zip'
        >>> strip_vsi_prefix("dataset.tacozip")
        'dataset.tacozip'
    """
    # /vsicurl/ wraps the full URL
    if path.startswith("/vsicurl/"):
        return path.replace("/vsicurl/", "", 1)

    # Other VSI prefixes map directly to protocols
    vsi_to_protocol = {
        "/vsis3/": "s3://",
        "/vsigs/": "gs://",
        "/vsiaz/": "az://",
        "/vsioss/": "oss://",
        "/vsiswift/": "swift://",
    }

    for vsi_prefix, protocol in vsi_to_protocol.items():
        if path.startswith(vsi_prefix):
            return path.replace(vsi_prefix, protocol, 1)

    # Not a VSI path or local path
    return path
