"""
GDAL VSI path utilities.

Pure functions for converting between standard paths and GDAL's Virtual File System (VSI) paths.
No I/O operations - only string transformations.
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
        >>> to_vsi_root("s3://bucket/data.tacozip")
        '/vsis3/bucket/data.tacozip'
        >>> to_vsi_root("gs://bucket/data.tacozip")
        '/vsigs/bucket/data.tacozip'
        >>> to_vsi_root("https://example.com/data.zip")
        '/vsicurl/https://example.com/data.zip'
        >>> to_vsi_root("dataset.tacozip")
        'dataset.tacozip'
    """
    if path.startswith("s3://"):
        return path.replace("s3://", "/vsis3/", 1)

    if path.startswith("gs://"):
        return path.replace("gs://", "/vsigs/", 1)

    if path.startswith("az://"):
        return path.replace("az://", "/vsiaz/", 1)

    if path.startswith("oss://"):
        return path.replace("oss://", "/vsioss/", 1)

    if path.startswith("swift://"):
        return path.replace("swift://", "/vsiswift/", 1)

    if path.startswith("http://") or path.startswith("https://"):
        return f"/vsicurl/{path}"

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
        >>> is_vsi_path("/vsicurl/https://example.com/data.zip")
        True
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

    Args:
        path: VSI path

    Returns:
        Path with original protocol restored

    Examples:
        >>> strip_vsi_prefix("/vsis3/bucket/data.tacozip")
        's3://bucket/data.tacozip'
        >>> strip_vsi_prefix("/vsigs/bucket/data.tacozip")
        'gs://bucket/data.tacozip'
        >>> strip_vsi_prefix("/vsicurl/https://example.com/data.zip")
        'https://example.com/data.zip'
        >>> strip_vsi_prefix("dataset.tacozip")
        'dataset.tacozip'
    """
    if path.startswith("/vsicurl/"):
        return path.replace("/vsicurl/", "", 1)

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

    return path


def parse_vsi_subfile(vsi_path: str) -> tuple[str, int, int]:
    """
    Parse /vsisubfile/ path to extract root path, offset, and size.

    Args:
        vsi_path: VSI subfile path like "/vsisubfile/1024_5000,/path/to/file.zip"

    Returns:
        Tuple of (root_path, offset, size)

    Raises:
        ValueError: If path format is invalid

    Examples:
        >>> parse_vsi_subfile("/vsisubfile/1024_5000,/vsis3/bucket/data.zip")
        ('/vsis3/bucket/data.zip', 1024, 5000)
        >>> parse_vsi_subfile("/vsisubfile/2737343662_3075,/home/user/file.zip")
        ('/home/user/file.zip', 2737343662, 3075)
    """
    if not vsi_path.startswith("/vsisubfile/"):
        raise ValueError(
            f"Invalid VSI subfile path: must start with '/vsisubfile/', got: {vsi_path}"
        )

    content = vsi_path[len("/vsisubfile/") :]

    if "," not in content:
        raise ValueError(
            f"Invalid VSI subfile path: missing comma separator, got: {vsi_path}"
        )

    offset_size_part, root_path = content.split(",", 1)

    if "_" not in offset_size_part:
        raise ValueError(
            f"Invalid VSI subfile path: missing underscore in offset_size, got: {vsi_path}"
        )

    try:
        offset_str, size_str = offset_size_part.split("_", 1)
        offset = int(offset_str)
        size = int(size_str)
    except ValueError as e:
        raise ValueError(
            f"Invalid VSI subfile path: offset or size not integers, got: {vsi_path}"
        ) from e

    return root_path, offset, size