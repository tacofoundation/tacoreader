"""
GDAL VSI path utilities.

Pure functions for converting between standard paths and GDAL Virtual File System paths.
No I/O operations - only string transformations.
"""


def to_vsi_root(path: str) -> str:
    """
    Convert storage path to GDAL VSI format.

    Local paths unchanged, cloud/HTTP transformed to VSI.

    Examples:
        to_vsi_root("s3://bucket/data.zip") -> "/vsis3/bucket/data.zip"
        to_vsi_root("https://example.com/data.zip") -> "/vsicurl/https://example.com/data.zip"
        to_vsi_root("dataset.zip") -> "dataset.zip"
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
    """Check if path already in VSI format."""
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
    Remove VSI prefix, restore original protocol.

    Examples:
        strip_vsi_prefix("/vsis3/bucket/data.zip") -> "s3://bucket/data.zip"
        strip_vsi_prefix("/vsicurl/https://example.com") -> "https://example.com"
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
    Parse /vsisubfile/ path to extract root, offset, size.

    Format: /vsisubfile/{offset}_{size},{root_path}

    Example:
        parse_vsi_subfile("/vsisubfile/1024_5000,/vsis3/bucket/data.zip")
        -> ("/vsis3/bucket/data.zip", 1024, 5000)
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
