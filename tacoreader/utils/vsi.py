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
        >>> strip_vsi_prefix("/vsicurl/https://example.com/data.zip")
        'https://example.com/data.zip'
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


def create_obstore_from_url(url: str):
    """
    Create obstore ObjectStore from URL.

    Handles different cloud storage protocols and HTTP(S) URLs by creating
    the appropriate obstore backend. For HTTP URLs, the entire URL is used
    as the store base since HTTPStore.from_url() handles the full path.

    Args:
        url: Cloud storage URL (s3://, gs://, az://, https://)
            - S3: "s3://bucket/path/to/file"
            - GCS: "gs://bucket/path/to/file"
            - Azure: "az://container/path/to/file"
            - HTTP: "https://example.com/path/to/file"

    Returns:
        obstore ObjectStore instance configured for the URL protocol

    Raises:
        ValueError: If URL scheme is not supported

    Examples:
        >>> store = create_obstore_from_url("s3://bucket/data.zip")
        >>> store = create_obstore_from_url("gs://bucket/data.zip")
        >>> store = create_obstore_from_url("https://example.com/data.zip")
    """
    import obstore as obs

    # S3 protocol
    if url.startswith("s3://"):
        return obs.store.S3Store.from_url(url)

    # Google Cloud Storage protocol
    elif url.startswith("gs://"):
        return obs.store.GCSStore.from_url(url)

    # Azure Blob Storage protocol
    elif url.startswith("az://") or url.startswith("azure://"):
        return obs.store.AzureStore.from_url(url)

    # HTTP/HTTPS protocol
    # HTTPStore.from_url() accepts the full file URL, not just base path
    elif url.startswith("http://") or url.startswith("https://"):
        return obs.store.HTTPStore.from_url(url)

    else:
        raise ValueError(
            f"Unsupported URL scheme: {url}\n"
            f"Supported schemes: s3://, gs://, az://, http://, https://"
        )


def extract_path_from_url(url: str) -> str:
    """
    Extract object path from full URL.

    Converts full cloud storage URLs into the relative path within the bucket
    or container. For HTTP URLs, returns empty string since HTTPStore.from_url()
    already handles the complete file path internally.

    Different cloud providers structure URLs differently:
    - S3/GCS: {protocol}://{bucket}/{path}
    - Azure: {protocol}://{container}/{path}
    - HTTP: Full URL is handled by store, no path extraction needed

    Args:
        url: Full URL like "s3://bucket/path/to/file.zip"

    Returns:
        Path within bucket/container, or empty string for HTTP URLs
        - S3/GCS: "path/to/file.zip"
        - Azure: "path/to/file.zip"
        - HTTP: "" (empty string)

    Examples:
        >>> extract_path_from_url("s3://bucket/path/to/file.zip")
        'path/to/file.zip'
        >>> extract_path_from_url("gs://my-bucket/data/archive.zip")
        'data/archive.zip'
        >>> extract_path_from_url("https://example.com/file.zip")
        ''
    """
    # S3 protocol: extract path after bucket name
    if url.startswith("s3://") or url.startswith("gs://"):
        parts = url[5:].split("/", 1)
        return parts[1] if len(parts) > 1 else ""

    # Azure Blob Storage protocol: extract path after container name
    elif url.startswith("az://") or url.startswith("azure://"):
        prefix_len = 5 if url.startswith("az://") else 8
        parts = url[prefix_len:].split("/", 1)
        return parts[1] if len(parts) > 1 else ""

    # HTTP/HTTPS protocol: return empty string
    # HTTPStore.from_url() already captures the full file URL,
    # so no additional path is needed for get_range() operations
    elif url.startswith("http://") or url.startswith("https://"):
        return ""

    else:
        raise ValueError(
            f"Unsupported URL scheme: {url}\n"
            f"Supported schemes: s3://, gs://, az://, http://, https://"
        )
