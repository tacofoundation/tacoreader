"""
GDAL VSI path utilities.

Pure functions for converting between standard paths and GDAL Virtual File System paths.
No I/O operations - only string transformations.
"""

from tacoreader._constants import ALL_VSI_PREFIXES, PROTOCOL_MAPPINGS


def to_vsi_root(path: str) -> str:
    """
    Convert storage path to GDAL VSI format.

    Local paths unchanged, cloud/HTTP transformed to VSI.
    """
    # Build protocol conversion mapping: standard → vsi
    protocol_to_vsi = {}

    for proto_config in PROTOCOL_MAPPINGS.values():
        standard = proto_config["standard"]
        vsi = proto_config["vsi"]
        protocol_to_vsi[standard] = vsi

        # Add alternative protocol if exists (e.g., azure://)
        if "alt" in proto_config:
            protocol_to_vsi[proto_config["alt"]] = vsi

    # Check each protocol
    for standard_proto, vsi_prefix in protocol_to_vsi.items():
        if path.startswith(standard_proto):
            # Special handling for HTTP/HTTPS (vsicurl wraps entire URL)
            if standard_proto in ("http://", "https://"):
                return f"{vsi_prefix}{path}"
            else:
                return path.replace(standard_proto, vsi_prefix, 1)

    return path


def is_vsi_path(path: str) -> bool:
    """Check if path already in VSI format."""
    return path.startswith(ALL_VSI_PREFIXES)


def strip_vsi_prefix(path: str) -> str:
    """
    Remove VSI prefix, restore original protocol.

    Converts GDAL VSI paths back to standard protocol URLs.
    """
    # Handle vsicurl special case (unwrap entire URL)
    if path.startswith("/vsicurl/"):
        return path.replace("/vsicurl/", "", 1)

    # Build VSI → standard protocol mapping
    vsi_to_standard = {}

    for proto_config in PROTOCOL_MAPPINGS.values():
        vsi = proto_config["vsi"]
        standard = proto_config["standard"]
        vsi_to_standard[vsi] = standard

    # Check each VSI prefix
    for vsi_prefix, standard_proto in vsi_to_standard.items():
        if path.startswith(vsi_prefix):
            return path.replace(vsi_prefix, standard_proto, 1)

    return path


def parse_vsi_subfile(vsi_path: str) -> tuple[str, int, int]:
    """
    Parse /vsisubfile/ path to extract root, offset, size.

    Format: /vsisubfile/{offset}_{size},{root_path}
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
