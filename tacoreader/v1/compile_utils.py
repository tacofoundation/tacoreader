import random
from pathlib import Path


def tortilla_message() -> str:
    """Get a random tortilla message"""

    tortilla_messages = [
        "Making a tortilla",
        "Making a tortilla 🫓",
        "Cooking a tortilla",
        "Making a tortilla 🫓",
        "Working on a tortilla",
        "Working on a tortilla 🫓",
        "Rolling out a tortilla",
        "Rolling out a tortilla 🫓",
        "Baking a tortilla",
        "Baking a tortilla 🫓",
        "Grilling a tortilla",
        "Grilling a tortilla 🫓",
        "Toasting a tortilla",
        "Toasting a tortilla 🫓",
    ]

    # Randomly accessing a message
    random_message = random.choice(tortilla_messages)
    return random_message


def human2bytes(size_str: str) -> int:
    """
    Converts a human-readable size string (e.g., "100MB") into bytes.
    Supported units: KB, MB, GB, TB, PB.
    """
    units = {"KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12, "PB": 10**15}
    size_str = size_str.strip().upper()

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                value = float(size_str[: -len(unit)].strip())
                return int(value * multiplier)
            except ValueError:
                raise ValueError(f"Invalid size value in '{size_str}'.")
    raise ValueError(
        f"Unsupported unit in '{size_str}'. Supported units are: {', '.join(units.keys())}."
    )


def transform_from_gdal_vfs(vfs_path: str) -> str:
    """
    Transforms a GDAL-compatible VFS path to its original remote path.

    Args:
        vfs_path (str): The GDAL-compatible VFS path.

    Returns:
        str: The original path (e.g., gs://, s3://, http://).

    """
    vfs_mapping = {
        "/vsis3/": "s3://",
        "/vsigs/": "gs://",
        "/vsicurl/http://": "http://",
        "/vsicurl/https://": "https://",
        "/vsicurl/ftp://": "ftp://",
    }

    for vfs, protocol in vfs_mapping.items():
        if vfs_path.startswith(vfs):
            return vfs_path.replace(vfs, protocol)

    # If no match was found, assume it's a local path
    if Path(vfs_path).exists():
        return vfs_path

    raise ValueError(
        f"Unsupported GDAL VFS path: {vfs_path}. Ensure the VFS path corresponds to a known protocol."
    )
