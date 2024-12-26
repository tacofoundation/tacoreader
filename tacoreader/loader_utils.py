from pathlib import Path

import geopandas as gpd
import shapely.wkt


def transform_to_gdal_vfs(path) -> str:
    """
    Transforms a remote path to a GDAL-compatible VFS path.

    Args:
        path (str): The original path (e.g., gs://, s3://, http://).

    Returns:
        str: A GDAL-compatible VFS path.
    """
    protocol_mapping = {
        "s3://": "/vsis3/",
        "gs://": "/vsigs/",
        "http://": "/vsicurl/http://",
        "https://": "/vsicurl/https://",
        "ftp://": "/vsicurl/ftp://",
        "hf://datasets/": "/vsihuggingface/",
    }

    if Path(path).exists():
        return path

    for protocol, vfs in protocol_mapping.items():
        if path.startswith(protocol):
            if protocol == "hf://datasets/":
                parts = path[len(protocol) :].split("/")
                if len(parts) < 3:
                    raise ValueError(
                        "Hugging Face path must include namespace, dataset, and file."
                    )
                namespace, dataset, file = parts[0], parts[1], "/".join(parts[2:])
                return f"https://huggingface.co/datasets/{namespace}/{dataset}/resolve/main/{file}"
            return path.replace(protocol, vfs)

    raise ValueError(
        f"Unsupported protocol: {path}. Supported protocols: "
        "s3, gs, az, oss, swift, http, https, ftp, and hf."
    )


def sort_columns_add_geometry(metadata):
    """Sort the columns of a metadata DataFrame.
    Also, convert the "stac:centroid" column to a geometry column.

    Args:
        metadata (pd.DataFrame): The metadata DataFrame.

    Returns:
        pd.DataFrame: The metadata DataFrame with sorted columns.
    """
    if "stac:centroid" in metadata.columns:
        metadata = gpd.GeoDataFrame(
            data=metadata,
            geometry=metadata["stac:centroid"].apply(shapely.wkt.loads),
            crs="EPSG:4326",
        )
    columns = metadata.columns
    prefixes = ["internal:", "tortilla:", "stac:", "rai:"]
    sorted_columns = [
        col for prefix in prefixes for col in columns if col.startswith(prefix)
    ]
    rest = [col for col in columns if col not in sorted_columns and col != "geometry"]
    columns = sorted_columns + rest + (["geometry"] if "geometry" in columns else [])
    return metadata[columns]
