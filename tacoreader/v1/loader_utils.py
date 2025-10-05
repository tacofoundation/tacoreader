from pathlib import Path

import requests


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
        f"File does not exist or unsupported protocol: {path}. Supported"
        " protocols: s3, gs, az, oss, swift, http, https, ftp, and hf."
    )


def load_tacofoundation_datasets() -> dict:
    """
    Load a TACO file from the TACO Foundation dataset.

    Returns:
        dict: A dictionary containing the TACO Foundation datasets.
    """
    dataset = "https://huggingface.co/datasets/tacofoundation/main/raw/main/tacos.json"
    with requests.get(dataset) as r:
        datasets = r.json()
    return datasets
