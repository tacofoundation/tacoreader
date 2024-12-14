import pathlib
import re
import urllib
from typing import List, Tuple, Union

import geopandas as gpd
import requests
import shapely.wkt


def is_valid_url(url: Union[str, List[str]]) -> bool:
    """Check if a URL or list of URLs is valid.

    Args:
        url (Union[str, List[str]]): The URL(s) to check. It can
            be a single URL or a list of URLs.

    Returns:
        bool: True if all URLs are valid, False otherwise.
    """
    if isinstance(url, pathlib.Path):
        return False

    if isinstance(url, list):
        return all(is_valid_url(single_url) for single_url in url)

    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme in ["http", "https"], result.netloc])
    except ValueError:
        return False


def split_name_and_path(file: Union[str, pathlib.Path]) -> Tuple[str, str]:
    """
    Split a file path or URL into its name and path components.

    Args:
        file (Union[str, pathlib.Path]): The input file or URL.

    Returns:
        Tuple[str, str]: A tuple containing the name and path.
    """
    if isinstance(file, (str, pathlib.Path)):
        # Handle URLs
        parsed = urllib.parse.urlparse(file)
        if parsed.scheme in {"http", "https"}:
            # Extract the name and path from a URL
            name = pathlib.PurePosixPath(parsed.path).name
            path = f"{parsed.scheme}://{parsed.netloc}{pathlib.PurePosixPath(parsed.path).parent.as_posix()}"
        else:
            # Handle local file paths
            file_path = pathlib.Path(file)
            name = file_path.name
            path = file_path.parent.as_posix()
        return name, path
    else:
        raise ValueError("Input must be a string or pathlib.Path.")


def snippet2files(
    file: Union[str, pathlib.Path, List[str], List[pathlib.Path]]
) -> Union[List[pathlib.Path], List[str], str, pathlib.Path]:
    """Convert snippets of a multi-part file to a list of files.

    Args:
        file (Union[str, pathlib.Path, List[str], List[pathlib.Path]]): A file, a
            list of files, or a snippet of a multi-part file.

    Raises:
        FileNotFoundError: If a file is not found.
        FileNotFoundError: If a partial file is missing.

    Returns:
        List[pathlib.Path]: A list of files or a single path. The path can be a
            local path or a URL.
    """

    # Check if the file is a list
    if isinstance(file, list):
        files = file
    else:
        # Does the file finish with: *.tortilla?
        if re.match(r".*\*\.tortilla$", str(file)):

            # Split in name and path
            name, path = split_name_and_path(file)

            # Get the filename without the snippet (i.e., *.tortilla)
            filename: str = re.sub(r"\*\.tortilla$", "", name)

            # check if file is a url
            if is_valid_url(path):
                # It is expected that the file is a multi-part file in the same url path
                dumbfile: str = f"{path}/{filename}.0000.part.tortilla"
                headers = {"Range": "bytes=42-50"}
                response: requests.Response = requests.get(dumbfile, headers=headers)
                npartitions: int = int.from_bytes(response.content, "little")

                # Check if all parts are there
                files = []
                for d in range(npartitions):
                    partial_file = f"{path}/{filename}.{str(d).zfill(4)}.part.tortilla"
                    files.append(partial_file)
            else:
                # Get all files in the directory
                file = pathlib.Path(file)
                filename = pathlib.Path(filename)

                # It is expected that the file is a multi-part file in the same directory
                dumbfile: pathlib.Path = file.resolve().parent / (
                    filename.stem + ".0000.part.tortilla"
                )

                # check how many parts are there
                with open(dumbfile, "rb") as f:
                    f.seek(42)
                    npartitions: int = int.from_bytes(f.read(8), "little")

                # Check if all parts are there
                files = []
                for d in range(npartitions):
                    partial_file = filename.with_suffix(
                        f".{str(d).zfill(4)}.part.tortilla"
                    )
                    if partial_file.exists():
                        files.append(partial_file)  # Add the file to the list
                    else:
                        raise FileNotFoundError(f"Missing partial file: {partial_file}")

        else:
            files = file

    return files


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
