import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union

import fsspec
import geopandas as gpd
import pandas as pd
from pyarrow import BufferReader
from pyarrow.parquet import read_table

from tacoreader.datasets import datasets
from tacoreader.loader_utils import (sort_columns_add_geometry,
                                     transform_to_gdal_vfs)


def load(file: Union[str, List[str], Path, List[Path]], **storage_options):
    """
    Load a TACO file and parse its metadata.

    Args:
        file (str): The path to the TACO file.
        storage_options (dict): Additional options for the storage

    Returns:
        pd.DataFrame: A dataframe containing the parsed metadata.
    """
    if isinstance(file, (str, Path)):
        # Check if the dataset is inside the taco foundation
        if file in datasets.keys():
            return load(datasets[file])
        return load_file(file, **storage_options)
    elif isinstance(file, list):
        return load_files(file, **storage_options)
    else:
        raise ValueError("Invalid file type: must be a string or a list of strings.")


def load_file(path, **storage_options) -> pd.DataFrame:
    """
    Load the TACO file content and parse its metadata.

    Args:
        path (str): The path to the TACO file.
        storage_options (dict): Additional options for the storage

    Returns:
        pd.DataFrame: A dataframe containing the parsed metadata.
    """

    vfs_path: str = transform_to_gdal_vfs(str(path))
    fs, fs_path = fsspec.core.url_to_fs(path, **storage_options)

    # Magick read for any backend
    with fs.open(fs_path, "rb") as f:
        header = f.read(18)
        magic, footer_offset, footer_length = header[:2], header[2:10], header[10:18]

        if magic not in {b"#y", b"WX"}:
            raise ValueError("Invalid file type: must be a Tortilla 🫓 or a TACO 🌮")

        footer_offset = int.from_bytes(footer_offset, "little")
        footer_length = int.from_bytes(footer_length, "little")

        f.seek(footer_offset)
        dataframe = read_table(BufferReader(f.read(footer_length))).to_pandas()

    # Set internal metadata
    dataframe["internal:subfile"] = dataframe.apply(
        lambda row: f"/vsisubfile/{row['tortilla:offset']}_{row['tortilla:length']},{vfs_path}",
        axis=1,
    )
    return TortillaDataFrame(dataframe)


def partial_load_file(offset: int, path: str, **storage_options) -> pd.DataFrame:
    """
    Load a TACO file partially and parse its metadata.

    Args:
        offset (int): The byte offset where the reading process will start.
        path (str): The path to the TACO file.
        storage_options (dict): Additional options for the storage

    Returns:
        pd.DataFrame: A dataframe containing the parsed metadata.
    """

    vfs_path: str = transform_to_gdal_vfs(path)
    fs, fs_path = fsspec.core.url_to_fs(path, **storage_options)

    # Magick read for any backend
    with fs.open(fs_path, "rb") as f:
        # Seek to the OFFSET
        f.seek(offset)

        static_bytes = f.read(18)

        # SPLIT the static bytes
        MB: bytes = static_bytes[:2]
        FO: bytes = static_bytes[2:10]
        FL: bytes = static_bytes[10:18]
        # DP: str = static_bytes[42:50]

        if MB not in {b"#y", b"WX"}:
            raise ValueError(
                "Invalid file type: must be either a Tortilla 🫓 or a TACO 🌮"
            )

        # Read the NEXT 8 bytes of the file
        footer_offset: int = int.from_bytes(FO, "little") + offset

        # Seek to the FOOTER offset
        f.seek(footer_offset)

        # Select the FOOTER length
        # Read the FOOTER
        footer_length: int = int.from_bytes(FL, "little")
        dataframe = read_table(BufferReader(f.read(footer_length))).to_pandas()

    # Fix the offset
    dataframe["tortilla:offset"] = dataframe["tortilla:offset"] + offset

    # Convert dataset to DataFrame
    dataframe["internal:subfile"] = dataframe.apply(
        lambda row: f"/vsisubfile/{row['tortilla:offset']}_{row['tortilla:length']},{vfs_path}",
        axis=1,
    )

    return TortillaDataFrame(dataframe)


def load_files(files: list, **storage_options) -> pd.DataFrame:
    """
    Load multiple TACO files in parallel and concatenate their metadata.

    Args:
        files (list): List of file paths or URIs.

    Returns:
        pd.DataFrame: A dataframe containing the concatenated metadata.
    """
    with ThreadPoolExecutor(max_workers=min(len(files), os.cpu_count())) as executor:
        futures = {
            executor.submit(load_file, file, **storage_options): file for file in files
        }
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")
    return TortillaDataFrame(pd.concat(results, ignore_index=True))


class TortillaDataFrame(gpd.GeoDataFrame):

    @property
    def _constructor(self):
        return TortillaDataFrame

    @staticmethod
    def get_internal_path(row):
        """
        Extract offset, length, and path from a row's internal subfile information.
        """
        pattern: re.Pattern = re.compile(r"/vsisubfile/(\d+)_(\d+),(.+)")
        offset, length, path = pattern.match(row["internal:subfile"]).groups()

        # Adjust path for curl files
        # Remove VFS prefix from path (supporting multiple protocols)
        if Path(path).is_file():
            path = path
        else:
            if path.startswith("/vsicurl/"):
                path = path[9:]
            elif path.startswith("/vsis3/"):
                path = path[7:]
            elif path.startswith("/vsigs/"):
                path = path[7:]
            elif path.startswith("/vsifs/"):
                path = path[7:]
            elif path.startswith("/vsiaz/"):
                path = path[7:]
            elif path.startswith("/vsioss/"):
                path = path[8:]
            elif path.startswith("/vsiswift/"):
                path = path[10:]
            else:
                raise ValueError(f"Unsupported GDAL VFS prefix: {path}")

        return int(offset), int(length), path

    def read(self, idx):
        """
        Read data based on the row's tortilla:file_format.
        """
        row = self.iloc[idx]
        if row["tortilla:file_format"] == "TORTILLA":
            offset, length, path = self.get_internal_path(row)
            return partial_load_file(row["tortilla:offset"], path)
        elif row["tortilla:file_format"] == "BYTES":
            # Obtain the offset, length and internal path
            offset, length, path = self.get_internal_path(row)
            # Fetch the bytes
            fs, fs_path = fsspec.core.url_to_fs(path)
            with fs.open(fs_path, "rb") as f:
                f.seek(int(offset))
                return f.read(int(length))
        else:
            return row["internal:subfile"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Automatically apply sort_columns_add_geometry
        sorted_metadata = sort_columns_add_geometry(self)
        self.__dict__.update(sorted_metadata.__dict__)


