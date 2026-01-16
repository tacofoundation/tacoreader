import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union

import fsspec
import pandas as pd
from pyarrow import BufferReader
from pyarrow.parquet import read_table

from tacoreader.v1.loader_utils import (
    load_tacofoundation_datasets,
    transform_to_gdal_vfs,
)
from tacoreader.v1.TortillaDataFrame import TortillaDataFrame


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
        # Does the file start with our protocol -> tacofoundation:?
        if file.startswith("tacofoundation:"):
            datasets = load_tacofoundation_datasets()
            return load_files(datasets[file[15:]])

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
    return TortillaDataFrame(
        pd.concat(results, ignore_index=True).sort_values(by=["tortilla:id"])
    )
