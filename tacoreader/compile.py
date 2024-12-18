import concurrent.futures
import mmap
import pathlib
import re
from typing import Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import tqdm

from tacoreader import compile_utils


def compile(
    dataframe: pd.DataFrame,
    output: Union[str, pathlib.Path],
    chunk_size_iter: str = "100MB",
    nworkers: int = 4,
    overwrite: bool = True,
    quiet: bool = False,
) -> pathlib.Path:
    """Select a subset of a Tortilla or TACO file and write a new "small"
    Tortilla file. If you want to convert this new file to a TACO file,
    you can use the `tacotoolbox.tortilla2taco` function.

    Args:
        metadata (pd.DataFrame): A subset of the Tortilla file.
        output_folder (Union[str, pathlib.Path]): The folder where the Tortilla file
            will be saved. If the folder does not exist, it will be created.
        chunk_size_iter (int, optional): The writting chunk size. By default,
            it is 100MB. Faster computers can use a larger chunk size.
        nworkers (int, optional): The number of workers to use when writing
            the tortilla. Defaults to 4.
        overwrite (bool, optional): If True, the function overwrites the file if it
            already exists. By default, it is True.
        quiet (bool, optional): If True, the function does not print any
            message. By default, it is False.
    Returns:
        pathlib.Path: The path to the new Tortilla file.
    """
    # From human-readable to bytes
    chunk_size_iter: int = compile_utils.human2bytes(chunk_size_iter)

    # If the folder does not exist, create it
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file already exists
    if output.exists() and overwrite:
        output.unlink()

    # Remove the index from the previous dataset
    dataframe = dataframe.copy()
    dataframe.sort_values("tortilla:offset", inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    # Compile your tortilla
    mode = dataframe["internal:mode"].iloc[0]
    if mode == "local":
        compile_local(dataframe, output, chunk_size_iter, nworkers, quiet)
    elif mode == "online":
        compile_online(dataframe, output, chunk_size_iter, nworkers, quiet)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return output


def compile_local(
    dataframe: pd.DataFrame,
    output: str,
    chunk_size_iter: int,
    nworkers: int,
    quiet: bool,
) -> pathlib.Path:
    """Prepare a subset of a local Tortilla file and write it to a new file.

    Args:
        dataframe (pd.DataFrame): The dataframe of a Tortilla file.
        output (str): The path to the Tortilla file.
        chunk_size_iter (int): The size of the chunks to use when writing
            the tortilla.
        nworkers (int): The number of workers to use when writing the tortilla.
        quiet (bool): Whether to suppress the progress bar.

    Returns:
        pathlib.Path: The path to the new Tortilla file.
    """

    # Estimate the new offset
    dataframe.loc[:, "tortilla:new_offset"] = (
        dataframe["tortilla:length"].shift(1, fill_value=0).cumsum() + 200
    )

    # Create the new FOOTER
    # Remove the columns generated on-the-fly by the load function (internal fields)
    new_footer = dataframe.copy()
    new_footer.drop(
        columns=[
            "geometry",
            "internal:mode",
            "internal:subfile",
            "tortilla:offset",
        ],
        errors="ignore",
        inplace=True,
    )
    new_footer.rename(columns={"tortilla:new_offset": "tortilla:offset"}, inplace=True)

    # Create an in-memory Parquet file with BufferOutputStream
    with pa.BufferOutputStream() as sink:
        pq.write_table(
            pa.Table.from_pandas(new_footer),
            sink,
            compression="zstd",  # Highly efficient codec
            compression_level=22,  # Maximum compression for Zstandard
            use_dictionary=False,  # Optimizes for repeated values
        )
        # return a blob of the in-memory Parquet file as bytes
        # Obtain the FOOTER metadata
        FOOTER: bytes = sink.getvalue().to_pybytes()

    # Calculate the bytes of the data blob (DATA)
    bytes_counter: int = (
        dataframe.iloc[-1]["tortilla:new_offset"]
        + dataframe.iloc[-1]["tortilla:length"]
    )

    # Prepare the static bytes
    MB: bytes = b"#y"
    FL: bytes = len(FOOTER).to_bytes(8, "little")
    FO: bytes = int(bytes_counter).to_bytes(8, "little")
    DP: bytes = int(1).to_bytes(8, "little")

    # Create the tortilla file (empty)
    with open(output, "wb") as f:
        f.truncate(bytes_counter + len(FOOTER))

    # Define the function to write into the main file
    def write_file(file, old_offset, length, new_offset):
        """read the file in chunks"""
        with open(file, "rb") as g:
            g.seek(old_offset)
            while True:
                # Iterate over the file in chunks until the length is 0
                if chunk_size_iter > length:
                    chunk = g.read(length)
                    length = 0
                else:
                    chunk = g.read(chunk_size_iter)
                    length -= chunk_size_iter

                # Write the chunk into the mmap
                mm[new_offset : (new_offset + len(chunk))] = chunk
                new_offset += len(chunk)

                if length == 0:
                    break

    # Cook the tortilla 🫓
    with open(output, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
            # Write the magic number
            mm[:2] = MB

            # Write the FOOTER offset
            mm[2:10] = FO

            # Write the FOOTER length
            mm[10:18] = FL

            # Write the DATA PARTITIONS
            mm[18:26] = DP

            # Write the free space
            mm[26:200] = b"\0" * 174

            # Write the DATA
            message = compile_utils.tortilla_message()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=nworkers
            ) as executor:
                futures = []
                for _, item in dataframe.iterrows():
                    old_offset = item["tortilla:offset"]
                    new_offset = item["tortilla:new_offset"]
                    length = item["tortilla:length"]
                    file = dataframe["internal:subfile"].iloc[0].split(",")[-1]
                    futures.append(
                        executor.submit(
                            write_file, file, old_offset, length, new_offset
                        )
                    )

                # Wait for all futures to complete
                if not quiet:
                    list(
                        tqdm.tqdm(
                            concurrent.futures.as_completed(futures),
                            total=len(futures),
                            desc=message,
                            unit="file",
                        )
                    )
                else:
                    concurrent.futures.wait(futures)

            # Write the FOOTER
            mm[bytes_counter : (bytes_counter + len(FOOTER))] = FOOTER

    return pathlib.Path(output)


def compile_online(
    dataframe: pd.DataFrame,
    output: str,
    chunk_size_iter: int,
    nworkers: int,
    quiet: bool,
) -> pathlib.Path:
    """Prepare a subset of an online Tortilla or TACO file and write it to
    a new local file. If you want to convert this new file to a TACO file,
    you can use the `tacotoolbox.tortilla2taco` function.

    Args:
        metadata (pd.DataFrame): The metadata of the Tortilla file.
        output (str): The path to the Tortilla file.
        chunk_size_iter (int, optional): The size of the chunks to use
            when writing the tortilla.
        quiet (bool): Whether to suppress the progress bar.

    Returns:
        pathlib.Path: The path to the new Tortilla file.
    """

    # Get the URL of the file
    url_pattern = r"(ftp|https?)://[^\s,]+"
    url = re.search(url_pattern, dataframe["internal:subfile"].iloc[0]).group(0)

    # Calculate the new offsets
    dataframe["tortilla:new_offset"] = (
        dataframe["tortilla:length"].shift(1, fill_value=0).cumsum() + 200
    )

    # Create the new FOOTER
    # Remove the columns generated on-the-fly by the load function
    new_footer = dataframe.copy()
    new_footer.drop(
        columns=[
            "geometry",
            "internal:mode",
            "internal:subfile",
            "tortilla:offset",
        ],
        errors="ignore",
        inplace=True,
    )
    new_footer.rename(columns={"tortilla:new_offset": "tortilla:offset"}, inplace=True)

    # Create an in-memory Parquet file with BufferOutputStream
    with pa.BufferOutputStream() as sink:
        pq.write_table(
            pa.Table.from_pandas(new_footer),
            sink,
            compression="zstd",  # Highly efficient codec
            compression_level=22,  # Maximum compression for Zstandard
            use_dictionary=False,  # Optimizes for repeated values
        )
        # return a blob of the in-memory Parquet file as bytes
        # Obtain the FOOTER metadata
        FOOTER: bytes = sink.getvalue().to_pybytes()

    # Calculate the total size of the data
    bytes_counter: int = (
        dataframe.iloc[-1]["tortilla:new_offset"] + dataframe.iloc[-1]["tortilla:length"]
    )

    # Prepare static bytes
    MB: bytes = b"#y"
    FL: bytes = len(FOOTER).to_bytes(8, "little")
    FO: bytes = int(bytes_counter).to_bytes(8, "little")
    DP: bytes = int(1).to_bytes(8, "little")


    # Define the function to write into the main file
    def write_file_url(url, header, new_offset):
        """read the url in chunks"""
        try:
            with requests.get(url, headers=header, stream=True, timeout=10) as response:
                response.raise_for_status()
                # Write the downloaded data in chunks and update the progress bar
                for chunk in response.iter_content(chunk_size=chunk_size_iter):
                    if chunk:  # Filter out keep-alive new chunks
                        mm[new_offset : (new_offset + len(chunk))] = chunk
                        new_offset += len(chunk)
        except requests.exceptions.RequestException as e:
            print("ddd")
            raise requests.exceptions.RequestException(
                f"Error downloading {url} with header {header}: {e}"
            )
                                                                             
    # Create the tortilla file (empty)
    with open(output, "wb") as f:
        f.truncate(bytes_counter + len(FOOTER))

    # Cook the tortilla 🫓
    with open(output, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
            # Write the magic number
            mm[:2] = MB

            # Write the FOOTER offset
            mm[2:10] = FO

            # Write the FOOTER length
            mm[10:18] = FL

            # Write the DATA PARTITIONS
            mm[18:26] = DP

            # Write the free space
            mm[26:200] = b"\0" * 174

            # Write the DATA
            message = compile_utils.tortilla_message()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=nworkers
            ) as executor:
                futures = []
                for _, item in dataframe.iterrows():
                    # Get the offset, length, and URL
                    new_offset = item["tortilla:new_offset"]
                    url = dataframe["internal:subfile"].iloc[0].split(",")[-1].replace("/vsicurl/", "")
                    
                    # Create the header
                    next_start = item["tortilla:offset"]
                    next_end = next_start + item["tortilla:length"]
                    header = {"Range": f"bytes={next_start}-{next_end - 1}"}

                    futures.append(
                        executor.submit(
                            write_file_url, url, header, new_offset
                        )
                    )

                # Wait for all futures to complete
                if not quiet:
                    list(
                        tqdm.tqdm(
                            concurrent.futures.as_completed(futures),
                            total=len(futures),
                            desc=message,
                            unit="file",
                        )
                    )
                else:
                    concurrent.futures.wait(futures)

            # Write the FOOTER
            mm[bytes_counter : (bytes_counter + len(FOOTER))] = FOOTER
        
    return pathlib.Path(output)