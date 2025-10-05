import concurrent.futures
import mmap
import pathlib
from typing import Union

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from tacoreader.v1 import compile_utils


def compile(
    dataframe: pd.DataFrame,
    output: Union[str, pathlib.Path],
    chunk_size_iter: str = "100MB",
    nworkers: int = 4,
    overwrite: bool = True,
    quiet: bool = False,
    **storage_options
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
        **storage_options: Additional options for the storage backend.
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
    parallel_compile(
        dataframe, output, chunk_size_iter, nworkers, quiet, **storage_options
    )

    return output


def parallel_compile(
    dataframe: pd.DataFrame,
    output: str,
    chunk_size_iter: int,
    nworkers: int,
    quiet: bool,
    **storage_options
) -> pathlib.Path:
    """Prepare a subset of a Tortilla file and write it to a new file.

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
    def write_file(fs, fs_file, old_offset, length, new_offset):
        """read the file in chunks"""
        with fs.open(fs_file, "rb") as g:
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

            # Convert the VFS path to the original file path
            message = compile_utils.tortilla_message()
            dataframe["internal:filepath"] = dataframe["internal:subfile"].apply(
                lambda x: compile_utils.transform_from_gdal_vfs(
                    vfs_path=x.split(",")[-1]
                )
            )
            dataframe.sort_values("internal:filepath", inplace=True)

            # Write the DATA
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=nworkers
            ) as executor:
                futures = []
                for idx, item in dataframe.iterrows():
                    # TODO: Check with more detail what this function does
                    fs, fs_file = fsspec.core.url_to_fs(
                        item["internal:filepath"], **storage_options
                    )

                    old_offset = item["tortilla:offset"]
                    new_offset = item["tortilla:new_offset"]
                    length = item["tortilla:length"]
                    futures.append(
                        executor.submit(
                            write_file, fs, fs_file, old_offset, length, new_offset
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
