import json
import pathlib
from typing import List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def local_file2dataframe(file: Union[str, pathlib.Path]):
    """Read the dataframe of tortilla file given a local path.

    Args:
        files (Union[str, pathlib.Path]): A local path pointing to the
            tortilla file.

    Returns:
        pd.DataFrame: The dataframe of the tortilla file.
    """
    with open(file, "rb") as f:
        static_bytes = f.read(50)

        # SPLIT the static bytes
        MB: bytes = static_bytes[:2]
        FO: bytes = static_bytes[2:10]
        FL: bytes = static_bytes[10:18]
        DF: str = static_bytes[18:42].strip().decode()
        # DP: str = static_bytes[42:50]

        if MB != b"#y":
            raise ValueError("You are not a tortilla 🫓 or a TACO 🌮")

        # Read the NEXT 8 bytes of the file
        footer_offset: int = int.from_bytes(FO, "little")

        # Seek to the FOOTER offset
        f.seek(footer_offset)

        # Select the FOOTER length
        # Read the FOOTER
        footer_length: int = int.from_bytes(FL, "little")
        dataframe = pq.read_table(pa.BufferReader(f.read(footer_length))).to_pandas()

        # Convert dataset to DataFrame
        dataframe["internal:file_format"] = DF
        dataframe["internal:mode"] = "local"
        dataframe["internal:subfile"] = dataframe.apply(
            lambda row: f"/vsisubfile/{row['tortilla:offset']}_{row['tortilla:length']},{file}",
            axis=1,
        )

    return dataframe


def local_files2dataframe(files: Union[List[str], List[pathlib.Path]]):
    """Read the dataframe of tortilla files given local paths.

    Args:
        files (Union[List[str], List[pathlib.Path]]): A list of local
            paths pointing to the tortilla files.

    Returns:
        pd.DataFrame: The dataframe of the tortilla file.
    """

    # Merge the dataframe of the files
    container = []
    for file in files:
        with open(file, "rb") as f:
            static_bytes = f.read(50)

            # SPLIT the static bytes
            MB: bytes = static_bytes[:2]
            FO: bytes = static_bytes[2:10]
            FL: bytes = static_bytes[10:18]
            DF: str = static_bytes[18:42].strip().decode()
            # DP: str = static_bytes[42:50]

            if MB != b"#y":
                raise ValueError("You are not a tortilla 🫓 or a TACO 🌮")

            # Read the NEXT 8 bytes of the file
            footer_offset: int = int.from_bytes(FO, "little")

            # Seek to the FOOTER offset
            f.seek(footer_offset)

            # Select the FOOTER length
            # Read the FOOTER
            footer_length: int = int.from_bytes(FL, "little")
            dataframe = pq.read_table(
                pa.BufferReader(f.read(footer_length))
            ).to_pandas()

            # Convert dataset to DataFrame
            dataframe["internal:file_format"] = DF
            dataframe["internal:mode"] = "local"
            dataframe["internal:subfile"] = dataframe.apply(
                lambda row: f"/vsisubfile/{row['tortilla:offset']}_{row['tortilla:length']},{file}",
                axis=1,
            )
            container.append(dataframe)

    return pd.concat(container, ignore_index=True)


def local_lazyfile2dataframe(
    offset: int, file: Union[str, pathlib.Path]
) -> pd.DataFrame:
    """Read the dataframe of tortilla file that is a subfile
    of a larger file.

    Args:
        offset (int): The offset of the subfile.
        file (Union[str, pathlib.Path]): A local path pointing to the
            main tortilla file.

    Returns:
        pd.DataFrame: The dataframe of the tortilla file.
    """

    with open(file, "rb") as f:
        # Seek to the OFFSET
        f.seek(offset)

        static_bytes = f.read(50)

        # SPLIT the static bytes
        MB: bytes = static_bytes[:2]
        FO: bytes = static_bytes[2:10]
        FL: bytes = static_bytes[10:18]
        DF: str = static_bytes[18:42].strip().decode()
        # DP: str = static_bytes[42:50]

        if MB != b"#y":
            raise ValueError("You are not a tortilla 🫓 or a TACO 🌮")

        # Read the NEXT 8 bytes of the file
        footer_offset: int = int.from_bytes(FO, "little") + offset

        # Seek to the FOOTER offset
        f.seek(footer_offset)

        # Select the FOOTER length
        # Read the FOOTER
        footer_length: int = int.from_bytes(FL, "little")
        dataframe = pq.read_table(pa.BufferReader(f.read(footer_length))).to_pandas()

        # Fix the offset
        dataframe["tortilla:offset"] = dataframe["tortilla:offset"] + offset

        # Convert dataset to DataFrame
        dataframe["internal:file_format"] = DF
        dataframe["internal:mode"] = "local"
        dataframe["internal:subfile"] = dataframe.apply(
            lambda row: f"/vsisubfile/{row['tortilla:offset']}_{row['tortilla:length']},{file}",
            axis=1,
        )

    return dataframe


def local_file2metadata(file: Union[str, pathlib.Path]) -> dict:
    """Read the dataframe of a taco file given a local path.

    Args:
        file (Union[str, pathlib.Path]): A local path pointing to the
            taco file.

    Returns:
        dict: The metadata of the taco file.
    """
    with open(file, "rb") as f:
        f.seek(50)

        # Read the Collection offset (CO)
        CO: int = int.from_bytes(f.read(8), "little")

        # Read the Collection length (CL)
        CL: int = int.from_bytes(f.read(8), "little")

        # Seek to the Collection offset
        f.seek(CO)

        # Read the Collection (JSON UTF-8 encoded)
        collection: dict = json.loads(f.read(CL).decode())

    return collection


def local_files2metadata(files: Union[List[str], List[pathlib.Path]]) -> dict:
    """Read the metadata of taco files given local paths.

    Args:
        files (Union[List[str], List[pathlib.Path]]): A list of local
            paths pointing to the taco files.

    Returns:
        dict: The metadata of the taco file.
    """

    return local_file2dataframe(files[0])
