import json
from pathlib import Path
from typing import List, Union

import fsspec


def load_metadata(
    file: Union[str, Path, List[Path], List[str]], **storage_options
) -> dict:
    """Load the metadata of a tortilla or taco file.

    Args:
        file (Union[str, pathlib.Path, List]): The path of
            the taco file. If the file is split into multiple
            parts, a list of paths is accepted. Also, multiple
            parts can be read by putting a asterisk (*) at the end
            of the file name. For example, "file*.tortilla". In this
            case, the function will create a list will all the partitions
            before the reading process.

    Returns:
        dict: The metadata of the taco file.
    """
    if isinstance(file, (str, Path)):
        return file2metadata(file, **storage_options)
    elif isinstance(file, list):
        return files2metadata(file, **storage_options)
    else:
        raise ValueError("Invalid file type: must be a string or a list of strings.")


def file2metadata(path: Union[str, Path], **storage_options) -> dict:
    """Read the dataframe of a taco file given a local path.

    Args:
        path (Union[str, pathlib.Path]): A local path pointing to the
            taco file.

    Returns:
        dict: The metadata of the taco file.
    """
    fs, fs_path = fsspec.core.url_to_fs(path, **storage_options)

    with fs.open(fs_path, "rb") as f:
        # Get all the bytes from 0 to 42
        header: bytes = f.read(42)

        # Read the magic number
        magic, COb, CLb = header[:2], header[26:34], header[34:42]
        CO: int = int.from_bytes(COb, "little")
        CL: int = int.from_bytes(CLb, "little")

        # Check if the file is a taco file
        if magic not in b"WX":
            raise ValueError("Invalid file type: must be a TACO 🌮")

        # Seek to the Collection offset
        f.seek(CO)

        # Read the Collection (JSON UTF-8 encoded)
        collection: dict = json.loads(f.read(CL).decode())

    return collection


def files2metadata(path: Union[str, Path], **storage_options) -> dict:
    """Read the metadata of a taco file given a list of paths from the
    same dataset.

    Args:
        path (Union[str, pathlib.Path]): A list of paths pointing to the
            taco file.

    Returns:
        dict: The metadata of the taco file.
    """
    # Get the list of files
    return file2metadata(path[0], **storage_options)
