import re
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from pyarrow import BufferReader
from pyarrow.parquet import read_table

from tacoreader.v1.loader_utils import transform_to_gdal_vfs


class TortillaDataSeries(pd.Series):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    @property
    def _constructor(self):
        return TortillaDataSeries

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

    def read(self):
        """Read data based on the row's tortilla:file_format."""
        if self["tortilla:file_format"] == "TORTILLA":
            offset, length, path = self.get_internal_path(self)
            return partial_load_file(self["tortilla:offset"], path)
        elif self["tortilla:file_format"] == "BYTES":
            # Obtain the offset, length and internal path
            offset, length, path = self.get_internal_path(self)
            # Fetch the bytes
            fs, fs_path = fsspec.core.url_to_fs(path)
            with fs.open(fs_path, "rb") as f:
                f.seek(int(offset))
                return f.read(int(length))
        else:
            return self["internal:subfile"]


class TortillaDataFrame(pd.DataFrame):
    def __init__(self, data=None, *args, **kwargs):
        # Apply sort_columns_add_geometry before passing
        # to the parent constructor
        dataclass: str = type(data).__name__
        if dataclass == "GeoDataFrame":
            data = data.drop("geometry", axis=1)
            data = sort_columns_add_geometry(data)
        elif dataclass == "DataFrame":
            data = sort_columns_add_geometry(data)
        elif dataclass == "TortillaDataFrame":
            pass
        else:
            raise ValueError(
                f"Unsupported data type: {dataclass}. Only GeoDataFrame, DataFrame, "
                "and TortillaDataFrame are supported."
            )
        super().__init__(data, *args, **kwargs)

    @property
    def _constructor(self):
        return TortillaDataFrame

    @property
    def _constructor_sliced(self):
        return TortillaDataSeries

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

    def to_geodataframe(self, inplace=False):
        """
        Convert the DataFrame to a GeoDataFrame.
        """
        try:
            from geopandas import GeoDataFrame, points_from_xy
        except ImportError:
            raise ImportError("geopandas is required to convert to GeoDataFrame")

        if inplace:
            self = GeoDataFrame(data=self, crs="EPSG:4326")
            return self

        return GeoDataFrame(
            data=self,
            geometry=points_from_xy(*parse_wkt_bulk(self["stac:centroid"])),
            crs="EPSG:4326",
        )

    def plot(self, *args, **kwargs):
        """Plot the GeoDataFrame."""
        self.to_geodataframe(inplace=False).plot(*args, **kwargs)


def sort_columns_add_geometry(metadata):
    """Sort the columns of a metadata DataFrame.
    Also, convert the "stac:centroid" column to a geometry column.

    Args:
        metadata (pd.DataFrame): The metadata DataFrame.

    Returns:
        pd.DataFrame: The metadata DataFrame with sorted columns.
    """
    columns = metadata.columns
    prefixes = ["internal:", "tortilla:", "stac:", "rai:"]
    sorted_columns = [
        col for prefix in prefixes for col in columns if col.startswith(prefix)
    ]
    rest = [col for col in columns if col not in sorted_columns]
    columns = sorted_columns + rest
    return metadata[columns]


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


def parse_wkt_bulk(wkt_series):
    # Concatenate all WKT strings into one large string, separated by spaces
    # Use slicing and string operations to extract coordinates efficiently
    # Example: "POINT (-97.6126 46.09836)" -> "-97.6126 46.09836" -> ["-97.6126", "46.09836"]
    points = " ".join(wkt_series).replace("POINT (", "").replace(")", "").split()

    # Convert the points list into a NumPy array and reshape it to (2, n)
    coords = np.array(points, dtype=float).reshape(-1, 2).T
    return coords
