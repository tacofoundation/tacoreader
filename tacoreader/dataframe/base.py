"""
Abstract base class for TacoDataFrame backends.

Each backend (PyArrow, Polars, Pandas) implements this interface
with their native DataFrame APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tacoreader._exceptions import TacoNavigationError


class TacoDataFrame(ABC):
    """
    Abstract base for hierarchical DataFrame navigation.

    Each backend implements:
    - DataFrame operations (filter, select, head, tail, etc.)
    - Hierarchical navigation (read())
    - Statistics aggregation (stats_*)
    - Properties (columns, shape)
    - Factory method (from_arrow classmethod)

    Shared across all backends:
    - read() navigation logic
    - _read_meta() parquet reading
    - _get_position() key resolution
    """

    def __init__(self, data: Any, format_type: str):
        """
        Initialize with backend-specific data structure.

        Args:
            data: PyArrow Table, Polars DataFrame, or Pandas DataFrame
            format_type: "zip", "folder", or "tacocat"
        """
        self._data = data
        self._format_type = format_type

    @classmethod
    @abstractmethod
    def from_arrow(cls, arrow_table, format_type: str) -> TacoDataFrame:
        """
        Convert PyArrow Table to backend-specific TacoDataFrame.

        This is the main entry point for creating TacoDataFrame instances
        from DuckDB query results (which are always PyArrow Tables).

        Args:
            arrow_table: PyArrow Table from DuckDB
            format_type: "zip", "folder", or "tacocat"

        Returns:
            Backend-specific TacoDataFrame instance
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Number of rows."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation with format info."""
        pass

    @abstractmethod
    def __getitem__(self, key):
        """Subscripting (row/column access)."""
        pass

    @property
    @abstractmethod
    def columns(self):
        """Column names."""
        pass

    @property
    @abstractmethod
    def shape(self):
        """Shape tuple: (rows, columns)."""
        pass

    @abstractmethod
    def head(self, n: int):
        """First n rows."""
        pass

    @abstractmethod
    def tail(self, n: int):
        """Last n rows."""
        pass

    @abstractmethod
    def _get_row(self, position: int) -> dict:
        """
        Get row as dictionary (backend-specific).

        PyArrow: table.to_pylist()[position]
        Polars: df.row(position, named=True)
        Pandas: df.iloc[position].to_dict()
        """
        pass

    @abstractmethod
    def _to_arrow_for_stats(self):
        """
        Convert current data to PyArrow for stats functions.

        Used when calling stats_*() methods - all stats functions
        expect PyArrow Tables internally.

        Returns:
            PyArrow Table representation of current data
        """
        pass

    def _get_position(self, key: int | str) -> int:
        """
        Convert key to integer position.

        This method is SHARED - works with any backend via _to_arrow_for_stats().
        Uses PyArrow for efficient ID search.
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise TacoNavigationError(
                    f"Position {key} out of range [0, {len(self)-1}]"
                )
            return key

        # Search by ID using PyArrow
        import pyarrow.compute as pc

        arrow_table = self._to_arrow_for_stats()

        if "id" not in arrow_table.column_names:
            raise TacoNavigationError("Cannot search by ID: 'id' column not found")

        id_col = arrow_table.column("id")
        mask = pc.equal(id_col, key)

        # Find first True index
        for i, val in enumerate(mask):
            if val.as_py():
                return i

        raise TacoNavigationError(f"ID '{key}' not found")

    def read(self, key: int | str) -> TacoDataFrame | str:
        """
        Navigate to child level by position or ID.

        FILE samples: returns GDAL VSI path as string
        FOLDER samples: reads __meta__ and returns TacoDataFrame with children

        This method is SHARED - same logic for all backends.
        """
        position = self._get_position(key)
        row = self._get_row(position)

        if row["type"] == "FILE":
            return row["internal:gdal_vsi"]

        return self._read_meta(row)

    def _read_meta(self, row: dict) -> TacoDataFrame:
        """
        Read __meta__ file for FOLDER sample.

        This method is SHARED but uses backend-specific from_arrow()
        to create the resulting TacoDataFrame.

        Handles:
        - /vsisubfile/ paths (ZIP, TacoCat): Read Parquet from offset
        - Direct paths (FOLDER): Read Parquet from filesystem

        Raises:
            TacoNavigationError: If required metadata columns are missing
        """
        vsi_path = row["internal:gdal_vsi"]

        if vsi_path.startswith("/vsisubfile/"):
            children_table = self._read_meta_from_archive(vsi_path)
        else:
            children_table = self._read_meta_from_folder(vsi_path)

        # Convert PyArrow Table to current backend using factory method
        return self.__class__.from_arrow(children_table, self._format_type)

    def _read_meta_from_archive(self, vsi_path: str):
        """
        Read __meta__ from ZIP or TacoCat archive.

        Parses /vsisubfile/ path, reads Parquet from offset, and builds
        VSI paths for children pointing back to archive.

        Args:
            vsi_path: /vsisubfile/{offset}_{size},{archive_path}

        Returns:
            PyArrow Table with children + internal:gdal_vsi column

        Raises:
            TacoNavigationError: If children missing internal:offset or internal:size
        """
        from io import BytesIO

        import pyarrow as pa
        import pyarrow.parquet as pq

        from tacoreader._format import is_remote
        from tacoreader._remote_io import download_range
        from tacoreader._vsi import parse_vsi_subfile, strip_vsi_prefix

        # Parse /vsisubfile/ path
        root_path, offset, size = parse_vsi_subfile(vsi_path)
        original_path = strip_vsi_prefix(root_path)

        # Read Parquet from offset
        if is_remote(original_path):
            parquet_bytes = download_range(original_path, offset, size)
        else:
            with open(original_path, "rb") as f:
                f.seek(offset)
                parquet_bytes = f.read(size)

        children_table = pq.read_table(BytesIO(parquet_bytes))

        # Build VSI paths with strict validation
        vsi_paths = []
        for i in range(children_table.num_rows):
            child_row = children_table.to_pylist()[i]

            # Missing required columns for ZIP/TacoCat
            if "internal:offset" not in child_row or "internal:size" not in child_row:
                raise TacoNavigationError(
                    f"Missing required metadata in ZIP/TacoCat format.\n"
                    f"Row {i} (id={child_row.get('id', 'unknown')}) is missing "
                    f"'internal:offset' or 'internal:size'.\n"
                    f"Dataset may be corrupted or created with incompatible version."
                )

            child_offset = child_row["internal:offset"]
            child_size = child_row["internal:size"]
            vsi_paths.append(f"/vsisubfile/{child_offset}_{child_size},{root_path}")

        vsi_array = pa.array(vsi_paths, type=pa.string())
        return children_table.append_column("internal:gdal_vsi", vsi_array)

    def _read_meta_from_folder(self, vsi_path: str):
        """
        Read __meta__ from FOLDER format.

        Reads Parquet from filesystem or remote storage, and builds
        direct paths to children.

        Args:
            vsi_path: Direct path to folder (may end with /__meta__)

        Returns:
            PyArrow Table with children + internal:gdal_vsi column

        Raises:
            TacoNavigationError: If children missing both internal:relative_path and id
        """
        from io import BytesIO
        from pathlib import Path

        import pyarrow as pa
        import pyarrow.parquet as pq

        from tacoreader._format import is_remote
        from tacoreader._remote_io import download_bytes

        # Read Parquet from __meta__
        if is_remote(vsi_path):
            if vsi_path.endswith("/__meta__"):
                meta_bytes = download_bytes(vsi_path)
            else:
                meta_bytes = download_bytes(vsi_path, "__meta__")

            children_table = pq.read_table(BytesIO(meta_bytes))
        else:
            meta_path = (
                vsi_path
                if vsi_path.endswith("/__meta__")
                else str(Path(vsi_path) / "__meta__")
            )

            children_table = pq.read_table(meta_path)

        # Strip /__meta__ suffix for path construction
        parent_path = vsi_path
        if parent_path.endswith("/__meta__"):
            parent_path = parent_path[:-9]

        # Build paths with strict validation
        vsi_paths = []
        for i in range(children_table.num_rows):
            child_row = children_table.to_pylist()[i]

            if "internal:relative_path" in child_row:
                relative = child_row["internal:relative_path"]
                vsi_paths.append(f"{parent_path}/{relative}")
            elif "id" in child_row:
                vsi_paths.append(f"{parent_path}/{child_row['id']}")
            else:
                # Missing path information
                raise TacoNavigationError(
                    f"Missing path information in FOLDER format.\n"
                    f"Row {i} (type={child_row.get('type', 'unknown')}) has neither "
                    f"'internal:relative_path' nor 'id'.\n"
                    f"Dataset may be corrupted or created with incompatible version."
                )

        vsi_array = pa.array(vsi_paths, type=pa.string())
        return children_table.append_column("internal:gdal_vsi", vsi_array)

    def stats_categorical(self):
        """Aggregate categorical probabilities using weighted average."""
        from tacoreader.dataframe._stats import stats_categorical

        arrow_table = self._to_arrow_for_stats()
        return stats_categorical(arrow_table)

    def stats_mean(self):
        """Aggregate means using weighted average."""
        from tacoreader.dataframe._stats import stats_mean

        arrow_table = self._to_arrow_for_stats()
        return stats_mean(arrow_table)

    def stats_std(self):
        """Aggregate standard deviations using pooled std formula."""
        from tacoreader.dataframe._stats import stats_std

        arrow_table = self._to_arrow_for_stats()
        return stats_std(arrow_table)

    def stats_min(self):
        """Aggregate minimums (global min across all rows)."""
        from tacoreader.dataframe._stats import stats_min

        arrow_table = self._to_arrow_for_stats()
        return stats_min(arrow_table)

    def stats_max(self):
        """Aggregate maximums (global max across all rows)."""
        from tacoreader.dataframe._stats import stats_max

        arrow_table = self._to_arrow_for_stats()
        return stats_max(arrow_table)

    def stats_p25(self):
        """Aggregate 25th percentiles using simple average."""
        from tacoreader.dataframe._stats import stats_p25

        arrow_table = self._to_arrow_for_stats()
        return stats_p25(arrow_table)

    def stats_p50(self):
        """Aggregate 50th percentiles (median) using simple average."""
        from tacoreader.dataframe._stats import stats_p50

        arrow_table = self._to_arrow_for_stats()
        return stats_p50(arrow_table)

    def stats_median(self):
        """Aggregate medians using simple average (alias for stats_p50)."""
        from tacoreader.dataframe._stats import stats_p50

        arrow_table = self._to_arrow_for_stats()
        return stats_p50(arrow_table)

    def stats_p75(self):
        """Aggregate 75th percentiles using simple average."""
        from tacoreader.dataframe._stats import stats_p75

        arrow_table = self._to_arrow_for_stats()
        return stats_p75(arrow_table)

    def stats_p95(self):
        """Aggregate 95th percentiles using simple average."""
        from tacoreader.dataframe._stats import stats_p95

        arrow_table = self._to_arrow_for_stats()
        return stats_p95(arrow_table)
