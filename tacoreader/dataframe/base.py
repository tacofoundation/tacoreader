"""Abstract base class for TacoDataFrame backends.

Each backend (PyArrow, Polars, Pandas) implements this interface
with their native DataFrame APIs.

Cascade Filter Navigation:
    When cascade filters are applied (level>0), filtered views are stored
    in _filtered_level_views. The read() method checks this dict and queries
    DuckDB instead of reading physical __meta__ files, ensuring only filtered
    children are returned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tacoreader._exceptions import TacoNavigationError


class TacoDataFrame(ABC):
    """Abstract base for hierarchical DataFrame navigation.

    Each backend implements:
    - DataFrame operations (filter, select, head, tail, etc.)
    - Hierarchical navigation (read())
    - Properties (columns, shape)
    - Factory method (from_arrow classmethod)

    Shared across all backends:
    - read() navigation logic (with cascade filter support)
    - _read_meta() parquet reading
    - _read_from_filtered_view() DuckDB query for filtered children
    - _get_position() key resolution

    Statistics:
        Use TacoDataset.stats_*(band=...) for statistics aggregation.
    """

    def __init__(
        self,
        data: Any,
        format_type: str,
        duckdb: Any = None,
        filtered_level_views: dict[int, str] | None = None,
        current_level: int = 0,
    ):
        """Initialize with backend-specific data structure.

        Args:
            data: PyArrow Table, Polars DataFrame, or Pandas DataFrame
            format_type: "zip", "folder", or "tacocat"
            duckdb: DuckDB connection for filtered view queries (optional)
            filtered_level_views: Dict mapping level -> filtered view name (optional)
            current_level: Current hierarchy level for navigation (default 0)
        """
        self._data = data
        self._format_type = format_type
        self._duckdb = duckdb
        self._filtered_level_views = filtered_level_views or {}
        self._current_level = current_level

    @classmethod
    @abstractmethod
    def from_arrow(
        cls,
        arrow_table,
        format_type: str,
        duckdb: Any = None,
        filtered_level_views: dict[int, str] | None = None,
        current_level: int = 0,
    ) -> TacoDataFrame:
        """Convert PyArrow Table to backend-specific TacoDataFrame.

        This is the main entry point for creating TacoDataFrame instances
        from DuckDB query results (which are always PyArrow Tables).

        Args:
            arrow_table: PyArrow Table from DuckDB
            format_type: "zip", "folder", or "tacocat"
            duckdb: DuckDB connection for filtered view queries (optional)
            filtered_level_views: Dict mapping level -> filtered view name (optional)
            current_level: Current hierarchy level for navigation (default 0)

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
        """Get row as dictionary (backend-specific).

        PyArrow: table.to_pylist()[position]
        Polars: df.row(position, named=True)
        Pandas: df.iloc[position].to_dict()
        """
        pass

    @abstractmethod
    def _to_arrow_for_stats(self):
        """Convert current data to PyArrow for stats functions.

        Returns:
            PyArrow Table representation of current data
        """
        pass

    def _get_position(self, key: int | str) -> int:
        """Convert key to integer position.

        This method is SHARED - works with any backend via _to_arrow_for_stats().
        Uses PyArrow for efficient ID search.
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise TacoNavigationError(f"Position {key} out of range [0, {len(self) - 1}]")
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
        """Navigate to child level by position or ID.

        FILE samples: returns GDAL VSI path as string
        FOLDER samples:
          - If filtered views exist for child level -> query DuckDB
          - Otherwise -> read physical __meta__ file (original behavior)

        This method is SHARED - same logic for all backends.
        The cascade filter fix routes to _read_from_filtered_view() when
        _filtered_level_views contains the child level.
        """
        position = self._get_position(key)
        row = self._get_row(position)

        if row["type"] == "FILE":
            return row["internal:gdal_vsi"]

        # Check if we have a filtered view for child level
        child_level = self._current_level + 1
        if child_level in self._filtered_level_views and self._duckdb is not None:
            return self._read_from_filtered_view(row, child_level)

        # Fallback: read physical __meta__ file (original behavior)
        return self._read_meta(row)

    def _read_from_filtered_view(self, row: dict, child_level: int) -> TacoDataFrame:
        """Read children from DuckDB filtered view instead of physical __meta__.

        This is KEY for the cascade filter. When cascade filtering
        has created filtered views, we query DuckDB to get only the children
        that matched the filter, instead of reading the physical __meta__ which
        contains ALL children.

        Args:
            row: Parent row dict with internal:current_id (and source_file for tacocat)
            child_level: Level of children to read

        Returns:
            TacoDataFrame with filtered children only
        """
        from tacoreader._constants import (
            METADATA_CURRENT_ID,
            METADATA_PARENT_ID,
            METADATA_SOURCE_FILE,
        )

        parent_id = row[METADATA_CURRENT_ID]
        filtered_view = self._filtered_level_views[child_level]

        # Build query - TacoCat needs source_file in WHERE condition
        if self._format_type == "tacocat":
            source_file = row[METADATA_SOURCE_FILE]
            # Escape single quotes in source_file path
            source_file_escaped = source_file.replace("'", "''")
            query = f"""
                SELECT * FROM {filtered_view}
                WHERE "{METADATA_PARENT_ID}" = {parent_id}
                AND "{METADATA_SOURCE_FILE}" = '{source_file_escaped}'
            """
        else:
            query = f"""
                SELECT * FROM {filtered_view}
                WHERE "{METADATA_PARENT_ID}" = {parent_id}
            """

        children_table = self._duckdb.execute(query).fetch_arrow_table()

        # Return new TacoDataFrame with same filtered views for deeper navigation
        return self.__class__.from_arrow(
            children_table,
            self._format_type,
            duckdb=self._duckdb,
            filtered_level_views=self._filtered_level_views,
            current_level=child_level,
        )

    def _read_meta(self, row: dict) -> TacoDataFrame:
        """Read __meta__ file for FOLDER sample (original behavior).

        This method is SHARED but uses backend-specific from_arrow()
        to create the resulting TacoDataFrame.

        Used when no filtered views exist for child level (level=0 filters
        or no cascade filtering applied).

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
        # Propagate duckdb and filtered_views for potential deeper navigation
        return self.__class__.from_arrow(
            children_table,
            self._format_type,
            duckdb=self._duckdb,
            filtered_level_views=self._filtered_level_views,
            current_level=self._current_level + 1,
        )

    def _read_meta_from_archive(self, vsi_path: str):
        """Read __meta__ from ZIP or TacoCat archive.

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
            parquet_bytes = download_range(original_path, offset, size)  # pragma: no cover
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
        """Read __meta__ from FOLDER format.

        Reads Parquet from filesystem or remote storage, and builds
        direct paths to children using their id.

        Args:
            vsi_path: Direct path to folder (may end with /__meta__)
                      May contain VSI prefix like /vsicurl/ for remote paths.

        Returns:
            PyArrow Table with children + internal:gdal_vsi column

        Raises:
            TacoNavigationError: If children missing id column
        """
        from io import BytesIO
        from pathlib import Path

        import pyarrow as pa
        import pyarrow.parquet as pq

        from tacoreader._format import is_remote
        from tacoreader._remote_io import download_bytes
        from tacoreader._vsi import strip_vsi_prefix

        # Strip VSI prefix for I/O operations
        io_path = strip_vsi_prefix(vsi_path)

        # Read Parquet from __meta__
        if is_remote(io_path):
            if io_path.endswith("/__meta__"):
                meta_bytes = download_bytes(io_path)
            else:
                meta_bytes = download_bytes(io_path, "__meta__")

            children_table = pq.read_table(BytesIO(meta_bytes))
        else:
            meta_path = io_path if io_path.endswith("/__meta__") else str(Path(io_path) / "__meta__")

            children_table = pq.read_table(meta_path)

        # Strip /__meta__ suffix for path construction
        # Keep original vsi_path for children (GDAL needs /vsicurl/ prefix)
        parent_path = vsi_path
        if parent_path.endswith("/__meta__"):
            parent_path = parent_path[:-9]

        # Build paths using id (in __meta__, children are always direct subdirs/files)
        vsi_paths = []
        for i in range(children_table.num_rows):
            child_row = children_table.to_pylist()[i]

            if "id" not in child_row:
                raise TacoNavigationError(
                    f"Missing 'id' in FOLDER format.\n"
                    f"Row {i} (type={child_row.get('type', 'unknown')}) has no 'id'.\n"
                    f"Dataset may be corrupted or created with incompatible version."
                )

            vsi_paths.append(f"{parent_path}/{child_row['id']}")

        vsi_array = pa.array(vsi_paths, type=pa.string())
        return children_table.append_column("internal:gdal_vsi", vsi_array)
