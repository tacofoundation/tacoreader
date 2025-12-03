"""
TacoDataFrame - hierarchical DataFrame wrapper for TACO navigation.

Wraps PyArrow Table with TACO-specific methods for hierarchical navigation
and statistics aggregation. Provides .read() for traversing nested structures
and stats_*() methods for aggregating pre-computed metadata.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tacoreader._constants import (
    DATAFRAME_DEFAULT_HEAD_ROWS,
    DATAFRAME_DEFAULT_TAIL_ROWS,
    NAVIGATION_REQUIRED_COLUMNS,
    PADDING_PREFIX,
    PROTECTED_COLUMNS,
)


def _is_protected_column(column_name: str) -> bool:
    """
    Check if column is protected and cannot be modified.

    Protected columns required for hierarchical navigation.
    Modifying these breaks .read() functionality.
    """
    return column_name in PROTECTED_COLUMNS or column_name.startswith("internal:")


def _validate_navigation_columns(table: pa.Table, operation: str) -> None:
    """
    Validate critical columns for navigation are present.

    Required: id, type, internal:gdal_vsi
    """
    current_cols = set(table.column_names)
    missing = NAVIGATION_REQUIRED_COLUMNS - current_cols

    if missing:
        raise ValueError(
            f"Operation '{operation}' removed required columns: {sorted(missing)}\n"
            f"\n"
            f"Required for navigation: id, type, internal:gdal_vsi\n"
            f"Current columns: {sorted(current_cols)}\n"
            f"\n"
            f"To drop these, convert to Arrow first:\n"
            f"  table = tdf.to_arrow().select(['custom_column'])"
        )


def _validate_not_protected(column_name: str) -> None:
    """
    Validate column not protected and can be modified.

    Protected: id, type, internal:*
    """
    if _is_protected_column(column_name):
        raise ValueError(
            f"Cannot modify protected column: '{column_name}'\n"
            f"\n"
            f"Protected: id, type, internal:*\n"
            f"Use a different name for derived columns."
        )


class TacoDataFrame:
    """
    Hierarchical DataFrame wrapper for TACO navigation.

    Wraps PyArrow Table with TACO-specific functionality:
    - Hierarchical navigation via read()
    - Statistics aggregation via stats_*()
    - Export to Arrow Table via to_arrow()

    Protected columns (cannot be modified):
        - id, type: Core navigation
        - internal:*: All internal columns

    Examples:
        tdf = dataset.data

        # Navigate
        child = tdf.read(0)  # By position
        child = tdf.read("sample_001")  # By ID
        vsi_path = tdf.read(5)  # Returns str if FILE

        # Export
        table = tdf.to_arrow()

        # Stats
        mean_values = tdf.stats_mean()
    """

    def __init__(self, data: pa.Table, format_type: str):
        """Initialize with materialized PyArrow Table."""
        self._data = data
        self._format_type = format_type

    def __len__(self) -> int:
        """Number of rows."""
        return self._data.num_rows

    def __repr__(self) -> str:
        """String representation with format info."""
        display_table = self._data

        # Filter out padding for cleaner display
        if "id" in self._data.column_names:
            id_column = self._data.column("id")
            mask = pc.invert(pc.match_substring(id_column, PADDING_PREFIX))
            display_table = self._data.filter(mask)

        # Simple string representation of Arrow Table
        base_repr = str(display_table)
        info = f"\n[TacoDataFrame: {len(self)} rows, format={self._format_type}]"
        return base_repr + info

    def __getitem__(self, key):
        """
        Subscripting like PyArrow Table.

        Supports: column selection, slicing.
        """
        return self._data[key]

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def columns(self):
        """Column names."""
        return self._data.column_names

    @property
    def shape(self):
        """Shape tuple: (rows, columns)."""
        return (self._data.num_rows, self._data.num_columns)

    def head(self, n: int = DATAFRAME_DEFAULT_HEAD_ROWS) -> pa.Table:
        """First n rows as PyArrow Table."""
        return self._data.slice(0, min(n, self._data.num_rows))

    def tail(self, n: int = DATAFRAME_DEFAULT_TAIL_ROWS) -> pa.Table:
        """Last n rows as PyArrow Table."""
        start = max(0, self._data.num_rows - n)
        return self._data.slice(start)

    def to_arrow(self) -> pa.Table:
        """Export as PyArrow Table."""
        return self._data

    # ========================================================================
    # HIERARCHICAL NAVIGATION
    # ========================================================================

    def read(self, key: int | str) -> TacoDataFrame | str:
        """
        Navigate to child level by position or ID.

        FILE samples: returns GDAL VSI path as string
        FOLDER samples: reads __meta__ and returns TacoDataFrame with children
        """
        position = self._get_position(key)
        row = self._data.to_pylist()[position]

        if row["type"] == "FILE":
            return row["internal:gdal_vsi"]

        return self._read_meta(row)

    def _get_position(self, key: int | str) -> int:
        """Convert key to integer position."""
        if isinstance(key, int):
            if key < 0 or key >= self._data.num_rows:
                raise IndexError(f"Position {key} out of range [0, {self._data.num_rows-1}]")
            return key

        # Search by ID
        ids = self._data.column("id").to_pylist()
        if key not in ids:
            raise KeyError(f"ID '{key}' not found")

        return ids.index(key)

    def _read_meta(self, row: dict) -> TacoDataFrame:  # noqa: C901
        """
        Read __meta__ file for FOLDER sample.

        Handles:
        - /vsisubfile/ paths (ZIP, TacoCat): Read Parquet from offset
        - Direct paths (FOLDER): Read Parquet from filesystem
        """
        from tacoreader.remote_io import download_bytes, download_range
        from tacoreader.utils.format import is_remote
        from tacoreader.utils.vsi import parse_vsi_subfile, strip_vsi_prefix

        vsi_path = row["internal:gdal_vsi"]

        if vsi_path.startswith("/vsisubfile/"):
            # ZIP or TacoCat: Read Parquet from offset
            root_path, offset, size = parse_vsi_subfile(vsi_path)
            original_path = strip_vsi_prefix(root_path)

            if is_remote(original_path):
                parquet_bytes = download_range(original_path, offset, size)
            else:
                with open(original_path, "rb") as f:
                    f.seek(offset)
                    parquet_bytes = f.read(size)

            children_table = pq.read_table(BytesIO(parquet_bytes))

            # Build VSI paths
            vsi_paths = []
            for i in range(children_table.num_rows):
                child_row = children_table.to_pylist()[i]
                if "internal:offset" in child_row and "internal:size" in child_row:
                    child_offset = child_row["internal:offset"]
                    child_size = child_row["internal:size"]
                    vsi_paths.append(
                        f"/vsisubfile/{child_offset}_{child_size},{root_path}"
                    )
                else:
                    vsi_paths.append("")

            vsi_array = pa.array(vsi_paths, type=pa.string())
            children_table = children_table.append_column("internal:gdal_vsi", vsi_array)

        else:
            # FOLDER: Read Parquet from __meta__
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

            # Build paths
            vsi_paths = []
            for i in range(children_table.num_rows):
                child_row = children_table.to_pylist()[i]
                if "internal:relative_path" in child_row:
                    relative = child_row["internal:relative_path"]
                    vsi_paths.append(f"{parent_path}/{relative}")
                elif "id" in child_row:
                    vsi_paths.append(f"{parent_path}/{child_row['id']}")
                else:
                    vsi_paths.append("")

            vsi_array = pa.array(vsi_paths, type=pa.string())
            children_table = children_table.append_column("internal:gdal_vsi", vsi_array)

        return TacoDataFrame(
            data=children_table,
            format_type=self._format_type,
        )

    # ========================================================================
    # STATISTICS AGGREGATION
    # ========================================================================

    def stats_categorical(self) -> np.ndarray:
        """Aggregate categorical probabilities using weighted average."""
        from tacoreader.utils.stats import stats_categorical

        return stats_categorical(self._data)

    def stats_mean(self) -> np.ndarray:
        """Aggregate means using weighted average."""
        from tacoreader.utils.stats import stats_mean

        return stats_mean(self._data)

    def stats_std(self) -> np.ndarray:
        """Aggregate standard deviations using pooled std formula."""
        from tacoreader.utils.stats import stats_std

        return stats_std(self._data)

    def stats_min(self) -> np.ndarray:
        """Aggregate minimums (global min across all rows)."""
        from tacoreader.utils.stats import stats_min

        return stats_min(self._data)

    def stats_max(self) -> np.ndarray:
        """Aggregate maximums (global max across all rows)."""
        from tacoreader.utils.stats import stats_max

        return stats_max(self._data)

    def stats_p25(self) -> np.ndarray:
        """Aggregate 25th percentiles using simple average."""
        from tacoreader.utils.stats import stats_p25

        return stats_p25(self._data)

    def stats_p50(self) -> np.ndarray:
        """Aggregate 50th percentiles (median) using simple average."""
        from tacoreader.utils.stats import stats_p50

        return stats_p50(self._data)

    def stats_median(self) -> np.ndarray:
        """Aggregate medians using simple average (alias for stats_p50)."""
        from tacoreader.utils.stats import stats_p50

        return stats_p50(self._data)

    def stats_p75(self) -> np.ndarray:
        """Aggregate 75th percentiles using simple average."""
        from tacoreader.utils.stats import stats_p75

        return stats_p75(self._data)

    def stats_p95(self) -> np.ndarray:
        """Aggregate 95th percentiles using simple average."""
        from tacoreader.utils.stats import stats_p95

        return stats_p95(self._data)