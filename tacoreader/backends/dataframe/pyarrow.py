"""
PyArrow backend for TacoDataFrame.

Default backend with no extra dependencies. Wraps PyArrow Table
with TACO-specific hierarchical navigation.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from tacoreader._constants import (
    DATAFRAME_DEFAULT_HEAD_ROWS,
    DATAFRAME_DEFAULT_TAIL_ROWS,
    PADDING_PREFIX,
)
from tacoreader.backends.dataframe.base import TacoDataFrame

# Maximum rows to display in __repr__
MAX_REPR_ROWS = 100


class TacoDataFrameArrow(TacoDataFrame):
    """
    PyArrow implementation of TacoDataFrame.

    Wraps PyArrow Table with TACO-specific functionality:
    - Hierarchical navigation via read()
    - Statistics aggregation via stats_*()
    - Export to Arrow Table via to_arrow()

    This is the default backend (no extra dependencies).

    Examples:
        tdf = dataset.data  # Returns TacoDataFrameArrow by default

        # Navigate
        child = tdf.read(0)  # By position
        child = tdf.read("sample_001")  # By ID
        vsi_path = tdf.read(5)  # Returns str if FILE

        # Export
        table = tdf.to_arrow()

        # Stats
        mean_values = tdf.stats_mean()
    """

    # ========================================================================
    # FACTORY METHOD
    # ========================================================================

    @classmethod
    def from_arrow(cls, arrow_table: pa.Table, format_type: str) -> TacoDataFrameArrow:
        """
        Create TacoDataFrameArrow from PyArrow Table.

        This is a no-op for PyArrow backend since we already have
        the correct type.

        Args:
            arrow_table: PyArrow Table from DuckDB
            format_type: Storage format ("zip", "folder", "tacocat")

        Returns:
            TacoDataFrameArrow instance
        """
        return cls(arrow_table, format_type)

    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================

    def __len__(self) -> int:
        """Number of rows."""
        return self._data.num_rows

    def __repr__(self) -> str:
        """
        String representation with format info.
        """
        # First slice to max display rows
        display_table = self._data.slice(0, min(MAX_REPR_ROWS, self._data.num_rows))

        # Then filter padding only from these few rows
        if "id" in display_table.column_names:
            id_column = display_table.column("id")
            mask = pc.invert(pc.match_substring(id_column, PADDING_PREFIX))
            display_table = display_table.filter(mask)

        # Simple string representation of Arrow Table
        base_repr = str(display_table)

        # Add metadata footer
        total_rows = len(self)
        if total_rows > MAX_REPR_ROWS:
            info = f"\n[TacoDataFrame: {total_rows} rows (showing first {MAX_REPR_ROWS}), format={self._format_type}]"
        else:
            info = f"\n[TacoDataFrame: {total_rows} rows, format={self._format_type}]"

        return base_repr + info

    def __getitem__(self, key):
        """
        Subscripting like PyArrow Table.

        Supports: column selection, slicing.
        """
        return self._data[key]

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

    def _get_row(self, position: int) -> dict:
        """
        Get row as dictionary.

        Args:
            position: Row index

        Returns:
            Dictionary with column names as keys
        """
        return self._data.to_pylist()[position]

    def _to_arrow_for_stats(self) -> pa.Table:
        """
        Convert to PyArrow for stats functions.

        For PyArrow backend, this is a no-op since we already
        have PyArrow Table.

        Returns:
            PyArrow Table (self._data)
        """
        return self._data

    # ========================================================================
    # PYARROW-SPECIFIC METHODS
    # ========================================================================

    def to_arrow(self) -> pa.Table:
        """
        Export as PyArrow Table.

        Returns:
            PyArrow Table (copy of internal data)
        """
        return self._data
