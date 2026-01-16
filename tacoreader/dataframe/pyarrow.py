"""PyArrow backend for TacoDataFrame.

Default backend, no extra dependencies. Wraps PyArrow Table with TACO navigation.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from tacoreader._constants import (
    DATAFRAME_DEFAULT_HEAD_ROWS,
    DATAFRAME_DEFAULT_TAIL_ROWS,
    DATAFRAME_MAX_REPR_ROWS,
    PADDING_PREFIX,
)
from tacoreader.dataframe.base import TacoDataFrame


class TacoDataFrameArrow(TacoDataFrame):
    """PyArrow Table wrapper with hierarchical navigation.

    Default backend - no external dependencies required.
    Provides minimal DataFrame API + TACO navigation via .read()

    Usage:
        tdf = dataset.data  # TacoDataFrameArrow by default
        child = tdf.read(0)  # Navigate hierarchy
        table = tdf.to_arrow()  # Export to PyArrow Table
    """

    @classmethod
    def from_arrow(
        cls,
        arrow_table: pa.Table,
        format_type: str,
        duckdb: Any = None,
        filtered_level_views: dict[int, str] | None = None,
        current_level: int = 0,
    ) -> TacoDataFrameArrow:
        """Create from PyArrow Table.

        Args:
            arrow_table: PyArrow Table from DuckDB query
            format_type: "zip", "folder", or "tacocat"
            duckdb: DuckDB connection for filtered view queries (optional)
            filtered_level_views: Dict mapping level -> filtered view name (optional)
            current_level: Current hierarchy level for navigation (default 0)

        Returns:
            TacoDataFrameArrow instance with cascade navigation support
        """
        return cls(
            data=arrow_table,
            format_type=format_type,
            duckdb=duckdb,
            filtered_level_views=filtered_level_views,
            current_level=current_level,
        )

    def __len__(self) -> int:
        return self._data.num_rows

    def __repr__(self) -> str:
        # Display first N rows, filter padding for cleaner output
        display_table = self._data.slice(0, min(DATAFRAME_MAX_REPR_ROWS, self._data.num_rows))

        if "id" in display_table.column_names:
            id_column = display_table.column("id")
            mask = pc.invert(pc.match_substring(id_column, PADDING_PREFIX))
            display_table = display_table.filter(mask)

        base_repr = str(display_table)

        total_rows = len(self)
        if total_rows > DATAFRAME_MAX_REPR_ROWS:
            info = f"\n[TacoDataFrame: {total_rows} rows (showing first {DATAFRAME_MAX_REPR_ROWS}), format={self._format_type}, backend=pyarrow]"
        else:
            info = f"\n[TacoDataFrame: {total_rows} rows, format={self._format_type}, backend=pyarrow]"

        return base_repr + info

    def __getitem__(self, key):
        """Standard PyArrow Table subscripting."""
        return self._data[key]

    @property
    def columns(self):
        return self._data.column_names

    @property
    def shape(self):
        return (self._data.num_rows, self._data.num_columns)

    def head(self, n: int = DATAFRAME_DEFAULT_HEAD_ROWS) -> pa.Table:
        return self._data.slice(0, min(n, self._data.num_rows))

    def tail(self, n: int = DATAFRAME_DEFAULT_TAIL_ROWS) -> pa.Table:
        start = max(0, self._data.num_rows - n)
        return self._data.slice(start)

    def _get_row(self, position: int) -> dict:
        """Row as dict for navigation. Used by base class .read()."""
        return self._data.to_pylist()[position]

    def _to_arrow_for_stats(self) -> pa.Table:
        """Convert to PyArrow for stats aggregation.

        No-op for PyArrow backend since we're already PyArrow.
        Required by base class for stats_mean(), stats_std(), etc.
        """
        return self._data

    def to_arrow(self) -> pa.Table:
        """Export as native PyArrow Table.

        Returns copy to prevent mutation of internal state.
        """
        return self._data
