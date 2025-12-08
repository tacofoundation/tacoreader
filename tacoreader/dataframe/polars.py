"""
Polars backend for TacoDataFrame.

Wraps Polars DataFrame with TACO-specific hierarchical navigation.
Requires polars package: pip install polars
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tacoreader._constants import (
    DATAFRAME_DEFAULT_HEAD_ROWS,
    DATAFRAME_DEFAULT_TAIL_ROWS,
    DATAFRAME_MAX_REPR_ROWS,
    PADDING_PREFIX,
    PROTECTED_COLUMNS,
)
from tacoreader._exceptions import TacoBackendError
from tacoreader.dataframe.base import TacoDataFrame

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

# Check Polars availability
try:
    import polars as pl
    import pyarrow as pa

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def _require_polars() -> None:
    """Raise ImportError if Polars is not available."""
    if not HAS_POLARS:
        raise TacoBackendError(
            "Polars backend requires polars package.\n"
            "Install with: pip install polars"
        )


class TacoDataFramePolars(TacoDataFrame):
    """
    Polars DataFrame wrapper with hierarchical navigation.

    Provides full Polars expression API + TACO navigation via .read()
    Requires: pip install polars

    Usage:
        tacoreader.use('polars')
        tdf = dataset.data  # TacoDataFramePolars

        # Polars expressions
        filtered = tdf.filter(pl.col("cloud_cover") < 10)

        # Navigation
        child = tdf.read(0)

        # Export to pure Polars
        df = tdf.to_polars()
    """

    @classmethod
    def from_arrow(cls, arrow_table: pa.Table, format_type: str) -> TacoDataFramePolars:
        """Convert PyArrow Table to Polars. Called by factory when backend='polars'."""
        _require_polars()
        polars_df = pl.from_arrow(arrow_table)
        return cls(polars_df, format_type)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        # Display first N rows, filter padding for cleaner output
        display_df = self._data.head(min(DATAFRAME_MAX_REPR_ROWS, len(self._data)))

        if "id" in display_df.columns:
            display_df = display_df.filter(~pl.col("id").str.contains(PADDING_PREFIX))

        base_repr = str(display_df)

        total_rows = len(self)
        if total_rows > DATAFRAME_MAX_REPR_ROWS:
            info = f"\n[TacoDataFrame: {total_rows} rows (showing first {DATAFRAME_MAX_REPR_ROWS}), format={self._format_type}, backend=polars]"
        else:
            info = f"\n[TacoDataFrame: {total_rows} rows, format={self._format_type}, backend=polars]"

        return base_repr + info

    def __getitem__(self, key):
        """Polars-style subscripting. Supports column selection, slicing, boolean masks."""
        return self._data[key]

    def __setitem__(self, key: str, value):
        """
        Modify column in-place. Protected columns (id, type, internal:*) cannot be modified.

        Raises ValueError if attempting to modify protected columns required for navigation.
        """
        if key in PROTECTED_COLUMNS or key.startswith("internal:"):
            raise ValueError(
                f"Cannot modify protected column: '{key}'\n"
                f"Protected: {sorted(PROTECTED_COLUMNS)} + internal:*\n"
                f"These columns are required for .read() navigation."
            )

        if isinstance(value, pl.Series | pl.Expr):
            self._data = self._data.with_columns(value.alias(key))
        else:
            self._data = self._data.with_columns(pl.Series(key, value))

    @property
    def columns(self):
        return self._data.columns

    @property
    def shape(self):
        return self._data.shape

    def head(self, n: int = DATAFRAME_DEFAULT_HEAD_ROWS) -> pl.DataFrame:
        return self._data.head(n)

    def tail(self, n: int = DATAFRAME_DEFAULT_TAIL_ROWS) -> pl.DataFrame:
        return self._data.tail(n)

    def _get_row(self, position: int) -> dict:
        """Row as dict for navigation. Used by base class .read()"""
        return self._data.row(position, named=True)

    def _to_arrow_for_stats(self) -> pa.Table:
        """
        Convert to PyArrow for stats aggregation.

        Stats functions internally use PyArrow, so we convert:
        Polars → PyArrow → stats computation

        Required by base class for stats_mean(), stats_std(), etc.
        """
        return self._data.to_arrow()

    def to_polars(self) -> pl.DataFrame:
        """
        Export as native Polars DataFrame.

        Returns clone to prevent mutation of internal state.
        Loses TACO navigation (.read() won't work on result).
        """
        return self._data.clone()

    # POLARS-SPECIFIC OPERATIONS

    def filter(self, *args, **kwargs) -> TacoDataFramePolars:
        """Filter with Polars expressions. Returns new TacoDataFramePolars."""
        filtered_df = self._data.filter(*args, **kwargs)
        return TacoDataFramePolars(filtered_df, self._format_type)

    def select(self, *args, **kwargs) -> TacoDataFramePolars:
        """
        Select columns with Polars expressions. Returns new TacoDataFramePolars.

        Note: Navigation requires id, type, internal:gdal_vsi columns.
        """
        selected_df = self._data.select(*args, **kwargs)
        return TacoDataFramePolars(selected_df, self._format_type)

    def with_columns(self, *args, **kwargs) -> TacoDataFramePolars:
        """Add/replace columns with Polars expressions. Returns new TacoDataFramePolars."""
        new_data = self._data.with_columns(*args, **kwargs)
        return TacoDataFramePolars(new_data, self._format_type)

    def sort(self, by, *args, **kwargs) -> TacoDataFramePolars:
        """Sort by column(s). Returns new TacoDataFramePolars."""
        sorted_df = self._data.sort(by, *args, **kwargs)
        return TacoDataFramePolars(sorted_df, self._format_type)

    def limit(self, n: int) -> TacoDataFramePolars:
        """Limit to first n rows. Returns new TacoDataFramePolars."""
        limited_df = self._data.limit(n)
        return TacoDataFramePolars(limited_df, self._format_type)

    def group_by(self, *args, **kwargs):
        """
        Group by column(s). Returns Polars GroupBy (NOT TacoDataFrame).

        Use for aggregations where navigation isn't needed.
        """
        return self._data.group_by(*args, **kwargs)
