"""
Pandas backend for TacoDataFrame.

Wraps Pandas DataFrame with TACO-specific hierarchical navigation.
Requires pandas package: pip install pandas
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
    import pandas as pd
    import pyarrow as pa

# Check Pandas availability
try:
    import pandas as pd
    import pyarrow as pa

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _require_pandas() -> None:
    """Raise ImportError if Pandas is not available."""
    if not HAS_PANDAS:
        raise TacoBackendError(
            "Pandas backend requires pandas package.\n"
            "Install with: pip install pandas"
        )


class TacoDataFramePandas(TacoDataFrame):
    """
    Pandas DataFrame wrapper with hierarchical navigation.

    Provides full Pandas API + TACO navigation via .read()
    Requires: pip install pandas

    Usage:
        tacoreader.use('pandas')
        tdf = dataset.data  # TacoDataFramePandas

        # Pandas operations
        filtered = tdf.query('cloud_cover < 10')
        filtered = tdf[tdf['cloud_cover'] < 10]

        # Navigation
        child = tdf.read(0)

        # Export to pure Pandas
        df = tdf.to_pandas()
    """

    @classmethod
    def from_arrow(cls, arrow_table: pa.Table, format_type: str) -> TacoDataFramePandas:
        """Convert PyArrow Table to Pandas. Called by factory when backend='pandas'."""
        _require_pandas()
        pandas_df = arrow_table.to_pandas()
        return cls(pandas_df, format_type)

    def __len__(self) -> int:
        """Number of rows."""
        return len(self._data)

    def __repr__(self) -> str:
        """String representation with format info."""
        # Slice to max display rows
        display_df = self._data.head(min(DATAFRAME_MAX_REPR_ROWS, len(self._data)))

        # Filter padding only from display rows
        if "id" in display_df.columns:
            display_df = display_df[~display_df["id"].str.contains(PADDING_PREFIX)]

        # Pandas string representation
        base_repr = str(display_df)

        # Add metadata footer
        total_rows = len(self)
        if total_rows > DATAFRAME_MAX_REPR_ROWS:
            info = f"\n[TacoDataFrame: {total_rows} rows (showing first {DATAFRAME_MAX_REPR_ROWS}), format={self._format_type}, backend=pandas]"
        else:
            info = f"\n[TacoDataFrame: {total_rows} rows, format={self._format_type}, backend=pandas]"

        return base_repr + info

    def __getitem__(self, key):
        """
        Pandas-style subscripting.

        Supports: column selection, boolean indexing, slicing.
        Returns TacoDataFramePandas if result is DataFrame, otherwise returns Series/scalar.
        """
        result = self._data[key]

        # Wrap DataFrames, return Series/scalars as-is
        if isinstance(result, pd.DataFrame):
            return TacoDataFramePandas(result, self._format_type)

        return result

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

        # Pandas handles assignment natively
        self._data[key] = value

    @property
    def columns(self):
        """Column names."""
        return self._data.columns.tolist()

    @property
    def shape(self):
        """Shape tuple: (rows, columns)."""
        return self._data.shape

    def head(self, n: int = DATAFRAME_DEFAULT_HEAD_ROWS) -> pd.DataFrame:
        """First n rows as Pandas DataFrame."""
        return self._data.head(n)

    def tail(self, n: int = DATAFRAME_DEFAULT_TAIL_ROWS) -> pd.DataFrame:
        """Last n rows as Pandas DataFrame."""
        return self._data.tail(n)

    def _get_row(self, position: int) -> dict:
        """Row as dict for navigation. Used by base class .read()"""
        return self._data.iloc[position].to_dict()

    def _to_arrow_for_stats(self) -> pa.Table:
        """
        Convert to PyArrow for stats aggregation.

        Stats functions internally use PyArrow, so we convert:
        Pandas → PyArrow → stats computation

        Required by base class for stats_mean(), stats_std(), etc.
        """
        return pa.Table.from_pandas(self._data)

    def to_pandas(self) -> pd.DataFrame:
        """
        Export as native Pandas DataFrame.

        Returns copy to prevent mutation of internal state.
        Loses TACO navigation (.read() won't work on result).
        """
        return self._data.copy()

    # PANDAS-SPECIFIC OPERATIONS

    def query(self, expr: str, **kwargs) -> TacoDataFramePandas:
        """Filter with Pandas query string. Returns new TacoDataFramePandas."""
        filtered_df = self._data.query(expr, **kwargs)
        return TacoDataFramePandas(filtered_df, self._format_type)

    def sort_values(self, by, ascending=True, **kwargs) -> TacoDataFramePandas:
        """Sort by column(s). Returns new TacoDataFramePandas."""
        sorted_df = self._data.sort_values(by=by, ascending=ascending, **kwargs)
        return TacoDataFramePandas(sorted_df, self._format_type)

    def assign(self, **kwargs) -> TacoDataFramePandas:
        """
        Add/replace columns immutably. Returns new TacoDataFramePandas.

        Validates protected columns (id, type, internal:*) cannot be modified.
        """
        # Validate protected columns
        for col_name in kwargs:
            if col_name in PROTECTED_COLUMNS or col_name.startswith("internal:"):
                raise ValueError(
                    f"Cannot modify protected column: '{col_name}'\n"
                    f"Protected: {sorted(PROTECTED_COLUMNS)} + internal:*"
                )

        new_data = self._data.assign(**kwargs)
        return TacoDataFramePandas(new_data, self._format_type)

    def groupby(self, by, **kwargs):
        """
        Group by column(s). Returns Pandas GroupBy (NOT TacoDataFrame).

        Use for aggregations where navigation isn't needed.
        """
        return self._data.groupby(by=by, **kwargs)

    # PANDAS INDEXERS (loc/iloc)

    @property
    def loc(self):
        """Label-based indexer. Returns Pandas indexer (NOT TacoDataFrame)."""
        return self._data.loc

    @property
    def iloc(self):
        """Integer position-based indexer. Returns Pandas indexer (NOT TacoDataFrame)."""
        return self._data.iloc
