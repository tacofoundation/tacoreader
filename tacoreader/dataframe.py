"""
TacoDataFrame - hierarchical DataFrame wrapper for TACO navigation.

Wraps Polars DataFrame with TACO-specific methods for hierarchical navigation
and statistics aggregation. Provides .read() for traversing nested structures
and stats_*() methods for aggregating pre-computed metadata.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from tacoreader._constants import (
    NAVIGATION_REQUIRED_COLUMNS,
    DATAFRAME_DEFAULT_HEAD_ROWS,
    DATAFRAME_DEFAULT_TAIL_ROWS,
    PROTECTED_COLUMNS,
    PADDING_PREFIX,
)


def _is_protected_column(column_name: str) -> bool:
    """
    Check if column is protected and cannot be modified.

    Protected columns required for hierarchical navigation.
    Modifying these breaks .read() functionality.
    """
    return column_name in PROTECTED_COLUMNS or column_name.startswith("internal:")


def _validate_navigation_columns(df: pl.DataFrame, operation: str) -> None:
    """
    Validate critical columns for navigation are present.

    Required: id, type, internal:gdal_vsi
    """
    current_cols = set(df.columns)
    missing = NAVIGATION_REQUIRED_COLUMNS - current_cols

    if missing:
        raise ValueError(
            f"Operation '{operation}' removed required columns: {sorted(missing)}\n"
            f"\n"
            f"Required for navigation: id, type, internal:gdal_vsi\n"
            f"Current columns: {sorted(current_cols)}\n"
            f"\n"
            f"To drop these, convert to Polars first:\n"
            f"  df = tdf.to_polars().select(['custom_column'])"
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

    Wraps Polars DataFrame with TACO-specific functionality:
    - Hierarchical navigation via read()
    - Statistics aggregation via stats_*()
    - Standard DataFrame operations (head, tail, filter, etc.)
    - Column modification with protection for critical columns

    Protected columns (cannot be modified):
        - id, type: Core navigation
        - internal:*: All internal columns

    Examples:
        tdf = dataset.collect()

        # Navigate
        child = tdf.read(0)  # By position
        child = tdf.read("sample_001")  # By ID
        vsi_path = tdf.read(5)  # Returns str if FILE

        # Modify columns (in-place)
        tdf["cloud_cover"] = tdf["cloud_cover"] * 100

        # Modify columns (immutable)
        tdf = tdf.with_column("timestamp", pl.col("timestamp").cast(pl.Datetime))

        # Stats
        mean_values = tdf.stats_mean()
    """

    def __init__(self, data: pl.DataFrame, format_type: str):
        """Initialize with materialized Polars DataFrame."""
        self._data = data
        self._format_type = format_type

    def __len__(self) -> int:
        """Number of rows."""
        return len(self._data)

    def __repr__(self) -> str:
        """String representation with format info."""
        display_df = self._data

        # Filter out padding for cleaner display
        if "id" in self._data.columns:
            display_df = self._data.filter(~pl.col("id").str.contains(PADDING_PREFIX))

        base_repr = display_df.__repr__()
        info = f"\n[TacoDataFrame: {len(self)} rows, format={self._format_type}]"
        return base_repr + info

    def __getitem__(self, key):
        """
        Subscripting like Polars DataFrame.

        Supports: row/column selection, boolean masks, slicing, tuple indexing.
        """
        return self._data[key]

    def __setitem__(self, key: str, value):
        """
        Modify column in-place (pandas-style).

        Protected columns (id, type, internal:*) cannot be modified.
        For immutable ops, use .with_column().
        """
        _validate_not_protected(key)

        if isinstance(value, pl.Series):
            self._data = self._data.with_columns(value.alias(key))
        else:
            self._data = self._data.with_columns(pl.Series(key, value))

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def columns(self):
        """Column names."""
        return self._data.columns

    @property
    def shape(self):
        """Shape tuple: (rows, columns)."""
        return self._data.shape

    def head(self, n: int = DATAFRAME_DEFAULT_HEAD_ROWS) -> pl.DataFrame:
        """First n rows as Polars DataFrame."""
        return self._data.head(n)

    def tail(self, n: int = DATAFRAME_DEFAULT_TAIL_ROWS) -> pl.DataFrame:
        """Last n rows as Polars DataFrame."""
        return self._data.tail(n)

    def to_polars(self) -> pl.DataFrame:
        """Export as Polars DataFrame."""
        return self._data.clone()

    def to_pandas(self) -> Any:
        """Export as pandas DataFrame (requires pandas)."""
        return self._data.to_pandas()

    # ========================================================================
    # DATAFRAME OPERATIONS
    # ========================================================================

    def filter(self, *args, **kwargs) -> TacoDataFrame:
        """
        Filter rows using Polars expressions.

        Returns new TacoDataFrame preserving navigation.
        """
        filtered_df = self._data.filter(*args, **kwargs)
        _validate_navigation_columns(filtered_df, "filter")

        return TacoDataFrame(
            data=filtered_df,
            format_type=self._format_type,
        )

    def select(self, *args, **kwargs) -> TacoDataFrame:
        """
        Select columns using Polars expressions.

        Navigation works if required columns present.
        """
        selected_df = self._data.select(*args, **kwargs)
        _validate_navigation_columns(selected_df, "select")

        return TacoDataFrame(
            data=selected_df,
            format_type=self._format_type,
        )

    def limit(self, n: int) -> TacoDataFrame:
        """Limit to first n rows."""
        limited_df = self._data.limit(n)
        _validate_navigation_columns(limited_df, "limit")

        return TacoDataFrame(
            data=limited_df,
            format_type=self._format_type,
        )

    def sort(self, by, *args, **kwargs) -> TacoDataFrame:
        """Sort rows by column(s)."""
        sorted_df = self._data.sort(by, *args, **kwargs)
        _validate_navigation_columns(sorted_df, "sort")

        return TacoDataFrame(
            data=sorted_df,
            format_type=self._format_type,
        )

    def with_column(self, name: str, expr) -> "TacoDataFrame":
        """
        Add or replace column (immutable, Polars-style).

        Protected columns (id, type, internal:*) cannot be modified.
        """
        _validate_not_protected(name)

        # Handle different input types
        if isinstance(expr, pl.Expr):
            new_data = self._data.with_columns(expr.alias(name))
        elif isinstance(expr, pl.Series):
            new_data = self._data.with_columns(expr.alias(name))
        else:
            new_data = self._data.with_columns(pl.Series(name, expr))

        return TacoDataFrame(
            data=new_data,
            format_type=self._format_type,
        )

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
        row = self._data.row(position, named=True)

        if row["type"] == "FILE":
            return row["internal:gdal_vsi"]

        return self._read_meta(row)

    def _get_position(self, key: int | str) -> int:
        """Convert key to integer position."""
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Position {key} out of range [0, {len(self)-1}]")
            return key

        # Search by ID
        ids = self._data["id"].to_list()
        if key not in ids:
            raise KeyError(f"ID '{key}' not found")

        return ids.index(key)

    def _read_meta(self, row: dict) -> TacoDataFrame:
        """
        Read __meta__ file for FOLDER sample.

        Handles:
        - /vsisubfile/ paths (ZIP, TacoCat): Read Parquet from offset
        - Direct paths (FOLDER): Read Parquet from filesystem
        """
        from tacoreader.io import download_range, download_bytes
        from tacoreader.utils.format import is_remote
        from tacoreader.utils.vsi import parse_vsi_subfile, strip_vsi_prefix

        vsi_path = row["internal:gdal_vsi"]

        if vsi_path.startswith("/vsisubfile/"):
            # ZIP or TacoCat: Read Parquet from offset
            root_path, offset, size = parse_vsi_subfile(vsi_path)
            original_path = strip_vsi_prefix(root_path)

            if is_remote(original_path):
                parquet_bytes = download_range(original_path, offset, size)
                children_df = pl.read_parquet(BytesIO(parquet_bytes))
            else:
                with open(original_path, "rb") as f:
                    f.seek(offset)
                    parquet_bytes = f.read(size)

                children_df = pl.read_parquet(BytesIO(parquet_bytes))

            # Construct VSI paths for children
            children_df = children_df.with_columns(
                pl.struct([pl.col("*")])
                .map_elements(
                    lambda row_struct: self._construct_vsi_path(row_struct, root_path),
                    return_dtype=pl.Utf8,
                )
                .alias("internal:gdal_vsi")
            )
        else:
            # FOLDER: Read Parquet from __meta__
            if is_remote(vsi_path):
                if vsi_path.endswith("/__meta__"):
                    meta_bytes = download_bytes(vsi_path)
                else:
                    meta_bytes = download_bytes(vsi_path, "__meta__")

                children_df = pl.read_parquet(BytesIO(meta_bytes))
            else:
                if vsi_path.endswith("/__meta__"):
                    meta_path = vsi_path
                else:
                    meta_path = str(Path(vsi_path) / "__meta__")

                children_df = pl.read_parquet(meta_path)

            # Construct direct paths for children
            children_df = children_df.with_columns(
                pl.struct([pl.col("*")])
                .map_elements(
                    lambda row_struct: self._construct_folder_path(
                        row_struct, vsi_path
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias("internal:gdal_vsi")
            )

        return TacoDataFrame(
            data=children_df,
            format_type=self._format_type,
        )

    def _construct_vsi_path(self, row_struct: dict, zip_path: str) -> str:
        """Construct /vsisubfile/ path for child in ZIP/TacoCat."""
        if "internal:offset" in row_struct and "internal:size" in row_struct:
            offset = row_struct["internal:offset"]
            size = row_struct["internal:size"]
            return f"/vsisubfile/{offset}_{size},{zip_path}"
        return ""

    def _construct_folder_path(self, row_struct: dict, parent_path: str) -> str:
        """Construct direct path for child in FOLDER."""
        if "internal:relative_path" in row_struct:
            relative = row_struct["internal:relative_path"]
            return f"{parent_path}/{relative}"
        elif "id" in row_struct:
            return f"{parent_path}/{row_struct['id']}"
        return ""

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
