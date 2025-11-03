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
import obstore as obs
import polars as pl

# ============================================================================
# PROTECTED COLUMNS
# ============================================================================

# Columns that cannot be modified (critical for TACO navigation)
PROTECTED_COLUMNS = frozenset(
    {
        # Core columns (modifying breaks references and navigation)
        "id",
        "type",
        # Internal columns (modifying breaks hierarchical navigation)
        "internal:parent_id",
        "internal:offset",
        "internal:size",
        "internal:gdal_vsi",
        "internal:source_file",
        "internal:relative_path",
    }
)


# ============================================================================
# TACODATAFRAME
# ============================================================================


class TacoDataFrame:
    """
    Hierarchical DataFrame wrapper for TACO navigation.

    Wraps a Polars DataFrame and provides TACO-specific functionality:
    - Hierarchical navigation via read() method
    - Statistics aggregation via stats_*() methods
    - Standard DataFrame operations (head, tail, shape, etc.)
    - Column modification with protection for critical columns

    The underlying DataFrame is always materialized in memory (Polars).
    Navigation to child levels loads child metadata on-demand.

    Protected columns (cannot be modified):
        - id, type: Core columns required for navigation
        - internal:*: All internal columns used for hierarchical navigation

    Attributes:
        columns: Column names from underlying DataFrame
        shape: Tuple of (rows, columns)

    Example:
        >>> tdf = dataset.collect()
        >>> print(tdf.shape)
        (1000, 15)
        >>>
        >>> # Navigate to child
        >>> child = tdf.read(0)  # By position
        >>> child = tdf.read("sample_001")  # By ID
        >>>
        >>> # Get file path
        >>> vsi_path = tdf.read(5)  # Returns str if FILE type
        >>>
        >>> # Modify columns (pandas-style, in-place)
        >>> tdf["cloud_cover"] = tdf["cloud_cover"] * 100
        >>>
        >>> # Modify columns (Polars-style, immutable)
        >>> tdf = tdf.with_column("timestamp", pl.col("timestamp").cast(pl.Datetime))
        >>>
        >>> # Statistics
        >>> mean_values = tdf.stats_mean()
    """

    def __init__(
        self,
        data: pl.DataFrame,
        format_type: str,
        consolidated_files: dict[str, str],
    ):
        """
        Initialize TacoDataFrame.

        Args:
            data: Materialized Polars DataFrame (in memory)
            format_type: Backend format ("zip", "folder", or "tacocat")
            consolidated_files: Paths to cached metadata files
        """
        self._data = data
        self._format_type = format_type
        self._consolidated_files = consolidated_files

    def __len__(self) -> int:
        """Number of rows in current level."""
        return len(self._data)

    def __repr__(self) -> str:
        """String representation with format info."""
        display_df = self._data

        # Filter out padding samples for cleaner display
        if "id" in self._data.columns:
            display_df = self._data.filter(~pl.col("id").str.contains("__TACOPAD__"))

        base_repr = display_df.__repr__()
        info = f"\n[TacoDataFrame: {len(self)} rows, format={self._format_type}]"
        return base_repr + info

    def __getitem__(self, key):
        """
        Enable subscripting like Polars DataFrame.

        Delegates to underlying Polars DataFrame for all indexing operations.
        Supports all Polars indexing patterns: row selection, column selection,
        boolean masks, slicing, and tuple indexing.

        Args:
            key: Index, column name, slice, tuple, or expression
                - Integer: row by position
                - String: column by name
                - Slice: multiple rows
                - Tuple: (row, column) indexing
                - Expression: Polars expression for filtering

        Returns:
            Selected data (scalar, Series, or DataFrame depending on key)

        Examples:
            >>> # Single cell access
            >>> tdf[0, "type"]
            'FOLDER'
            >>>
            >>> # Row access
            >>> tdf[0]
            {'id': 'sample_001', 'type': 'FOLDER', ...}
            >>>
            >>> # Column access
            >>> tdf["type"]
            Series(['FOLDER', 'FILE', 'FILE', ...])
            >>>
            >>> # Slicing
            >>> tdf[0:5]
            DataFrame(...)
            >>>
            >>> # Multiple columns
            >>> tdf[["id", "type"]]
            DataFrame(...)
        """
        return self._data[key]

    def __setitem__(self, key: str, value):
        """
        Modify column (in-place mutation, pandas-style).

        Allows modifying metadata columns but protects critical columns
        required for hierarchical navigation. For immutable operations,
        use .with_column() instead.

        Protected columns (cannot be modified):
            - id, type: Core navigation columns
            - internal:*: All internal columns

        Args:
            key: Column name to modify
            value: New values (Polars Series, list, numpy array, etc.)

        Raises:
            ValueError: If attempting to modify protected column

        Examples:
            >>> # Convert timestamp string to datetime
            >>> tdf["timestamp"] = pd.to_datetime(tdf["timestamp"])
            >>>
            >>> # Scale values
            >>> tdf["cloud_cover"] = tdf["cloud_cover"] * 100
            >>>
            >>> # Add new column
            >>> tdf["normalized"] = tdf["value"] / tdf["value"].max()
            >>>
            >>> # PROTECTED: Cannot modify critical columns
            >>> tdf["id"] = "new_id"  # ValueError!
            >>> tdf["internal:offset"] = 0  # ValueError!
        """
        # Check if column is protected
        if key in PROTECTED_COLUMNS or key.startswith("internal:"):
            raise ValueError(
                f"Cannot modify protected column: '{key}'\n"
                f"Protected columns are required for hierarchical navigation:\n"
                f"  - Core: id, type\n"
                f"  - Internal: internal:parent_id, internal:offset, internal:size, "
                f"internal:gdal_vsi, internal:source_file, internal:relative_path\n"
                f"To create derived columns, use a different name."
            )

        # Convert to Polars Series if needed
        if isinstance(value, pl.Series):
            self._data = self._data.with_columns(value.alias(key))
        else:
            # Handle pandas Series, numpy arrays, lists, etc.
            self._data = self._data.with_columns(pl.Series(key, value))

    # ========================================================================
    # DATAFRAME PROPERTIES
    # ========================================================================

    @property
    def columns(self):
        """Column names from underlying DataFrame."""
        return self._data.columns

    @property
    def shape(self):
        """Shape tuple: (rows, columns)."""
        return self._data.shape

    def head(self, n: int = 5) -> pl.DataFrame:
        """
        First n rows as Polars DataFrame.

        Args:
            n: Number of rows (default: 5)

        Returns:
            Polars DataFrame with first n rows
        """
        return self._data.head(n)

    def tail(self, n: int = 5) -> pl.DataFrame:
        """
        Last n rows as Polars DataFrame.

        Args:
            n: Number of rows (default: 5)

        Returns:
            Polars DataFrame with last n rows
        """
        return self._data.tail(n)

    def to_polars(self) -> pl.DataFrame:
        """
        Export as Polars DataFrame.

        Returns:
            Clone of underlying Polars DataFrame
        """
        return self._data.clone()

    def to_pandas(self) -> Any:
        """
        Export as pandas DataFrame.

        Requires pandas installed.

        Returns:
            pandas DataFrame

        Raises:
            ImportError: If pandas not installed
        """
        return self._data.to_pandas()

    # ========================================================================
    # DATAFRAME OPERATIONS (return TacoDataFrame for chaining)
    # ========================================================================

    def filter(self, *args, **kwargs) -> TacoDataFrame:
        """
        Filter rows using Polars expressions.

        Returns new TacoDataFrame preserving navigation capabilities.
        All filtered rows can still use .read() for hierarchical navigation.

        Args:
            *args: Polars expressions or predicates
            **kwargs: Additional Polars filter arguments

        Returns:
            New TacoDataFrame with filtered rows

        Examples:
            >>> # Filter by column value
            >>> large = tdf.filter(pl.col("internal:size") > 1000000)
            >>> large.read(0)

            >>> # Multiple conditions
            >>> filtered = tdf.filter(
            ...     (pl.col("type") == "FILE") &
            ...     (pl.col("internal:size") < 5000)
            ... )

            >>> # Chain operations
            >>> tdf.filter(pl.col("type") == "FOLDER").limit(10).read(0)
        """
        filtered_df = self._data.filter(*args, **kwargs)
        return TacoDataFrame(
            data=filtered_df,
            format_type=self._format_type,
            consolidated_files=self._consolidated_files,
        )

    def select(self, *args, **kwargs) -> TacoDataFrame:
        """
        Select columns using Polars expressions.

        Returns new TacoDataFrame with selected columns.
        Navigation with .read() still works if required columns present.

        Args:
            *args: Column names or Polars expressions
            **kwargs: Additional Polars select arguments

        Returns:
            New TacoDataFrame with selected columns

        Examples:
            >>> # Select specific columns
            >>> subset = tdf.select(["id", "type", "internal:gdal_vsi"])
            >>>
            >>> # Using expressions
            >>> tdf.select(pl.col("id"), pl.col("type"))
            >>>
            >>> # Chain with filter
            >>> tdf.filter(pl.col("type") == "FILE").select(["id", "internal:size"])
        """
        selected_df = self._data.select(*args, **kwargs)
        return TacoDataFrame(
            data=selected_df,
            format_type=self._format_type,
            consolidated_files=self._consolidated_files,
        )

    def limit(self, n: int) -> TacoDataFrame:
        """
        Limit to first n rows.

        Returns new TacoDataFrame with at most n rows.
        Alias for head() that returns TacoDataFrame instead of pl.DataFrame.

        Args:
            n: Maximum number of rows

        Returns:
            New TacoDataFrame with at most n rows

        Examples:
            >>> # Get first 10 rows
            >>> subset = tdf.limit(10)
            >>> subset.read(0)
            >>>
            >>> # Chain with filter
            >>> tdf.filter(pl.col("type") == "FOLDER").limit(5)
        """
        limited_df = self._data.limit(n)
        return TacoDataFrame(
            data=limited_df,
            format_type=self._format_type,
            consolidated_files=self._consolidated_files,
        )

    def sort(self, by, *args, **kwargs) -> TacoDataFrame:
        """
        Sort rows by column(s).

        Returns new TacoDataFrame with sorted rows.
        Navigation with .read() works on sorted order.

        Args:
            by: Column name(s) or expression(s) to sort by
            *args: Additional sort expressions
            **kwargs: Additional Polars sort arguments (descending, nulls_last, etc.)

        Returns:
            New TacoDataFrame with sorted rows

        Examples:
            >>> # Sort by single column
            >>> sorted_df = tdf.sort("id")
            >>>
            >>> # Sort descending
            >>> largest = tdf.sort("internal:size", descending=True)
            >>> largest.read(0)  # Navigate to largest file
            >>>
            >>> # Multiple columns
            >>> tdf.sort(["type", "id"])
            >>>
            >>> # Chain operations
            >>> tdf.filter(pl.col("type") == "FILE").sort("internal:size").limit(10)
        """
        sorted_df = self._data.sort(by, *args, **kwargs)
        return TacoDataFrame(
            data=sorted_df,
            format_type=self._format_type,
            consolidated_files=self._consolidated_files,
        )

    def with_column(self, name: str, expr) -> "TacoDataFrame":
        """
        Add or replace column (immutable, Polars-style).

        Returns new TacoDataFrame with modified column, leaving original unchanged.
        Protects critical columns required for hierarchical navigation.

        This is the recommended way to modify columns as it follows Polars'
        functional paradigm and prevents accidental mutations.

        Protected columns (cannot be modified):
            - id, type: Core navigation columns
            - internal:*: All internal columns

        Args:
            name: Column name to add/replace
            expr: Polars expression, Series, or values

        Returns:
            New TacoDataFrame with modified column

        Raises:
            ValueError: If attempting to modify protected column

        Examples:
            >>> # Cast column type (Polars expression)
            >>> tdf = tdf.with_column("timestamp", pl.col("timestamp").cast(pl.Datetime))
            >>>
            >>> # Add computed column
            >>> tdf = tdf.with_column("normalized", pl.col("value") / pl.col("value").max())
            >>>
            >>> # Replace with new values (Series or list)
            >>> tdf = tdf.with_column("cloud_cover", pl.Series([10, 20, 30]))
            >>>
            >>> # Chain operations
            >>> tdf = (tdf
            ...     .with_column("timestamp", pl.col("timestamp").cast(pl.Datetime))
            ...     .with_column("cloud_cover", pl.col("cloud_cover") * 100)
            ... )
            >>>
            >>> # PROTECTED: Cannot modify critical columns
            >>> tdf = tdf.with_column("id", "new_id")  # ValueError!
        """
        # Check if column is protected
        if name in PROTECTED_COLUMNS or name.startswith("internal:"):
            raise ValueError(
                f"Cannot modify protected column: '{name}'\n"
                f"Protected columns are required for hierarchical navigation:\n"
                f"  - Core: id, type\n"
                f"  - Internal: internal:parent_id, internal:offset, internal:size, "
                f"internal:gdal_vsi, internal:source_file, internal:relative_path\n"
                f"To create derived columns, use a different name."
            )

        # Handle different input types
        if isinstance(expr, pl.Expr):
            # Polars expression: pl.col("x") * 2
            new_data = self._data.with_columns(expr.alias(name))
        elif isinstance(expr, pl.Series):
            # Polars Series
            new_data = self._data.with_columns(expr.alias(name))
        else:
            # List, numpy array, etc.
            new_data = self._data.with_columns(pl.Series(name, expr))

        return TacoDataFrame(
            data=new_data,
            format_type=self._format_type,
            consolidated_files=self._consolidated_files,
        )

    # ========================================================================
    # HIERARCHICAL NAVIGATION
    # ========================================================================

    def read(self, key: int | str) -> TacoDataFrame | str:
        """
        Navigate to child level by position or ID.

        For FILE samples, returns GDAL VSI path as string.
        For FOLDER samples, reads __meta__ file and returns TacoDataFrame with children.

        Args:
            key: Row position (int) or sample ID (str)

        Returns:
            str: GDAL VSI path if sample type is FILE
            TacoDataFrame: Child samples if sample type is FOLDER

        Raises:
            IndexError: If position is out of range
            KeyError: If ID not found

        Example:
            >>> # Navigate by position
            >>> child = tdf.read(0)
            >>>
            >>> # Navigate by ID
            >>> child = tdf.read("sample_001")
            >>>
            >>> # FILE type returns path
            >>> vsi_path = tdf.read(5)
            >>> print(vsi_path)
            '/vsisubfile/1024_5000,/vsis3/bucket/data.tacozip'
        """
        position = self._get_position(key)
        row = self._data.row(position, named=True)

        if row["type"] == "FILE":
            return row["internal:gdal_vsi"]

        return self._read_meta(row)

    def _get_position(self, key: int | str) -> int:
        """
        Convert key to integer position.

        Args:
            key: Position (int) or ID (str)

        Returns:
            Integer row position

        Raises:
            IndexError: If position out of range
            KeyError: If ID not found
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Position {key} out of range [0, {len(self)-1}]")
            return key

        # Search by ID - convert to list and use Python's index()
        ids = self._data["id"].to_list()
        if key not in ids:
            raise KeyError(f"ID '{key}' not found in current level")

        return ids.index(key)

    def _read_meta(self, row: dict) -> TacoDataFrame:
        """
        Read __meta__ file for FOLDER sample.

        Handles both formats:
        - /vsisubfile/ paths (ZIP, TacoCat): Read Parquet from offset
        - Direct paths (FOLDER): Read Avro from filesystem

        Args:
            row: Row dictionary with sample metadata

        Returns:
            TacoDataFrame with child samples
        """
        from tacoreader.utils.format import is_remote
        from tacoreader.utils.vsi import (
            create_obstore_from_url,
            extract_path_from_url,
            parse_vsi_subfile,
            strip_vsi_prefix,
        )

        vsi_path = row["internal:gdal_vsi"]

        if vsi_path.startswith("/vsisubfile/"):
            # ZIP or TacoCat format: Read Parquet from offset
            root_path, offset, size = parse_vsi_subfile(vsi_path)
            original_path = strip_vsi_prefix(root_path)

            if is_remote(original_path):
                store = create_obstore_from_url(original_path)
                object_path = extract_path_from_url(original_path)

                parquet_bytes = obs.get_range(
                    store, object_path, start=offset, length=size
                )
                children_df = pl.read_parquet(BytesIO(bytes(parquet_bytes)))
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
            # FOLDER format: Read Avro from __meta__ file
            if is_remote(vsi_path):
                store = create_obstore_from_url(vsi_path)
                base_path = extract_path_from_url(vsi_path)
                meta_path = f"{base_path}/__meta__"

                response = obs.get(store, meta_path)
                meta_bytes = response.bytes()

                children_df = pl.read_avro(BytesIO(bytes(meta_bytes)))
            else:
                meta_path = str(Path(vsi_path) / "__meta__")
                children_df = pl.read_avro(meta_path)

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
            consolidated_files=self._consolidated_files,
        )

    def _construct_vsi_path(self, row_struct: dict, zip_path: str) -> str:
        """
        Construct /vsisubfile/ path for child in ZIP/TacoCat.

        Args:
            row_struct: Child row as dictionary
            zip_path: Parent ZIP path

        Returns:
            VSI path string or empty string
        """
        if "internal:offset" in row_struct and "internal:size" in row_struct:
            offset = row_struct["internal:offset"]
            size = row_struct["internal:size"]
            return f"/vsisubfile/{offset}_{size},{zip_path}"
        return ""

    def _construct_folder_path(self, row_struct: dict, parent_path: str) -> str:
        """
        Construct direct path for child in FOLDER.

        Args:
            row_struct: Child row as dictionary
            parent_path: Parent directory path

        Returns:
            Direct path string or empty string
        """
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
        """
        Aggregate categorical probabilities using weighted average.

        Returns:
            numpy array with aggregated categorical probabilities
        """
        from tacoreader.utils.stats import stats_categorical

        return stats_categorical(self._data.to_pandas())

    def stats_mean(self) -> np.ndarray:
        """
        Aggregate means using weighted average.

        Returns:
            numpy array with aggregated mean values
        """
        from tacoreader.utils.stats import stats_mean

        return stats_mean(self._data.to_pandas())

    def stats_std(self) -> np.ndarray:
        """
        Aggregate standard deviations using pooled std formula.

        Returns:
            numpy array with aggregated standard deviations
        """
        from tacoreader.utils.stats import stats_std

        return stats_std(self._data.to_pandas())

    def stats_min(self) -> np.ndarray:
        """
        Aggregate minimums (global min across all rows).

        Returns:
            numpy array with minimum values per band
        """
        from tacoreader.utils.stats import stats_min

        return stats_min(self._data.to_pandas())

    def stats_max(self) -> np.ndarray:
        """
        Aggregate maximums (global max across all rows).

        Returns:
            numpy array with maximum values per band
        """
        from tacoreader.utils.stats import stats_max

        return stats_max(self._data.to_pandas())

    def stats_p25(self) -> np.ndarray:
        """
        Aggregate 25th percentiles using simple average.

        Returns:
            numpy array with 25th percentile values
        """
        from tacoreader.utils.stats import stats_p25

        return stats_p25(self._data.to_pandas())

    def stats_p50(self) -> np.ndarray:
        """
        Aggregate 50th percentiles (median) using simple average.

        Returns:
            numpy array with median values
        """
        from tacoreader.utils.stats import stats_p50

        return stats_p50(self._data.to_pandas())

    def stats_median(self) -> np.ndarray:
        """
        Aggregate medians using simple average (alias for stats_p50).

        Returns:
            numpy array with median values
        """
        from tacoreader.utils.stats import stats_p50

        return stats_p50(self._data.to_pandas())

    def stats_p75(self) -> np.ndarray:
        """
        Aggregate 75th percentiles using simple average.

        Returns:
            numpy array with 75th percentile values
        """
        from tacoreader.utils.stats import stats_p75

        return stats_p75(self._data.to_pandas())

    def stats_p95(self) -> np.ndarray:
        """
        Aggregate 95th percentiles using simple average.

        Returns:
            numpy array with 95th percentile values
        """
        from tacoreader.utils.stats import stats_p95

        return stats_p95(self._data.to_pandas())