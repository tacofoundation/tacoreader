from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from tacoreader._schema import PITSchema
from tacoreader._stats import (
    stats_categorical,
    stats_max,
    stats_mean,
    stats_min,
    stats_p25,
    stats_p50,
    stats_p75,
    stats_p95,
    stats_std,
)


class TacoDataFrame(pd.DataFrame):
    """
    Hierarchical DataFrame using Position-Isomorphic Tree navigation.

    Supports three navigation modes:
    1. By position: df.read(10).read(4).read(2)
    2. By ID: df.read("landslide_001").read("imagery")
    3. By path: df.read("10:4:2")

    Returns DataFrame if navigating to FOLDER, VSI string if FILE.
    """

    _metadata: ClassVar[list[str]] = [
        "_all_levels",
        "_schema",
        "_current_depth",
        "_root_path",
        "_slice_offset",
    ]

    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        columns: Any = None,
        dtype: Any = None,
        copy: bool | None = None,
        all_levels: list[pd.DataFrame] | None = None,
        schema: PITSchema | None = None,
        current_depth: int = 0,
        root_path: str = "",
        slice_offset: int = 0,
    ) -> None:
        """
        Initialize TacoDataFrame.

        Args:
            data: DataFrame data (current level)
            index: Index for DataFrame
            columns: Column labels for DataFrame
            dtype: Data type to force
            copy: Whether to copy data
            all_levels: List of all DataFrames [level0, level1, ...]
            schema: PIT schema for navigation
            current_depth: Current depth (0 = root)
            root_path: VSI root path
            slice_offset: Global offset of first row in this DataFrame
        """
        # Initialize parent DataFrame with positional arguments (mypy compatibility)
        super().__init__(data, index, columns, dtype, copy)  # type: ignore[call-arg]

        # Set TacoDataFrame-specific attributes after parent initialization
        self._all_levels = all_levels if all_levels is not None else []
        self._schema = schema
        self._current_depth = current_depth
        self._root_path = root_path
        self._slice_offset = slice_offset

    @property
    def _constructor(self) -> type:
        return TacoDataFrame

    @property
    def _constructor_sliced(self) -> type:
        return pd.Series

    @property
    def tree_depth(self) -> int:
        """Get total tree depth."""
        return len(self._all_levels)

    @property
    def max_depth(self) -> int:
        """Get maximum navigable depth."""
        return len(self._all_levels) - 1

    def flat(
        self,
        depth: int | None = None,
        include: list[str] | None = None,
        where: str | None = None,
    ) -> TacoDataFrame:
        """
        Flatten hierarchical structure into flat TacoDataFrame.

        Expands all levels and merges into a flat structure. Always filters
        padding samples (__TACOPAD__) and returns only FILE nodes.

        Args:
            depth: Maximum depth to expand (None = all levels)
            include: Extra columns beyond defaults (id, type, internal:gdal_vsi)
            where: Regex pattern to filter 'id' column (None = include all)

        Returns:
            Flat TacoDataFrame (no hierarchy, but still supports .read())

        Examples:
            >>> df.flat()  # All files
            >>> df.flat(where='cloudmask|s2')  # Only cloudmask or s2
            >>> df.flat(where='^(?!.*thumbnail)')  # Exclude thumbnails
            >>> df.flat(include=['stac:time_start'], where='cloudmask')
        """
        if self._schema is None:
            raise ValueError("Schema is required for flat() operation")

        # Determine target depth
        target_depth = depth if depth is not None else self.max_depth

        if target_depth < self._current_depth:
            raise ValueError(
                f"depth ({target_depth}) cannot be less than current depth ({self._current_depth})"
            )

        # Step 1: Collect global indices from current subset
        # Use actual pandas index values to handle non-contiguous selections
        global_indices = [self._slice_offset + int(idx) for idx in self.index]

        # Step 2: Expand indices level by level using PIT arithmetic
        indices_by_level = self._expand_indices_to_depth(global_indices, target_depth)

        # Step 3: Extract DataFrames and repeat parent rows to align with children
        level_dfs = []
        for depth_level in range(self._current_depth, target_depth + 1):
            indices = indices_by_level[depth_level]
            level_df = self._all_levels[depth_level].iloc[indices].reset_index(drop=True)
            
            # Calculate how many times each row should be repeated
            if depth_level < target_depth:
                repeat_counts = self._calculate_repeat_counts(indices, depth_level)
                # Repeat rows to align with children at the next level
                level_df = level_df.loc[level_df.index.repeat(repeat_counts)].reset_index(drop=True)
            
            level_dfs.append((depth_level, level_df))

        # Step 4: Merge horizontally with proper column renaming
        merged_df = self._merge_levels_horizontal(level_dfs, include)

        # Step 5: Filter padding samples
        if "id" in merged_df.columns:
            merged_df = merged_df[~merged_df["id"].str.contains("__TACOPAD__", na=False)]

        # Step 6: Always filter only FILEs
        if "type" in merged_df.columns:
            merged_df = merged_df[merged_df["type"] == "FILE"]

        # Step 7: Apply regex filter on 'id' if provided
        if where is not None and "id" in merged_df.columns:
            merged_df = merged_df[merged_df["id"].str.contains(where, regex=True, na=False)]

        # Return as TacoDataFrame (flattened, no hierarchy metadata needed)
        return TacoDataFrame(
            merged_df.reset_index(drop=True),
            all_levels=[],  # Flat structure has no levels
            schema=None,  # No schema for flat structure
            current_depth=0,
            root_path=self._root_path,
            slice_offset=0,
        )

    def stats_categorical(self) -> np.ndarray:
        """Aggregate categorical probabilities using weighted average."""
        return stats_categorical(self)

    def stats_mean(self) -> np.ndarray:
        """Aggregate means using weighted average."""
        return stats_mean(self)

    def stats_std(self) -> np.ndarray:
        """Aggregate standard deviations using pooled std formula."""
        return stats_std(self)

    def stats_min(self) -> np.ndarray:
        """Aggregate minimums (global min across all rows)."""
        return stats_min(self)

    def stats_max(self) -> np.ndarray:
        """Aggregate maximums (global max across all rows)."""
        return stats_max(self)

    def stats_p25(self) -> np.ndarray:
        """Aggregate 25th percentiles using simple average."""
        return stats_p25(self)

    def stats_p50(self) -> np.ndarray:
        """Aggregate 50th percentiles (median) using simple average."""
        return stats_p50(self)

    def stats_median(self) -> np.ndarray:
        """Aggregate medians using simple average (alias for stats_p50)."""
        return stats_p50(self)

    def stats_p75(self) -> np.ndarray:
        """Aggregate 75th percentiles using simple average."""
        return stats_p75(self)

    def stats_p95(self) -> np.ndarray:
        """Aggregate 95th percentiles using simple average."""
        return stats_p95(self)

    def _calculate_repeat_counts(self, parent_indices: list[int], parent_depth: int) -> list[int]:
        """
        Calculate how many children each parent has.
        
        This is used to repeat parent rows so they align with their children
        when merging levels horizontally.
        
        Args:
            parent_indices: Global indices of parents at parent_depth
            parent_depth: Depth of parent level
            
        Returns:
            List of repeat counts (one per parent index)
        """
        if self._schema is None:
            raise ValueError("Schema is required for repeat count calculation")
        
        child_depth = parent_depth + 1
        repeat_counts = []
        
        for parent_idx in parent_indices:
            # Determine which pattern this parent uses for its children
            pattern_index = self._determine_pattern_index(parent_idx, child_depth)
            child_patterns = self._schema.hierarchy[str(child_depth)]
            pattern = child_patterns[pattern_index]["children"]
            children_count = len(pattern)
            repeat_counts.append(children_count)
        
        return repeat_counts

    def _expand_indices_to_depth(
        self, start_indices: list[int], target_depth: int
    ) -> dict[int, list[int]]:
        """
        Expand parent indices to all child indices level by level.

        Uses PIT arithmetic to calculate child offsets without recursion.

        Args:
            start_indices: Initial indices at current depth
            target_depth: Target depth to expand to

        Returns:
            Dictionary mapping depth -> list of indices at that depth
        """
        indices_by_level: dict[int, list[int]] = {self._current_depth: start_indices}

        for depth in range(self._current_depth, target_depth):
            parent_indices = indices_by_level[depth]
            child_indices: list[int] = []

            for parent_idx in parent_indices:
                # Determine pattern index for this parent
                pattern_index = self._determine_pattern_index(parent_idx, depth + 1)

                # Get pattern for children
                child_patterns = self._schema.hierarchy[str(depth + 1)]  # type: ignore[union-attr]
                pattern = child_patterns[pattern_index]["children"]
                children_count = len(pattern)

                # Calculate child offset using PIT arithmetic
                child_offset = self._calculate_child_offset(parent_idx, depth + 1)

                # Add all children indices
                child_indices.extend(range(child_offset, child_offset + children_count))

            indices_by_level[depth + 1] = child_indices

        return indices_by_level

    def _merge_levels_horizontal(
        self, level_dfs: list[tuple[int, pd.DataFrame]], extra_columns: list[str] | None
    ) -> pd.DataFrame:
        """
        Merge multiple level DataFrames horizontally with proper column handling.

        Creates merged id column (level0___level1___level2), takes last type,
        preserves internal:gdal_vsi, and includes requested extra columns.

        Args:
            level_dfs: List of (depth, DataFrame) tuples
            extra_columns: Extra columns to include from deepest level

        Returns:
            Merged DataFrame with id, type, gdal_vsi, and extra columns
        """
        if not level_dfs:
            return pd.DataFrame()

        # Collect id columns from all levels
        id_parts = []
        last_type = None
        gdal_vsi = None

        for depth, df in level_dfs:
            if "id" in df.columns:
                id_parts.append(df["id"])

            # Keep updating type (last one wins)
            if "type" in df.columns:
                last_type = df["type"]

            # Get gdal_vsi from deepest level
            if "internal:gdal_vsi" in df.columns:
                gdal_vsi = df["internal:gdal_vsi"]

        # Build result DataFrame
        result_data = {}

        # Merged id with ___
        if id_parts:
            result_data["id"] = id_parts[0].str.cat(id_parts[1:], sep="___")

        # Last type
        if last_type is not None:
            result_data["type"] = last_type

        # GDAL VSI
        if gdal_vsi is not None:
            result_data["internal:gdal_vsi"] = gdal_vsi

        # Add extra columns from deepest level
        if extra_columns:
            deepest_depth, deepest_df = level_dfs[-1]
            for col in extra_columns:
                if col in deepest_df.columns:
                    result_data[col] = deepest_df[col]

        return pd.DataFrame(result_data)

    def read(self, index: int | str) -> TacoDataFrame | str:
        """
        Navigate to child by position, ID, or path.

        Args:
            index: Position (int), ID (str), or path (str with ":")

        Returns:
            TacoDataFrame if FOLDER, VSI path if FILE

        Examples:
            >>> df.read(10)           # Position
            >>> df.read("sample_001") # ID
            >>> df.read("10:4:2")     # Path
        """
        if isinstance(index, str) and ":" in index:
            return self._read_path(index)

        if isinstance(index, str):
            return self._read_by_id(index)

        return self._read_by_position(index)

    def _read_path(self, path: str) -> TacoDataFrame | str:
        """Navigate using colon-separated path."""
        positions = [int(p) for p in path.split(":")]

        current: TacoDataFrame | str = TacoDataFrame(
            self._all_levels[0],
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=0,
            root_path=self._root_path,
            slice_offset=0,
        )

        for pos in positions:
            if isinstance(current, str):
                return current
            current = current._read_by_position(pos)

        return current

    def _read_by_id(self, node_id: str) -> TacoDataFrame | str:
        """Navigate by node ID in current DataFrame."""
        if "id" not in self.columns:
            raise ValueError("Current level has no 'id' column")

        matches = self[self["id"] == node_id]

        if len(matches) == 0:
            raise ValueError(f"Node '{node_id}' not found in current level")

        if len(matches) > 1:
            raise ValueError(f"Multiple nodes with id '{node_id}' found")

        row_idx = matches.index[0]
        node_type = matches.iloc[0]["type"]

        if node_type != "FOLDER":
            return str(matches.iloc[0]["internal:gdal_vsi"])

        child_depth = self._current_depth + 1

        if child_depth > self.max_depth:
            raise ValueError(f"FOLDER '{node_id}' has no children")

        # Cast row_idx to int - pandas index can be various types
        position_in_current = row_idx if isinstance(row_idx, int) else int(row_idx)  # type: ignore[arg-type]

        if self._schema is None:
            raise ValueError("Schema is required for navigation")

        # Calculate global position
        global_position: int = int(self._slice_offset) + position_in_current

        # Determine pattern_index
        pattern_index = self._determine_pattern_index(global_position, child_depth)

        # Get pattern
        child_patterns = self._schema.hierarchy[str(child_depth)]
        pattern = child_patterns[pattern_index]["children"]
        children_count = len(pattern)

        # Calculate child offset
        child_offset = self._calculate_child_offset(global_position, child_depth)

        next_level = self._all_levels[child_depth]
        children = next_level.iloc[child_offset : child_offset + children_count]

        return TacoDataFrame(
            children.reset_index(drop=True),
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=child_depth,
            root_path=self._root_path,
            slice_offset=child_offset,
        )

    def _read_by_position(self, position: int) -> TacoDataFrame | str:
        """Navigate by position using PIT arithmetic."""
        if position >= len(self):
            raise IndexError(f"Position {position} out of range (max {len(self)-1})")

        node_type = self.iloc[position]["type"]

        if node_type != "FOLDER":
            return str(self.iloc[position]["internal:gdal_vsi"])

        child_depth = self._current_depth + 1

        if child_depth > self.max_depth:
            raise ValueError(f"Node at position {position} has no children")

        if self._schema is None:
            raise ValueError("Schema is required for navigation")

        # Calculate global position: slice_offset + row position
        # Ensure both are int to avoid mypy errors
        global_position: int = int(self._slice_offset) + int(position)

        # Determine pattern_index based on global position
        pattern_index = self._determine_pattern_index(global_position, child_depth)

        # Get the correct pattern from child level
        child_patterns = self._schema.hierarchy[str(child_depth)]
        pattern = child_patterns[pattern_index]["children"]
        children_count = len(pattern)

        # Calculate global offset for children
        child_offset = self._calculate_child_offset(global_position, child_depth)

        next_level = self._all_levels[child_depth]
        children = next_level.iloc[child_offset : child_offset + children_count]

        return TacoDataFrame(
            children.reset_index(drop=True),
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=child_depth,
            root_path=self._root_path,
            slice_offset=child_offset,
        )

    def _determine_pattern_index(self, global_position: int, child_depth: int) -> int:
        """
        Determine which pattern to use based on global position.

        For level 0 → level 1: always use pattern 0 (Level-1 Uniformity)
        For level > 0: use position modulo parent pattern length
        """
        if self._schema is None:
            raise ValueError("Schema is required for pattern determination")

        if self._current_depth == 0:
            # Level-1 Uniformity: all children of TACO use same pattern
            return 0

        # Get parent level pattern
        parent_schema = self._schema.hierarchy[str(self._current_depth)]
        parent_pattern = parent_schema[0]["children"]
        parent_pattern_length = len(parent_pattern)

        # Pattern index is determined by position within repeating cycle
        pattern_index = global_position % parent_pattern_length

        # Validate pattern_index doesn't exceed available patterns
        child_patterns = self._schema.hierarchy.get(str(child_depth), [])
        if pattern_index >= len(child_patterns):
            pattern_index = 0  # Fallback to first pattern

        return pattern_index

    def _calculate_child_offset(self, parent_global_pos: int, child_depth: int) -> int:
        """
        Calculate starting offset for children in child level.

        Accounts for variable pattern sizes across different parent positions.
        Uses pure arithmetic based on schema - no lookups needed.
        """
        if self._schema is None:
            raise ValueError("Schema is required for offset calculation")

        if self._current_depth == 0:
            # Level 0 → Level 1: simple case (Level-1 Uniformity)
            parent_pattern = self._schema.hierarchy["1"][0]["children"]
            return parent_global_pos * len(parent_pattern)

        # Level > 0: need to count all children before this parent
        child_patterns = self._schema.hierarchy[str(child_depth)]
        parent_pattern = self._schema.hierarchy[str(self._current_depth)][0]["children"]
        parent_cycle_size = len(parent_pattern)

        # How many complete parent cycles before this position?
        full_cycles = parent_global_pos // parent_cycle_size
        position_in_cycle = parent_global_pos % parent_cycle_size

        # Count all nodes in complete cycles
        total_per_cycle = sum(len(p["children"]) for p in child_patterns)
        offset = full_cycles * total_per_cycle

        # Add nodes from positions before this one in the current cycle
        for i in range(position_in_cycle):
            if i < len(child_patterns):
                offset += len(child_patterns[i]["children"])

        return offset

    def get_level(self, depth: int) -> TacoDataFrame:
        """Get DataFrame at specific depth."""
        if depth < 0 or depth > self.max_depth:
            raise IndexError(f"Depth {depth} out of range [0, {self.max_depth}]")

        level_df = self._all_levels[depth]

        return TacoDataFrame(
            level_df,
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=depth,
            root_path=self._root_path,
            slice_offset=0,
        )

    def __getitem__(self, key: Any) -> TacoDataFrame | pd.Series | Any:  # type: ignore[override]
        """Preserve metadata on slicing/filtering."""
        result = super().__getitem__(key)

        if isinstance(result, pd.DataFrame):
            return TacoDataFrame(
                result,
                all_levels=self._all_levels,
                schema=self._schema,
                current_depth=self._current_depth,
                root_path=self._root_path,
                slice_offset=self._slice_offset,
            )

        return result

    def query(self, expr: str, **kwargs: Any) -> TacoDataFrame:  # type: ignore[override]
        """Query with metadata preservation."""
        result = super().query(expr, **kwargs)
        return TacoDataFrame(
            result,
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=self._current_depth,
            root_path=self._root_path,
            slice_offset=self._slice_offset,
        )

    def head(self, n: int = 5) -> TacoDataFrame:
        """Head with metadata preservation."""
        result = super().head(n)
        return TacoDataFrame(
            result,
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=self._current_depth,
            root_path=self._root_path,
            slice_offset=self._slice_offset,
        )

    def tail(self, n: int = 5) -> TacoDataFrame:
        """Tail with metadata preservation."""
        result = super().tail(n)
        # Note: tail changes offset
        new_offset = self._slice_offset + max(0, len(self) - n)
        return TacoDataFrame(
            result,
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=self._current_depth,
            root_path=self._root_path,
            slice_offset=new_offset,
        )

    def sample(self, *args: Any, **kwargs: Any) -> TacoDataFrame:
        """Sample with metadata preservation."""
        result = super().sample(*args, **kwargs)
        return TacoDataFrame(
            result,
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=self._current_depth,
            root_path=self._root_path,
            slice_offset=self._slice_offset,
        )

    def copy(self, deep: bool = True) -> TacoDataFrame:
        """Copy with metadata preservation."""
        copied_data = super().copy(deep=deep)

        if deep and self._all_levels:
            copied_levels = [df.copy(deep=True) for df in self._all_levels]
        else:
            copied_levels = self._all_levels

        return TacoDataFrame(
            copied_data,
            all_levels=copied_levels,
            schema=self._schema,
            current_depth=self._current_depth,
            root_path=self._root_path,
            slice_offset=self._slice_offset,
        )

    def reset_index(self, *args: Any, **kwargs: Any) -> TacoDataFrame | None:  # type: ignore[override]
        """
        Reset index - NOT ALLOWED for TacoDataFrame.

        Raises:
            ValueError: Always raises - reset_index breaks PIT navigation
        """
        raise ValueError(
            "reset_index() is not supported on TacoDataFrame as it breaks PIT navigation. "
            "If you need to reset the index, convert to regular DataFrame first: "
            "pd.DataFrame(tree).reset_index()"
        )

    def __repr__(self) -> str:
        """Custom representation (filters out padding samples for display)."""
        # Filter padding samples for display only (using ternary operator)
        display_df = (
            self[~self["id"].str.contains("__TACOPAD__", na=False)]
            if "id" in self.columns
            else self
        )

        # Use regular pandas repr (without TacoDataFrame metadata)
        base_repr = pd.DataFrame(display_df).__repr__()

        # Add tree info (different for flat vs hierarchical)
        if not self._all_levels:
            # Flat DataFrame
            tree_info = f"\n[TacoDataFrame: {len(self)} rows (flat)]"
        else:
            # Hierarchical DataFrame
            tree_info = (
                f"\n[TacoDataFrame: {len(self)} rows, "
                f"depth={self._current_depth}, "
                f"max_depth={self.max_depth}, "
                f"offset={self._slice_offset}]"
            )
        return base_repr + tree_info