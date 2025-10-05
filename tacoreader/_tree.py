from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from tacoreader._schema import PITSchema


class TacoDataFrame(pd.DataFrame):
    """
    Hierarchical DataFrame using Position-Isomorphic Tree navigation.

    Supports three navigation modes:
    1. By position: df.read(10).read(4).read(2)
    2. By ID: df.read("landslide_001").read("imagery")
    3. By path: df.read("10:4:2")

    Returns DataFrame if navigating to TORTILLA, VSI string if SAMPLE.
    """

    _metadata: ClassVar[list[str]] = ["_all_levels", "_schema", "_current_depth", "_root_path"]

    def __init__(
        self,
        data: Any = None,
        all_levels: list[pd.DataFrame] | None = None,
        schema: PITSchema | None = None,
        current_depth: int = 0,
        root_path: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize TacoDataFrame.

        Args:
            data: DataFrame data (current level)
            all_levels: List of all DataFrames [level0, level1, ...]
            schema: PIT schema for navigation
            current_depth: Current depth (0 = root)
            root_path: VSI root path
        """
        super().__init__(data, *args, **kwargs)

        self._all_levels = all_levels if all_levels is not None else []
        self._schema = schema
        self._current_depth = current_depth
        self._root_path = root_path

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

    def read(self, index: int | str) -> TacoDataFrame | str:
        """
        Navigate to child by position, ID, or path.

        Args:
            index: Position (int), ID (str), or path (str with ":")

        Returns:
            TacoDataFrame if TORTILLA, VSI path if SAMPLE

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

        # Search in current DataFrame
        matches = self[self["id"] == node_id]

        if len(matches) == 0:
            raise ValueError(f"Node '{node_id}' not found in current level")

        if len(matches) > 1:
            raise ValueError(f"Multiple nodes with id '{node_id}' found")

        # Get the matched row
        row_idx = matches.index[0]
        node_type = matches.iloc[0]["type"]

        # If SAMPLE, return VSI path
        if node_type != "TORTILLA":
            return str(matches.iloc[0]["internal:gdal_vsi"])

        # If TORTILLA, get its children from next level
        child_depth = self._current_depth + 1

        if child_depth > self.max_depth:
            raise ValueError(f"TORTILLA '{node_id}' has no children")

        # Find the position of this TORTILLA in the current DataFrame
        position_in_current = row_idx

        # Schema is required for navigation
        if self._schema is None:
            raise ValueError("Schema is required for navigation")

        # Calculate children offset using PIT
        pattern = self._schema.get_pattern(child_depth, pattern_index=0)
        children_count = len(pattern)
        child_offset = self._schema.calculate_child_offset(
            position_in_current, children_count
        )

        next_level = self._all_levels[child_depth]
        children = next_level.iloc[child_offset : child_offset + children_count]

        return TacoDataFrame(
            children.reset_index(drop=True),
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=child_depth,
            root_path=self._root_path,
        )

    def _read_by_position(self, position: int) -> TacoDataFrame | str:
        """Navigate by position using PIT arithmetic."""
        if position >= len(self):
            raise IndexError(f"Position {position} out of range (max {len(self)-1})")

        node_type = self.iloc[position]["type"]

        # If current node is not TORTILLA, just return its VSI
        if node_type != "TORTILLA":
            return str(self.iloc[position]["internal:gdal_vsi"])

        # TORTILLA: get its children from next level
        child_depth = self._current_depth + 1

        if child_depth > self.max_depth:
            raise ValueError(f"Node at position {position} has no children")

        # Schema is required for navigation
        if self._schema is None:
            raise ValueError("Schema is required for navigation")

        # Calculate children offset using PIT
        pattern = self._schema.get_pattern(child_depth, pattern_index=0)
        children_count = len(pattern)
        child_offset = self._schema.calculate_child_offset(position, children_count)

        next_level = self._all_levels[child_depth]
        children = next_level.iloc[child_offset : child_offset + children_count]

        return TacoDataFrame(
            children.reset_index(drop=True),
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=child_depth,
            root_path=self._root_path,
        )

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
        )

    def __getitem__(self, key: Any) -> TacoDataFrame | pd.Series | Any:
        """Preserve metadata on slicing/filtering."""
        result = super().__getitem__(key)

        if isinstance(result, pd.DataFrame):
            return TacoDataFrame(
                result,
                all_levels=self._all_levels,
                schema=self._schema,
                current_depth=self._current_depth,
                root_path=self._root_path,
            )

        return result

    def query(self, expr: str, **kwargs: Any) -> TacoDataFrame:
        """Query with metadata preservation."""
        result = super().query(expr, **kwargs)
        return TacoDataFrame(
            result,
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=self._current_depth,
            root_path=self._root_path,
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
        )

    def tail(self, n: int = 5) -> TacoDataFrame:
        """Tail with metadata preservation."""
        result = super().tail(n)
        return TacoDataFrame(
            result,
            all_levels=self._all_levels,
            schema=self._schema,
            current_depth=self._current_depth,
            root_path=self._root_path,
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
        )

    def reset_index(self, *args: Any, **kwargs: Any) -> TacoDataFrame | None:
        """
        Reset index - NOT RECOMMENDED for TacoDataFrame.

        Raises:
            ValueError: Always raises - reset_index breaks PIT navigation
        """
        raise ValueError(
            "reset_index() is not supported on TacoDataFrame as it breaks PIT navigation. "
            "If you need to reset the index, convert to regular DataFrame first: "
            "pd.DataFrame(tree).reset_index()"
        )

    def __repr__(self) -> str:
        """Custom representation."""
        base_repr = super().__repr__()
        tree_info = (
            f"\n[TacoDataFrame: {len(self)} rows, "
            f"depth={self._current_depth}, "
            f"max_depth={self.max_depth}]"
        )
        return base_repr + tree_info
