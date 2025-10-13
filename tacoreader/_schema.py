from typing import Any, Literal, TypedDict


class PITRootLevel(TypedDict):
    """Root level descriptor (level 0 - the collection)."""

    n: int  # Number of items in collection
    type: Literal["FOLDER", "FILE"]  # Node type


class PITPattern(TypedDict):
    """Pattern descriptor for a position in the hierarchy."""

    n: int  # Total nodes at this depth for this pattern
    children: list[str]  # Ordered array of child types


class PITSchemaDict(TypedDict):
    """PIT schema dictionary structure."""

    root: PITRootLevel
    hierarchy: dict[str, list[PITPattern]]  # depth_str -> patterns array


class PITSchema:
    """
    Position-Isomorphic Tree schema for deterministic navigation.

    Provides O(1) structure lookups and arithmetic parent/child calculations.
    """

    def __init__(self, schema_dict: PITSchemaDict) -> None:
        """
        Initialize PIT schema.

        Args:
            schema_dict: Schema dictionary with root and hierarchy

        Raises:
            ValueError: If schema structure is invalid
        """
        self.root = schema_dict["root"]
        self.hierarchy = schema_dict["hierarchy"]
        self._validate()

    def _validate(self) -> None:
        """Validate schema structure and consistency."""
        # Validate root
        if self.root["type"] not in ("FOLDER", "FILE"):
            raise ValueError(
                f"Invalid root type: {self.root['type']}. " "Must be 'FOLDER' or 'FILE'"
            )

        # Validate hierarchy levels
        for depth_str, patterns in self.hierarchy.items():
            if not depth_str.isdigit():
                raise ValueError(
                    f"Invalid depth key: {depth_str}. Must be numeric string"
                )

            for i, pattern in enumerate(patterns):
                if not pattern["children"]:
                    raise ValueError(
                        f"Depth {depth_str}, pattern {i}: children cannot be empty"
                    )

                if len(pattern["children"]) == 0:
                    raise ValueError(
                        f"Depth {depth_str}, pattern {i}: must have at least one child"
                    )

                # Validate child types
                for child_type in pattern["children"]:
                    if child_type not in ("FOLDER", "FILE"):
                        raise ValueError(
                            f"Depth {depth_str}, pattern {i}: "
                            f"invalid child type '{child_type}'"
                        )

    def get_pattern(self, depth: int, pattern_index: int = 0) -> list[str]:
        """
        Get child type pattern at given depth and pattern index.

        Args:
            depth: Depth in hierarchy (1, 2, 3, ...)
            pattern_index: Index into patterns array at this depth (default 0)

        Returns:
            List of child types (e.g., ['FILE', 'FOLDER', 'FILE'])

        Raises:
            ValueError: If depth or pattern_index is out of range

        Examples:
            >>> schema.get_pattern(1, 0)
            ['FILE', 'FOLDER']
            >>> schema.get_pattern(2, 0)
            ['FILE', 'FILE']
        """
        depth_str = str(depth)
        if depth_str not in self.hierarchy:
            raise ValueError(f"No patterns defined for depth {depth}")

        patterns = self.hierarchy[depth_str]
        if pattern_index >= len(patterns):
            raise ValueError(
                f"Pattern index {pattern_index} out of range for depth {depth}"
            )

        return patterns[pattern_index]["children"]

    def get_children_count(self, depth: int, pattern_index: int = 0) -> int:
        """
        Get number of children for a pattern at given depth.

        Args:
            depth: Depth in hierarchy
            pattern_index: Index into patterns array at this depth

        Returns:
            Number of children in the pattern

        Examples:
            >>> schema.get_children_count(1, 0)
            2  # ['FILE', 'FOLDER'] -> 2 children
        """
        pattern = self.get_pattern(depth, pattern_index)
        return len(pattern)

    def calculate_parent(self, child_row: int, children_per_parent: int) -> int:
        """
        Calculate parent row from child row using PIT arithmetic.

        Core PIT operation: parent_row = floor(child_row / children_per_parent)

        Args:
            child_row: Row index in child level
            children_per_parent: Number of children each parent has

        Returns:
            Row index of parent in parent level

        Examples:
            >>> schema.calculate_parent(4, 2)
            2  # Row 4 with 2 children per parent -> parent at row 2
            >>> schema.calculate_parent(1, 2)
            0  # Row 1 with 2 children per parent -> parent at row 0
        """
        if children_per_parent <= 0:
            raise ValueError("children_per_parent must be positive")
        if child_row < 0:
            raise ValueError("child_row must be non-negative")

        return child_row // children_per_parent

    def calculate_child_offset(self, parent_row: int, children_per_parent: int) -> int:
        """
        Calculate starting row index for parent's children.

        Args:
            parent_row: Row index of parent
            children_per_parent: Number of children each parent has

        Returns:
            Starting row index in child level

        Examples:
            >>> schema.calculate_child_offset(2, 3)
            6  # Parent at row 2 with 3 children -> children start at row 6
        """
        if children_per_parent <= 0:
            raise ValueError("children_per_parent must be positive")
        if parent_row < 0:
            raise ValueError("parent_row must be non-negative")

        return parent_row * children_per_parent

    def is_compatible(self, other: "PITSchema") -> bool:
        """
        Check if another schema is structurally identical to this one.

        Only compares structure (types and patterns), NOT counts (n values).
        Used for multi-file validation - all files must have identical structure.

        Args:
            other: Another PITSchema to compare

        Returns:
            True if schemas are structurally identical, False otherwise

        Examples:
            >>> schema1.is_compatible(schema2)
            True
        """
        # Compare root type only (ignore n)
        if self.root["type"] != other.root["type"]:
            return False

        # Compare hierarchy depths
        if set(self.hierarchy.keys()) != set(other.hierarchy.keys()):
            return False

        # Compare patterns at each depth (ignore n values)
        for depth_str in self.hierarchy:
            # Extract only the children patterns (not n)
            self_patterns = [p["children"] for p in self.hierarchy[depth_str]]
            other_patterns = [p["children"] for p in other.hierarchy[depth_str]]

            if self_patterns != other_patterns:
                return False

        return True

    def max_depth(self) -> int:
        """
        Get maximum depth in the hierarchy.

        Returns:
            Maximum depth (0 if only root, 1+ for hierarchies)

        Examples:
            >>> schema.max_depth()
            2  # Has depth 1 and 2 in hierarchy
        """
        if not self.hierarchy:
            return 0
        return max(int(depth_str) for depth_str in self.hierarchy)

    def to_dict(self) -> PITSchemaDict:
        """
        Convert schema back to dictionary format.

        Returns:
            Schema as dictionary
        """
        return {"root": self.root, "hierarchy": self.hierarchy}

    def __repr__(self) -> str:
        """String representation."""
        return f"PITSchema(root={self.root['type']}, max_depth={self.max_depth()})"


def validate_schemas(schemas: list[PITSchema]) -> None:
    """
    Validate that all schemas are structurally identical.

    Used when loading multiple files - they must have compatible schemas.

    Args:
        schemas: List of PITSchema objects to compare

    Raises:
        ValueError: If any schemas differ

    Examples:
        >>> validate_schemas([schema1, schema2, schema3])
        # Raises if any differ
    """
    if len(schemas) < 2:
        return  # Nothing to validate

    reference = schemas[0]
    for i, schema in enumerate(schemas[1:], start=1):
        if not reference.is_compatible(schema):
            raise ValueError(
                f"Schema mismatch: File 0 and File {i} have different PIT schemas. "
                "All files must have identical schemas to be loaded together."
            )


def extract_schema_from_collection(collection: dict[str, Any]) -> PITSchema:
    """
    Extract PIT schema from COLLECTION.json data.

    Args:
        collection: COLLECTION.json dictionary

    Returns:
        PITSchema object

    Raises:
        ValueError: If taco:pit_schema is missing or invalid

    Examples:
        >>> schema = extract_schema_from_collection(collection_dict)
    """
    if "taco:pit_schema" not in collection:
        raise ValueError("COLLECTION.json missing 'taco:pit_schema' field")

    schema_dict = collection["taco:pit_schema"]

    if "root" not in schema_dict or "hierarchy" not in schema_dict:
        raise ValueError(
            "Invalid PIT schema: must contain 'root' and 'hierarchy' fields"
        )

    return PITSchema(schema_dict)


def merge_schemas(schemas: list[PITSchema]) -> PITSchema:
    """
    Merge multiple schemas by summing 'n' values.

    Validates all schemas are structurally identical (same patterns, same types),
    then creates a merged schema by summing all node counts.

    Used when loading multiple TACO files - they must have compatible schemas
    but can have different numbers of nodes.

    Args:
        schemas: List of PITSchema objects to merge

    Returns:
        Merged PITSchema with summed 'n' values

    Raises:
        ValueError: If schemas are not structurally identical
        ValueError: If schemas list is empty

    Examples:
        >>> # Two files with same structure, different counts
        >>> schema1 = PITSchema({"root": {"n": 500, "type": "FOLDER"}, ...})
        >>> schema2 = PITSchema({"root": {"n": 300, "type": "FOLDER"}, ...})
        >>> merged = merge_schemas([schema1, schema2])
        >>> merged.root["n"]
        800
    """
    if not schemas:
        raise ValueError("Cannot merge empty list of schemas")

    if len(schemas) == 1:
        return schemas[0]

    # Validate all schemas are structurally identical
    validate_schemas(schemas)

    # Use first schema as template
    merged_dict = schemas[0].to_dict()

    # Sum root 'n' values
    merged_dict["root"]["n"] = sum(s.root["n"] for s in schemas)

    # Sum 'n' values in each hierarchy pattern
    for depth_str in merged_dict["hierarchy"]:
        patterns = merged_dict["hierarchy"][depth_str]
        for pattern_idx in range(len(patterns)):
            # Sum 'n' across all schemas for this specific pattern
            merged_dict["hierarchy"][depth_str][pattern_idx]["n"] = sum(
                s.hierarchy[depth_str][pattern_idx]["n"] for s in schemas
            )

    return PITSchema(merged_dict)
