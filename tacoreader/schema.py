"""
Position-Isomorphic Tree (PIT) schema for TACO datasets.

The PIT schema enforces structural homogeneity across hierarchical datasets,
enabling predictable navigation and efficient vectorized operations. All samples
at the same level must have identical structure (same types, same child counts,
same child IDs).

Core Concepts:
    - Position-Isomorphic: All nodes at same level have identical structure
    - Homogeneity: All folders at same level have same children (type and ID)
    - Padding: __TACOPAD__ samples maintain uniformity when counts differ
    - Compatibility: Datasets can be concatenated if schemas are structurally identical

PIT Structure Example:
    {
        "root": {
            "n": 1000,              # Total samples at level 0
            "type": "FOLDER"        # All level 0 samples are FOLDERs
        },
        "hierarchy": {
            "1": [                  # Level 1 (children of level 0)
                {
                    "n": 3000,      # Total samples at level 1
                    "type": ["FILE", "FILE", "FILE"],  # Pattern: 3 FILEs per folder
                    "id": ["red", "green", "blue"]     # IDs must match across all folders
                }
            ],
            "2": [                  # Level 2 (children of level 1, if any)
                {
                    "n": 6000,
                    "type": ["FILE", "FILE"],
                    "id": ["metadata", "thumbnail"]
                }
            ]
        }
    }

This structure guarantees that:
    - Every folder at level 0 has exactly 3 children: red, green, blue (all FILEs)
    - Every folder at level 1 has exactly 2 children: metadata, thumbnail (all FILEs)
    - Queries can safely assume structure without checking each folder

Main classes:
    PITRootLevel: Root level descriptor (TypedDict)
    PITPattern: Pattern descriptor for hierarchy levels (TypedDict)
    PITSchemaDict: Complete schema structure (TypedDict)
    PITSchema: Schema validator and compatibility checker

Example:
    >>> from tacoreader.schema import PITSchema
    >>> 
    >>> schema_dict = {
    ...     "root": {"n": 100, "type": "FOLDER"},
    ...     "hierarchy": {
    ...         "1": [{"n": 300, "type": ["FILE", "FILE", "FILE"], "id": ["r", "g", "b"]}]
    ...     }
    ... }
    >>> schema = PITSchema(schema_dict)
    >>> print(schema)
    PITSchema(root=FOLDER, max_depth=1)
    >>> 
    >>> # Check compatibility for concatenation
    >>> schema2 = PITSchema(other_dict)
    >>> if schema.is_compatible(schema2):
    ...     combined = concat([ds1, ds2])
"""

from typing import Literal, TypedDict

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================


class PITRootLevel(TypedDict):
    """
    Root level descriptor (level 0).

    Describes the top-level samples in the dataset. All samples at
    root level must have the same type (either all FILE or all FOLDER).

    Attributes:
        n: Total number of samples at root level
        type: Sample type - either "FILE" or "FOLDER"

    Example:
        >>> root = {"n": 100, "type": "FOLDER"}
        >>> # This means: 100 folders at level 0, each with identical structure
    """

    n: int
    type: Literal["FOLDER", "FILE"]


class PITPattern(TypedDict):
    """
    Pattern descriptor for a specific position in the hierarchy.

    Defines the structure that ALL parents at a given level share.
    The type and id arrays describe the children that every parent has.

    Position-isomorphic means: if folder A has children [X, Y, Z],
    then ALL folders at that level have children [X, Y, Z].

    Attributes:
        n: Total number of samples at this level (across all parents)
        type: Array of child types (FILE or FOLDER) - defines pattern
        id: Array of child IDs - must match across all parents

    Example:
        >>> pattern = {
        ...     "n": 300,
        ...     "type": ["FILE", "FILE", "FILE"],
        ...     "id": ["red", "green", "blue"]
        ... }
        >>> # This means: Every parent has 3 children (red, green, blue - all FILEs)
        >>> # With 100 parents, total samples = 100 * 3 = 300
    """

    n: int
    type: list[str]
    id: list[str]


class PITSchemaDict(TypedDict):
    """
    Complete PIT schema structure.

    Combines root descriptor with hierarchy patterns to define
    the complete tree structure. Hierarchy keys are string integers
    representing depth levels ("1", "2", "3", etc.).

    Attributes:
        root: Root level descriptor
        hierarchy: Mapping from depth to list of patterns at that depth

    Example:
        >>> schema = {
        ...     "root": {"n": 100, "type": "FOLDER"},
        ...     "hierarchy": {
        ...         "1": [{"n": 300, "type": ["FILE"] * 3, "id": ["r", "g", "b"]}],
        ...         "2": []  # Level 1 FILEs have no children
        ...     }
        ... }
    """

    root: PITRootLevel
    hierarchy: dict[str, list[PITPattern]]


# ============================================================================
# PIT SCHEMA CLASS
# ============================================================================


class PITSchema:
    """
    Position-Isomorphic Tree schema validator and compatibility checker.

    Validates PIT schema structure and provides methods for checking
    compatibility between datasets. Two datasets can be concatenated
    only if their schemas are structurally identical (ignoring 'n' values).

    The PIT schema ensures:
    - All samples at same level have same type
    - All folders at same level have identical children (type and ID)
    - Predictable structure for efficient queries
    - Safe concatenation of compatible datasets

    Attributes:
        root: Root level descriptor (PITRootLevel)
        hierarchy: Hierarchy patterns by depth (dict[str, list[PITPattern]])

    Example:
        >>> schema = PITSchema({
        ...     "root": {"n": 100, "type": "FOLDER"},
        ...     "hierarchy": {
        ...         "1": [{"n": 300, "type": ["FILE", "FILE"], "id": ["a", "b"]}]
        ...     }
        ... })
        >>>
        >>> # Validate structure
        >>> schema._validate()  # Raises ValueError if invalid
        >>>
        >>> # Check compatibility
        >>> other = PITSchema(other_dict)
        >>> if schema.is_compatible(other):
        ...     print("Can concatenate these datasets!")
        >>>
        >>> # Update counts
        >>> filtered = schema.with_n(50)
        >>> print(filtered.root["n"])
        50
    """

    def __init__(self, schema_dict: PITSchemaDict) -> None:
        """
        Initialize PIT schema from dictionary.

        Args:
            schema_dict: Dictionary with 'root' and 'hierarchy' keys

        Raises:
            ValueError: If schema structure is invalid

        Example:
            >>> schema = PITSchema({
            ...     "root": {"n": 100, "type": "FOLDER"},
            ...     "hierarchy": {"1": [...]}
            ... })
        """
        self.root = schema_dict["root"]
        self.hierarchy = schema_dict["hierarchy"]
        self._validate()

    def _validate(self) -> None:
        """
        Validate schema structure and constraints.

        Checks:
        - Root type is valid (FILE or FOLDER)
        - Hierarchy keys are numeric strings
        - Each pattern has required fields (type, id)
        - Type and id arrays are non-empty and same length
        - All child types are valid (FILE or FOLDER)

        Raises:
            ValueError: If any validation check fails, with detailed message

        Example:
            >>> schema._validate()  # Silent if valid
            >>>
            >>> # Invalid schema raises error
            >>> bad = PITSchema({"root": {"n": 1, "type": "INVALID"}})
            ValueError: Invalid root type: INVALID
        """
        # Validate root type
        if self.root["type"] not in ("FOLDER", "FILE"):
            raise ValueError(
                f"Invalid root type: {self.root['type']}\n"
                f"Root type must be 'FOLDER' or 'FILE'"
            )

        # Validate hierarchy structure
        for depth_str, patterns in self.hierarchy.items():
            # Check depth key is numeric
            if not depth_str.isdigit():
                raise ValueError(
                    f"Invalid depth key: {depth_str}\n"
                    f"Depth keys must be numeric strings (e.g., '1', '2', '3')"
                )

            # Validate each pattern at this depth
            for i, pattern in enumerate(patterns):
                # Check required fields
                if "type" not in pattern or "id" not in pattern:
                    raise ValueError(
                        f"Depth {depth_str}, pattern {i}: missing required field\n"
                        f"Patterns must have both 'type' and 'id' fields"
                    )

                types = pattern["type"]
                ids = pattern["id"]

                # Check non-empty arrays
                if not types or len(types) == 0:
                    raise ValueError(
                        f"Depth {depth_str}, pattern {i}: type array is empty\n"
                        f"Pattern must define at least one child type"
                    )

                if not ids or len(ids) == 0:
                    raise ValueError(
                        f"Depth {depth_str}, pattern {i}: id array is empty\n"
                        f"Pattern must define at least one child ID"
                    )

                # Check arrays have same length
                if len(types) != len(ids):
                    raise ValueError(
                        f"Depth {depth_str}, pattern {i}: type and id arrays have different lengths\n"
                        f"Type array length: {len(types)}\n"
                        f"ID array length: {len(ids)}\n"
                        f"Both arrays must have same length"
                    )

                # Validate each child type
                for child_type in types:
                    if child_type not in ("FOLDER", "FILE"):
                        raise ValueError(
                            f"Depth {depth_str}, pattern {i}: invalid child type '{child_type}'\n"
                            f"Child types must be 'FOLDER' or 'FILE'"
                        )

    def is_compatible(self, other: "PITSchema") -> bool:
        """
        Check if another schema is structurally identical.

        Two schemas are compatible if they have the same structure,
        ignoring the 'n' (count) values. This means:
        - Same root type
        - Same hierarchy depths
        - Same patterns at each depth (type and id arrays match)

        Compatible schemas can be safely concatenated, with counts summed.

        Args:
            other: Another PITSchema to compare

        Returns:
            True if schemas are structurally identical, False otherwise

        Example:
            >>> schema1 = PITSchema({"root": {"n": 100, "type": "FOLDER"}, "hierarchy": {...}})
            >>> schema2 = PITSchema({"root": {"n": 200, "type": "FOLDER"}, "hierarchy": {...}})
            >>>
            >>> if schema1.is_compatible(schema2):
            ...     # Can concatenate: n values will be summed (100 + 200 = 300)
            ...     combined = concat([ds1, ds2])
            ... else:
            ...     print("Incompatible structures - cannot concatenate")
        """
        # Check root type (ignoring n)
        if self.root["type"] != other.root["type"]:
            return False

        # Check same hierarchy depths
        if set(self.hierarchy.keys()) != set(other.hierarchy.keys()):
            return False

        # Check patterns at each depth
        for depth_str in self.hierarchy:
            self_patterns = self.hierarchy[depth_str]
            other_patterns = other.hierarchy[depth_str]

            # Check same number of patterns
            if len(self_patterns) != len(other_patterns):
                return False

            # Check each pattern matches
            for self_p, other_p in zip(self_patterns, other_patterns, strict=False):
                # Type arrays must match exactly
                if self_p["type"] != other_p["type"]:
                    return False

                # ID arrays must match exactly
                if self_p["id"] != other_p["id"]:
                    return False

        return True

    def to_dict(self) -> PITSchemaDict:
        """
        Convert schema to dictionary format.

        Returns:
            Dictionary with 'root' and 'hierarchy' keys

        Example:
            >>> schema_dict = schema.to_dict()
            >>> print(schema_dict["root"])
            {'n': 100, 'type': 'FOLDER'}
        """
        return {"root": self.root, "hierarchy": self.hierarchy}

    def with_n(self, new_n: int) -> "PITSchema":
        """
        Create new schema with updated root count.

        Useful after filtering operations that change the number of
        samples at root level. Creates a new PITSchema instance with
        the same structure but different count.

        Args:
            new_n: New count for root level

        Returns:
            New PITSchema instance with updated root['n']

        Example:
            >>> original = PITSchema({"root": {"n": 1000, "type": "FOLDER"}, ...})
            >>> print(original.root["n"])
            1000
            >>>
            >>> # After SQL filter that returns 100 rows
            >>> filtered = original.with_n(100)
            >>> print(filtered.root["n"])
            100
            >>>
            >>> # Original unchanged
            >>> print(original.root["n"])
            1000
        """
        import copy

        schema_dict = copy.deepcopy(self.to_dict())
        schema_dict["root"]["n"] = new_n
        return PITSchema(schema_dict)

    def max_depth(self) -> int:
        """
        Calculate maximum hierarchy depth.

        Returns the deepest level in the hierarchy. Root level (0)
        is not counted, so max_depth=2 means 3 total levels (0, 1, 2).

        Returns:
            Maximum depth as integer (0-5)

        Example:
            >>> schema = PITSchema({
            ...     "root": {"n": 100, "type": "FOLDER"},
            ...     "hierarchy": {
            ...         "1": [...],
            ...         "2": [...]
            ...     }
            ... })
            >>> print(schema.max_depth())
            2  # Levels: 0 (root), 1, 2
        """
        if not self.hierarchy:
            return 0
        return max((int(k) for k in self.hierarchy.keys()), default=0)

    def __repr__(self) -> str:
        """
        String representation showing root type and max depth.

        Returns:
            String like "PITSchema(root=FOLDER, max_depth=2)"
        """
        max_depth = self.max_depth()
        return f"PITSchema(root={self.root['type']}, max_depth={max_depth})"
