"""
Position-Invariant Tree (PIT) schema for TACO datasets.

Enforces structural homogeneity: all samples at same level have identical structure
(same types, same child counts, same child IDs). Enables predictable navigation
and efficient vectorized operations.

Core concepts:
    - Position-Invariant: All nodes at same level have identical structure
    - Homogeneity: All folders at same level have same children (type and ID)
    - Padding: __TACOPAD__ samples maintain uniformity when counts differ
    - Compatibility: Datasets can be concatenated if schemas match structurally

PIT example:
    {
        "root": {"n": 1000, "type": "FOLDER"},
        "hierarchy": {
            "1": [{
                "n": 3000,
                "type": ["FILE", "FILE", "FILE"],
                "id": ["red", "green", "blue"]
            }]
        }
    }

This guarantees every level-0 folder has exactly 3 children: red, green, blue (all FILEs).

Main classes:
    PITSchema: Schema validator and compatibility checker
"""

from typing import Literal, TypedDict

from tacoreader._constants import (
    SAMPLE_TYPE_FILE,
    SAMPLE_TYPE_FOLDER,
    VALID_SAMPLE_TYPES,
)
from tacoreader._exceptions import TacoSchemaError


class PITRootLevel(TypedDict):
    """Root level descriptor (level 0)."""

    n: int
    type: Literal["FOLDER", "FILE"]


class PITPattern(TypedDict):
    """
    Pattern descriptor for hierarchy level.

    Position-Invariant: if folder A has children [X, Y, Z],
    ALL folders at that level have children [X, Y, Z].
    """

    n: int
    type: list[str]
    id: list[str]


class PITSchemaDict(TypedDict):
    """Complete PIT schema structure."""

    root: PITRootLevel
    hierarchy: dict[str, list[PITPattern]]


class PITSchema:
    """
    PIT schema validator and compatibility checker.

    Validates structure and checks compatibility between datasets.
    Two datasets can be concatenated only if schemas are structurally
    identical (ignoring 'n' values).
    """

    def __init__(self, schema_dict: PITSchemaDict) -> None:
        """Initialize and validate PIT schema."""
        self.root = schema_dict["root"]
        self.hierarchy = schema_dict["hierarchy"]
        self._validate()

    def _validate_pattern(
        self, depth_str: str, pattern_idx: int, pattern: PITPattern
    ) -> None:
        """Validate a single pattern at given depth."""
        # Check required fields
        if "type" not in pattern or "id" not in pattern:
            raise TacoSchemaError(
                f"Depth {depth_str}, pattern {pattern_idx}: missing required field\n"
                f"Patterns must have both 'type' and 'id'"
            )

        types = pattern["type"]
        ids = pattern["id"]

        # Check non-empty arrays
        if not types:
            raise TacoSchemaError(
                f"Depth {depth_str}, pattern {pattern_idx}: type array empty"
            )

        if not ids:
            raise TacoSchemaError(
                f"Depth {depth_str}, pattern {pattern_idx}: id array empty"
            )

        # Check same length
        if len(types) != len(ids):
            raise TacoSchemaError(
                f"Depth {depth_str}, pattern {pattern_idx}: type and id arrays differ\n"
                f"Type: {len(types)}, ID: {len(ids)}"
            )

        # Validate child types
        for child_type in types:
            if child_type not in VALID_SAMPLE_TYPES:
                raise TacoSchemaError(
                    f"Depth {depth_str}, pattern {pattern_idx}: invalid type '{child_type}'"
                )

    def _validate(self) -> None:
        """
        Validate schema structure.

        Checks:
        - Root type is FILE or FOLDER
        - Hierarchy keys are numeric strings
        - Each pattern has type/id fields
        - Type and id arrays are non-empty and same length
        - All child types are valid
        """
        # Validate root type
        if self.root["type"] not in VALID_SAMPLE_TYPES:
            raise TacoSchemaError(
                f"Invalid root type: {self.root['type']}\n"
                f"Must be '{SAMPLE_TYPE_FILE}' or '{SAMPLE_TYPE_FOLDER}'"
            )

        # Validate hierarchy
        for depth_str, patterns in self.hierarchy.items():
            if not depth_str.isdigit():
                raise TacoSchemaError(
                    f"Invalid depth key: {depth_str}\n"
                    f"Depth keys must be numeric strings"
                )

            for i, pattern in enumerate(patterns):
                self._validate_pattern(depth_str, i, pattern)

    def is_compatible(self, other: "PITSchema") -> bool:
        """
        Check if another schema is structurally identical.

        Ignores 'n' values - only checks structure (root type, hierarchy
        depths, type/id patterns). Compatible schemas can be concatenated.
        """
        # Check root type (ignore n)
        if self.root["type"] != other.root["type"]:
            return False

        # Check same hierarchy depths
        if set(self.hierarchy.keys()) != set(other.hierarchy.keys()):
            return False

        # Check patterns at each depth
        for depth_str in self.hierarchy:
            self_patterns = self.hierarchy[depth_str]
            other_patterns = other.hierarchy[depth_str]

            if len(self_patterns) != len(other_patterns):
                return False

            for self_p, other_p in zip(self_patterns, other_patterns, strict=False):
                if self_p["type"] != other_p["type"]:
                    return False

                if self_p["id"] != other_p["id"]:
                    return False

        return True

    def to_dict(self) -> PITSchemaDict:
        """Convert schema to dictionary."""
        return {"root": self.root, "hierarchy": self.hierarchy}

    def with_n(self, new_n: int) -> "PITSchema":
        """
        Create new schema with updated root count.

        Useful after filtering - creates new instance with same structure,
        different count. Original unchanged.
        """
        import copy

        schema_dict = copy.deepcopy(self.to_dict())
        schema_dict["root"]["n"] = new_n
        return PITSchema(schema_dict)

    def max_depth(self) -> int:
        """
        Max hierarchy depth.

        Root level (0) not counted, so max_depth=2 means 3 total levels (0, 1, 2).
        """
        if not self.hierarchy:
            return 0
        return max((int(k) for k in self.hierarchy), default=0)

    def __repr__(self) -> str:
        """String representation showing root type and max depth."""
        max_depth = self.max_depth()
        return f"PITSchema(root={self.root['type']}, max_depth={max_depth})"
