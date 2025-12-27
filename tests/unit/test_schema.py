# tests/unit/test_schema.py
"""Unit tests for PITSchema validation."""

import pytest

from tacoreader.schema import PITSchema
from tacoreader._exceptions import TacoSchemaError


class TestPITSchemaValidation:
    """Test PITSchema initialization and validation."""

    def test_valid_file_root(self):
        """Valid schema with FILE root, no hierarchy."""
        schema = PITSchema({
            "root": {"n": 100, "type": "FILE"},
            "hierarchy": {},
        })
        assert schema.root["type"] == "FILE"
        assert schema.root["n"] == 100
        assert schema.max_depth() == 0

    def test_valid_folder_root_with_hierarchy(self):
        """Valid schema with FOLDER root and children."""
        schema = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {
                "1": [{
                    "n": 30,
                    "type": ["FILE", "FILE", "FILE"],
                    "id": ["red", "green", "blue"],
                }]
            },
        })
        assert schema.root["type"] == "FOLDER"
        assert schema.max_depth() == 1

    def test_invalid_root_type_raises(self):
        """Invalid root type raises TacoSchemaError."""
        with pytest.raises(TacoSchemaError, match="Invalid root type"):
            PITSchema({
                "root": {"n": 10, "type": "INVALID"},
                "hierarchy": {},
            })

    def test_invalid_child_type_raises(self):
        """Invalid child type raises TacoSchemaError."""
        with pytest.raises(TacoSchemaError, match="invalid type"):
            PITSchema({
                "root": {"n": 10, "type": "FOLDER"},
                "hierarchy": {
                    "1": [{"n": 10, "type": ["BADTYPE"], "id": ["x"]}]
                },
            })

    def test_mismatched_type_id_length_raises(self):
        """Mismatched type/id arrays raise TacoSchemaError."""
        with pytest.raises(TacoSchemaError, match="type and id arrays differ"):
            PITSchema({
                "root": {"n": 10, "type": "FOLDER"},
                "hierarchy": {
                    "1": [{"n": 10, "type": ["FILE", "FILE"], "id": ["only_one"]}]
                },
            })


class TestPITSchemaCompatibility:
    """Test schema compatibility for concat."""

    def test_compatible_same_structure(self):
        """Same structure, different n → compatible."""
        schema_a = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [{"n": 20, "type": ["FILE"], "id": ["data"]}]},
        })
        schema_b = PITSchema({
            "root": {"n": 5, "type": "FOLDER"},
            "hierarchy": {"1": [{"n": 10, "type": ["FILE"], "id": ["data"]}]},
        })
        assert schema_a.is_compatible(schema_b)

    def test_incompatible_different_root_type(self):
        """Different root type → incompatible."""
        schema_a = PITSchema({"root": {"n": 10, "type": "FOLDER"}, "hierarchy": {}})
        schema_b = PITSchema({"root": {"n": 10, "type": "FILE"}, "hierarchy": {}})
        assert not schema_a.is_compatible(schema_b)

    def test_incompatible_different_hierarchy(self):
        """Different hierarchy depth → incompatible."""
        schema_a = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [{"n": 10, "type": ["FILE"], "id": ["x"]}]},
        })
        schema_b = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {},
        })
        assert not schema_a.is_compatible(schema_b)


class TestPITSchemaOperations:
    """Test schema utility methods."""

    def test_with_n_creates_new_schema(self):
        """with_n() creates copy, doesn't modify original."""
        original = PITSchema({"root": {"n": 100, "type": "FILE"}, "hierarchy": {}})
        new = original.with_n(50)

        assert new.root["n"] == 50
        assert original.root["n"] == 100  # unchanged

    def test_max_depth_empty_hierarchy(self):
        """max_depth() returns 0 for empty hierarchy."""
        schema = PITSchema({"root": {"n": 5, "type": "FILE"}, "hierarchy": {}})
        assert schema.max_depth() == 0

    def test_max_depth_multiple_levels(self):
        """max_depth() returns correct depth."""
        schema = PITSchema({
            "root": {"n": 5, "type": "FOLDER"},
            "hierarchy": {
                "1": [{"n": 10, "type": ["FOLDER"], "id": ["a"]}],
                "2": [{"n": 20, "type": ["FILE"], "id": ["b"]}],
                "3": [{"n": 40, "type": ["FILE"], "id": ["c"]}],
            },
        })
        assert schema.max_depth() == 3

    def test_to_dict_roundtrip(self):
        """to_dict() produces valid input for PITSchema."""
        original = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [{"n": 20, "type": ["FILE"], "id": ["x"]}]},
        })
        reconstructed = PITSchema(original.to_dict())
        assert original.is_compatible(reconstructed)