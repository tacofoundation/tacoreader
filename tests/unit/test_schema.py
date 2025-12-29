"""Unit tests for PITSchema validation."""

import pytest

from tacoreader.schema import PITSchema
from tacoreader._exceptions import TacoSchemaError


class TestPITSchemaValidation:

    def test_valid_file_root(self):
        schema = PITSchema({
            "root": {"n": 100, "type": "FILE"},
            "hierarchy": {},
        })
        assert schema.root["type"] == "FILE"
        assert schema.root["n"] == 100
        assert schema.max_depth() == 0

    def test_valid_folder_root_with_hierarchy(self):
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
        with pytest.raises(TacoSchemaError, match="Invalid root type"):
            PITSchema({
                "root": {"n": 10, "type": "INVALID"},
                "hierarchy": {},
            })

    def test_invalid_child_type_raises(self):
        with pytest.raises(TacoSchemaError, match="invalid type"):
            PITSchema({
                "root": {"n": 10, "type": "FOLDER"},
                "hierarchy": {
                    "1": [{"n": 10, "type": ["BADTYPE"], "id": ["x"]}]
                },
            })

    def test_mismatched_type_id_length_raises(self):
        with pytest.raises(TacoSchemaError, match="type and id arrays differ"):
            PITSchema({
                "root": {"n": 10, "type": "FOLDER"},
                "hierarchy": {
                    "1": [{"n": 10, "type": ["FILE", "FILE"], "id": ["only_one"]}]
                },
            })

    def test_missing_type_field_raises(self):
        with pytest.raises(TacoSchemaError, match="missing required field"):
            PITSchema({
                "root": {"n": 10, "type": "FOLDER"},
                "hierarchy": {"1": [{"n": 10, "id": ["x"]}]},
            })

    def test_missing_id_field_raises(self):
        with pytest.raises(TacoSchemaError, match="missing required field"):
            PITSchema({
                "root": {"n": 10, "type": "FOLDER"},
                "hierarchy": {"1": [{"n": 10, "type": ["FILE"]}]},
            })

    def test_empty_type_array_raises(self):
        with pytest.raises(TacoSchemaError, match="type array empty"):
            PITSchema({
                "root": {"n": 10, "type": "FOLDER"},
                "hierarchy": {"1": [{"n": 10, "type": [], "id": ["x"]}]},
            })

    def test_empty_id_array_raises(self):
        with pytest.raises(TacoSchemaError, match="id array empty"):
            PITSchema({
                "root": {"n": 10, "type": "FOLDER"},
                "hierarchy": {"1": [{"n": 10, "type": ["FILE"], "id": []}]},
            })

    def test_non_numeric_depth_key_raises(self):
        with pytest.raises(TacoSchemaError, match="must be numeric"):
            PITSchema({
                "root": {"n": 10, "type": "FOLDER"},
                "hierarchy": {"abc": [{"n": 10, "type": ["FILE"], "id": ["x"]}]},
            })


class TestPITSchemaCompatibility:

    def test_compatible_same_structure_different_n(self):
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
        schema_a = PITSchema({"root": {"n": 10, "type": "FOLDER"}, "hierarchy": {}})
        schema_b = PITSchema({"root": {"n": 10, "type": "FILE"}, "hierarchy": {}})
        assert not schema_a.is_compatible(schema_b)

    def test_incompatible_different_hierarchy_depth(self):
        schema_a = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [{"n": 10, "type": ["FILE"], "id": ["x"]}]},
        })
        schema_b = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {},
        })
        assert not schema_a.is_compatible(schema_b)

    def test_incompatible_different_pattern_count(self):
        schema_a = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [
                {"n": 10, "type": ["FILE"], "id": ["x"]},
                {"n": 10, "type": ["FILE"], "id": ["y"]},
            ]},
        })
        schema_b = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [
                {"n": 10, "type": ["FILE"], "id": ["x"]},
            ]},
        })
        assert not schema_a.is_compatible(schema_b)

    def test_incompatible_different_pattern_types(self):
        schema_a = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [{"n": 10, "type": ["FILE"], "id": ["x"]}]},
        })
        schema_b = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [{"n": 10, "type": ["FOLDER"], "id": ["x"]}]},
        })
        assert not schema_a.is_compatible(schema_b)

    def test_incompatible_different_pattern_ids(self):
        schema_a = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [{"n": 10, "type": ["FILE"], "id": ["x"]}]},
        })
        schema_b = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [{"n": 10, "type": ["FILE"], "id": ["y"]}]},
        })
        assert not schema_a.is_compatible(schema_b)


class TestPITSchemaOperations:

    def test_with_n_creates_new_instance(self):
        original = PITSchema({"root": {"n": 100, "type": "FILE"}, "hierarchy": {}})
        new = original.with_n(50)

        assert new.root["n"] == 50
        assert original.root["n"] == 100

    def test_max_depth_empty_hierarchy(self):
        schema = PITSchema({"root": {"n": 5, "type": "FILE"}, "hierarchy": {}})
        assert schema.max_depth() == 0

    def test_max_depth_multiple_levels(self):
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
        original = PITSchema({
            "root": {"n": 10, "type": "FOLDER"},
            "hierarchy": {"1": [{"n": 20, "type": ["FILE"], "id": ["x"]}]},
        })
        reconstructed = PITSchema(original.to_dict())
        assert original.is_compatible(reconstructed)

    def test_repr_contains_root_type_and_depth(self):
        schema = PITSchema({
            "root": {"n": 5, "type": "FOLDER"},
            "hierarchy": {
                "1": [{"n": 10, "type": ["FOLDER"], "id": ["a"]}],
                "2": [{"n": 20, "type": ["FILE"], "id": ["b"]}],
            },
        })
        r = repr(schema)
        assert "FOLDER" in r
        assert "max_depth=2" in r