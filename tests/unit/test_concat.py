"""Tests for tacoreader.concat() function."""

import warnings

import pytest

import tacoreader
from tacoreader.concat import concat
from tacoreader._exceptions import TacoQueryError, TacoSchemaError, TacoBackendError


class TestConcatBasic:

    def test_concat_combines_all_rows(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        result = concat([ds1, ds2])

        n1 = len(ds1.data)
        n2 = len(ds2.data)
        assert len(result.data) == n1 + n2

    def test_concat_preserves_format(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        result = concat([ds1, ds2])

        assert result._format == "zip"

    def test_concat_result_is_queryable(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        result = concat([ds1, ds2])
        data = result.data

        assert "id" in data.columns
        assert "type" in data.columns


class TestConcatFormats:

    def test_concat_zip_format(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        result = concat([ds1, ds2])

        assert result._format == "zip"
        assert len(result.data) == len(ds1.data) + len(ds2.data)

    def test_concat_folder_format(self, folder_nested):
        """FOLDER format uses relative_path instead of offset/size."""
        ds1 = tacoreader.load(str(folder_nested))
        ds2 = tacoreader.load(str(folder_nested))

        result = concat([ds1, ds2])

        assert result._format == "folder"
        assert len(result.data) == len(ds1.data) * 2

    def test_concat_tacocat_format(self, tacocat_deep):
        ds1 = tacoreader.load(str(tacocat_deep))
        ds2 = tacoreader.load(str(tacocat_deep))

        result = concat([ds1, ds2])

        assert result._format == "tacocat"
        assert len(result.data) == len(ds1.data) * 2


class TestConcatErrors:

    def test_single_dataset_raises(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))

        with pytest.raises(TacoQueryError, match="at least 2"):
            concat([ds])

    def test_empty_list_raises(self):
        with pytest.raises(TacoQueryError, match="at least 2"):
            concat([])


class TestConcatValidation:

    def test_different_formats_raises(self, zip_flat, folder_flat):
        ds_zip = tacoreader.load(str(zip_flat))
        ds_folder = tacoreader.load(str(folder_flat))

        with pytest.raises(TacoSchemaError, match="different formats"):
            concat([ds_zip, ds_folder])

    def test_incompatible_schemas_raises(self, zip_flat, zip_nested):
        ds_flat = tacoreader.load(str(zip_flat))
        ds_nested = tacoreader.load(str(zip_nested))

        with pytest.raises(TacoSchemaError, match="incompatible schema"):
            concat([ds_flat, ds_nested])


class TestConcatColumnModes:

    def test_intersection_is_default(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        result = concat([ds1, ds2])

        assert len(result.data) > 0

    def test_invalid_mode_raises(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        with pytest.raises(TacoQueryError, match="Invalid column_mode"):
            concat([ds1, ds2], column_mode="invalid")

    def test_fill_missing_mode_accepted(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        result = concat([ds1, ds2], column_mode="fill_missing")

        assert len(result.data) > 0

    def test_strict_mode_with_identical_columns(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        result = concat([ds1, ds2], column_mode="strict")

        assert len(result.data) > 0


class TestValidateDatasets:
    """Unit tests for _validation module."""

    @staticmethod
    def _make_mock_dataset(backend: str = "pyarrow", format_type: str = "zip"):
        """Create minimal mock dataset for validation tests."""
        from unittest.mock import Mock
        
        mock_ds = Mock()
        mock_ds._dataframe_backend = backend
        mock_ds._format = format_type
        mock_ds._path = "/fake/path.tacozip"
        mock_ds.pit_schema.is_compatible.return_value = True
        
        return mock_ds

    def test_different_backends_raises(self):
        """Validates backend mismatch without requiring polars installed."""
        from tacoreader.concat._validation import _validate_backends
        
        ds1 = self._make_mock_dataset(backend="pyarrow")
        ds2 = self._make_mock_dataset(backend="polars")

        with pytest.raises(TacoBackendError, match="different.*backends"):
            _validate_backends([ds1, ds2])


class TestValidateColumnCompatibility:
    """Unit tests for _columns.validate_column_compatibility.
    
    Tests column validation logic directly without full concat pipeline.
    Uses minimal mocks to isolate _columns.py behavior.
    """

    @staticmethod
    def _make_mock_dataset(columns: list[str], format_type: str = "zip"):
        """Create minimal mock dataset for column validation."""
        import duckdb
        import pyarrow as pa
        from unittest.mock import Mock
        
        # Create arrow table with specified columns
        data = {col: ["val"] for col in columns}
        arrow_table = pa.table(data)
        
        # Mock DuckDB connection
        db = duckdb.connect(":memory:")
        db.register("level0", arrow_table)
        
        # Mock dataset
        mock_ds = Mock()
        mock_ds._format = format_type
        mock_ds._duckdb = db
        mock_ds.pit_schema.max_depth.return_value = 0
        
        return mock_ds

    def test_intersection_drops_extra_columns(self):
        from tacoreader.concat._columns import validate_column_compatibility
        
        base_cols = ["id", "type", "internal:parent_id", "internal:offset", "internal:size"]
        ds1 = self._make_mock_dataset(base_cols + ["extra_col"])
        ds2 = self._make_mock_dataset(base_cols)

        with pytest.warns(UserWarning, match="dropped.*column"):
            result = validate_column_compatibility([ds1, ds2], mode="intersection")

        assert "extra_col" not in result["level0"]

    def test_fill_missing_keeps_all_columns(self):
        from tacoreader.concat._columns import validate_column_compatibility
        
        base_cols = ["id", "type", "internal:parent_id", "internal:offset", "internal:size"]
        ds1 = self._make_mock_dataset(base_cols + ["extra_col"])
        ds2 = self._make_mock_dataset(base_cols)

        with pytest.warns(UserWarning, match="filling missing columns"):
            result = validate_column_compatibility([ds1, ds2], mode="fill_missing")

        assert "extra_col" in result["level0"]

    def test_strict_raises_on_column_mismatch(self):
        from tacoreader.concat._columns import validate_column_compatibility
        
        base_cols = ["id", "type", "internal:parent_id", "internal:offset", "internal:size"]
        ds1 = self._make_mock_dataset(base_cols + ["extra_col"])
        ds2 = self._make_mock_dataset(base_cols)

        with pytest.raises(TacoSchemaError, match="Column mismatch"):
            validate_column_compatibility([ds1, ds2], mode="strict")

    def test_missing_critical_columns_raises(self):
        """ZIP format requires offset/size columns."""
        from tacoreader.concat._columns import validate_column_compatibility
        
        ds1 = self._make_mock_dataset(["id", "type", "internal:offset", "internal:size"])
        ds2 = self._make_mock_dataset(["id", "type"])  # Missing offset/size

        with pytest.raises(TacoSchemaError, match="Critical columns missing"):
            validate_column_compatibility([ds1, ds2])

    def test_folder_format_does_not_require_offset_size(self):
        """FOLDER format uses relative_path, not offset/size."""
        from tacoreader.concat._columns import validate_column_compatibility
        
        folder_cols = ["id", "type", "internal:parent_id"]
        ds1 = self._make_mock_dataset(folder_cols, format_type="folder")
        ds2 = self._make_mock_dataset(folder_cols, format_type="folder")

        # Should NOT raise - folder doesn't need offset/size
        result = validate_column_compatibility([ds1, ds2])
        
        assert "level0" in result

    def test_identical_columns_no_warning(self):
        from tacoreader.concat._columns import validate_column_compatibility
        import warnings
        
        cols = ["id", "type", "internal:parent_id", "internal:offset", "internal:size"]
        ds1 = self._make_mock_dataset(cols)
        ds2 = self._make_mock_dataset(cols)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should NOT raise any warnings
            result = validate_column_compatibility([ds1, ds2])
        
        assert set(result["level0"]) == set(cols)