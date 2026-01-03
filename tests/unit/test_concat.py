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


class TestConcatSourcePath:

    def test_concat_preserves_source_paths(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        n1 = len(ds1.data)
        n2 = len(ds2.data)

        result = concat([ds1, ds2])

        table = result._duckdb.execute("SELECT * FROM level0").fetch_arrow_table()
        assert "internal:source_path" in table.column_names

        source_paths = table.column("internal:source_path").to_pylist()
        
        paths_ds1 = [p for p in source_paths if str(zip_deep_part1) in p]
        paths_ds2 = [p for p in source_paths if str(zip_deep_part2) in p]

        assert len(paths_ds1) == n1
        assert len(paths_ds2) == n2

    def test_concat_vsi_paths_use_correct_source(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        result = concat([ds1, ds2])

        table = result._duckdb.execute("SELECT * FROM level0").fetch_arrow_table()
        
        for row in table.to_pylist():
            vsi_path = row["internal:gdal_vsi"]
            source_path = row["internal:source_path"]
            
            assert source_path in vsi_path

    def test_concat_navigation_works_across_sources(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        result = concat([ds1, ds2])
        tdf = result.data

        first_sample = tdf.read(0)
        assert first_sample is not None

        last_sample = tdf.read(len(tdf) - 1)
        assert last_sample is not None

    def test_concat_folder_preserves_source_paths(self, folder_nested):
        ds1 = tacoreader.load(str(folder_nested))
        ds2 = tacoreader.load(str(folder_nested))

        result = concat([ds1, ds2])

        table = result._duckdb.execute("SELECT * FROM level0").fetch_arrow_table()
        assert "internal:source_path" in table.column_names

    def test_concat_tacocat_preserves_source_paths(self, tacocat_deep):
        ds1 = tacoreader.load(str(tacocat_deep))
        ds2 = tacoreader.load(str(tacocat_deep))

        result = concat([ds1, ds2])

        table = result._duckdb.execute("SELECT * FROM level0").fetch_arrow_table()
        assert "internal:source_path" in table.column_names


class TestConcatPrefiltered:
    """Tests for filter → concat workflow (Mode 2).
    
    Verifies that .sql() filters are respected when concatenating
    """

    def test_concat_prefiltered_level0(self, zip_deep_part1, zip_deep_part2):
        """Filtered datasets should keep filters after concat."""
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        # Get first ID from each dataset via DuckDB
        id1 = ds1._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]
        id2 = ds2._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]

        # Filter to single sample each
        ds1_filtered = ds1.sql(f"SELECT * FROM data WHERE id = '{id1}'")
        ds2_filtered = ds2.sql(f"SELECT * FROM data WHERE id = '{id2}'")

        assert ds1_filtered.pit_schema.root["n"] == 1
        assert ds2_filtered.pit_schema.root["n"] == 1

        # Concat filtered datasets
        combined = concat([ds1_filtered, ds2_filtered])

        # Should have 2 rows (1 + 1), NOT all rows from both datasets
        assert combined.pit_schema.root["n"] == 2
        assert len(combined.data) == 2

        # Verify correct IDs
        result_ids = set(combined._duckdb.execute("SELECT id FROM level0").fetch_arrow_table().column("id").to_pylist())
        assert result_ids == {id1, id2}

    def test_concat_prefiltered_propagates_to_level1(self, zip_deep_part1, zip_deep_part2):
        """Level1 should only contain children of filtered level0 samples."""
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        # Get first tile ID from each
        id1 = ds1._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]
        id2 = ds2._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]

        # Filter to single tile each
        ds1_filtered = ds1.sql(f"SELECT * FROM data WHERE id = '{id1}'")
        ds2_filtered = ds2.sql(f"SELECT * FROM data WHERE id = '{id2}'")

        combined = concat([ds1_filtered, ds2_filtered])

        # Level0: should have 2 tiles
        level0_count = combined._duckdb.execute("SELECT COUNT(*) FROM level0").fetchone()[0]
        assert level0_count == 2

        # Level1: should only have children of those 2 tiles
        # Get the internal:current_id values from level0 to compare with level1 parent_ids
        level0_current_ids = set(
            combined._duckdb.execute("SELECT \"internal:current_id\" FROM level0")
            .fetch_arrow_table().column("internal:current_id").to_pylist()
        )
        
        level1_df = combined._duckdb.execute("SELECT * FROM level1").fetch_arrow_table()
        level1_parent_ids = set(level1_df.column("internal:parent_id").to_pylist())

        # All parent_ids in level1 should be in level0's current_ids
        assert level1_parent_ids.issubset(level0_current_ids)

        # deep fixture: each tile has 2 sensors (sensor_A, sensor_B)
        # So level1 should have 2 tiles × 2 sensors = 4 rows
        assert level1_df.num_rows == 4

    def test_concat_unfiltered_backward_compat(self, zip_deep_part1, zip_deep_part2):
        """Unfiltered concat should work as before (Mode 1)."""
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        n1 = ds1.pit_schema.root["n"]
        n2 = ds2.pit_schema.root["n"]

        # Concat without filtering
        combined = concat([ds1, ds2])

        # Should have all rows
        assert combined.pit_schema.root["n"] == n1 + n2

        # Filter AFTER concat still works
        first_id = combined._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]
        filtered = combined.sql(f"SELECT * FROM data WHERE id = '{first_id}'")
        
        # Should find that ID in both datasets (if same) or just one
        assert filtered.pit_schema.root["n"] >= 1

    def test_concat_mixed_filtered_unfiltered(self, zip_deep_part1, zip_deep_part2):
        """Can concat filtered dataset with unfiltered dataset."""
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        n2_original = ds2.pit_schema.root["n"]

        # Filter only ds1 to single sample
        id1 = ds1._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]
        ds1_filtered = ds1.sql(f"SELECT * FROM data WHERE id = '{id1}'")

        # Concat: 1 filtered + all of ds2
        combined = concat([ds1_filtered, ds2])

        # Should have 1 + n2_original
        assert combined.pit_schema.root["n"] == 1 + n2_original

    def test_concat_empty_filter_result(self, zip_deep_part1, zip_deep_part2):
        """Concat handles datasets filtered to zero rows."""
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        # Filter ds1 to impossible condition
        ds1_empty = ds1.sql("SELECT * FROM data WHERE id = 'nonexistent_id_xyz'")
        assert ds1_empty.pit_schema.root["n"] == 0

        # Filter ds2 to single sample
        id2 = ds2._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]
        ds2_filtered = ds2.sql(f"SELECT * FROM data WHERE id = '{id2}'")

        combined = concat([ds1_empty, ds2_filtered])

        # Should have only ds2's filtered row
        assert combined.pit_schema.root["n"] == 1
        
        result_ids = combined._duckdb.execute("SELECT id FROM level0").fetch_arrow_table().column("id").to_pylist()
        assert result_ids == [id2]

    def test_concat_chained_sql_filters(self, zip_deep_part1, zip_deep_part2):
        """Chained .sql() calls should all be respected."""
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        # Get ID for chained filtering
        id1 = ds1._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]

        # Chain multiple filters on ds1
        ds1_step1 = ds1.sql("SELECT * FROM data WHERE id LIKE 'tile_%'")
        ds1_step2 = ds1_step1.sql(f"SELECT * FROM data WHERE id = '{id1}'")

        assert ds1_step2.pit_schema.root["n"] == 1

        # Single filter on ds2
        id2 = ds2._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]
        ds2_filtered = ds2.sql(f"SELECT * FROM data WHERE id = '{id2}'")

        combined = concat([ds1_step2, ds2_filtered])

        assert combined.pit_schema.root["n"] == 2

    def test_concat_prefiltered_navigation_works(self, zip_deep_part1, zip_deep_part2):
        """Can navigate to children after filter → concat."""
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))

        id1 = ds1._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]
        id2 = ds2._duckdb.execute("SELECT id FROM level0 LIMIT 1").fetchone()[0]

        ds1_filtered = ds1.sql(f"SELECT * FROM data WHERE id = '{id1}'")
        ds2_filtered = ds2.sql(f"SELECT * FROM data WHERE id = '{id2}'")

        combined = concat([ds1_filtered, ds2_filtered])

        # Should be able to navigate to children
        tdf = combined.data
        assert len(tdf) == 2

        # Read children of first sample
        children = tdf.read(0)
        assert children is not None
        assert len(children) > 0  # deep: 2 sensors per tile

        # Read children of second sample
        children2 = tdf.read(1)
        assert children2 is not None
        assert len(children2) > 0

    def test_concat_prefiltered_folder_format(self, folder_nested):
        """Filter → concat works with FOLDER format."""
        ds1 = tacoreader.load(str(folder_nested))
        ds2 = tacoreader.load(str(folder_nested))

        # nested: level0 has europe, americas, asia
        ds1_filtered = ds1.sql("SELECT * FROM data WHERE id = 'europe'")
        ds2_filtered = ds2.sql("SELECT * FROM data WHERE id = 'asia'")

        assert ds1_filtered.pit_schema.root["n"] == 1
        assert ds2_filtered.pit_schema.root["n"] == 1

        combined = concat([ds1_filtered, ds2_filtered])

        assert combined.pit_schema.root["n"] == 2
        assert combined._format == "folder"

        result_ids = set(combined._duckdb.execute("SELECT id FROM level0").fetch_arrow_table().column("id").to_pylist())
        assert result_ids == {"europe", "asia"}

    def test_concat_prefiltered_level1_propagation_folder(self, folder_nested):
        """Level1 filtering propagates correctly in FOLDER format."""
        ds1 = tacoreader.load(str(folder_nested))
        ds2 = tacoreader.load(str(folder_nested))

        ds1_filtered = ds1.sql("SELECT * FROM data WHERE id = 'europe'")
        ds2_filtered = ds2.sql("SELECT * FROM data WHERE id = 'americas'")

        combined = concat([ds1_filtered, ds2_filtered])

        # Get current_ids from level0 to verify level1 parent relationships
        level0_current_ids = set(
            combined._duckdb.execute("SELECT \"internal:current_id\" FROM level0")
            .fetch_arrow_table().column("internal:current_id").to_pylist()
        )

        # Level1: should only have children of europe and americas
        level1_df = combined._duckdb.execute("SELECT * FROM level1").fetch_arrow_table()
        level1_parent_ids = set(level1_df.column("internal:parent_id").to_pylist())

        # All parent_ids (int64) should be in level0's current_ids (int64)
        assert level1_parent_ids.issubset(level0_current_ids)
        
        # Should have exactly 2 unique parent_ids (one from each filtered dataset)
        assert len(level1_parent_ids) == 2

        # nested: each region has 3 items
        # 2 regions × 3 items = 6 rows
        assert level1_df.num_rows == 6


class TestValidateDatasets:

    @staticmethod
    def _make_mock_dataset(backend: str = "pyarrow", format_type: str = "zip"):
        from unittest.mock import Mock
        
        mock_ds = Mock()
        mock_ds._dataframe_backend = backend
        mock_ds._format = format_type
        mock_ds._path = "/fake/path.tacozip"
        mock_ds.pit_schema.is_compatible.return_value = True
        
        return mock_ds

    def test_different_backends_raises(self):
        from tacoreader.concat._validation import _validate_backends
        
        ds1 = self._make_mock_dataset(backend="pyarrow")
        ds2 = self._make_mock_dataset(backend="polars")

        with pytest.raises(TacoBackendError, match="different.*backends"):
            _validate_backends([ds1, ds2])


class TestValidateColumnCompatibility:

    @staticmethod
    def _make_mock_dataset(columns: list[str], format_type: str = "zip"):
        import duckdb
        import pyarrow as pa
        from unittest.mock import Mock
        
        data = {col: ["val"] for col in columns}
        arrow_table = pa.table(data)
        
        db = duckdb.connect(":memory:")
        db.register("level0_table", arrow_table)
        
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
        from tacoreader.concat._columns import validate_column_compatibility
        
        ds1 = self._make_mock_dataset(["id", "type", "internal:offset", "internal:size"])
        ds2 = self._make_mock_dataset(["id", "type"])

        with pytest.raises(TacoSchemaError, match="Critical columns missing"):
            validate_column_compatibility([ds1, ds2])

    def test_folder_format_does_not_require_offset_size(self):
        from tacoreader.concat._columns import validate_column_compatibility
        
        folder_cols = ["id", "type", "internal:parent_id"]
        ds1 = self._make_mock_dataset(folder_cols, format_type="folder")
        ds2 = self._make_mock_dataset(folder_cols, format_type="folder")

        result = validate_column_compatibility([ds1, ds2])
        
        assert "level0" in result

    def test_identical_columns_no_warning(self):
        from tacoreader.concat._columns import validate_column_compatibility
        
        cols = ["id", "type", "internal:parent_id", "internal:offset", "internal:size"]
        ds1 = self._make_mock_dataset(cols)
        ds2 = self._make_mock_dataset(cols)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = validate_column_compatibility([ds1, ds2])
        
        assert set(result["level0"]) == set(cols)