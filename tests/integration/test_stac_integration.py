"""Integration tests for STAC filtering with real fixtures."""

from datetime import datetime, timezone

import pytest

from tacoreader._exceptions import TacoQueryError


VALENCIA_BBOX = (-0.5, 39.3, -0.2, 39.6)
EUROPE_BBOX = (-0.5, 39.3, 13.6, 52.7)
NYC_BBOX = (-74.2, 40.5, -73.8, 40.9)

SPAIN_BBOX = (-10, 36, 4, 44)
MADRID_BBOX = (-3.9, 40.3, -3.5, 40.5)
JAPAN_BBOX = (129, 31, 146, 46)
TOKYO_BBOX = (139.5, 35.5, 139.9, 35.9)
NOWHERE_BBOX = (50, 50, 51, 51)


class TestFilterBboxSimple:

    def test_single_match(self, ds_zip_flat):
        result = ds_zip_flat.filter_bbox(*VALENCIA_BBOX)
        df = result.data
        assert len(df) == 1
        assert df.head(1).to_pylist()[0]["location"] == "valencia"

    def test_no_match_returns_empty(self, ds_zip_flat):
        result = ds_zip_flat.filter_bbox(*NOWHERE_BBOX)
        assert len(result.data) == 0

    def test_multiple_matches(self, ds_folder_nested):
        result = ds_folder_nested.filter_bbox(*EUROPE_BBOX)
        assert len(result.data) == 1
        assert result.data.head(1).to_pylist()[0]["id"] == "europe"

    def test_explicit_geometry_col(self, ds_zip_flat):
        result = ds_zip_flat.filter_bbox(*VALENCIA_BBOX, geometry_col="istac:geometry")
        assert len(result.data) == 1

    def test_invalid_geometry_col_raises(self, ds_zip_flat):
        with pytest.raises(TacoQueryError, match="Column.*not found"):
            ds_zip_flat.filter_bbox(*VALENCIA_BBOX, geometry_col="nonexistent")

    def test_chained_with_sql(self, ds_zip_flat):
        result = (
            ds_zip_flat
            .filter_bbox(*VALENCIA_BBOX)
            .sql("SELECT * FROM data WHERE cloud_cover < 50")
        )
        assert len(result.data) == 1


class TestFilterDatetimeSimple:

    def test_range_single_match(self, ds_zip_flat):
        result = ds_zip_flat.filter_datetime("2023-01-01/2023-01-15")
        df = result.data
        assert len(df) == 1
        assert df.head(1).to_pylist()[0]["location"] == "valencia"

    def test_range_multiple_matches(self, ds_zip_flat):
        result = ds_zip_flat.filter_datetime("2023-01-01/2023-02-15")
        assert len(result.data) == 2

    def test_single_date_point_query(self, ds_zip_flat):
        result = ds_zip_flat.filter_datetime("2023-01-01")
        df = result.data
        assert len(df) == 1
        assert df.head(1).to_pylist()[0]["location"] == "valencia"

    def test_datetime_object(self, ds_zip_flat):
        dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        result = ds_zip_flat.filter_datetime(dt)
        assert len(result.data) == 1

    def test_datetime_tuple(self, ds_zip_flat):
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 2, 15, tzinfo=timezone.utc)
        result = ds_zip_flat.filter_datetime((start, end))
        assert len(result.data) == 2

    def test_no_match_returns_empty(self, ds_zip_flat):
        result = ds_zip_flat.filter_datetime("2020-01-01/2020-12-31")
        assert len(result.data) == 0

    def test_explicit_time_col(self, ds_zip_flat):
        result = ds_zip_flat.filter_datetime("2023-01-01/2023-01-15", time_col="istac:time_start")
        assert len(result.data) == 1

    def test_invalid_time_col_raises(self, ds_zip_flat):
        with pytest.raises(TacoQueryError, match="Column.*not found"):
            ds_zip_flat.filter_datetime("2023-01-01/2023-12-31", time_col="nope")


class TestFilterBboxCascadeNested:

    def test_level1_filters_by_children(self, ds_folder_nested):
        result = ds_folder_nested.filter_bbox(*NYC_BBOX, level=1)
        df = result.data
        assert len(df) == 1
        assert df.head(1).to_pylist()[0]["id"] == "americas"

    def test_level1_no_match(self, ds_folder_nested):
        result = ds_folder_nested.filter_bbox(*NOWHERE_BBOX, level=1)
        assert len(result.data) == 0

    def test_invalid_level_raises(self, ds_folder_nested):
        with pytest.raises(TacoQueryError, match="Level .* does not exist"):
            ds_folder_nested.filter_bbox(*VALENCIA_BBOX, level=99)

    def test_explicit_geometry_col_level1(self, ds_folder_nested):
        result = ds_folder_nested.filter_bbox(*NYC_BBOX, geometry_col="istac:geometry", level=1)
        assert len(result.data) == 1

    def test_invalid_geometry_col_level1_raises(self, ds_folder_nested):
        with pytest.raises(TacoQueryError, match="Column.*not found"):
            ds_folder_nested.filter_bbox(*NYC_BBOX, geometry_col="fake_geom", level=1)


class TestFilterDatetimeCascadeNested:

    def test_level1_filters_by_children(self, ds_folder_nested):
        result = ds_folder_nested.filter_datetime("2023-01-01/2023-01-15", level=1)
        df = result.data
        assert len(df) == 3

    def test_level1_single_date(self, ds_folder_nested):
        result = ds_folder_nested.filter_datetime("2023-01-01", level=1)
        assert len(result.data) == 3

    def test_invalid_level_raises(self, ds_folder_nested):
        with pytest.raises(TacoQueryError, match="Level .* does not exist"):
            ds_folder_nested.filter_datetime("2023-01-01", level=99)

    def test_invalid_time_col_level1_raises(self, ds_folder_nested):
        with pytest.raises(TacoQueryError, match="Column.*not found"):
            ds_folder_nested.filter_datetime("2023-01-01", time_col="fake_time_column", level=1)


class TestFilterBboxDeep:

    def test_level1_no_geometry_raises(self, ds_folder_deep):
        with pytest.raises(TacoQueryError, match="No geometry column found"):
            ds_folder_deep.filter_bbox(*VALENCIA_BBOX, level=1)


class TestFilterBboxTacocatDeep:

    def test_level0_tacocat(self, ds_tacocat):
        result = ds_tacocat.filter_bbox(*VALENCIA_BBOX)
        df = result.data
        locations = [r["location"] for r in df.head(20).to_pylist()]
        assert "valencia" in locations

    def test_level1_no_geometry_raises(self, ds_tacocat):
        with pytest.raises(TacoQueryError, match="No geometry column found"):
            ds_tacocat.filter_bbox(*VALENCIA_BBOX, level=1)


class TestCascadeLevel1:

    def test_zip_cascade_level1(self, ds_zip_cascade):
        result = ds_zip_cascade.filter_bbox(*SPAIN_BBOX, level=1)
        df = result.data
        ids = [r["id"] for r in df.head(10).to_pylist()]
        assert "europe" in ids
        assert "asia" not in ids

    def test_tacocat_cascade_level1(self, ds_tacocat_cascade):
        result = ds_tacocat_cascade.filter_bbox(*SPAIN_BBOX, level=1)
        df = result.data
        ids = [r["id"] for r in df.head(10).to_pylist()]
        assert "europe" in ids

    def test_tacocat_cascade_level1_asia(self, ds_tacocat_cascade):
        result = ds_tacocat_cascade.filter_bbox(*JAPAN_BBOX, level=1)
        df = result.data
        ids = [r["id"] for r in df.head(10).to_pylist()]
        assert "asia" in ids
        assert "europe" not in ids


class TestCascadeLevel2:
    """Covers L280-285 in _stac_cascade.py (chained JOINs for level >= 2)."""

    def test_zip_cascade_level2(self, ds_zip_cascade):
        result = ds_zip_cascade.filter_bbox(*MADRID_BBOX, level=2)
        df = result.data
        ids = [r["id"] for r in df.head(10).to_pylist()]
        assert "europe" in ids

    def test_tacocat_cascade_level2(self, ds_tacocat_cascade):
        """TacoCat level2 JOIN - verifies asia is found via tokyo bbox."""
        result = ds_tacocat_cascade.filter_bbox(*TOKYO_BBOX, level=2)
        df = result.data
        ids = [r["id"] for r in df.head(10).to_pylist()]
        assert "asia" in ids

    def test_cascade_level2_no_match(self, ds_zip_cascade):
        result = ds_zip_cascade.filter_bbox(*NOWHERE_BBOX, level=2)
        assert len(result.data) == 0


class TestCascadeDatetimeMultilevel:

    def test_datetime_level1(self, ds_zip_cascade):
        result = ds_zip_cascade.filter_datetime("2023-01-01/2023-01-15", level=1)
        df = result.data
        assert len(df) >= 1
        ids = [r["id"] for r in df.head(10).to_pylist()]
        assert "europe" in ids

    def test_datetime_level2(self, ds_zip_cascade):
        result = ds_zip_cascade.filter_datetime("2023-01-01", level=2)
        df = result.data
        ids = [r["id"] for r in df.head(10).to_pylist()]
        assert "europe" in ids


class TestHelperFunctions:

    def test_get_columns_for_level(self, ds_folder_deep):
        from tacoreader._stac_cascade import get_columns_for_level

        cols_0 = get_columns_for_level(ds_folder_deep, 0)
        cols_1 = get_columns_for_level(ds_folder_deep, 1)
        cols_2 = get_columns_for_level(ds_folder_deep, 2)

        assert "istac:geometry" in cols_0
        assert "resolution_m" in cols_1
        assert "wavelength_nm" in cols_2

    def test_validate_level_exists(self, ds_folder_nested):
        from tacoreader._stac_cascade import validate_level_exists

        validate_level_exists(ds_folder_nested, 0)
        validate_level_exists(ds_folder_nested, 1)

        with pytest.raises(TacoQueryError, match="Level .* does not exist"):
            validate_level_exists(ds_folder_nested, 5)


class TestFilterCombined:

    def test_bbox_then_datetime(self, ds_zip_flat):
        result = (
            ds_zip_flat
            .filter_bbox(-80, -15, -70, 0)
            .filter_datetime("2023-01-01/2023-06-01")
        )
        df = result.data
        assert len(df) == 1
        assert df.head(1).to_pylist()[0]["location"] == "lima"

    def test_datetime_then_bbox(self, ds_zip_flat):
        result = (
            ds_zip_flat
            .filter_datetime("2023-01-01/2023-06-01")
            .filter_bbox(-80, -15, -70, 0)
        )
        df = result.data
        assert len(df) == 1
        assert df.head(1).to_pylist()[0]["location"] == "lima"

    def test_filter_then_sql(self, ds_zip_flat):
        result = (
            ds_zip_flat
            .filter_datetime("2023-01-01/2023-12-31")
            .sql("SELECT id, location, cloud_cover FROM data WHERE cloud_cover < 30")
        )
        df = result.data
        for row in df.head(10).to_pylist():
            assert row["cloud_cover"] < 30