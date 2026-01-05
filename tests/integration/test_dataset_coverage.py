"""Integration tests for TacoDataset."""

import pytest
import tacoreader
from tacoreader._exceptions import TacoQueryError


class TestClose:

    def test_idempotent(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        ds.close()
        ds.close()

    def test_child_view_cleanup(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        child = ds.sql("SELECT * FROM data WHERE cloud_cover < 50")
        assert child._view_name != ds._view_name
        child.close()
        ds.close()

    def test_context_manager(self, zip_flat):
        with tacoreader.load(str(zip_flat)) as ds:
            _ = ds.sql("SELECT * FROM data LIMIT 1")


class TestProperties:

    def test_field_schema_returns_dict(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        assert isinstance(ds.field_schema, dict)
        ds.close()

    def test_collection_returns_copy(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        c1 = ds.collection
        c2 = ds.collection
        assert c1 is not c2
        assert c1 == c2
        ds.close()


class TestFilterDatetime:

    def test_string_range(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        filtered = ds.filter_datetime("2023-01-01/2023-03-01")
        assert filtered.pit_schema.root["n"] <= ds.pit_schema.root["n"]
        ds.close()

    def test_cascade_level1(self, zip_nested):
        ds = tacoreader.load(str(zip_nested))
        filtered = ds.filter_datetime("2023-01-01/2023-06-01", level=1)
        assert filtered.pit_schema.root["n"] >= 0
        ds.close()

    def test_chained_with_sql(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        filtered = ds.filter_datetime("2023-01-01/2023-06-01").sql(
            "SELECT * FROM data WHERE cloud_cover < 30"
        )
        assert filtered.pit_schema.root["n"] <= ds.pit_schema.root["n"]
        ds.close()


class TestFilterBbox:

    def test_cascade_level1(self, zip_nested):
        ds = tacoreader.load(str(zip_nested))
        filtered = ds.filter_bbox(-1, 39, 15, 53, level=1)
        assert filtered.pit_schema.root["n"] >= 0
        ds.close()


class TestStatsErrors:

    def test_level_without_stats_column(self, zip_nested):
        ds = tacoreader.load(str(zip_nested))
        with pytest.raises(TacoQueryError, match="does not contain statistics"):
            ds.stats_mean(band=0, level=0)
        ds.close()

    def test_band_out_of_range(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        with pytest.raises(TacoQueryError, match="out of range"):
            ds.stats_mean(band=99)
        ds.close()

    def test_band_list_out_of_range(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        with pytest.raises(TacoQueryError, match="out of range"):
            ds.stats_mean(band=[0, 1, 99])
        ds.close()


class TestStatsWithLevel:

    def test_level1_requires_id(self, zip_nested):
        ds = tacoreader.load(str(zip_nested))
        result = ds.stats_mean(band=0, level=1, id="americas")
        assert result is not None
        ds.close()

    def test_level0_ignores_id(self, zip_flat):
        """id parameter is ignored at level 0 (logged as debug)."""
        ds = tacoreader.load(str(zip_flat))
        # id is ignored at level=0, just verify it doesn't error
        result = ds.stats_mean(band=0, level=0, id="sample_0")
        assert result is not None
        ds.close()


class TestRepr:

    def test_contains_metadata(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        r = repr(ds)
        assert "TacoDataset" in r
        assert ds.id in r
        assert "Temporal" in r
        ds.close()

    def test_repr_html(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        html = ds._repr_html_()
        assert "<" in html
        assert ds.id in html
        ds.close()


class TestFormatTemporalString:

    def test_midnight_returns_date_only(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        assert ds._format_temporal_string("2023-01-15T00:00:00Z") == "2023-01-15"
        ds.close()

    def test_non_midnight_returns_full_datetime(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        assert ds._format_temporal_string("2023-01-15T14:30:00Z") == "2023-01-15 14:30:00"
        ds.close()

    def test_truncates_microseconds(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        assert ds._format_temporal_string("2023-01-15T14:30:00.123456Z") == "2023-01-15 14:30:00"
        ds.close()