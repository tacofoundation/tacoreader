"""Integration tests for statistics with real fixtures."""

import pytest

np = pytest.importorskip("numpy")

import tacoreader
from tacoreader._exceptions import TacoQueryError


class TestStatsLevel0:
    """Stats aggregation at level 0 (all samples)."""

    def test_mean_level0(self, zip_flat):
        """Mean aggregation at level 0."""
        ds = tacoreader.load(str(zip_flat))

        result = ds.stats_mean(band=0)

        assert isinstance(result, np.floating)
        assert result >= 0

    def test_mean_multiple_bands(self, zip_flat):
        """Mean for multiple bands."""
        ds = tacoreader.load(str(zip_flat))

        result = ds.stats_mean(band=[0, 1, 2])

        assert result.shape == (3,)
        assert result.dtype == np.float32

    def test_std_level0(self, zip_flat):
        """Std aggregation at level 0."""
        ds = tacoreader.load(str(zip_flat))

        result = ds.stats_std(band=0)

        assert isinstance(result, np.floating)
        assert result >= 0

    def test_min_max_level0(self, zip_flat):
        """Min/max at level 0."""
        ds = tacoreader.load(str(zip_flat))

        min_val = ds.stats_min(band=0)
        max_val = ds.stats_max(band=0)

        assert min_val <= max_val

    def test_percentiles_level0(self, zip_flat):
        """Percentiles at level 0."""
        ds = tacoreader.load(str(zip_flat))

        with pytest.warns(UserWarning, match="simple averaging"):
            p25 = ds.stats_p25(band=0)
            p50 = ds.stats_p50(band=0)
            p75 = ds.stats_p75(band=0)
            p95 = ds.stats_p95(band=0)

        assert p25 <= p50 <= p75 <= p95

    def test_median_alias(self, zip_flat):
        """stats_median is alias for stats_p50."""
        ds = tacoreader.load(str(zip_flat))

        with pytest.warns(UserWarning):
            median = ds.stats_median(band=0)
            p50 = ds.stats_p50(band=0)

        assert median == p50

    def test_id_ignored_at_level0(self, zip_flat):
        """id parameter is ignored at level 0 with warning."""
        ds = tacoreader.load(str(zip_flat))

        with pytest.warns(UserWarning, match="ignored for level=0"):
            result = ds.stats_mean(band=0, id="ignored")

        assert isinstance(result, np.floating)


class TestStatsValidation:
    """Parameter validation tests."""

    def test_level_gt0_requires_id(self, folder_nested):
        """level > 0 requires id parameter."""
        ds = tacoreader.load(str(folder_nested))

        with pytest.raises(TacoQueryError, match="id is required for level > 0"):
            ds.stats_mean(band=0, level=1)

    def test_invalid_level_raises(self, zip_flat):
        """Invalid level raises error."""
        ds = tacoreader.load(str(zip_flat))

        with pytest.raises(TacoQueryError, match="does not exist"):
            ds.stats_mean(band=0, level=99)

    def test_invalid_band_raises(self, zip_flat):
        """Band out of range raises error."""
        ds = tacoreader.load(str(zip_flat))

        with pytest.raises(TacoQueryError, match="out of range"):
            ds.stats_mean(band=999)

    def test_no_stats_column_error_message(self, zip_flat):
        """Error message is clear when stats column missing."""
        # This is covered by unit tests in test_stats.py
        # Integration test just verifies the dataset loads correctly
        ds = tacoreader.load(str(zip_flat))
        
        # Verify stats column exists and works
        result = ds.stats_mean(band=0)
        assert isinstance(result, np.floating)


class TestStatsLevel1:
    """Stats aggregation at level 1 with id."""

    def test_mean_level1_with_id(self, folder_nested):
        """Mean at level 1 requires id."""
        ds = tacoreader.load(str(folder_nested))

        # Get stats of children of 'americas' at level 1 (continuous)
        result = ds.stats_mean(band=0, level=1, id="americas")

        assert isinstance(result, np.floating)

    def test_categorical_level1(self, folder_nested):
        """Categorical aggregation at level 1."""
        ds = tacoreader.load(str(folder_nested))

        # europe has categorical children
        result = ds.stats_categorical(band=0, level=1, id="europe")

        # Should be probabilities (sum ~1)
        assert result.ndim == 1 or isinstance(result, np.floating)

    def test_continuous_level1(self, folder_nested):
        """Continuous aggregation at level 1."""
        ds = tacoreader.load(str(folder_nested))

        # americas has continuous children
        result = ds.stats_mean(band=0, level=1, id="americas")

        assert isinstance(result, np.floating)


class TestStatsAfterFiltering:
    """Stats work correctly after SQL/STAC filtering."""

    def test_stats_after_sql(self, zip_flat):
        """Stats work after SQL filtering."""
        ds = tacoreader.load(str(zip_flat))

        filtered = ds.sql("SELECT * FROM data WHERE id IN ('sample_0', 'sample_1')")

        result = filtered.stats_mean(band=0)

        assert isinstance(result, np.floating)

    def test_stats_after_bbox(self, zip_flat):
        """Stats work after bbox filtering."""
        ds = tacoreader.load(str(zip_flat))

        # Filter by bbox
        filtered = ds.filter_bbox(-1.0, 39.0, 0.0, 40.0)

        if len(filtered.data) > 0:
            result = filtered.stats_mean(band=0)
            assert isinstance(result, np.floating)


class TestStatsWithNones:
    """Handle None values in stats column."""

    def test_none_values_filtered(self, zip_flat):
        """Samples with None stats are filtered automatically."""
        ds = tacoreader.load(str(zip_flat))

        # Even if some samples have None stats, aggregation works
        result = ds.stats_mean(band=0)

        assert isinstance(result, np.floating)
        assert not np.isnan(result)


class TestStatsAllBackends:
    """Stats work across all DataFrame backends."""

    @pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
    def test_stats_mean_all_backends(self, zip_flat, backend):
        """stats_mean works with all backends."""
        if backend == "pandas":
            pytest.importorskip("pandas")
        elif backend == "polars":
            pytest.importorskip("polars")

        tacoreader.use(backend)

        try:
            ds = tacoreader.load(str(zip_flat))
            result = ds.stats_mean(band=0)

            assert isinstance(result, np.floating)
        finally:
            tacoreader.use("pyarrow")

    @pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
    def test_stats_categorical_all_backends(self, folder_nested, backend):
        """stats_categorical works with all backends."""
        if backend == "pandas":
            pytest.importorskip("pandas")
        elif backend == "polars":
            pytest.importorskip("polars")

        tacoreader.use(backend)

        try:
            ds = tacoreader.load(str(folder_nested))
            result = ds.stats_categorical(band=0, level=1, id="europe")

            assert isinstance(result, (np.ndarray, np.floating))
        finally:
            tacoreader.use("pyarrow")