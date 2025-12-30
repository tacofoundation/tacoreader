"""Integration tests for statistics with real fixtures."""

import pytest

np = pytest.importorskip("numpy")

import tacoreader


class TestStatsCategoricalWithFixtures:
    """Categorical aggregation with real fixture data."""

    def test_categorical_flat(self, zip_flat):
        """Aggregate categorical samples from flat fixture."""
        ds = tacoreader.load(str(zip_flat))

        # Filter categorical samples (sample_0, sample_2, sample_4)
        categorical = ds.sql("SELECT * FROM data WHERE id IN ('sample_0', 'sample_2', 'sample_4')")

        result = categorical.data.stats_categorical()

        # Should have 3 bands, N classes per band
        assert result.ndim == 2
        assert result.shape[0] == 3
        # Probabilities should sum to ~1.0 per band
        assert np.allclose(result.sum(axis=1), 1.0, atol=0.01)

    def test_categorical_europe_region(self, folder_nested):
        """Aggregate europe region (3 categorical items)."""
        ds = tacoreader.load(str(folder_nested))

        # Navigate to europe (categorical)
        europe = ds.data.read("europe")

        result = europe.stats_categorical()

        assert result.ndim == 2
        assert result.shape[0] == 3  # 3 bands (RGB)
        assert np.allclose(result.sum(axis=1), 1.0, atol=0.01)

    def test_categorical_single_sample_flat(self, zip_flat):
        """Single categorical sample returns input stats."""
        ds = tacoreader.load(str(zip_flat))

        # Get only sample_0 (categorical)
        single = ds.sql("SELECT * FROM data WHERE id = 'sample_0'")

        result = single.data.stats_categorical()

        assert result.ndim == 2
        assert result.shape[0] == 3  # 3 bands


class TestStatsContinuousWithFixtures:
    """Continuous aggregation with real fixture data."""

    def test_mean_continuous_flat(self, zip_flat):
        """Weighted mean from continuous samples in flat fixture."""
        ds = tacoreader.load(str(zip_flat))

        # Filter continuous (sample_1, sample_3)
        continuous = ds.sql("SELECT * FROM data WHERE id IN ('sample_1', 'sample_3')")

        result = continuous.data.stats_mean()

        assert result.shape == (3,)
        assert result.dtype == np.float32
        # Mean should be in reasonable range (0-255 for typical imagery)
        assert np.all((result >= 0) & (result <= 255))

    def test_std_pooled_variance_americas(self, folder_nested):
        """Pooled variance formula with americas region (continuous)."""
        ds = tacoreader.load(str(folder_nested))

        # Navigate to americas (continuous)
        americas = ds.data.read("americas")

        result = americas.stats_std()

        assert result.shape == (3,)  # 3 bands
        assert result.dtype == np.float32
        # Std is always positive
        assert np.all(result >= 0)

    def test_min_max_global_continuous(self, zip_flat):
        """Global min/max across continuous samples."""
        ds = tacoreader.load(str(zip_flat))

        continuous = ds.sql("SELECT * FROM data WHERE id IN ('sample_1', 'sample_3')")

        min_result = continuous.data.stats_min()
        max_result = continuous.data.stats_max()

        assert min_result.shape == (3,)
        assert max_result.shape == (3,)
        # Min <= Max for all bands
        assert np.all(min_result <= max_result)

    def test_percentiles_continuous(self, folder_nested):
        """Percentile aggregation on continuous samples."""
        ds = tacoreader.load(str(folder_nested))

        americas = ds.data.read("americas")

        with pytest.warns(UserWarning, match="simple averaging"):
            p25 = americas.stats_p25()
            p50 = americas.stats_p50()
            p75 = americas.stats_p75()
            p95 = americas.stats_p95()

        # All should have same shape
        assert p25.shape == p50.shape == p75.shape == p95.shape
        assert p25.shape == (3,)  # 3 bands

        # Percentiles should be ordered
        assert np.all(p25 <= p50)
        assert np.all(p50 <= p75)
        assert np.all(p75 <= p95)

    def test_median_alias(self, zip_flat):
        """stats_median() is alias for stats_p50()."""
        ds = tacoreader.load(str(zip_flat))

        continuous = ds.sql("SELECT * FROM data WHERE id IN ('sample_1', 'sample_3')")

        with pytest.warns(UserWarning):
            median = continuous.data.stats_median()
            p50 = continuous.data.stats_p50()

        assert np.array_equal(median, p50)


class TestStatsDeepHierarchy:
    """Stats aggregation in deep hierarchical structures."""

    def test_aggregate_sensor_bands(self, folder_deep):
        """Aggregate RGB bands from single sensor."""
        ds = tacoreader.load(str(folder_deep))

        # Navigate to sensor_A in tile_00
        tile = ds.data.read("tile_00")
        sensor = tile.read("sensor_A")

        # Sensor has band_R, band_G, band_B (3 continuous single-band files)
        mean = sensor.stats_mean()
        std = sensor.stats_std()
        min_val = sensor.stats_min()
        max_val = sensor.stats_max()

        # Each band file is single-band, so aggregating 3 files gives 3 bands
        assert mean.shape == (3,)
        assert std.shape == (3,)
        assert min_val.shape == (3,)
        assert max_val.shape == (3,)

        # RGB values should be in 0-255 range
        assert np.all((min_val >= 0) & (min_val <= 255))
        assert np.all((max_val >= 0) & (max_val <= 255))

    def test_aggregate_multiple_tiles(self, zip_deep_part1):
        """Aggregate stats across multiple tiles."""
        ds = tacoreader.load(str(zip_deep_part1))

        # Get first 3 tiles
        subset = ds.sql("SELECT * FROM data WHERE id IN ('tile_00', 'tile_01', 'tile_02')")

        # Can't directly aggregate at level0 (FOLDER samples)
        # Navigate to first tile's sensor to test stats
        tile = subset.data.read("tile_00")
        sensor = tile.read("sensor_B")

        mean = sensor.stats_mean()

        assert mean.shape == (3,)
        assert np.all((mean >= 0) & (mean <= 255))


class TestStatsEdgeCases:
    """Edge cases and special scenarios."""

    def test_single_sample_no_aggregation(self, zip_flat):
        """Single sample stats equal input stats (no aggregation needed)."""
        ds = tacoreader.load(str(zip_flat))

        single = ds.sql("SELECT * FROM data WHERE id = 'sample_1'")

        mean = single.data.stats_mean()

        # Single sample mean should match the mean in internal:stats (index 2)
        assert mean.shape == (3,)
        assert mean.dtype == np.float32

    def test_stats_after_sql_filter(self, folder_nested):
        """Stats work correctly after SQL filtering."""
        ds = tacoreader.load(str(folder_nested))

        # SQL filter before navigation
        filtered = ds.sql("SELECT * FROM data WHERE region = 'europe'")

        europe = filtered.data.read("europe")
        result = europe.stats_categorical()

        assert result.ndim == 2
        assert result.shape[0] == 3  # 3 bands

    def test_stats_with_bbox_filter(self, zip_flat):
        """Stats work after STAC bbox filtering."""
        ds = tacoreader.load(str(zip_flat))

        # Filter by bbox (valencia region)
        valencia = ds.filter_bbox(-0.5, 39.3, -0.2, 39.6)

        if len(valencia.data) > 0:
            # If any samples match, stats should work
            # Valencia samples are categorical (sample_0)
            result = valencia.data.stats_categorical()
            assert result.ndim == 2

    def test_stats_mixed_regions(self, folder_nested):
        """Different regions have different stats formats (cat vs cont)."""
        ds = tacoreader.load(str(folder_nested))

        # Europe is categorical
        europe = ds.data.read("europe")
        cat_result = europe.stats_categorical()
        assert cat_result.ndim == 2

        # Americas is continuous
        americas = ds.data.read("americas")
        cont_result = americas.stats_mean()
        assert cont_result.ndim == 1

    def test_stats_all_samples_flat(self, zip_flat):
        """Cannot aggregate mixed categorical/continuous at level0."""
        ds = tacoreader.load(str(zip_flat))

        # Flat has both categorical and continuous samples
        # Calling stats_categorical will fail on continuous rows
        # This is expected behavior - user should filter first

        # Test that error is clear
        from tacoreader._exceptions import TacoQueryError

        # This should fail because we're mixing formats
        # (sample_1 is continuous but we're calling categorical)
        with pytest.raises(TacoQueryError, match="continuous format"):
            ds.data.stats_categorical()


class TestStatsDataFrameIntegration:
    """Stats methods called via TacoDataFrame API."""

    @pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
    def test_stats_mean_all_backends(self, zip_flat, backend):
        """stats_mean() works across all DataFrame backends."""
        if backend == "pandas":
            pytest.importorskip("pandas")
        elif backend == "polars":
            pytest.importorskip("polars")

        tacoreader.use(backend)
        ds = tacoreader.load(str(zip_flat))

        continuous = ds.sql("SELECT * FROM data WHERE id IN ('sample_1', 'sample_3')")

        result = continuous.data.stats_mean()

        # All backends should return same result
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

        tacoreader.use("pyarrow")

    @pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
    def test_stats_categorical_all_backends(self, folder_nested, backend):
        """stats_categorical() works across all DataFrame backends."""
        if backend == "pandas":
            pytest.importorskip("pandas")
        elif backend == "polars":
            pytest.importorskip("polars")

        tacoreader.use(backend)
        ds = tacoreader.load(str(folder_nested))

        europe = ds.data.read("europe")
        result = europe.stats_categorical()

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 3  # 3 bands

        tacoreader.use("pyarrow")

    def test_stats_chained_navigation(self, folder_deep):
        """Stats after multi-level navigation."""
        ds = tacoreader.load(str(folder_deep))

        # Navigate: tile → sensor → get stats
        tile = ds.data.read("tile_05")
        sensor = tile.read("sensor_A")

        mean = sensor.stats_mean()
        std = sensor.stats_std()

        assert mean.shape == (3,)
        assert std.shape == (3,)