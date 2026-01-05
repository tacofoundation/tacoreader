"""Unit tests for statistics aggregation."""

import pytest
import pyarrow as pa

np = pytest.importorskip("numpy")

from tacoreader._exceptions import TacoQueryError
from tacoreader.dataframe._stats import (
    _aggregate_categorical,
    _aggregate_continuous,
    _aggregate_std,
    _extract_stats_array,
    _extract_weights,
    _is_categorical,
)


class TestExtractStatsArray:
    """Core extraction logic with None filtering."""

    def test_extract_basic(self):
        """Extract stats as 3D array."""
        table = pa.table({
            "geotiff:stats": [
                [[10.0, 20.0, 15.0, 2.0, 100.0, 12.0, 15.0, 18.0, 19.0]],
                [[5.0, 15.0, 10.0, 1.5, 100.0, 8.0, 10.0, 12.0, 14.0]],
            ],
        })

        stats_3d, valid_mask = _extract_stats_array(table, "geotiff:stats")

        assert stats_3d.shape == (2, 1, 9)
        assert valid_mask.shape == (2,)
        assert valid_mask.all()

    def test_extract_filters_nones(self):
        """None values are filtered out."""
        table = pa.table({
            "geotiff:stats": [
                [[10.0, 20.0, 15.0, 2.0, 100.0, 12.0, 15.0, 18.0, 19.0]],
                None,
                [[5.0, 15.0, 10.0, 1.5, 100.0, 8.0, 10.0, 12.0, 14.0]],
            ],
        })

        stats_3d, valid_mask = _extract_stats_array(table, "geotiff:stats")

        assert stats_3d.shape == (2, 1, 9)  # Only 2 valid rows
        assert valid_mask.shape == (3,)
        assert valid_mask.tolist() == [True, False, True]

    def test_extract_filters_empty_lists(self):
        """Empty lists are filtered out."""
        table = pa.table({
            "geotiff:stats": [
                [[10.0, 20.0, 15.0, 2.0, 100.0, 12.0, 15.0, 18.0, 19.0]],
                [],
                [[5.0, 15.0, 10.0, 1.5, 100.0, 8.0, 10.0, 12.0, 14.0]],
            ],
        })

        stats_3d, valid_mask = _extract_stats_array(table, "geotiff:stats")

        assert stats_3d.shape == (2, 1, 9)
        assert valid_mask.tolist() == [True, False, True]

    def test_extract_all_none_raises(self):
        """All None values raises error."""
        table = pa.table({
            "geotiff:stats": [None, None],
        })

        with pytest.raises(TacoQueryError, match="No valid stats found"):
            _extract_stats_array(table, "geotiff:stats")

    def test_extract_missing_column_raises(self):
        """Missing stats column raises error."""
        table = pa.table({
            "id": ["A", "B"],
        })

        with pytest.raises(TacoQueryError, match="not found in table"):
            _extract_stats_array(table, "geotiff:stats")


class TestExtractWeights:
    """Weight extraction with valid_mask support."""

    def test_weights_from_tensor_shape(self):
        """Compute pixel-based weights from tensor_shape."""
        table = pa.table({
            "stac:tensor_shape": [[1, 100, 100], [1, 200, 200]],
        })

        weights = _extract_weights(table)

        assert weights.shape == (2,)
        assert weights[0] == 10000
        assert weights[1] == 40000

    def test_weights_with_valid_mask(self):
        """Apply valid_mask to filter weights."""
        table = pa.table({
            "stac:tensor_shape": [[1, 100, 100], [1, 200, 200], [1, 300, 300]],
        })
        valid_mask = np.array([True, False, True])

        weights = _extract_weights(table, valid_mask)

        assert weights.shape == (2,)  # Only 2 valid
        assert weights[0] == 10000
        assert weights[1] == 90000

    def test_weights_missing_column(self):
        """Missing tensor_shape uses equal weights (logged as debug)."""
        table = pa.table({
            "id": ["A", "B"],
        })

        # No warning expected, just debug logging
        weights = _extract_weights(table)

        assert np.all(weights == 1)

    def test_weights_invalid_shape(self):
        """1D tensor_shape uses weight=1 (logged as debug)."""
        table = pa.table({
            "stac:tensor_shape": [[100], [1, 200, 200]],
        })

        # No warning expected, just debug logging
        weights = _extract_weights(table)

        assert weights[0] == 1
        assert weights[1] == 40000


class TestIsCategorical:
    """Detection of categorical vs continuous format."""

    def test_categorical_3_values(self):
        """3 values per band indicates categorical."""
        stats = np.array([[[0.5, 0.3, 0.2]]])
        assert _is_categorical(stats) is True

    def test_categorical_5_values(self):
        """5 values per band indicates categorical."""
        stats = np.array([[[0.2, 0.2, 0.2, 0.2, 0.2]]])
        assert _is_categorical(stats) is True

    def test_continuous_9_values(self):
        """9 values per band indicates continuous."""
        stats = np.array([[[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]])
        assert _is_categorical(stats) is False

    def test_empty_raises(self):
        """Empty stats raises error."""
        with pytest.raises(TacoQueryError, match="Empty stats"):
            _is_categorical(np.array([]))


class TestAggregateCategorical:
    """Categorical probability aggregation."""

    def test_categorical_single_sample(self):
        """Single sample returns input unchanged."""
        table = pa.table({
            "geotiff:stats": [[[0.5, 0.3, 0.2]]],
            "stac:tensor_shape": [[1, 100, 100]],
        })

        result = _aggregate_categorical(table, "geotiff:stats")

        assert result.shape == (1, 3)
        assert np.allclose(result[0], [0.5, 0.3, 0.2])

    def test_categorical_weighted_average(self):
        """Weighted average based on pixel counts."""
        table = pa.table({
            "geotiff:stats": [
                [[0.6, 0.3, 0.1]],
                [[0.2, 0.5, 0.3]],
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 100, 100]],
        })

        result = _aggregate_categorical(table, "geotiff:stats")

        expected = np.array([0.4, 0.4, 0.2])
        assert result.shape == (1, 3)
        assert np.allclose(result[0], expected)

    def test_categorical_multiband(self):
        """Categorical with multiple bands."""
        table = pa.table({
            "geotiff:stats": [
                [[0.5, 0.3, 0.2], [0.7, 0.2, 0.1]],
                [[0.3, 0.4, 0.3], [0.5, 0.3, 0.2]],
            ],
            "stac:tensor_shape": [[2, 100, 100], [2, 100, 100]],
        })

        result = _aggregate_categorical(table, "geotiff:stats")

        assert result.shape == (2, 3)
        assert np.allclose(result.sum(axis=1), 1.0)

    def test_categorical_on_continuous_raises(self):
        """Calling categorical on continuous raises error."""
        table = pa.table({
            "geotiff:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
            ],
            "stac:tensor_shape": [[1, 100, 100]],
        })

        with pytest.raises(TacoQueryError, match="continuous format"):
            _aggregate_categorical(table, "geotiff:stats")


class TestAggregateContinuousMean:
    """Weighted mean aggregation."""

    def test_mean_single_sample(self):
        """Single sample extracts mean value (index 2)."""
        table = pa.table({
            "geotiff:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
            ],
            "stac:tensor_shape": [[1, 100, 100]],
        })

        result = _aggregate_continuous(table, "geotiff:stats", "mean")

        assert result.shape == (1,)
        assert result[0] == 50.0

    def test_mean_weighted_average(self):
        """Weighted average based on pixel counts."""
        table = pa.table({
            "geotiff:stats": [
                [[0.0, 100.0, 40.0, 10.0, 98.0, 35.0, 40.0, 45.0, 60.0]],
                [[0.0, 100.0, 60.0, 10.0, 98.0, 55.0, 60.0, 65.0, 80.0]],
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 300, 300]],
        })

        result = _aggregate_continuous(table, "geotiff:stats", "mean")

        weight1 = 10000
        weight2 = 90000
        expected = (40.0 * weight1 + 60.0 * weight2) / (weight1 + weight2)

        assert result.shape == (1,)
        assert np.isclose(result[0], expected)

    def test_mean_multiband(self):
        """Mean with multiple bands."""
        table = pa.table({
            "geotiff:stats": [
                [
                    [0.0, 100.0, 30.0, 10.0, 98.0, 25.0, 30.0, 35.0, 50.0],
                    [0.0, 200.0, 100.0, 20.0, 98.0, 80.0, 100.0, 120.0, 150.0],
                ],
            ],
            "stac:tensor_shape": [[2, 100, 100]],
        })

        result = _aggregate_continuous(table, "geotiff:stats", "mean")

        assert result.shape == (2,)
        assert result[0] == 30.0
        assert result[1] == 100.0

    def test_mean_on_categorical_raises(self):
        """Calling mean on categorical raises error."""
        table = pa.table({
            "geotiff:stats": [[[0.5, 0.3, 0.2]]],
            "stac:tensor_shape": [[1, 100, 100]],
        })

        with pytest.raises(TacoQueryError, match="categorical format"):
            _aggregate_continuous(table, "geotiff:stats", "mean")


class TestAggregateContinuousMinMax:
    """Global min/max aggregation."""

    def test_min_global(self):
        """Min returns global minimum across samples."""
        table = pa.table({
            "geotiff:stats": [
                [[10.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
                [[5.0, 90.0, 45.0, 10.0, 98.0, 35.0, 45.0, 55.0, 75.0]],
                [[15.0, 110.0, 55.0, 10.0, 98.0, 45.0, 55.0, 65.0, 85.0]],
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 100, 100], [1, 100, 100]],
        })

        result = _aggregate_continuous(table, "geotiff:stats", "min")

        assert result.shape == (1,)
        assert result[0] == 5.0

    def test_max_global(self):
        """Max returns global maximum across samples."""
        table = pa.table({
            "geotiff:stats": [
                [[10.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
                [[5.0, 120.0, 45.0, 10.0, 98.0, 35.0, 45.0, 55.0, 75.0]],
                [[15.0, 110.0, 55.0, 10.0, 98.0, 45.0, 55.0, 65.0, 85.0]],
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 100, 100], [1, 100, 100]],
        })

        result = _aggregate_continuous(table, "geotiff:stats", "max")

        assert result.shape == (1,)
        assert result[0] == 120.0


class TestAggregateStd:
    """Pooled standard deviation calculation."""

    def test_std_single_sample(self):
        """Single sample extracts std value (index 3)."""
        table = pa.table({
            "geotiff:stats": [
                [[0.0, 100.0, 50.0, 15.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
            ],
            "stac:tensor_shape": [[1, 100, 100]],
        })

        result = _aggregate_std(table, "geotiff:stats")

        assert result.shape == (1,)
        assert result[0] == 15.0

    def test_std_pooled_variance(self):
        """Pooled variance with multiple samples."""
        table = pa.table({
            "geotiff:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
                [[0.0, 100.0, 50.0, 20.0, 98.0, 30.0, 50.0, 70.0, 90.0]],
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 100, 100]],
        })

        result = _aggregate_std(table, "geotiff:stats")

        assert result.shape == (1,)
        assert result[0] > 0

    def test_std_on_categorical_raises(self):
        """Calling std on categorical raises error."""
        table = pa.table({
            "geotiff:stats": [[[0.5, 0.3, 0.2]]],
            "stac:tensor_shape": [[1, 100, 100]],
        })

        with pytest.raises(TacoQueryError, match="categorical format"):
            _aggregate_std(table, "geotiff:stats")


class TestAggregateContinuousPercentiles:
    """Percentile aggregation (uses averaging approximation, logged as debug)."""

    def test_p25_returns_value(self):
        """p25 returns expected value."""
        table = pa.table({
            "geotiff:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
            ],
            "stac:tensor_shape": [[1, 100, 100]],
        })

        result = _aggregate_continuous(table, "geotiff:stats", "p25")

        assert result.shape == (1,)
        assert result[0] == 40.0

    def test_p50_returns_value(self):
        """p50 returns expected value."""
        table = pa.table({
            "geotiff:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
            ],
            "stac:tensor_shape": [[1, 100, 100]],
        })

        result = _aggregate_continuous(table, "geotiff:stats", "p50")

        assert result.shape == (1,)
        assert result[0] == 50.0

    def test_percentile_average_across_samples(self):
        """Percentiles average across samples."""
        table = pa.table({
            "geotiff:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 30.0, 50.0, 70.0, 90.0]],
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 100, 100]],
        })

        result = _aggregate_continuous(table, "geotiff:stats", "p25")

        expected = (30.0 + 40.0) / 2
        assert result.shape == (1,)
        assert result[0] == expected


class TestNumPyRequirement:
    """NumPy dependency validation."""

    def test_numpy_required(self, monkeypatch):
        """Stats functions require NumPy."""
        import tacoreader.dataframe._stats as stats_module

        monkeypatch.setattr(stats_module, "HAS_NUMPY", False)

        table = pa.table({
            "geotiff:stats": [[[0.5, 0.3, 0.2]]],
            "stac:tensor_shape": [[1, 100, 100]],
        })

        with pytest.raises(ImportError, match="requires NumPy"):
            _aggregate_categorical(table, "geotiff:stats")