"""Unit tests for statistics aggregation."""

import pytest
import pyarrow as pa

np = pytest.importorskip("numpy")

from tacoreader._exceptions import TacoQueryError
from tacoreader.dataframe._stats import (
    _extract_stats_and_weights,
    _is_categorical,
    stats_categorical,
    stats_max,
    stats_mean,
    stats_min,
    stats_p25,
    stats_p50,
    stats_p75,
    stats_p95,
    stats_std,
)


class TestExtractStatsAndWeights:
    """Core extraction logic with weights calculation."""

    def test_extract_with_tensor_shape(self):
        """Extract stats and compute pixel-based weights from tensor_shape."""
        table = pa.table({
            "internal:stats": [
                [[10.0, 20.0, 15.0, 2.0, 100.0, 12.0, 15.0, 18.0, 19.0]],
                [[5.0, 15.0, 10.0, 1.5, 100.0, 8.0, 10.0, 12.0, 14.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 200, 200]]
        })

        stats_3d, weights = _extract_stats_and_weights(table)

        assert stats_3d.shape == (2, 1, 9)
        assert weights.shape == (2,)
        assert weights[0] == 10000
        assert weights[1] == 40000

    def test_extract_without_tensor_shape(self):
        """Missing tensor_shape uses equal weights with warning."""
        table = pa.table({
            "internal:stats": [
                [[10.0, 20.0, 15.0, 2.0, 100.0, 12.0, 15.0, 18.0, 19.0]],
                [[5.0, 15.0, 10.0, 1.5, 100.0, 8.0, 10.0, 12.0, 14.0]]
            ]
        })

        with pytest.warns(UserWarning, match="stac:tensor_shape.*not found"):
            stats_3d, weights = _extract_stats_and_weights(table)

        assert np.all(weights == 1)

    def test_extract_invalid_tensor_shape(self):
        """1D tensor_shape uses weight=1 with warning."""
        table = pa.table({
            "internal:stats": [
                [[10.0, 20.0, 15.0, 2.0, 100.0, 12.0, 15.0, 18.0, 19.0]],
                [[5.0, 15.0, 10.0, 1.5, 100.0, 8.0, 10.0, 12.0, 14.0]]
            ],
            "stac:tensor_shape": [[100], [1, 200, 200]]
        })

        with pytest.warns(UserWarning, match="<2 dimensions"):
            stats_3d, weights = _extract_stats_and_weights(table)

        assert weights[0] == 1
        assert weights[1] == 40000

    def test_extract_missing_stats_column(self):
        """Missing internal:stats column raises error."""
        table = pa.table({
            "id": ["A", "B"],
            "stac:tensor_shape": [[1, 100, 100], [1, 200, 200]]
        })

        with pytest.raises(TacoQueryError, match="must contain 'internal:stats'"):
            _extract_stats_and_weights(table)


class TestIsCategorical:
    """Detection of categorical vs continuous format."""

    def test_is_categorical_true(self):
        """3 values per band indicates categorical (class probabilities)."""
        stats = [[0.5, 0.3, 0.2]]
        assert _is_categorical(stats) is True

    def test_is_categorical_true_multiband(self):
        """Multiple bands with non-9 values is categorical."""
        stats = [
            [0.5, 0.3, 0.2],
            [0.6, 0.2, 0.2],
            [0.4, 0.4, 0.2]
        ]
        assert _is_categorical(stats) is True

    def test_is_categorical_false(self):
        """9 values per band indicates continuous (min/max/mean/std/...)."""
        stats = [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
        assert _is_categorical(stats) is False

    def test_is_categorical_empty_raises(self):
        """Empty stats raises error."""
        with pytest.raises(TacoQueryError, match="Empty stats"):
            _is_categorical([])

    def test_is_categorical_empty_band_raises(self):
        """Empty band raises error."""
        with pytest.raises(TacoQueryError, match="Empty stats"):
            _is_categorical([[]])


class TestStatsCategorical:
    """Categorical probability aggregation."""

    def test_categorical_single_sample(self):
        """Single sample categorical returns input unchanged."""
        table = pa.table({
            "internal:stats": [
                [[0.5, 0.3, 0.2]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        result = stats_categorical(table)

        assert result.shape == (1, 3)
        assert np.allclose(result[0], [0.5, 0.3, 0.2])

    def test_categorical_weighted_average(self):
        """Weighted average based on pixel counts."""
        table = pa.table({
            "internal:stats": [
                [[0.6, 0.3, 0.1]],
                [[0.2, 0.5, 0.3]]
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 100, 100]]
        })

        result = stats_categorical(table)

        expected = np.array([0.4, 0.4, 0.2])
        assert result.shape == (1, 3)
        assert np.allclose(result[0], expected)

    def test_categorical_multiband(self):
        """Categorical with multiple bands."""
        table = pa.table({
            "internal:stats": [
                [[0.5, 0.3, 0.2], [0.7, 0.2, 0.1]],
                [[0.3, 0.4, 0.3], [0.5, 0.3, 0.2]]
            ],
            "stac:tensor_shape": [[2, 100, 100], [2, 100, 100]]
        })

        result = stats_categorical(table)

        assert result.shape == (2, 3)
        assert np.allclose(result.sum(axis=1), 1.0)

    def test_categorical_on_continuous_raises(self):
        """Calling categorical on continuous format raises error."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        with pytest.raises(TacoQueryError, match="continuous format.*Use stats_mean"):
            stats_categorical(table)


class TestStatsMean:
    """Weighted mean aggregation."""

    def test_mean_single_sample(self):
        """Single sample mean extracts mean value (index 2)."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        result = stats_mean(table)

        assert result.shape == (1,)
        assert result[0] == 50.0

    def test_mean_weighted_average(self):
        """Weighted average based on pixel counts."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 100.0, 40.0, 10.0, 98.0, 35.0, 40.0, 45.0, 60.0]],
                [[0.0, 100.0, 60.0, 10.0, 98.0, 55.0, 60.0, 65.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 300, 300]]
        })

        result = stats_mean(table)

        weight1 = 10000
        weight2 = 90000
        expected = (40.0 * weight1 + 60.0 * weight2) / (weight1 + weight2)

        assert result.shape == (1,)
        assert np.isclose(result[0], expected)

    def test_mean_multiband(self):
        """Mean aggregation with multiple bands."""
        table = pa.table({
            "internal:stats": [
                [
                    [0.0, 100.0, 30.0, 10.0, 98.0, 25.0, 30.0, 35.0, 50.0],
                    [0.0, 200.0, 100.0, 20.0, 98.0, 80.0, 100.0, 120.0, 150.0]
                ]
            ],
            "stac:tensor_shape": [[2, 100, 100]]
        })

        result = stats_mean(table)

        assert result.shape == (2,)
        assert result[0] == 30.0
        assert result[1] == 100.0

    def test_mean_on_categorical_raises(self):
        """Calling mean on categorical format raises error."""
        table = pa.table({
            "internal:stats": [
                [[0.5, 0.3, 0.2]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        with pytest.raises(TacoQueryError, match="categorical format.*Use stats_categorical"):
            stats_mean(table)


class TestStatsStd:
    """Pooled standard deviation calculation."""

    def test_std_single_sample(self):
        """Single sample std extracts std value (index 3)."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 100.0, 50.0, 15.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        result = stats_std(table)

        assert result.shape == (1,)
        assert result[0] == 15.0

    def test_std_pooled_variance(self):
        """Pooled variance formula with multiple samples."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
                [[0.0, 100.0, 50.0, 20.0, 98.0, 30.0, 50.0, 70.0, 90.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 100, 100]]
        })

        result = stats_std(table)

        assert result.shape == (1,)
        assert result[0] > 0

    def test_std_on_categorical_raises(self):
        """Calling std on categorical format raises error."""
        table = pa.table({
            "internal:stats": [
                [[0.5, 0.3, 0.2]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        with pytest.raises(TacoQueryError, match="categorical format"):
            stats_std(table)


class TestStatsMinMax:
    """Global min/max aggregation."""

    def test_min_single_sample(self):
        """Single sample min extracts min value (index 0)."""
        table = pa.table({
            "internal:stats": [
                [[5.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        result = stats_min(table)

        assert result.shape == (1,)
        assert result[0] == 5.0

    def test_max_single_sample(self):
        """Single sample max extracts max value (index 1)."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 95.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        result = stats_max(table)

        assert result.shape == (1,)
        assert result[0] == 95.0

    def test_min_global_across_samples(self):
        """Min returns global minimum across all samples."""
        table = pa.table({
            "internal:stats": [
                [[10.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
                [[5.0, 90.0, 45.0, 10.0, 98.0, 35.0, 45.0, 55.0, 75.0]],
                [[15.0, 110.0, 55.0, 10.0, 98.0, 45.0, 55.0, 65.0, 85.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 100, 100], [1, 100, 100]]
        })

        result = stats_min(table)

        assert result.shape == (1,)
        assert result[0] == 5.0

    def test_max_global_across_samples(self):
        """Max returns global maximum across all samples."""
        table = pa.table({
            "internal:stats": [
                [[10.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]],
                [[5.0, 120.0, 45.0, 10.0, 98.0, 35.0, 45.0, 55.0, 75.0]],
                [[15.0, 110.0, 55.0, 10.0, 98.0, 45.0, 55.0, 65.0, 85.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 100, 100], [1, 100, 100]]
        })

        result = stats_max(table)

        assert result.shape == (1,)
        assert result[0] == 120.0


class TestStatsPercentiles:
    """Percentile aggregation with approximation warnings."""

    def test_p25_with_warning(self):
        """p25 shows approximation warning."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        with pytest.warns(UserWarning, match="simple averaging.*NOT exact"):
            result = stats_p25(table)

        assert result.shape == (1,)
        assert result[0] == 40.0

    def test_p50_median_with_warning(self):
        """p50 (median) shows approximation warning."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        with pytest.warns(UserWarning, match="simple averaging.*NOT exact"):
            result = stats_p50(table)

        assert result.shape == (1,)
        assert result[0] == 50.0

    def test_p75_with_warning(self):
        """p75 shows approximation warning."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        with pytest.warns(UserWarning, match="simple averaging.*NOT exact"):
            result = stats_p75(table)

        assert result.shape == (1,)
        assert result[0] == 60.0

    def test_p95_with_warning(self):
        """p95 shows approximation warning."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        with pytest.warns(UserWarning, match="simple averaging.*NOT exact"):
            result = stats_p95(table)

        assert result.shape == (1,)
        assert result[0] == 80.0

    def test_percentile_average_across_samples(self):
        """Percentiles average across multiple samples."""
        table = pa.table({
            "internal:stats": [
                [[0.0, 100.0, 50.0, 10.0, 98.0, 30.0, 50.0, 70.0, 90.0]],
                [[0.0, 100.0, 50.0, 10.0, 98.0, 40.0, 50.0, 60.0, 80.0]]
            ],
            "stac:tensor_shape": [[1, 100, 100], [1, 100, 100]]
        })

        with pytest.warns(UserWarning):
            result = stats_p25(table)

        expected = (30.0 + 40.0) / 2
        assert result.shape == (1,)
        assert result[0] == expected


class TestNumPyRequirement:
    """NumPy dependency validation."""

    def test_numpy_required_for_stats(self, monkeypatch):
        """Stats functions require NumPy."""
        import tacoreader.dataframe._stats as stats_module
        monkeypatch.setattr(stats_module, "HAS_NUMPY", False)

        table = pa.table({
            "internal:stats": [
                [[0.5, 0.3, 0.2]]
            ],
            "stac:tensor_shape": [[1, 100, 100]]
        })

        with pytest.raises(ImportError, match="requires NumPy"):
            stats_mean(table)