"""
Statistics aggregation for TACO datasets.

Weighted aggregation of pre-computed stats from internal:stats column.
Operates on Polars DataFrames for performance.
"""

import warnings

import numpy as np
import polars as pl

from tacoreader._constants import STATS_CONTINUOUS_LENGTH


def _extract_stats_and_weights(
    df: pl.DataFrame,
) -> tuple[list[list[list[float]]], list[int]]:
    """
    Extract internal:stats and calculate pixel counts as weights.

    Uses stac:tensor_shape for weights if available, otherwise equal weights.
    """
    if "internal:stats" not in df.columns:
        raise ValueError("DataFrame must contain 'internal:stats' column")

    # Fast extraction (no iterrows overhead)
    all_stats = df["internal:stats"].to_list()

    # Calculate weights from tensor_shape if available
    if "stac:tensor_shape" in df.columns:
        shapes = df["stac:tensor_shape"].to_list()
        weights = []
        for i, shape in enumerate(shapes):
            if len(shape) >= 2:
                weights.append(int(shape[-2] * shape[-1]))
            else:
                warnings.warn(
                    f"Row {i}: stac:tensor_shape has <2 dimensions. Using weight=1.",
                    UserWarning,
                    stacklevel=4,
                )
                weights.append(1)
    else:
        warnings.warn(
            "Column 'stac:tensor_shape' not found. Using equal weights. "
            "Results may be inaccurate if files have different sizes.",
            UserWarning,
            stacklevel=4,
        )
        weights = [1] * len(df)

    return all_stats, weights


def _is_categorical(stats: list[list[float]]) -> bool:
    """
    Detect categorical vs continuous stats.

    Continuous: 9 values [min, max, mean, std, valid%, p25, p50, p75, p95]
    Categorical: N values [prob_class_0, ..., prob_class_N]
    """
    if len(stats) == 0 or len(stats[0]) == 0:
        raise ValueError("Empty stats provided")

    return len(stats[0]) != STATS_CONTINUOUS_LENGTH


def stats_categorical(df: pl.DataFrame) -> np.ndarray:
    """
    Aggregate categorical probabilities using weighted average.

    Returns: Array [n_bands, n_classes] with averaged probabilities
    """
    all_stats, weights = _extract_stats_and_weights(df)

    if not _is_categorical(all_stats[0]):
        raise ValueError(
            f"Stats appear to be continuous format ({STATS_CONTINUOUS_LENGTH} values per band). "
            "Use stats_mean(), stats_std(), etc. for continuous data."
        )

    n_bands = len(all_stats[0])
    n_classes = len(all_stats[0][0])

    # Validate structure consistency
    for i, stats in enumerate(all_stats):
        if len(stats) != n_bands:
            raise ValueError(f"Row {i}: Expected {n_bands} bands, got {len(stats)}")
        for band_idx, band_stats in enumerate(stats):
            if len(band_stats) != n_classes:
                raise ValueError(
                    f"Row {i}, band {band_idx}: Expected {n_classes} classes, got {len(band_stats)}"
                )

    # Weighted average
    result = np.zeros((n_bands, n_classes), dtype=np.float32)
    total_weight = sum(weights)

    for band_idx in range(n_bands):
        for class_idx in range(n_classes):
            weighted_sum = sum(
                stats[band_idx][class_idx] * weight
                for stats, weight in zip(all_stats, weights, strict=False)
            )
            result[band_idx, class_idx] = weighted_sum / total_weight

    return result


def stats_mean(df: pl.DataFrame) -> np.ndarray:
    """Aggregate means using weighted average."""
    all_stats, weights = _extract_stats_and_weights(df)

    if _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    n_bands = len(all_stats[0])
    result = np.zeros(n_bands, dtype=np.float32)
    total_weight = sum(weights)

    for band_idx in range(n_bands):
        weighted_sum = sum(
            stats[band_idx][2] * weight  # mean at index 2
            for stats, weight in zip(all_stats, weights, strict=False)
        )
        result[band_idx] = weighted_sum / total_weight

    return result


def stats_std(df: pl.DataFrame) -> np.ndarray:
    """Aggregate standard deviations using pooled variance formula."""
    all_stats, weights = _extract_stats_and_weights(df)

    if _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    n_bands = len(all_stats[0])
    global_means = stats_mean(df)
    result = np.zeros(n_bands, dtype=np.float32)
    total_weight = sum(weights)

    for band_idx in range(n_bands):
        variance_sum = 0.0
        for stats, weight in zip(all_stats, weights, strict=False):
            mean_i = stats[band_idx][2]  # mean at index 2
            std_i = stats[band_idx][3]  # std at index 3
            variance_sum += (weight - 1) * (std_i**2)
            variance_sum += weight * ((mean_i - global_means[band_idx]) ** 2)

        pooled_variance = variance_sum / (total_weight - 1)
        result[band_idx] = np.sqrt(pooled_variance)

    return result


def stats_min(df: pl.DataFrame) -> np.ndarray:
    """Aggregate minimums (global min across all rows)."""
    all_stats, _ = _extract_stats_and_weights(df)

    if _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    n_bands = len(all_stats[0])
    result = np.full(n_bands, np.inf, dtype=np.float32)

    for band_idx in range(n_bands):
        for stats in all_stats:
            result[band_idx] = min(result[band_idx], stats[band_idx][0])

    return result


def stats_max(df: pl.DataFrame) -> np.ndarray:
    """Aggregate maximums (global max across all rows)."""
    all_stats, _ = _extract_stats_and_weights(df)

    if _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    n_bands = len(all_stats[0])
    result = np.full(n_bands, -np.inf, dtype=np.float32)

    for band_idx in range(n_bands):
        for stats in all_stats:
            result[band_idx] = max(result[band_idx], stats[band_idx][1])

    return result


def _stats_percentile(
    df: pl.DataFrame, percentile_idx: int, percentile_name: str
) -> np.ndarray:
    """
    Aggregate percentiles using simple average (APPROXIMATION).

    WARNING: Not statistically exact. For critical analysis, recompute from raw data.
    """
    all_stats, _ = _extract_stats_and_weights(df)

    if _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    warnings.warn(
        f"stats_{percentile_name}() uses simple averaging, NOT exact. "
        "For critical analysis, recompute from raw data.",
        UserWarning,
        stacklevel=3,
    )

    n_bands = len(all_stats[0])
    n_samples = len(all_stats)
    result = np.zeros(n_bands, dtype=np.float32)

    for band_idx in range(n_bands):
        percentile_sum = sum(stats[band_idx][percentile_idx] for stats in all_stats)
        result[band_idx] = percentile_sum / n_samples

    return result


def stats_p25(df: pl.DataFrame) -> np.ndarray:
    """Aggregate 25th percentiles (approximation)."""
    return _stats_percentile(df, 5, "p25")


def stats_p50(df: pl.DataFrame) -> np.ndarray:
    """Aggregate 50th percentiles / median (approximation)."""
    return _stats_percentile(df, 6, "p50")


def stats_p75(df: pl.DataFrame) -> np.ndarray:
    """Aggregate 75th percentiles (approximation)."""
    return _stats_percentile(df, 7, "p75")


def stats_p95(df: pl.DataFrame) -> np.ndarray:
    """Aggregate 95th percentiles (approximation)."""
    return _stats_percentile(df, 8, "p95")
