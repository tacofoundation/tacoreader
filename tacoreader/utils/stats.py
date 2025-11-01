import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _extract_stats_and_weights(
    df: "pd.DataFrame",
) -> tuple[list[list[list[float]]], list[int]]:
    """
    Extract internal:stats and calculate pixel counts (weights) from DataFrame.

    Args:
        df: DataFrame with 'internal:stats' column

    Returns:
        Tuple of:
        - List of stats per row: [[[band0_stats], [band1_stats], ...], ...]
        - List of pixel counts (weights) per row

    Raises:
        ValueError: If 'internal:stats' column not found
    """
    if "internal:stats" not in df.columns:
        raise ValueError("DataFrame must contain 'internal:stats' column")

    all_stats = []
    weights = []
    has_tensor_shape = "stac:tensor_shape" in df.columns

    if not has_tensor_shape:
        warnings.warn(
            "Column 'stac:tensor_shape' not found. Using equal weights for all rows. "
            "Results may be inaccurate if files have different sizes.",
            UserWarning,
            stacklevel=3,
        )

    for idx, row in df.iterrows():
        stats = row["internal:stats"]
        all_stats.append(stats)

        if has_tensor_shape:
            tensor_shape = row["stac:tensor_shape"]
            # Get spatial dimensions (last 2 dimensions: rows * cols)
            if len(tensor_shape) >= 2:
                n_pixels = int(tensor_shape[-2] * tensor_shape[-1])
            else:
                n_pixels = 1
                warnings.warn(
                    f"Row {idx}: stac:tensor_shape has less than 2 dimensions. Using weight=1.",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            n_pixels = 1

        weights.append(n_pixels)

    return all_stats, weights


def _is_categorical(stats: list[list[float]]) -> bool:
    """
    Detect if stats are categorical or continuous.

    Continuous: len(band_stats) == 9 [min, max, mean, std, valid%, p25, p50, p75, p95]
    Categorical: len(band_stats) != 9 [prob_class_0, prob_class_1, ...]

    Args:
        stats: Stats for all bands from a single row

    Returns:
        True if categorical, False if continuous
    """
    # Handle both lists and numpy arrays
    if len(stats) == 0:
        raise ValueError("Empty stats provided")

    first_band = stats[0]
    if len(first_band) == 0:
        raise ValueError("Empty stats provided")

    return len(first_band) != 9


def stats_categorical(df: "pd.DataFrame") -> np.ndarray:
    """
    Aggregate categorical probabilities using weighted average.

    Computes weighted average of class probabilities across all rows,
    using pixel counts from stac:tensor_shape as weights.

    Args:
        df: DataFrame with 'internal:stats' column (categorical)

    Returns:
        Array of shape [n_bands, n_classes] with averaged probabilities

    Raises:
        ValueError: If stats are not categorical format

    Examples:
        >>> probs = df.stats_categorical()
        >>> probs.shape
        (3, 5)  # 3 bands, 5 classes
        >>> probs[0]  # Band 0 probabilities
        array([0.65, 0.25, 0.08, 0.01, 0.01])
    """
    all_stats, weights = _extract_stats_and_weights(df)

    # Validate categorical format
    if not _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be continuous format (9 values per band). "
            "Use stats_mean(), stats_std(), etc. for continuous data."
        )

    n_bands = len(all_stats[0])
    n_classes = len(all_stats[0][0])

    # Validate all rows have same structure
    for i, stats in enumerate(all_stats):
        if len(stats) != n_bands:
            raise ValueError(f"Row {i}: Expected {n_bands} bands, got {len(stats)}")
        for band_idx, band_stats in enumerate(stats):
            if len(band_stats) != n_classes:
                raise ValueError(
                    f"Row {i}, band {band_idx}: Expected {n_classes} classes, got {len(band_stats)}"
                )

    # Compute weighted average for each band and class
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


def stats_mean(df: "pd.DataFrame") -> np.ndarray:
    """
    Aggregate means using weighted average.

    Computes: sum(mean_i * n_pixels_i) / sum(n_pixels_i)

    Args:
        df: DataFrame with 'internal:stats' column (continuous)

    Returns:
        Array of shape [n_bands] with aggregated means

    Examples:
        >>> means = df.stats_mean()
        >>> means
        array([0.234, 0.456, 0.123])  # 3 bands
    """
    all_stats, weights = _extract_stats_and_weights(df)

    # Validate continuous format
    if _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    n_bands = len(all_stats[0])
    result = np.zeros(n_bands, dtype=np.float32)
    total_weight = sum(weights)

    for band_idx in range(n_bands):
        weighted_sum = sum(
            stats[band_idx][2] * weight  # mean is at index 2
            for stats, weight in zip(all_stats, weights, strict=False)
        )
        result[band_idx] = weighted_sum / total_weight

    return result


def stats_std(df: "pd.DataFrame") -> np.ndarray:
    """
    Aggregate standard deviations using pooled std formula.

    Computes pooled std: sqrt(sum((n_i-1)*std_i^2 + n_i*(mean_i - mu_global)^2) / (sum(n_i) - 1))

    Args:
        df: DataFrame with 'internal:stats' column (continuous)

    Returns:
        Array of shape [n_bands] with aggregated standard deviations

    Examples:
        >>> stds = df.stats_std()
        >>> stds
        array([0.045, 0.067, 0.023])  # 3 bands
    """
    all_stats, weights = _extract_stats_and_weights(df)

    # Validate continuous format
    if _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    n_bands = len(all_stats[0])

    # First compute global means
    global_means = stats_mean(df)

    result = np.zeros(n_bands, dtype=np.float32)
    total_weight = sum(weights)

    for band_idx in range(n_bands):
        variance_sum = 0.0

        for stats, weight in zip(all_stats, weights, strict=False):
            mean_i = stats[band_idx][2]  # mean at index 2
            std_i = stats[band_idx][3]  # std at index 3

            # Pooled variance formula
            variance_sum += (weight - 1) * (std_i**2)
            variance_sum += weight * ((mean_i - global_means[band_idx]) ** 2)

        pooled_variance = variance_sum / (total_weight - 1)
        result[band_idx] = np.sqrt(pooled_variance)

    return result


def stats_min(df: "pd.DataFrame") -> np.ndarray:
    """
    Aggregate minimums (global min across all rows).

    Args:
        df: DataFrame with 'internal:stats' column (continuous)

    Returns:
        Array of shape [n_bands] with global minimums

    Examples:
        >>> mins = df.stats_min()
        >>> mins
        array([0.001, 0.002, 0.000])  # 3 bands
    """
    all_stats, _ = _extract_stats_and_weights(df)

    # Validate continuous format
    if _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    n_bands = len(all_stats[0])
    result = np.full(n_bands, np.inf, dtype=np.float32)

    for band_idx in range(n_bands):
        for stats in all_stats:
            min_val = stats[band_idx][0]  # min is at index 0
            result[band_idx] = min(result[band_idx], min_val)

    return result


def stats_max(df: "pd.DataFrame") -> np.ndarray:
    """
    Aggregate maximums (global max across all rows).

    Args:
        df: DataFrame with 'internal:stats' column (continuous)

    Returns:
        Array of shape [n_bands] with global maximums

    Examples:
        >>> maxs = df.stats_max()
        >>> maxs
        array([0.998, 0.995, 0.987])  # 3 bands
    """
    all_stats, _ = _extract_stats_and_weights(df)

    # Validate continuous format
    if _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    n_bands = len(all_stats[0])
    result = np.full(n_bands, -np.inf, dtype=np.float32)

    for band_idx in range(n_bands):
        for stats in all_stats:
            max_val = stats[band_idx][1]  # max is at index 1
            result[band_idx] = max(result[band_idx], max_val)

    return result


def _stats_percentile(
    df: "pd.DataFrame", percentile_idx: int, percentile_name: str
) -> np.ndarray:
    """
    Internal helper to aggregate percentiles using simple average.

    WARNING: This is NOT statistically exact - it's an approximation.

    Args:
        df: DataFrame with 'internal:stats' column (continuous)
        percentile_idx: Index in stats array (5=p25, 6=p50, 7=p75, 8=p95)
        percentile_name: Name for warning message (e.g., "p25")

    Returns:
        Array of shape [n_bands] with averaged percentiles
    """
    all_stats, _ = _extract_stats_and_weights(df)

    # Validate continuous format
    if _is_categorical(all_stats[0]):
        raise ValueError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    warnings.warn(
        f"stats_{percentile_name}() uses simple averaging of percentiles, "
        f"which is NOT statistically exact. Use with caution. "
        f"For more accurate results, recompute percentiles from raw data.",
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


def stats_p25(df: "pd.DataFrame") -> np.ndarray:
    """
    Aggregate 25th percentiles using simple average.

    WARNING: This is NOT statistically exact - it's an approximation.
    Avoid using this for critical analysis. Prefer recomputing from raw data.

    Args:
        df: DataFrame with 'internal:stats' column (continuous)

    Returns:
        Array of shape [n_bands] with averaged 25th percentiles

    Examples:
        >>> p25 = df.stats_p25()
        >>> p25
        array([0.145, 0.234, 0.089])  # 3 bands
    """
    return _stats_percentile(df, 5, "p25")


def stats_p50(df: "pd.DataFrame") -> np.ndarray:
    """
    Aggregate 50th percentiles (median) using simple average.

    WARNING: This is NOT statistically exact - it's an approximation.
    Avoid using this for critical analysis. Prefer recomputing from raw data.

    Args:
        df: DataFrame with 'internal:stats' column (continuous)

    Returns:
        Array of shape [n_bands] with averaged 50th percentiles

    Examples:
        >>> p50 = df.stats_p50()
        >>> p50
        array([0.234, 0.345, 0.156])  # 3 bands
    """
    return _stats_percentile(df, 6, "p50")


def stats_p75(df: "pd.DataFrame") -> np.ndarray:
    """
    Aggregate 75th percentiles using simple average.

    WARNING: This is NOT statistically exact - it's an approximation.
    Avoid using this for critical analysis. Prefer recomputing from raw data.

    Args:
        df: DataFrame with 'internal:stats' column (continuous)

    Returns:
        Array of shape [n_bands] with averaged 75th percentiles

    Examples:
        >>> p75 = df.stats_p75()
        >>> p75
        array([0.345, 0.456, 0.234])  # 3 bands
    """
    return _stats_percentile(df, 7, "p75")


def stats_p95(df: "pd.DataFrame") -> np.ndarray:
    """
    Aggregate 95th percentiles using simple average.

    WARNING: This is NOT statistically exact - it's an approximation.
    Avoid using this for critical analysis. Prefer recomputing from raw data.

    Args:
        df: DataFrame with 'internal:stats' column (continuous)

    Returns:
        Array of shape [n_bands] with averaged 95th percentiles

    Examples:
        >>> p95 = df.stats_p95()
        >>> p95
        array([0.567, 0.678, 0.456])  # 3 bands
    """
    return _stats_percentile(df, 8, "p95")
