"""
Statistics aggregation for TACO datasets.

Weighted aggregation of pre-computed stats from internal:stats column.
Operates on PyArrow Tables.
"""

import warnings

import pyarrow as pa
import pyarrow.compute as pc

from tacoreader._constants import STATS_CONTINUOUS_LENGTH
from tacoreader._exceptions import TacoQueryError

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _require_numpy(func_name: str) -> None:
    """Raise ImportError if NumPy is not available."""
    if not HAS_NUMPY:
        raise ImportError(
            f"{func_name}() requires NumPy. Install it with: pip install numpy"
        )


def _extract_stats_and_weights(
    table: pa.Table,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract internal:stats and calculate pixel counts as weights.

    Returns:
        stats_3d: Shape (n_rows, n_bands, 9)
        weights: Shape (n_rows,)
    """
    _require_numpy("_extract_stats_and_weights")

    if "internal:stats" not in table.schema.names:
        raise TacoQueryError("Table must contain 'internal:stats' column")

    stats_list = table.column("internal:stats").to_pylist()
    stats_3d = np.array(stats_list, dtype=np.float32)

    if "stac:tensor_shape" in table.schema.names:
        shapes_col = table.column("stac:tensor_shape")

        try:
            heights = pc.list_element(shapes_col, -2)
            widths = pc.list_element(shapes_col, -1)
            weights_arr = pc.multiply(heights, widths)
            weights = weights_arr.to_numpy(zero_copy_only=False).astype(np.int64)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            shapes = shapes_col.to_pylist()
            weights = np.array(
                [int(s[-2] * s[-1]) if len(s) >= 2 else 1 for s in shapes],
                dtype=np.int64,
            )

            if any(len(s) < 2 for s in shapes):
                warnings.warn(
                    "Some rows have stac:tensor_shape with <2 dimensions. Using weight=1.",
                    UserWarning,
                    stacklevel=4,
                )
    else:
        warnings.warn(
            "Column 'stac:tensor_shape' not found. Using equal weights. "
            "Results may be inaccurate if files have different sizes.",
            UserWarning,
            stacklevel=4,
        )
        weights = np.ones(len(stats_3d), dtype=np.int64)

    return stats_3d, weights


def _is_categorical(stats: list[list[float]]) -> bool:
    """
    Detect categorical vs continuous stats.

    Continuous: 9 values [min, max, mean, std, valid%, p25, p50, p75, p95]
    Categorical: N values [prob_class_0, ..., prob_class_N]
    """
    if len(stats) == 0 or len(stats[0]) == 0:
        raise TacoQueryError("Empty stats provided")

    return len(stats[0]) != STATS_CONTINUOUS_LENGTH


def stats_categorical(table: pa.Table) -> np.ndarray:
    """
    Aggregate categorical probabilities using weighted average.

    Returns: Array [n_bands, n_classes]
    """
    _require_numpy("stats_categorical")

    all_stats = table.column("internal:stats").to_pylist()

    if not _is_categorical(all_stats[0]):
        raise TacoQueryError(
            f"Stats appear to be continuous format ({STATS_CONTINUOUS_LENGTH} values per band). "
            "Use stats_mean(), stats_std(), etc. for continuous data."
        )

    stats_3d = np.array(all_stats, dtype=np.float32)

    if "stac:tensor_shape" in table.schema.names:
        shapes_col = table.column("stac:tensor_shape")
        try:
            heights = pc.list_element(shapes_col, -2)
            widths = pc.list_element(shapes_col, -1)
            weights_arr = pc.multiply(heights, widths)
            weights = weights_arr.to_numpy(zero_copy_only=False).astype(np.float32)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            shapes = shapes_col.to_pylist()
            weights = np.array(
                [s[-2] * s[-1] if len(s) >= 2 else 1 for s in shapes], dtype=np.float32
            )
    else:
        weights = np.ones(len(stats_3d), dtype=np.float32)

    weights_expanded = weights[:, np.newaxis, np.newaxis]
    weighted_stats = stats_3d * weights_expanded
    result = weighted_stats.sum(axis=0) / weights.sum()

    return result


def stats_mean(table: pa.Table) -> np.ndarray:
    """Aggregate means using weighted average."""
    _require_numpy("stats_mean")

    stats_3d, weights = _extract_stats_and_weights(table)

    if _is_categorical(stats_3d.tolist()):
        raise TacoQueryError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    means = stats_3d[:, :, 2]
    result = np.average(means, axis=0, weights=weights)

    return result.astype(np.float32)


def stats_std(table: pa.Table) -> np.ndarray:
    """Aggregate standard deviations using pooled variance formula."""
    _require_numpy("stats_std")

    stats_3d, weights = _extract_stats_and_weights(table)

    if _is_categorical(stats_3d.tolist()):
        raise TacoQueryError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    means = stats_3d[:, :, 2]
    stds = stats_3d[:, :, 3]

    global_mean = np.average(means, axis=0, weights=weights)

    weights_expanded = weights[:, np.newaxis]
    variance_terms = (weights_expanded - 1) * (stds**2)
    variance_terms += weights_expanded * ((means - global_mean) ** 2)

    pooled_variance = variance_terms.sum(axis=0) / (weights.sum() - 1)
    result = np.sqrt(pooled_variance)

    return result.astype(np.float32)


def stats_min(table: pa.Table) -> np.ndarray:
    """Aggregate minimums (global min across all rows)."""
    _require_numpy("stats_min")

    stats_3d, _ = _extract_stats_and_weights(table)

    if _is_categorical(stats_3d.tolist()):
        raise TacoQueryError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    mins = stats_3d[:, :, 0]
    result = mins.min(axis=0)

    return result.astype(np.float32)


def stats_max(table: pa.Table) -> np.ndarray:
    """Aggregate maximums (global max across all rows)."""
    _require_numpy("stats_max")

    stats_3d, _ = _extract_stats_and_weights(table)

    if _is_categorical(stats_3d.tolist()):
        raise TacoQueryError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    maxs = stats_3d[:, :, 1]
    result = maxs.max(axis=0)

    return result.astype(np.float32)


def _stats_percentile(
    table: pa.Table, percentile_idx: int, percentile_name: str
) -> np.ndarray:
    """
    Aggregate percentiles using simple average.

    WARNING: Not statistically exact. For critical analysis, recompute from raw data.
    """
    _require_numpy("_stats_percentile")

    stats_3d, _ = _extract_stats_and_weights(table)

    if _is_categorical(stats_3d.tolist()):
        raise TacoQueryError(
            "Stats appear to be categorical format. Use stats_categorical() instead."
        )

    warnings.warn(
        f"stats_{percentile_name}() uses simple averaging, NOT exact. "
        "For critical analysis, recompute from raw data.",
        UserWarning,
        stacklevel=3,
    )

    percentiles = stats_3d[:, :, percentile_idx]
    result = percentiles.mean(axis=0)

    return result.astype(np.float32)


def stats_p25(table: pa.Table) -> np.ndarray:
    """Aggregate 25th percentiles (approximation)."""
    return _stats_percentile(table, 5, "p25")


def stats_p50(table: pa.Table) -> np.ndarray:
    """Aggregate 50th percentiles / median (approximation)."""
    return _stats_percentile(table, 6, "p50")


def stats_p75(table: pa.Table) -> np.ndarray:
    """Aggregate 75th percentiles (approximation)."""
    return _stats_percentile(table, 7, "p75")


def stats_p95(table: pa.Table) -> np.ndarray:
    """Aggregate 95th percentiles (approximation)."""
    return _stats_percentile(table, 8, "p95")
