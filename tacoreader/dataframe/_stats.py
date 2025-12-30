"""
Statistics aggregation for TACO datasets.

Weighted aggregation of pre-computed stats from format-specific columns
(geotiff:stats, netcdf:stats, zarr:stats).

Core functions for TacoDataset.stats_*() methods:
- _aggregate_continuous: mean, min, max, percentiles
- _aggregate_std: pooled standard deviation
- _aggregate_categorical: weighted class probabilities
"""

import warnings

import pyarrow as pa
import pyarrow.compute as pc

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from tacoreader._constants import (
    STATS_CONTINUOUS_INDICES,
    STATS_CONTINUOUS_LENGTH,
    STATS_WEIGHT_COLUMN,
)
from tacoreader._exceptions import TacoQueryError


def _require_numpy(func_name: str) -> None:
    """Raise ImportError if NumPy is not available."""
    if not HAS_NUMPY:
        raise ImportError(f"{func_name}() requires NumPy. Install with: pip install numpy")


def _extract_weights(table: pa.Table, valid_mask: "np.ndarray | None" = None) -> "np.ndarray":
    """
    Extract pixel counts as weights from stac:tensor_shape column.

    Returns array of weights (one per row). If column missing, returns ones.

    Args:
        table: PyArrow table
        valid_mask: Optional boolean mask to filter weights (from _extract_stats_array)
    """
    _require_numpy("_extract_weights")

    if STATS_WEIGHT_COLUMN not in table.schema.names:
        warnings.warn(
            f"Column '{STATS_WEIGHT_COLUMN}' not found. Using equal weights. "
            f"Results may be inaccurate if files have different sizes.",
            UserWarning,
            stacklevel=4,
        )
        n_rows = valid_mask.sum() if valid_mask is not None else table.num_rows
        return np.ones(n_rows, dtype=np.float64)

    shapes_col = table.column(STATS_WEIGHT_COLUMN)

    try:
        heights = pc.list_element(shapes_col, -2)
        widths = pc.list_element(shapes_col, -1)
        weights_arr = pc.multiply(heights, widths)
        weights = weights_arr.to_numpy(zero_copy_only=False).astype(np.float64)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        shapes = shapes_col.to_pylist()
        weights = np.array(
            [float(s[-2] * s[-1]) if s and len(s) >= 2 else 1.0 for s in shapes],
            dtype=np.float64,
        )
        if any(not s or len(s) < 2 for s in shapes):
            warnings.warn(
                f"Some rows have {STATS_WEIGHT_COLUMN} with <2 dimensions. Using weight=1.",
                UserWarning,
                stacklevel=4,
            )

    # Apply mask if provided
    if valid_mask is not None:
        weights = weights[valid_mask]

    return weights


def _extract_stats_array(table: pa.Table, stats_col: str) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Extract stats as 3D numpy array: (n_rows, n_bands, n_values).

    Filters out None values and returns valid indices for weight alignment.

    Args:
        table: PyArrow table with stats column
        stats_col: Name of stats column (e.g., "geotiff:stats")

    Returns:
        Tuple of (stats_3d, valid_mask) where:
        - stats_3d: np.ndarray with shape (n_valid_rows, n_bands, n_values)
        - valid_mask: boolean np.ndarray with shape (n_rows,) indicating valid rows
    """
    _require_numpy("_extract_stats_array")

    if stats_col not in table.schema.names:
        raise TacoQueryError(f"Stats column '{stats_col}' not found in table")

    stats_list = table.column(stats_col).to_pylist()

    if not stats_list:
        raise TacoQueryError("No stats data found (empty table)")

    # Filter out Nones
    valid_mask = np.array([s is not None and len(s) > 0 for s in stats_list], dtype=bool)
    valid_stats = [s for s in stats_list if s is not None and len(s) > 0]

    if not valid_stats:
        raise TacoQueryError("No valid stats found (all samples have None)")

    return np.array(valid_stats, dtype=np.float32), valid_mask


def _is_categorical(stats_3d: "np.ndarray") -> bool:
    """
    Detect categorical vs continuous stats based on array shape.

    Continuous: 9 values [min, max, mean, std, valid%, p25, p50, p75, p95]
    Categorical: N values [prob_class_0, ..., prob_class_N]
    """
    if stats_3d.size == 0:
        raise TacoQueryError("Empty stats array")

    n_values = stats_3d.shape[-1]
    return n_values != STATS_CONTINUOUS_LENGTH


def _aggregate_continuous(
    table: pa.Table,
    stats_col: str,
    stat_name: str,
) -> "np.ndarray":
    """
    Aggregate continuous statistics across samples.

    Args:
        table: PyArrow table with stats and weight columns
        stats_col: Name of stats column
        stat_name: One of "min", "max", "mean", "p25", "p50", "p75", "p95"

    Returns:
        np.ndarray with shape (n_bands,)
    """
    stats_3d, valid_mask = _extract_stats_array(table, stats_col)

    if _is_categorical(stats_3d):
        raise TacoQueryError("Stats appear to be categorical format. Use stats_categorical() instead.")

    idx = STATS_CONTINUOUS_INDICES.get(stat_name)
    if idx is None:
        raise TacoQueryError(f"Unknown stat: '{stat_name}'. Available: {list(STATS_CONTINUOUS_INDICES.keys())}")

    values = stats_3d[:, :, idx]  # shape: (n_valid_rows, n_bands)

    # min/max: global across all rows
    if stat_name == "min":
        return values.min(axis=0).astype(np.float32)
    elif stat_name == "max":
        return values.max(axis=0).astype(np.float32)

    # mean: weighted average
    if stat_name == "mean":
        weights = _extract_weights(table, valid_mask)
        return np.average(values, axis=0, weights=weights).astype(np.float32)

    # percentiles: simple average (approximation)
    if stat_name in ("p25", "p50", "p75", "p95"):
        warnings.warn(
            f"stats_{stat_name}() uses simple averaging, NOT exact. For critical analysis, recompute from raw data.",
            UserWarning,
            stacklevel=4,
        )
        return values.mean(axis=0).astype(np.float32)

    raise TacoQueryError(f"Unhandled stat: '{stat_name}'")


def _aggregate_std(table: pa.Table, stats_col: str) -> "np.ndarray":
    """
    Aggregate standard deviations using pooled variance formula.

    Pooled variance accounts for both within-sample variance and
    between-sample mean differences.

    Args:
        table: PyArrow table with stats and weight columns
        stats_col: Name of stats column

    Returns:
        np.ndarray with shape (n_bands,)
    """
    stats_3d, valid_mask = _extract_stats_array(table, stats_col)

    if _is_categorical(stats_3d):
        raise TacoQueryError("Stats appear to be categorical format. Use stats_categorical() instead.")

    weights = _extract_weights(table, valid_mask)

    means = stats_3d[:, :, STATS_CONTINUOUS_INDICES["mean"]]
    stds = stats_3d[:, :, STATS_CONTINUOUS_INDICES["std"]]

    # Global mean (weighted)
    global_mean = np.average(means, axis=0, weights=weights)

    # Pooled variance formula
    weights_expanded = weights[:, np.newaxis]
    variance_terms = (weights_expanded - 1) * (stds**2)
    variance_terms += weights_expanded * ((means - global_mean) ** 2)

    pooled_variance = variance_terms.sum(axis=0) / (weights.sum() - 1)
    return np.sqrt(pooled_variance).astype(np.float32)


def _aggregate_categorical(table: pa.Table, stats_col: str) -> "np.ndarray":
    """
    Aggregate categorical probabilities using weighted average.

    Args:
        table: PyArrow table with stats and weight columns
        stats_col: Name of stats column

    Returns:
        np.ndarray with shape (n_bands, n_classes)
    """
    stats_3d, valid_mask = _extract_stats_array(table, stats_col)

    if not _is_categorical(stats_3d):
        raise TacoQueryError(
            f"Stats appear to be continuous format ({STATS_CONTINUOUS_LENGTH} values per band). "
            f"Use stats_mean(), stats_std(), etc. for continuous data."
        )

    weights = _extract_weights(table, valid_mask)

    # Weighted average: (n_valid_rows, n_bands, n_classes) -> (n_bands, n_classes)
    weights_expanded = weights[:, np.newaxis, np.newaxis]
    weighted_stats = stats_3d * weights_expanded
    result = weighted_stats.sum(axis=0) / weights.sum()

    return result.astype(np.float32)
