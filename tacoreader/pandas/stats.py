import warnings
import numpy as np
import pandas as pd


def _get_stats_array(df: pd.DataFrame) -> np.ndarray:
    """Extract and validate internal:stats data as numpy array."""
    if 'internal:stats' not in df.columns:
        raise ValueError("DataFrame must contain 'internal:stats' column")
    
    if df['internal:stats'].isnull().any():
        raise ValueError("internal:stats column contains None values")
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    stats_list = df['internal:stats'].tolist()
    
    if not stats_list:
        raise ValueError("No stats data found")
    
    # Validate consistent structure
    reference_bands = len(stats_list[0])
    reference_stats_per_band = len(stats_list[0][0])
    
    for i, sample_stats in enumerate(stats_list):
        if len(sample_stats) != reference_bands:
            raise ValueError(f"Sample {i} has {len(sample_stats)} bands, expected {reference_bands}")
        
        for band_idx, band_stats in enumerate(sample_stats):
            if len(band_stats) != reference_stats_per_band:
                raise ValueError(
                    f"Sample {i}, band {band_idx} has {len(band_stats)} stats, "
                    f"expected {reference_stats_per_band}"
                )
    
    # Convert to numpy array: Shape (n_samples, n_bands, n_stats)
    return np.array(stats_list, dtype=np.float32)


def _validate_continuous(stats_array: np.ndarray) -> None:
    """Validate array contains continuous stats (9 values per band)."""
    if stats_array.shape[2] != 9:
        raise ValueError("Function requires continuous stats (9 values per band)")


def _validate_categorical(stats_array: np.ndarray) -> None:
    """Validate array contains categorical stats (not 9 values per band).""" 
    if stats_array.shape[2] == 9:
        raise ValueError("Function cannot be used with continuous stats (9 values per band)")


def _get_band_result(result_array: np.ndarray, band: int | None) -> list[float] | float:
    """Return full array or specific band based on band parameter."""
    if band is not None:
        if band >= len(result_array) or band < 0:
            raise IndexError(f"Band {band} out of range, only {len(result_array)} bands available")
        return float(result_array[band])
    return result_array.tolist()


def aggregate_min(df: pd.DataFrame, band: int | None = None) -> list[float] | float:
    """Aggregate minimum values across all samples."""
    stats_array = _get_stats_array(df)
    _validate_continuous(stats_array)
    
    result = np.min(stats_array[:, :, 0], axis=0)  # Index 0 = min
    return _get_band_result(result, band)


def aggregate_max(df: pd.DataFrame, band: int | None = None) -> list[float] | float:
    """Aggregate maximum values across all samples."""
    stats_array = _get_stats_array(df)
    _validate_continuous(stats_array)
    
    result = np.max(stats_array[:, :, 1], axis=0)  # Index 1 = max
    return _get_band_result(result, band)


def aggregate_mean(df: pd.DataFrame, band: int | None = None) -> list[float] | float:
    """Aggregate mean values across all samples."""
    stats_array = _get_stats_array(df)
    _validate_continuous(stats_array)
    
    result = np.mean(stats_array[:, :, 2], axis=0)  # Index 2 = mean
    return _get_band_result(result, band)


def aggregate_std_approximation(df: pd.DataFrame, band: int | None = None, warn: bool = True) -> list[float] | float:
    """Approximate standard deviation aggregation."""
    if warn:
        warnings.warn(
            "aggregate_std_approximation is a simple average of local std values. "
            "True global standard deviation requires raw pixel data.",
            UserWarning,
            stacklevel=2
        )
    
    stats_array = _get_stats_array(df)
    _validate_continuous(stats_array)
    
    result = np.mean(stats_array[:, :, 3], axis=0)  # Index 3 = std
    return _get_band_result(result, band)


def aggregate_valid_pct(df: pd.DataFrame, band: int | None = None) -> list[float] | float:
    """Aggregate valid pixel percentages across all samples."""
    stats_array = _get_stats_array(df)
    _validate_continuous(stats_array)
    
    result = np.mean(stats_array[:, :, 4], axis=0)  # Index 4 = valid_pct
    return _get_band_result(result, band)


def aggregate_p25_approximation(df: pd.DataFrame, band: int | None = None, warn: bool = True) -> list[float] | float:
    """Approximate 25th percentile aggregation."""
    if warn:
        warnings.warn(
            "aggregate_p25_approximation averages local percentiles. "
            "True global percentile requires raw pixel data.",
            UserWarning,
            stacklevel=2
        )
    
    stats_array = _get_stats_array(df)
    _validate_continuous(stats_array)
    
    result = np.mean(stats_array[:, :, 5], axis=0)  # Index 5 = p25
    return _get_band_result(result, band)


def aggregate_p50_approximation(df: pd.DataFrame, band: int | None = None, warn: bool = True) -> list[float] | float:
    """Approximate 50th percentile (median) aggregation."""
    if warn:
        warnings.warn(
            "aggregate_p50_approximation averages local medians. "
            "True global median requires raw pixel data.",
            UserWarning,
            stacklevel=2
        )
    
    stats_array = _get_stats_array(df)
    _validate_continuous(stats_array)
    
    result = np.mean(stats_array[:, :, 6], axis=0)  # Index 6 = p50
    return _get_band_result(result, band)


def aggregate_p75_approximation(df: pd.DataFrame, band: int | None = None, warn: bool = True) -> list[float] | float:
    """Approximate 75th percentile aggregation."""
    if warn:
        warnings.warn(
            "aggregate_p75_approximation averages local percentiles. "
            "True global percentile requires raw pixel data.",
            UserWarning,
            stacklevel=2
        )
    
    stats_array = _get_stats_array(df)
    _validate_continuous(stats_array)
    
    result = np.mean(stats_array[:, :, 7], axis=0)  # Index 7 = p75
    return _get_band_result(result, band)


def aggregate_p95_approximation(df: pd.DataFrame, band: int | None = None, warn: bool = True) -> list[float] | float:
    """Approximate 95th percentile aggregation."""
    if warn:
        warnings.warn(
            "aggregate_p95_approximation averages local percentiles. "
            "True global percentile requires raw pixel data.",
            UserWarning,
            stacklevel=2
        )
    
    stats_array = _get_stats_array(df)
    _validate_continuous(stats_array)
    
    result = np.mean(stats_array[:, :, 8], axis=0)  # Index 8 = p95
    return _get_band_result(result, band)


def aggregate_categorical(df: pd.DataFrame, band: int | None = None) -> list[list[float]] | list[float]:
    """Aggregate categorical probability distributions."""
    stats_array = _get_stats_array(df)
    _validate_categorical(stats_array)
    
    # Average across samples: (n_samples, n_bands, n_classes) -> (n_bands, n_classes)
    result = np.mean(stats_array, axis=0)
    
    if band is not None:
        if band >= result.shape[0] or band < 0:
            raise IndexError(f"Band {band} out of range, only {result.shape[0]} bands available")
        return result[band].tolist()
    
    return [band_probs.tolist() for band_probs in result]