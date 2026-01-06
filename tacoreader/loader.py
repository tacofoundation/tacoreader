"""Load TACO datasets from any format.

Main entry point. Auto-detects format and dispatches to backend.
Supports single files or lists (automatically concatenated).
"""

from pathlib import Path

from tqdm import tqdm

from tacoreader._exceptions import TacoQueryError
from tacoreader._legacy import is_legacy_format, raise_legacy_error
from tacoreader._path import TacoPath
from tacoreader.dataset import TacoDataset


def load(
    path: str | Path | list[str | Path],
    base_path: str | Path | None = None,
    backend: str | None = None,
    cache: bool = True,
) -> TacoDataset:
    """Load TACO dataset(s) with auto format detection.

    Returns TacoDataset with lazy SQL interface. Multiple files are
    automatically concatenated if schemas are compatible.

    Args:
        path: Single path or list of paths to load
            Formats: .tacozip (ZIP), directory (FOLDER), .tacocat
            Examples: "data.tacozip", ["part1.tacozip", "part2.tacozip"]
        base_path: Override base path for TacoCat ZIP files (TacoCat only)
            Use when .tacocat and .tacozip files are in different locations
        backend: DataFrame backend to use ('pyarrow', 'polars', 'pandas')
            If None, uses global backend set by tacoreader.use()
            Default: 'pyarrow'
        cache: Use disk cache for remote TacoCat datasets (default True)

    Returns:
        TacoDataset with lazy SQL interface

    Examples:
        reader = tacoreader.load('data.taco')
        reader = tacoreader.load('data.taco', backend='polars')
        ds = tacoreader.load('https://.../.tacocat')
        ds = tacoreader.load('https://.../.tacocat', cache=False)
    """
    from tacoreader import _DATAFRAME_BACKEND

    backend = backend or _DATAFRAME_BACKEND

    # Normalize to strings
    path = [str(p) for p in path] if isinstance(path, list) else str(path)
    if base_path is not None:
        base_path = str(base_path)

    # Handle list of paths
    if isinstance(path, list):
        if len(path) == 0:
            raise TacoQueryError("Cannot load empty list of paths")

        if len(path) == 1:
            return load(path[0], base_path, backend, cache)

        datasets = []
        path_iter = tqdm(path, desc="Loading datasets", unit="ds") if len(path) >= 3 else path
        for p in path_iter:
            datasets.append(load(p, base_path, backend, cache))

        from tacoreader.concat import concat

        return concat(datasets)

    # Check for legacy format
    if is_legacy_format(path):
        raise_legacy_error(path)

    # Load dataset
    tp = TacoPath(path, base_path)
    dataset = tp.load(cache=cache)
    dataset._dataframe_backend = backend

    return dataset
