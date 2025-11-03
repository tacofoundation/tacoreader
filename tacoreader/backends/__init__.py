"""
TACO format backends.

Factory functions for creating backend instances.
"""

from pathlib import Path

from tacoreader.backends.base import TacoBackend
from tacoreader.backends.folder import FolderBackend
from tacoreader.backends.tacocat import TacoCatBackend
from tacoreader.backends.zip import ZipBackend
from tacoreader.dataset import TacoDataset

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_backend(format_type: str) -> TacoBackend:
    """
    Create backend instance for specified format.

    Args:
        format_type: Format identifier ('zip', 'folder', or 'tacocat')

    Returns:
        Backend instance

    Raises:
        ValueError: If format_type is unknown

    Example:
        >>> backend = create_backend("zip")
        >>> dataset = backend.load("data.tacozip")
    """
    backends = {
        "zip": ZipBackend,
        "folder": FolderBackend,
        "tacocat": TacoCatBackend,
    }

    backend_class = backends.get(format_type)
    if backend_class is None:
        raise ValueError(
            f"Unknown format: {format_type}\n"
            f"Supported formats: {', '.join(backends.keys())}"
        )

    return backend_class()


def load_dataset(
    path: str, format_type: str, cache_dir: Path | None = None
) -> TacoDataset:
    """
    Load dataset using specified backend.

    Args:
        path: Dataset path
        format_type: Format identifier ('zip', 'folder', or 'tacocat')
        cache_dir: Optional cache directory

    Returns:
        TacoDataset with lazy SQL interface

    Example:
        >>> ds = load_dataset("data.tacozip", "zip")
        >>> ds = load_dataset("data/", "folder")
        >>> ds = load_dataset("__tacocat__", "tacocat")
    """
    backend = create_backend(format_type)
    return backend.load(path, cache_dir)


__all__ = [
    "FolderBackend",
    "TacoBackend",
    "TacoCatBackend",
    "ZipBackend",
    "create_backend",
    "load_dataset",
]
