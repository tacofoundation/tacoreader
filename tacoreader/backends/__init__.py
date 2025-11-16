"""
TACO format backends.

Factory functions for creating backend instances.
"""

from tacoreader.backends.base import TacoBackend
from tacoreader.backends.folder import FolderBackend
from tacoreader.backends.tacocat import TacoCatBackend
from tacoreader.backends.zip import ZipBackend
from tacoreader.dataset import TacoDataset


def create_backend(format_type: str) -> TacoBackend:
    """Create backend instance: 'zip', 'folder', or 'tacocat'."""
    backends = {
        "zip": ZipBackend,
        "folder": FolderBackend,
        "tacocat": TacoCatBackend,
    }

    backend_class = backends.get(format_type)
    if backend_class is None:
        raise ValueError(
            f"Unknown format: {format_type}\n"
            f"Supported: {', '.join(backends.keys())}"
        )

    return backend_class()


def load_dataset(path: str, format_type: str) -> TacoDataset:
    """Load dataset using specified backend."""
    backend = create_backend(format_type)
    return backend.load(path)


__all__ = [
    "FolderBackend",
    "TacoBackend",
    "TacoCatBackend",
    "ZipBackend",
    "create_backend",
    "load_dataset",
]
