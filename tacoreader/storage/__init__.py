"""
TACO format backends.

Factory functions for creating backend instances.
"""

from tacoreader._exceptions import TacoFormatError
from tacoreader.dataset import TacoDataset
from tacoreader.storage.base import TacoBackend
from tacoreader.storage.folder import FolderBackend
from tacoreader.storage.tacocat import TacoCatBackend
from tacoreader.storage.zip import ZipBackend


def create_backend(format_type: str) -> TacoBackend:
    """Create backend instance: 'zip', 'folder', or 'tacocat'."""
    backends: dict[str, type[TacoBackend]] = {
        "zip": ZipBackend,
        "folder": FolderBackend,
        "tacocat": TacoCatBackend,
    }

    backend_class = backends.get(format_type)
    if backend_class is None:
        raise TacoFormatError(
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
    "TacoCatBackend",
    "ZipBackend",
    "create_backend",
    "load_dataset",
]
