"""
DataFrame backend registry and factory.

Provides centralized backend registration and creation of TacoDataFrame
instances. Dataset.py uses create_dataframe() to remain backend-agnostic.
"""

from typing import Any, Protocol

from tacoreader._constants import AVAILABLE_BACKENDS, DataFrameBackend
from tacoreader._exceptions import TacoBackendError

# Re-export base class for type hints and isinstance checks
from tacoreader.dataframe.base import TacoDataFrame


class BackendFactory(Protocol):
    """Protocol for backend factory functions."""

    def __call__(self, arrow_table: Any, format_type: str) -> Any:
        """Create TacoDataFrame from PyArrow Table."""
        ...


# Backend registry: name -> factory function
_BACKENDS: dict[DataFrameBackend, BackendFactory] = {}


def register_backend(name: DataFrameBackend, factory_fn: BackendFactory) -> None:
    """
    Register a DataFrame backend.

    Args:
        name: Backend name from DataFrameBackend literal
        factory_fn: Factory function that takes (arrow_table, format_type) -> TacoDataFrame

    Example:
        from .pyarrow import TacoDataFrameArrow
        register_backend('pyarrow', TacoDataFrameArrow.from_arrow)
    """
    _BACKENDS[name] = factory_fn


def create_dataframe(backend: str, arrow_table: Any, format_type: str):
    """
    Factory function to create TacoDataFrame from PyArrow Table.

    This is the main entry point used by TacoDataset.data property
    to convert DuckDB results (PyArrow Tables) into backend-specific
    TacoDataFrame instances.

    Args:
        backend: Backend name ("pyarrow", "polars", "pandas")
        arrow_table: PyArrow Table from DuckDB query
        format_type: Storage format ("zip", "folder", "tacocat")

    Returns:
        Backend-specific TacoDataFrame instance

    Raises:
        TacoBackendError: If backend is not registered or unknown

    Example:
        # In dataset.py
        from tacoreader.dataframe import create_dataframe

        arrow_table = self._duckdb.execute(query).fetch_arrow_table()
        return create_dataframe('pyarrow', arrow_table, self._format)
    """
    # Validate backend type first
    if backend not in AVAILABLE_BACKENDS:
        raise TacoBackendError(
            f"Unknown backend: '{backend}'\n"
            f"Available backends: {AVAILABLE_BACKENDS}\n"
            f"\n"
            f"To use additional backends, install required packages:\n"
            f"  pip install polars  # For Polars backend\n"
            f"  pip install pandas  # For Pandas backend"
        )

    # Check if backend is registered
    if backend not in _BACKENDS:
        raise TacoBackendError(
            f"Backend '{backend}' is not registered.\n"
            f"Registered backends: {list(_BACKENDS.keys())}\n"
            f"\n"
            f"The backend may require additional dependencies:\n"
            f"  pip install {backend}"
        )

    factory = _BACKENDS[backend]  # type: ignore[index]
    return factory(arrow_table, format_type)


def get_available_backends() -> list[DataFrameBackend]:
    """
    Get list of currently registered backends.

    Returns:
        List of registered backend names
    """
    return list(_BACKENDS.keys())


def _register_all_backends() -> None:
    """Register all available backends on module import."""
    # PyArrow (default, always available)
    from tacoreader.dataframe.pyarrow import TacoDataFrameArrow

    register_backend("pyarrow", TacoDataFrameArrow.from_arrow)

    # Polars (optional)
    from tacoreader.dataframe.polars import TacoDataFramePolars

    register_backend("polars", TacoDataFramePolars.from_arrow)

    # Pandas (optional)
    from tacoreader.dataframe.pandas import TacoDataFramePandas

    register_backend("pandas", TacoDataFramePandas.from_arrow)


# Auto-register on module import
_register_all_backends()

__all__ = [
    "TacoDataFrame",
    "create_dataframe",
    "get_available_backends",
    "register_backend",
]
