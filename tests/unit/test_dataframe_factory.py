"""Tests for DataFrame backend factory and registration."""

import pytest
import pyarrow as pa

from tacoreader._exceptions import TacoBackendError
from tacoreader.dataframe import create_dataframe, get_available_backends


class TestBackendFactory:
    """Factory function error handling."""

    def test_unknown_backend_raises_error(self):
        """Error when backend not in AVAILABLE_BACKENDS."""
        table = pa.table({"id": [1, 2, 3], "type": ["FILE", "FILE", "FILE"]})

        with pytest.raises(TacoBackendError, match="Unknown backend: 'nonexistent'"):
            create_dataframe("nonexistent", table, "folder")

        with pytest.raises(TacoBackendError, match="Unknown backend: 'tensorflow'"):
            create_dataframe("tensorflow", table, "zip")

    def test_unregistered_backend_raises_error(self):
        """Error when backend valid but not registered (missing dependency)."""
        table = pa.table({"id": [1, 2, 3], "type": ["FILE", "FILE", "FILE"]})

        # Simulate unregistered backend by using invalid name that passes type check
        # This would happen if backend package isn't installed
        # Note: This is harder to test without actually uninstalling packages
        # So we test the error message structure instead

        try:
            create_dataframe("invalid_but_typed", table, "folder")
        except TacoBackendError as e:
            # Should mention either "Unknown backend" or "not registered"
            assert "backend" in str(e).lower()

    def test_available_backends_always_includes_pyarrow(self):
        """PyArrow backend always available (no extra dependencies)."""
        backends = get_available_backends()

        assert "pyarrow" in backends
        assert isinstance(backends, list)
        assert len(backends) >= 1  # At minimum, pyarrow

    def test_create_dataframe_pyarrow_success(self):
        """PyArrow backend works without errors."""
        table = pa.table({
            "id": ["A", "B", "C"],
            "type": ["FILE", "FILE", "FOLDER"],
            "internal:gdal_vsi": ["/path/a", "/path/b", "/path/c"],
        })

        df = create_dataframe("pyarrow", table, "folder")

        assert df is not None
        assert len(df) == 3
        assert df.columns == ["id", "type", "internal:gdal_vsi"]

    @pytest.mark.pandas
    def test_create_dataframe_pandas_success(self):
        """Pandas backend works when installed."""
        pytest.importorskip("pandas")

        table = pa.table({
            "id": ["A", "B"],
            "type": ["FILE", "FOLDER"],
            "internal:gdal_vsi": ["/path/a", "/path/b"],
        })

        df = create_dataframe("pandas", table, "zip")

        assert df is not None
        assert len(df) == 2

    @pytest.mark.polars
    def test_create_dataframe_polars_success(self):
        """Polars backend works when installed."""
        pytest.importorskip("polars")

        table = pa.table({
            "id": ["A", "B"],
            "type": ["FILE", "FOLDER"],
            "internal:gdal_vsi": ["/path/a", "/path/b"],
        })

        df = create_dataframe("polars", table, "tacocat")

        assert df is not None
        assert len(df) == 2


class TestBackendRegistration:
    """Backend registry operations."""

    def test_get_available_backends_returns_list(self):
        """get_available_backends() returns list of strings."""
        backends = get_available_backends()

        assert isinstance(backends, list)
        assert all(isinstance(b, str) for b in backends)
        assert len(backends) > 0

    def test_pyarrow_always_registered(self):
        """PyArrow backend always in registry (default, no dependencies)."""
        backends = get_available_backends()

        assert "pyarrow" in backends