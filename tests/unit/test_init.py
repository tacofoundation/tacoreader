"""Tests for tacoreader.__init__ public API."""

import pytest

import tacoreader
from tacoreader._exceptions import TacoBackendError


class TestBackendSelection:
    """Backend global state via use() and get_backend()."""

    def test_use_and_get_backend_roundtrip(self):
        tacoreader.use("pyarrow")
        assert tacoreader.get_backend() == "pyarrow"

    def test_use_invalid_backend_raises_with_helpful_message(self):
        with pytest.raises(TacoBackendError) as exc_info:
            tacoreader.use("nonexistent")

        msg = str(exc_info.value)
        assert "nonexistent" in msg
        assert "pip install" in msg

    @pytest.mark.polars
    def test_use_polars_accepted_when_installed(self):
        pytest.importorskip("polars")
        tacoreader.use("polars")
        assert tacoreader.get_backend() == "polars"

    @pytest.mark.pandas
    def test_use_pandas_accepted_when_installed(self):
        pytest.importorskip("pandas")
        tacoreader.use("pandas")
        assert tacoreader.get_backend() == "pandas"


class TestVerbose:
    """verbose() logging configuration."""

    @pytest.mark.parametrize("level", [True, False, "info", "debug"])
    def test_valid_levels_accepted(self, level):
        tacoreader.verbose(level)

    @pytest.mark.parametrize("invalid", ["warning", "error", 42, None, []])
    def test_invalid_level_raises_valueerror(self, invalid):
        with pytest.raises(ValueError) as exc_info:
            tacoreader.verbose(invalid)

        assert "Invalid verbose level" in str(exc_info.value)


class TestClearCache:
    """clear_cache() smoke test."""

    def test_clear_cache_does_not_raise(self):
        tacoreader.clear_cache()