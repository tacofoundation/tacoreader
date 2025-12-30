"""Tests for tacoreader._logging module."""

import logging

import pytest

from tacoreader._logging import (
    disable_logging,
    enable_debug_logging,
    get_logger,
    setup_basic_logging,
)


class TestGetLogger:
    """get_logger() namespace handling."""

    def test_already_namespaced_unchanged(self):
        logger = get_logger("tacoreader.storage.zip")
        assert logger.name == "tacoreader.storage.zip"

    def test_bare_name_gets_prefixed(self):
        logger = get_logger("mymodule")
        assert logger.name == "tacoreader.mymodule"

    def test_dunder_main_becomes_root(self):
        logger = get_logger("__main__")
        assert logger.name == "tacoreader"


class TestSetupBasicLogging:
    """setup_basic_logging() configuration."""

    @pytest.fixture(autouse=True)
    def cleanup_handlers(self):
        """Remove handlers added during tests."""
        logger = logging.getLogger("tacoreader")
        original_handlers = logger.handlers[:]
        original_level = logger.level
        yield
        logger.handlers = original_handlers
        logger.level = original_level

    def test_sets_requested_level(self):
        setup_basic_logging(level=logging.WARNING)
        logger = logging.getLogger("tacoreader")
        assert logger.level == logging.WARNING

    def test_does_not_duplicate_handlers_on_repeated_calls(self):
        setup_basic_logging()
        setup_basic_logging()
        setup_basic_logging()

        logger = logging.getLogger("tacoreader")
        assert len(logger.handlers) == 1

    def test_disables_propagation(self):
        setup_basic_logging()
        logger = logging.getLogger("tacoreader")
        assert logger.propagate is False


class TestConvenienceFunctions:
    """enable_debug_logging() and disable_logging()."""

    @pytest.fixture(autouse=True)
    def cleanup_logger(self):
        logger = logging.getLogger("tacoreader")
        original_level = logger.level
        yield
        logger.level = original_level

    def test_enable_debug_sets_debug_level(self):
        enable_debug_logging()
        logger = logging.getLogger("tacoreader")
        assert logger.level == logging.DEBUG

    def test_disable_logging_silences_completely(self):
        disable_logging()
        logger = logging.getLogger("tacoreader")
        assert logger.level > logging.CRITICAL