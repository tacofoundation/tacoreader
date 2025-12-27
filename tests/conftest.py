# tests/conftest.py
"""
Pytest configuration and fixtures for tacoreader tests.

Fixtures load pre-generated test datasets from tests/fixtures/.
To regenerate fixtures, run: python tests/fixtures/regenerate.py
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
ZIP_FIXTURES = FIXTURES_DIR / "zip"
FOLDER_FIXTURES = FIXTURES_DIR / "folder"


# ---------------------------------------------------------------------------
# Pytest Configuration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: unit tests (no I/O)")
    config.addinivalue_line("markers", "integration: integration tests with fixtures")
    config.addinivalue_line("markers", "slow: slow tests (network, large data)")
    config.addinivalue_line("markers", "polars: requires polars package")
    config.addinivalue_line("markers", "pandas: requires pandas package")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on directory."""
    for item in items:
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# ---------------------------------------------------------------------------
# ZIP Fixtures (paths)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def zip_flat() -> Path:
    """ZIP flat: 5 FILE samples, 1 level."""
    path = ZIP_FIXTURES / "flat" / "flat.tacozip"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def zip_nested() -> Path:
    """ZIP nested: 3 FOLDER → 3 FILE each, 2 levels."""
    path = ZIP_FIXTURES / "nested" / "nested.tacozip"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def zip_deep_part1() -> Path:
    """ZIP deep part 1: for concat tests."""
    path = ZIP_FIXTURES / "deep" / "deep_part0001.tacozip"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def tacocat_deep() -> Path:
    """TacoCat consolidated from deep parts."""
    path = ZIP_FIXTURES / "deep" / ".tacocat"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


# ---------------------------------------------------------------------------
# FOLDER Fixtures (paths)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def folder_flat() -> Path:
    """FOLDER flat: 5 FILE samples, 1 level."""
    path = FOLDER_FIXTURES / "flat"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def folder_nested() -> Path:
    """FOLDER nested: 3 FOLDER → 3 FILE each, 2 levels."""
    path = FOLDER_FIXTURES / "nested"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def folder_deep() -> Path:
    """FOLDER deep: 4 levels, mixed hierarchy."""
    path = FOLDER_FIXTURES / "deep"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Loaded Dataset Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ds_zip_flat(zip_flat):
    """Loaded TacoDataset from flat ZIP."""
    import tacoreader
    return tacoreader.load(str(zip_flat))


@pytest.fixture
def ds_zip_nested(zip_nested):
    """Loaded TacoDataset from nested ZIP."""
    import tacoreader
    return tacoreader.load(str(zip_nested))


@pytest.fixture
def ds_folder_flat(folder_flat):
    """Loaded TacoDataset from flat FOLDER."""
    import tacoreader
    return tacoreader.load(str(folder_flat))


@pytest.fixture
def ds_folder_nested(folder_nested):
    """Loaded TacoDataset from nested FOLDER."""
    import tacoreader
    return tacoreader.load(str(folder_nested))


@pytest.fixture
def ds_tacocat(tacocat_deep):
    """Loaded TacoDataset from TacoCat."""
    import tacoreader
    return tacoreader.load(str(tacocat_deep))


# ---------------------------------------------------------------------------
# Backend Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["pyarrow", "polars", "pandas"])
def all_backends(request):
    """Parametrized fixture for testing all backends."""
    backend = request.param
    if backend == "polars":
        pytest.importorskip("polars")
    elif backend == "pandas":
        pytest.importorskip("pandas")
    return backend


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_tacoreader():
    """Reset global state after each test."""
    yield
    import tacoreader
    tacoreader.use("pyarrow")
    tacoreader.clear_cache()