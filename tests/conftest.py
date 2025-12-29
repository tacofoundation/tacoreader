"""Pytest fixtures for tacoreader tests."""

from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
ZIP_FIXTURES = FIXTURES_DIR / "zip"
FOLDER_FIXTURES = FIXTURES_DIR / "folder"


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: unit tests (no I/O)")
    config.addinivalue_line("markers", "integration: integration tests with fixtures")
    config.addinivalue_line("markers", "slow: slow tests (network, large data)")
    config.addinivalue_line("markers", "polars: requires polars package")
    config.addinivalue_line("markers", "pandas: requires pandas package")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# ZIP fixtures

@pytest.fixture(scope="session")
def zip_flat() -> Path:
    """5 FILE samples, 1 level."""
    path = ZIP_FIXTURES / "flat" / "flat.tacozip"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def zip_nested() -> Path:
    """3 FOLDER -> 3 FILE each, 2 levels."""
    path = ZIP_FIXTURES / "nested" / "nested.tacozip"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def zip_deep_part1() -> Path:
    """Part 1 of split dataset for concat tests."""
    path = ZIP_FIXTURES / "deep" / "deep_part0001.tacozip"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def zip_deep_part2() -> Path:
    """Part 2 of split dataset for concat tests."""
    path = ZIP_FIXTURES / "deep" / "deep_part0002.tacozip"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def tacocat_deep() -> Path:
    """TacoCat consolidating deep_part0001 + deep_part0002."""
    path = ZIP_FIXTURES / "deep" / ".tacocat"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


# FOLDER fixtures

@pytest.fixture(scope="session")
def folder_flat() -> Path:
    """5 FILE samples, 1 level."""
    path = FOLDER_FIXTURES / "flat"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def folder_nested() -> Path:
    """3 FOLDER -> 3 FILE each, 2 levels."""
    path = FOLDER_FIXTURES / "nested"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture(scope="session")
def folder_deep() -> Path:
    """4 levels, mixed hierarchy."""
    path = FOLDER_FIXTURES / "deep"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return path


# Loaded datasets (function scope for isolation)

@pytest.fixture
def ds_zip_flat(zip_flat):
    import tacoreader
    return tacoreader.load(str(zip_flat))


@pytest.fixture
def ds_zip_nested(zip_nested):
    import tacoreader
    return tacoreader.load(str(zip_nested))


@pytest.fixture
def ds_folder_flat(folder_flat):
    import tacoreader
    return tacoreader.load(str(folder_flat))


@pytest.fixture
def ds_folder_nested(folder_nested):
    import tacoreader
    return tacoreader.load(str(folder_nested))


@pytest.fixture
def ds_folder_deep(folder_deep):
    import tacoreader
    return tacoreader.load(str(folder_deep))


@pytest.fixture
def ds_tacocat(tacocat_deep):
    import tacoreader
    return tacoreader.load(str(tacocat_deep))


# Backend parametrization

@pytest.fixture(params=["pyarrow", "polars", "pandas"])
def all_backends(request):
    backend = request.param
    if backend == "polars":
        pytest.importorskip("polars")
    elif backend == "pandas":
        pytest.importorskip("pandas")
    return backend


@pytest.fixture(autouse=True)
def reset_tacoreader():
    """Reset tacoreader state after each test."""
    yield
    import tacoreader
    tacoreader.use("pyarrow")
    tacoreader.clear_cache()