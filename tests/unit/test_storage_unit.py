"""Unit tests for storage module."""

import json

import pytest

from tacoreader._constants import COLUMN_ID, PADDING_PREFIX, ZIP_MAX_GAP_SIZE
from tacoreader._exceptions import TacoFormatError
from tacoreader.storage import create_backend
from tacoreader.storage.folder import FolderBackend
from tacoreader.storage.tacocat import TacoCatBackend
from tacoreader.storage.zip import ZipBackend


@pytest.mark.parametrize(
    "format_type,expected_class",
    [
        ("zip", ZipBackend),
        ("folder", FolderBackend),
        ("tacocat", TacoCatBackend),
    ],
)
def test_create_backend_returns_correct_instance(format_type, expected_class):
    assert isinstance(create_backend(format_type), expected_class)


def test_create_backend_unknown_format_raises():
    with pytest.raises(TacoFormatError, match="Unknown format.*gibberish"):
        create_backend("gibberish")


def test_view_filter_excludes_padding_samples():
    sql = ZipBackend._build_view_filter()
    assert PADDING_PREFIX in sql
    assert COLUMN_ID in sql
    assert "NOT LIKE" in sql
    assert sql.endswith("%'")


class TestParseCollectionJson:
    @pytest.fixture
    def backend(self):
        return ZipBackend()

    def test_valid_json(self, backend):
        data = {"id": "test", "version": "1.0"}
        result = backend._parse_collection_json(json.dumps(data).encode(), "/path")
        assert result == data

    def test_malformed_json_includes_context_in_error(self, backend):
        with pytest.raises(json.JSONDecodeError) as exc_info:
            backend._parse_collection_json(b"not valid json", "/some/path.tacozip")
        assert "/some/path.tacozip" in str(exc_info.value)


class TestZipFileGrouping:
    """
    ZIP backend groups nearby files into single HTTP requests.
    Threshold: gap < 4MB AND gap < 50% of useful data.
    """

    @pytest.fixture
    def backend(self):
        return ZipBackend()

    def test_contiguous_files_grouped(self, backend):
        file1 = (0, 0, 1000)
        file2 = (1, 1000, 1000)
        assert backend._should_group(file1, file2) is True

    def test_small_gap_large_files_grouped(self, backend):
        # 2MB gap, 5MB files → 20% ratio
        file1 = (0, 0, 5 * 1024 * 1024)
        file2 = (1, 7 * 1024 * 1024, 5 * 1024 * 1024)
        assert backend._should_group(file1, file2) is True

    def test_large_gap_small_files_not_grouped(self, backend):
        # 3.9MB gap, 1KB files → ~1950% ratio, wastes bandwidth
        file1 = (0, 0, 1024)
        file2 = (1, int(3.9 * 1024 * 1024), 1024)
        assert backend._should_group(file1, file2) is False

    def test_gap_at_max_limit_not_grouped(self, backend):
        file1 = (0, 0, 10 * 1024 * 1024)
        file2 = (1, 10 * 1024 * 1024 + ZIP_MAX_GAP_SIZE, 10 * 1024 * 1024)
        assert backend._should_group(file1, file2) is False

    def test_empty_header_returns_empty_groups(self, backend):
        assert backend._group_files_by_proximity([]) == []

    def test_single_file_returns_single_group(self, backend):
        header = [(0, 1000)]
        groups = backend._group_files_by_proximity(header)
        assert len(groups) == 1
        assert groups[0] == [(0, 0, 1000)]

    def test_zero_size_files_ignored(self, backend):
        header = [(0, 1000), (1000, 0), (1000, 500)]
        groups = backend._group_files_by_proximity(header)
        files_in_groups = [f for g in groups for f in g]
        assert all(size > 0 for _, _, size in files_in_groups)

    def test_distant_files_split_into_separate_groups(self, backend):
        # Files 100MB apart
        header = [(0, 1000), (100 * 1024 * 1024, 1000)]
        groups = backend._group_files_by_proximity(header)
        assert len(groups) == 2


class TestTacoCatBasePath:
    @pytest.fixture
    def backend(self):
        return TacoCatBackend()

    @pytest.mark.parametrize(
        "input_path,expected",
        [
            ("/vsis3/bucket/data/.tacocat/", "/vsis3/bucket/data/"),
            ("/vsis3/bucket/data/.tacocat", "/vsis3/bucket/data/"),
            ("/local/path/.tacocat", "/local/path/"),
            ("/local/path/.tacocat/", "/local/path/"),
            ("/already/clean/path", "/already/clean/path/"),
        ],
    )
    def test_extract_base_path(self, backend, input_path, expected):
        assert backend._extract_base_path(input_path) == expected