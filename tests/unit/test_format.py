"""Tests for format detection and path resolution."""

import pytest

from tacoreader._format import (
    detect_and_resolve_format,
    detect_format,
    is_local,
    is_remote,
)


class TestDetectFormat:
    """Pure string logic - no I/O."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("data.tacozip", "zip"),
            ("data.zip", "zip"),
            ("/path/to/data.tacozip", "zip"),
            ("s3://bucket/data.zip", "zip"),
            (".tacocat", "tacocat"),
            (".tacocat/", "tacocat"),
            ("path/to/.tacocat", "tacocat"),
            ("/vsis3/bucket/.tacocat/", "tacocat"),
            ("folder", "folder"),
            ("folder/", "folder"),
            ("/absolute/path", "folder"),
            ("s3://bucket/dataset/", "folder"),
        ],
    )
    def test_detect_format(self, path, expected):
        assert detect_format(path) == expected

    def test_trailing_slash_normalized(self):
        """Trailing slashes don't affect detection."""
        assert detect_format("data/.tacocat") == detect_format("data/.tacocat/")
        assert detect_format("folder") == detect_format("folder/")

    @pytest.mark.parametrize(
        "path",
        [
            "s3://bucket/.tacocat",
            "gs://bucket/data.tacozip",
            "azure://container/data.zip",
            "/vsis3/bucket/.tacocat",
        ],
    )
    def test_remote_paths_work(self, path):
        """Remote protocols don't break detection."""
        result = detect_format(path)
        assert result in ("zip", "folder", "tacocat")


class TestDetectAndResolveFormat:
    """Integration tests with real fixtures and tmp_path."""

    def test_explicit_zip_unchanged(self, zip_flat):
        """ZIP format returned as-is with no modification."""
        fmt, resolved = detect_and_resolve_format(str(zip_flat))
        assert fmt == "zip"
        assert resolved == str(zip_flat)

    def test_explicit_tacocat_unchanged(self, tacocat_deep):
        """TacoCat format returned as-is with no modification."""
        fmt, resolved = detect_and_resolve_format(str(tacocat_deep))
        assert fmt == "tacocat"
        assert resolved == str(tacocat_deep)

    def test_folder_with_collection_unchanged(self, folder_flat):
        """FOLDER with COLLECTION.json in root returned unchanged."""
        fmt, resolved = detect_and_resolve_format(str(folder_flat))
        assert fmt == "folder"
        assert resolved == str(folder_flat)

    def test_tacocat_fallback_local(self, tmp_path):
        """Fallback to .tacocat/ subfolder when root missing COLLECTION.json."""
        # Create .tacocat/ with COLLECTION.json
        tacocat_dir = tmp_path / ".tacocat"
        tacocat_dir.mkdir()
        (tacocat_dir / "COLLECTION.json").write_text('{"id": "test"}')

        # Don't create COLLECTION.json in root
        # Ensure _file_exists() will fail for root but succeed for .tacocat/

        fmt, resolved = detect_and_resolve_format(str(tmp_path))

        assert fmt == "tacocat"
        assert ".tacocat" in resolved
        assert str(tacocat_dir) == resolved

    def test_folder_missing_collection_returns_folder(self, tmp_path):
        """Missing COLLECTION.json returns folder format (backend will fail later)."""
        # Empty directory, no COLLECTION.json anywhere
        fmt, resolved = detect_and_resolve_format(str(tmp_path))

        assert fmt == "folder"
        assert resolved == str(tmp_path)


class TestIsRemote:
    """Protocol detection for remote vs local paths."""

    @pytest.mark.parametrize(
        "path",
        [
            "s3://bucket/data.tacozip",
            "gs://bucket/folder/",
            "az://container/data/",
            "azure://container/data/",
            "https://example.com/data.zip",
            "http://example.com/data/",
            "/vsis3/bucket/data.zip",
            "/vsigs/bucket/folder/",
            "/vsiaz/container/data/",
            "/vsicurl/https://example.com/data.zip",
        ],
    )
    def test_is_remote(self, path):
        assert is_remote(path) is True
        assert is_local(path) is False

    @pytest.mark.parametrize(
        "path",
        [
            "/absolute/path/data.tacozip",
            "relative/path/folder",
            "../parent/data",
            "C:\\Windows\\path\\data.zip",  # Windows
            "~/home/user/data",
        ],
    )
    def test_is_local(self, path):
        assert is_local(path) is True
        assert is_remote(path) is False