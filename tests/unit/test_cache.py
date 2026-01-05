"""Unit tests for TacoCat remote cache management."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tacoreader._cache import (
    clear_tacocat_cache,
    get_cache_dir,
    get_cache_stats,
    get_cached_path,
    get_remote_metadata,
    get_tacocat_cache_dir,
    invalidate_cache,
    is_cache_valid,
    load_from_cache,
    read_cache_meta,
    save_to_cache,
    url_to_cache_key,
    write_cache_meta,
)
from tacoreader._constants import CACHE_DIR_NAME, CACHE_ENV_VAR, CACHE_META_FILENAME


class TestGetCacheDir:
    """Tests for get_cache_dir()."""

    def test_uses_env_var_when_set(self, tmp_path, monkeypatch):
        """Environment variable overrides default."""
        custom_path = tmp_path / "custom_cache"
        monkeypatch.setenv(CACHE_ENV_VAR, str(custom_path))
        assert get_cache_dir() == custom_path

    def test_uses_platformdirs_when_no_env(self, monkeypatch):
        """Falls back to platformdirs when env not set."""
        monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
        result = get_cache_dir()
        assert CACHE_DIR_NAME in str(result)


class TestGetTacocatCacheDir:
    """Tests for get_tacocat_cache_dir()."""

    def test_returns_tacocat_subdir(self, tmp_path, monkeypatch):
        """Returns tacocat subdirectory under cache dir."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        result = get_tacocat_cache_dir()
        assert result == tmp_path / "tacocat"


class TestUrlToCacheKey:
    """Tests for url_to_cache_key()."""

    def test_returns_16_char_hash(self):
        """Hash is exactly 16 characters."""
        key = url_to_cache_key("https://example.com/data.tacocat")
        assert len(key) == 16
        assert key.isalnum()

    def test_normalizes_trailing_slash(self):
        """Trailing slashes are stripped before hashing."""
        key1 = url_to_cache_key("https://example.com/data/")
        key2 = url_to_cache_key("https://example.com/data")
        assert key1 == key2

    def test_different_urls_different_keys(self):
        """Different URLs produce different keys."""
        key1 = url_to_cache_key("https://example.com/a")
        key2 = url_to_cache_key("https://example.com/b")
        assert key1 != key2


class TestGetCachedPath:
    """Tests for get_cached_path()."""

    def test_returns_path_with_hash(self, tmp_path, monkeypatch):
        """Returns path under tacocat cache with URL hash."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/dataset"
        result = get_cached_path(url)
        expected_hash = url_to_cache_key(url)
        assert result == tmp_path / "tacocat" / expected_hash


class TestReadCacheMeta:
    """Tests for read_cache_meta()."""

    def test_returns_none_when_not_exists(self, tmp_path, monkeypatch):
        """Returns None when meta file doesn't exist."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        result = read_cache_meta("https://nonexistent.com/data")
        assert result is None

    def test_returns_dict_when_exists(self, tmp_path, monkeypatch):
        """Returns parsed dict when meta file exists."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        cache_dir = get_cached_path(url)
        cache_dir.mkdir(parents=True)
        meta = {"url": url, "etag": "abc123", "size": 1000, "files": ["level0.parquet"]}
        (cache_dir / CACHE_META_FILENAME).write_text(json.dumps(meta))

        result = read_cache_meta(url)
        assert result == meta

    def test_returns_none_on_json_decode_error(self, tmp_path, monkeypatch):
        """Returns None when meta file is invalid JSON."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        cache_dir = get_cached_path(url)
        cache_dir.mkdir(parents=True)
        (cache_dir / CACHE_META_FILENAME).write_text("not valid json {{{")

        result = read_cache_meta(url)
        assert result is None

    def test_returns_none_on_os_error(self, tmp_path, monkeypatch):
        """Returns None on OSError during read."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        cache_dir = get_cached_path(url)
        cache_dir.mkdir(parents=True)
        meta_path = cache_dir / CACHE_META_FILENAME
        meta_path.write_text("{}")

        with patch.object(Path, "read_text", side_effect=OSError("Permission denied")):
            result = read_cache_meta(url)
        assert result is None


class TestWriteCacheMeta:
    """Tests for write_cache_meta()."""

    def test_creates_meta_file(self, tmp_path, monkeypatch):
        """Creates .meta.json with correct content."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        etag = "abc123"
        size = 5000
        files = ["level0.parquet", "COLLECTION.json"]

        write_cache_meta(url, etag, size, files)

        meta_path = get_cached_path(url) / CACHE_META_FILENAME
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["url"] == url
        assert meta["etag"] == etag
        assert meta["size"] == size
        assert meta["files"] == files
        assert "cached_at" in meta

    def test_creates_parent_dirs(self, tmp_path, monkeypatch):
        """Creates parent directories if they don't exist."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/new/data"

        write_cache_meta(url, None, None, [])

        cache_dir = get_cached_path(url)
        assert cache_dir.exists()


class TestIsCacheValid:
    """Tests for is_cache_valid()."""

    def test_returns_false_when_no_meta(self, tmp_path, monkeypatch):
        """Returns False when cache meta doesn't exist."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        result = is_cache_valid("https://example.com/data", "etag123", 1000)
        assert result is False

    def test_etag_match_returns_true(self, tmp_path, monkeypatch):
        """Returns True when ETags match."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        etag = "abc123"
        write_cache_meta(url, etag, 1000, ["file.parquet"])

        result = is_cache_valid(url, etag, None)
        assert result is True

    def test_etag_mismatch_returns_false(self, tmp_path, monkeypatch):
        """Returns False when ETags don't match."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        write_cache_meta(url, "old_etag", 1000, ["file.parquet"])

        result = is_cache_valid(url, "new_etag", None)
        assert result is False

    def test_size_fallback_match_returns_true(self, tmp_path, monkeypatch):
        """Returns True when sizes match (no ETag available)."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        size = 5000
        write_cache_meta(url, None, size, ["file.parquet"])

        result = is_cache_valid(url, None, size)
        assert result is True

    def test_size_fallback_mismatch_returns_false(self, tmp_path, monkeypatch):
        """Returns False when sizes don't match."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        write_cache_meta(url, None, 5000, ["file.parquet"])

        result = is_cache_valid(url, None, 9999)
        assert result is False

    def test_no_remote_metadata_trusts_cache(self, tmp_path, monkeypatch):
        """Returns True when no remote metadata and cache exists."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        write_cache_meta(url, "etag", 1000, ["file.parquet"])

        result = is_cache_valid(url, None, None)
        assert result is True

    def test_no_cached_etag_or_size_returns_false(self, tmp_path, monkeypatch):
        """Returns False when cache has no etag/size and remote has partial."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        write_cache_meta(url, None, None, ["file.parquet"])

        result = is_cache_valid(url, "remote_etag", None)
        assert result is False

    def test_remote_size_but_no_cached_size_returns_false(self, tmp_path, monkeypatch):
        """Returns False when remote has size but cache doesn't."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        write_cache_meta(url, None, None, ["file.parquet"])

        result = is_cache_valid(url, None, 5000)
        assert result is False


class TestGetRemoteMetadata:
    """Tests for get_remote_metadata()."""

    def test_returns_etag_and_size_on_success(self, tmp_path, monkeypatch):
        """Returns dict with etag and size on successful HEAD."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))

        mock_head = MagicMock()
        mock_head.e_tag = "etag123"
        mock_head.size = 5000

        with patch("tacoreader._remote_io._create_store") as mock_store, \
             patch("obstore.head", return_value=mock_head):
            mock_store.return_value = MagicMock()
            result = get_remote_metadata("https://example.com/.tacocat")

        assert result == {"etag": "etag123", "size": 5000}

    def test_uses_etag_fallback_attribute(self, tmp_path, monkeypatch):
        """Falls back to 'etag' attribute if 'e_tag' not present."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))

        # Simple class without e_tag, only etag
        class HeadResponse:
            etag = "fallback_etag"
            size = 3000

        with patch("tacoreader._remote_io._create_store") as mock_store, \
             patch("obstore.head", return_value=HeadResponse()):
            mock_store.return_value = MagicMock()
            result = get_remote_metadata("https://example.com/.tacocat")

        assert result["etag"] == "fallback_etag"
        assert result["size"] == 3000

    def test_returns_none_on_exception(self, tmp_path, monkeypatch):
        """Returns None when HEAD request fails."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))

        with patch("tacoreader._remote_io._create_store", side_effect=Exception("Network error")):
            result = get_remote_metadata("https://example.com/.tacocat")

        assert result is None


class TestLoadFromCache:
    """Tests for load_from_cache()."""

    def test_returns_none_when_no_meta(self, tmp_path, monkeypatch):
        """Returns None when cache meta doesn't exist."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        result = load_from_cache("https://nonexistent.com/data")
        assert result is None

    def test_returns_none_when_empty_files_list(self, tmp_path, monkeypatch):
        """Returns None when files list is empty."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        write_cache_meta(url, "etag", 1000, [])

        result = load_from_cache(url)
        assert result is None

    def test_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        """Returns None when cached file is missing."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        write_cache_meta(url, "etag", 1000, ["missing.parquet"])

        result = load_from_cache(url)
        assert result is None

    def test_returns_files_dict_on_success(self, tmp_path, monkeypatch):
        """Returns dict of filename -> bytes on success."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        cache_dir = get_cached_path(url)
        cache_dir.mkdir(parents=True)

        file1_content = b"parquet data"
        file2_content = b'{"id": "test"}'
        (cache_dir / "level0.parquet").write_bytes(file1_content)
        (cache_dir / "COLLECTION.json").write_bytes(file2_content)
        write_cache_meta(url, "etag", 1000, ["level0.parquet", "COLLECTION.json"])

        result = load_from_cache(url)

        assert result is not None
        assert result["level0.parquet"] == file1_content
        assert result["COLLECTION.json"] == file2_content


class TestSaveToCache:
    """Tests for save_to_cache()."""

    def test_saves_files_and_meta(self, tmp_path, monkeypatch):
        """Saves files and writes meta."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        files = {
            "level0.parquet": b"parquet bytes",
            "COLLECTION.json": b'{"id": "test"}',
        }

        save_to_cache(url, files, "etag123", 5000)

        cache_dir = get_cached_path(url)
        assert (cache_dir / "level0.parquet").read_bytes() == b"parquet bytes"
        assert (cache_dir / "COLLECTION.json").read_bytes() == b'{"id": "test"}'

        meta = read_cache_meta(url)
        assert meta["etag"] == "etag123"
        assert meta["size"] == 5000
        assert set(meta["files"]) == {"level0.parquet", "COLLECTION.json"}


class TestInvalidateCache:
    """Tests for invalidate_cache()."""

    def test_deletes_cache_dir(self, tmp_path, monkeypatch):
        """Deletes entire cache directory for URL."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        url = "https://example.com/data"
        save_to_cache(url, {"file.txt": b"data"}, "etag", 100)
        cache_dir = get_cached_path(url)
        assert cache_dir.exists()

        invalidate_cache(url)

        assert not cache_dir.exists()

    def test_noop_when_not_exists(self, tmp_path, monkeypatch):
        """Does nothing when cache doesn't exist."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        invalidate_cache("https://nonexistent.com/data")  # Should not raise


class TestClearTacocatCache:
    """Tests for clear_tacocat_cache()."""

    def test_clears_all_cached_datasets(self, tmp_path, monkeypatch):
        """Clears entire tacocat cache directory."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        save_to_cache("https://example.com/a", {"f.txt": b"a"}, "e1", 10)
        save_to_cache("https://example.com/b", {"f.txt": b"b"}, "e2", 20)

        tacocat_dir = get_tacocat_cache_dir()
        assert tacocat_dir.exists()

        clear_tacocat_cache()

        assert not tacocat_dir.exists()

    def test_noop_when_cache_empty(self, tmp_path, monkeypatch):
        """Does nothing when cache directory doesn't exist."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        clear_tacocat_cache()  # Should not raise


class TestGetCacheStats:
    """Tests for get_cache_stats()."""

    def test_empty_cache(self, tmp_path, monkeypatch):
        """Returns zeros when cache is empty."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        result = get_cache_stats()
        assert result["entries"] == 0
        assert result["size_mb"] == 0.0

    def test_with_cached_entries(self, tmp_path, monkeypatch):
        """Returns correct stats with cached data."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
        # Use larger files so size_mb > 0 after rounding (need > 5KB for 0.01 MB)
        save_to_cache("https://example.com/a", {"f.txt": b"a" * 100_000}, "e1", 10)
        save_to_cache("https://example.com/b", {"f.txt": b"b" * 200_000}, "e2", 20)

        result = get_cache_stats()

        assert result["entries"] == 2
        assert result["size_mb"] > 0
        assert "path" in result