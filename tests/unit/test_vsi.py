"""Tests for tacoreader._vsi path utilities."""

from pathlib import Path

import pytest

from tacoreader._exceptions import TacoFormatError
from tacoreader._vsi import (
    is_vsi_path,
    parse_vsi_subfile,
    strip_vsi_prefix,
    to_vsi_root,
)


class TestToVsiRoot:
    """to_vsi_root() protocol conversion."""

    @pytest.mark.parametrize(
        "input_path,expected_prefix",
        [
            ("s3://bucket/key", "/vsis3/bucket/key"),
            ("gs://bucket/key", "/vsigs/bucket/key"),
            ("az://container/blob", "/vsiaz/container/blob"),
            ("azure://container/blob", "/vsiaz/container/blob"),  # alias
        ],
    )
    def test_cloud_protocols(self, input_path, expected_prefix):
        assert to_vsi_root(input_path) == expected_prefix

    @pytest.mark.parametrize(
        "input_path",
        [
            "https://example.com/file.zip",
            "http://example.com/file.zip",
        ],
    )
    def test_http_protocols_wrap_full_url(self, input_path):
        result = to_vsi_root(input_path)
        assert result.startswith("/vsicurl/")
        assert input_path in result

    def test_local_relative_path_becomes_absolute(self, tmp_path):
        test_file = tmp_path / "data.taco"
        test_file.touch()

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = to_vsi_root("data.taco")
            assert Path(result).is_absolute()
            assert result == str(test_file)
        finally:
            os.chdir(original_cwd)

    def test_local_absolute_path_unchanged(self):
        abs_path = "/data/datasets/file.tacozip"
        result = to_vsi_root(abs_path)
        assert Path(result).is_absolute()


class TestIsVsiPath:
    """is_vsi_path() detection."""

    @pytest.mark.parametrize(
        "vsi_path",
        [
            "/vsis3/bucket/key",
            "/vsigs/bucket/key",
            "/vsiaz/container/blob",
            "/vsicurl/https://example.com",
            "/vsizip/archive.zip/file",
            "/vsisubfile/0_100,/path/to/file",
        ],
    )
    def test_recognizes_vsi_prefixes(self, vsi_path):
        assert is_vsi_path(vsi_path) is True

    @pytest.mark.parametrize(
        "non_vsi_path",
        [
            "s3://bucket/key",
            "/local/path/file",
            "relative/path",
            "",
            "vsis3/missing/slash",
        ],
    )
    def test_rejects_non_vsi_paths(self, non_vsi_path):
        assert is_vsi_path(non_vsi_path) is False


class TestStripVsiPrefix:
    """strip_vsi_prefix() inverse conversion."""

    @pytest.mark.parametrize(
        "vsi_path,expected",
        [
            ("/vsis3/bucket/key", "s3://bucket/key"),
            ("/vsigs/bucket/key", "gs://bucket/key"),
            ("/vsiaz/container/blob", "az://container/blob"),
        ],
    )
    def test_cloud_vsi_to_standard(self, vsi_path, expected):
        assert strip_vsi_prefix(vsi_path) == expected

    def test_vsicurl_unwraps_full_url(self):
        vsi = "/vsicurl/https://example.com/file.zip"
        assert strip_vsi_prefix(vsi) == "https://example.com/file.zip"

    def test_non_vsi_path_unchanged(self):
        path = "/local/path/file"
        assert strip_vsi_prefix(path) == path


class TestParseVsiSubfile:
    """parse_vsi_subfile() extraction."""

    def test_parses_valid_subfile_path(self):
        vsi = "/vsisubfile/1024_2048,/vsis3/bucket/file.zip"
        root, offset, size = parse_vsi_subfile(vsi)

        assert root == "/vsis3/bucket/file.zip"
        assert offset == 1024
        assert size == 2048

    def test_handles_large_offsets(self):
        large_offset = 5_000_000_000
        large_size = 100_000_000
        vsi = f"/vsisubfile/{large_offset}_{large_size},/path/huge.zip"

        root, offset, size = parse_vsi_subfile(vsi)
        assert offset == large_offset
        assert size == large_size

    def test_handles_local_path_with_commas_in_root(self):
        vsi = "/vsisubfile/100_200,/path/with,comma/file.zip"
        root, offset, size = parse_vsi_subfile(vsi)

        assert root == "/path/with,comma/file.zip"
        assert offset == 100
        assert size == 200

    # --- Error cases ---

    def test_missing_vsisubfile_prefix_raises(self):
        with pytest.raises(TacoFormatError) as exc_info:
            parse_vsi_subfile("/vsis3/bucket/file")

        assert "must start with '/vsisubfile/'" in str(exc_info.value)

    def test_missing_comma_separator_raises(self):
        with pytest.raises(TacoFormatError) as exc_info:
            parse_vsi_subfile("/vsisubfile/1024_2048")

        assert "missing comma separator" in str(exc_info.value)

    def test_missing_underscore_raises(self):
        with pytest.raises(TacoFormatError) as exc_info:
            parse_vsi_subfile("/vsisubfile/10242048,/path/file")

        assert "missing underscore" in str(exc_info.value)

    def test_non_integer_offset_raises(self):
        with pytest.raises(TacoFormatError) as exc_info:
            parse_vsi_subfile("/vsisubfile/abc_2048,/path/file")

        assert "not integers" in str(exc_info.value)

    def test_non_integer_size_raises(self):
        with pytest.raises(TacoFormatError) as exc_info:
            parse_vsi_subfile("/vsisubfile/1024_xyz,/path/file")

        assert "not integers" in str(exc_info.value)