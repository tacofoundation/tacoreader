"""Tests for tacoreader.load() function."""

from pathlib import Path

import pytest
import tacoreader
from tacoreader._exceptions import TacoQueryError, TacoFormatError, TacoIOError
from tacoreader.dataset import TacoDataset


class TestLoadFormats:

    def test_load_zip(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        assert isinstance(ds, TacoDataset)
        assert ds._format == "zip"

    def test_load_zip_nested_has_hierarchy(self, zip_nested):
        ds = tacoreader.load(str(zip_nested))
        assert ds.pit_schema.max_depth() >= 1
        assert ds.pit_schema.root["type"] == "FOLDER"

    def test_load_folder(self, folder_flat):
        ds = tacoreader.load(str(folder_flat))
        assert isinstance(ds, TacoDataset)
        assert ds._format == "folder"

    def test_load_folder_deep_has_multiple_levels(self, folder_deep):
        ds = tacoreader.load(str(folder_deep))
        assert ds.pit_schema.max_depth() >= 2

    def test_load_tacocat(self, tacocat_deep):
        ds = tacoreader.load(str(tacocat_deep))
        assert isinstance(ds, TacoDataset)
        assert ds._format == "tacocat"

    def test_loaded_dataset_has_data_property(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        data = ds.data
        assert len(data) > 0
        assert "id" in data.columns


class TestLoadPathTypes:
    """Test that load() accepts both str and Path objects."""

    def test_load_path_object(self, zip_flat):
        path = Path(zip_flat)
        ds = tacoreader.load(path)
        assert isinstance(ds, TacoDataset)
        assert ds._format == "zip"

    def test_load_path_object_folder(self, folder_flat):
        path = Path(folder_flat)
        ds = tacoreader.load(path)
        assert isinstance(ds, TacoDataset)
        assert ds._format == "folder"

    def test_load_list_of_paths(self, zip_deep_part1, zip_deep_part2):
        paths = [Path(zip_deep_part1), Path(zip_deep_part2)]
        ds = tacoreader.load(paths)
        assert isinstance(ds, TacoDataset)

    def test_load_mixed_str_and_path(self, zip_deep_part1, zip_deep_part2):
        paths = [str(zip_deep_part1), Path(zip_deep_part2)]
        ds = tacoreader.load(paths)
        assert isinstance(ds, TacoDataset)

    def test_load_single_path_in_list(self, zip_flat):
        ds = tacoreader.load([Path(zip_flat)])
        assert isinstance(ds, TacoDataset)

    def test_base_path_accepts_path_object(self, tacocat_deep, tmp_path):
        ds = tacoreader.load(Path(tacocat_deep), base_path=Path(tmp_path))
        assert str(tmp_path) in ds._vsi_base_path


class TestLoadBackend:

    def test_explicit_backend_overrides_global(self, zip_flat):
        tacoreader.use("pyarrow")
        ds = tacoreader.load(str(zip_flat), backend="pyarrow")
        assert ds._dataframe_backend == "pyarrow"

    def test_uses_global_backend_by_default(self, zip_flat):
        tacoreader.use("pyarrow")
        ds = tacoreader.load(str(zip_flat))
        assert ds._dataframe_backend == "pyarrow"

    def test_all_backends_load_successfully(self, zip_flat, all_backends):
        ds = tacoreader.load(str(zip_flat), backend=all_backends)
        assert ds._dataframe_backend == all_backends
        assert len(ds.data) > 0


class TestLoadList:

    def test_empty_list_raises(self):
        with pytest.raises(TacoQueryError, match="empty list"):
            tacoreader.load([])

    def test_single_element_list_same_as_direct(self, zip_flat):
        ds_direct = tacoreader.load(str(zip_flat))
        ds_list = tacoreader.load([str(zip_flat)])
        assert ds_direct.pit_schema.root["n"] == ds_list.pit_schema.root["n"]

    def test_multiple_paths_concatenates(self, zip_deep_part1, zip_deep_part2):
        ds1 = tacoreader.load(str(zip_deep_part1))
        ds2 = tacoreader.load(str(zip_deep_part2))
        ds_concat = tacoreader.load([str(zip_deep_part1), str(zip_deep_part2)])

        expected_n = ds1.pit_schema.root["n"] + ds2.pit_schema.root["n"]
        assert ds_concat.pit_schema.root["n"] == expected_n


class TestLoadTacoCatBasePath:

    def test_base_path_changes_root_path(self, tacocat_deep, tmp_path):
        # base_path permite que el tacocat apunte a ZIPs en otra ubicación
        ds_default = tacoreader.load(str(tacocat_deep))
        ds_custom = tacoreader.load(str(tacocat_deep), base_path=str(tmp_path))
        
        assert ds_custom._vsi_base_path != ds_default._vsi_base_path
        assert str(tmp_path) in ds_custom._vsi_base_path


class TestLoadLegacy:

    def test_taco_extension_raises(self):
        with pytest.raises(TacoFormatError, match="Legacy"):
            tacoreader.load("dataset.taco")

    def test_tortilla_extension_raises(self):
        with pytest.raises(TacoFormatError, match="Legacy"):
            tacoreader.load("dataset.tortilla")

    def test_tacofoundation_prefix_raises(self):
        with pytest.raises(TacoFormatError, match="Legacy"):
            tacoreader.load("tacofoundation:some/path")


class TestLoadErrors:

    def test_nonexistent_path_raises(self):
        with pytest.raises((TacoIOError, TacoFormatError, FileNotFoundError)):
            tacoreader.load("/nonexistent/path/data.tacozip")

    def test_nonexistent_folder_raises(self):
        with pytest.raises((TacoIOError, TacoFormatError, FileNotFoundError)):
            tacoreader.load("/nonexistent/folder/dataset/")