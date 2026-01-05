"""Integration tests for storage backends."""

import pytest

import tacoreader
from tacoreader._constants import PADDING_PREFIX
from tacoreader._exceptions import TacoFormatError


class TestZipBackend:
    def test_load_flat_dataset(self, zip_flat):
        ds = tacoreader.load(str(zip_flat))
        assert ds.id is not None
        assert len(ds.data) == 5

    def test_read_collection_standalone(self, zip_flat):
        from tacoreader.storage.zip import ZipBackend
        backend = ZipBackend()
        collection = backend.read_collection(str(zip_flat))
        assert "id" in collection
        assert "taco:pit_schema" in collection

    def test_load_nested_and_navigate(self, zip_nested):
        ds = tacoreader.load(str(zip_nested))
        tdf = ds.data
        assert len(tdf) == 3

        child = tdf.read(0)
        assert len(child) == 3
        # Leaf node returns VSI path
        vsi_path = child.read(0)
        assert isinstance(vsi_path, str)
        assert "/vsisubfile/" in vsi_path

    def test_padding_filtered_from_views(self, zip_nested):
        ds = tacoreader.load(str(zip_nested))
        ids = [row["id"] for row in ds.data.head(100).to_pylist()]
        assert not any(id.startswith(PADDING_PREFIX) for id in ids)


class TestFolderBackend:
    def test_load_flat_dataset(self, folder_flat):
        ds = tacoreader.load(str(folder_flat))
        assert ds.id is not None
        assert len(ds.data) == 5

    def test_load_nested_and_navigate(self, folder_nested):
        ds = tacoreader.load(str(folder_nested))
        tdf = ds.data
        assert len(tdf) == 3

        child = tdf.read(0)
        assert len(child) == 3
        vsi_path = child.read(0)
        assert isinstance(vsi_path, str)
        assert "/vsisubfile/" not in vsi_path  # folder uses direct paths


class TestTacoCatBackend:
    def test_load_consolidated_dataset(self, tacocat_deep):
        ds = tacoreader.load(str(tacocat_deep))
        assert ds.id is not None
        assert ds._format == "tacocat"

    def test_vsi_paths_point_to_source_zips(self, tacocat_deep):
        ds = tacoreader.load(str(tacocat_deep))
        tdf = ds.data
        vsi_col = tdf["internal:gdal_vsi"]
        
        # All paths should reference .tacozip files
        for path in vsi_col.to_pylist():
            assert ".tacozip" in path
            assert "/vsisubfile/" in path

    def test_source_file_column_present(self, tacocat_deep):
        ds = tacoreader.load(str(tacocat_deep))
        # Use .data_raw since .data filters internal:* columns
        assert "internal:source_file" in ds.data_raw.columns


class TestBackendConsistency:
    """Same dataset loaded via different formats should have equivalent data."""

    def test_flat_zip_and_folder_have_same_sample_count(self, zip_flat, folder_flat):
        ds_zip = tacoreader.load(str(zip_flat))
        ds_folder = tacoreader.load(str(folder_flat))
        assert len(ds_zip.data) == len(ds_folder.data)

    def test_nested_zip_and_folder_have_same_ids(self, zip_nested, folder_nested):
        ds_zip = tacoreader.load(str(zip_nested))
        ds_folder = tacoreader.load(str(folder_nested))
        
        ids_zip = set(ds_zip.data["id"].to_pylist())
        ids_folder = set(ds_folder.data["id"].to_pylist())
        assert ids_zip == ids_folder


class TestErrorPaths:
    def test_folder_missing_collection_json(self, tmp_path):
        (tmp_path / "METADATA").mkdir()
        with pytest.raises(TacoFormatError, match="COLLECTION.json not found"):
            tacoreader.load(str(tmp_path))

    def test_folder_missing_metadata_dir(self, tmp_path):
        (tmp_path / "COLLECTION.json").write_text('{"id": "test", "taco:pit_schema": {"root": {"n": 1, "type": "FILE"}, "hierarchy": {}}}')
        with pytest.raises(TacoFormatError, match="METADATA directory not found"):
            tacoreader.load(str(tmp_path))