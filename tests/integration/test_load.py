# tests/integration/test_load.py
"""Integration tests for loading TACO datasets."""

import pytest

import tacoreader
from tacoreader.dataset import TacoDataset


class TestLoadZip:
    """Test loading ZIP format datasets."""

    def test_load_flat(self, zip_flat):
        """Load flat ZIP returns TacoDataset."""
        ds = tacoreader.load(str(zip_flat))
        assert isinstance(ds, TacoDataset)
        assert ds._format == "zip"

    def test_load_nested(self, zip_nested):
        """Load nested ZIP returns TacoDataset."""
        ds = tacoreader.load(str(zip_nested))
        assert isinstance(ds, TacoDataset)
        assert ds.pit_schema.max_depth() >= 1

    def test_data_returns_dataframe(self, ds_zip_flat):
        """Accessing .data returns TacoDataFrame."""
        tdf = ds_zip_flat.data
        assert len(tdf) > 0
        assert "id" in tdf.columns
        assert "type" in tdf.columns

    def test_has_gdal_vsi_column(self, ds_zip_flat):
        """ZIP data has internal:gdal_vsi for GDAL access."""
        tdf = ds_zip_flat.data
        assert "internal:gdal_vsi" in tdf.columns


class TestLoadFolder:
    """Test loading FOLDER format datasets."""

    def test_load_flat(self, folder_flat):
        """Load flat FOLDER returns TacoDataset."""
        ds = tacoreader.load(str(folder_flat))
        assert isinstance(ds, TacoDataset)
        assert ds._format == "folder"

    def test_load_nested(self, folder_nested):
        """Load nested FOLDER returns TacoDataset."""
        ds = tacoreader.load(str(folder_nested))
        assert isinstance(ds, TacoDataset)
        assert ds.pit_schema.max_depth() >= 1

    def test_data_returns_dataframe(self, ds_folder_flat):
        """Accessing .data returns TacoDataFrame."""
        tdf = ds_folder_flat.data
        assert len(tdf) > 0
        assert "id" in tdf.columns


class TestLoadTacoCat:
    """Test loading TacoCat format datasets."""

    def test_load_tacocat(self, tacocat_deep):
        """Load TacoCat returns TacoDataset."""
        ds = tacoreader.load(str(tacocat_deep))
        assert isinstance(ds, TacoDataset)
        assert ds._format == "tacocat"

    def test_has_source_file(self, ds_tacocat):
        """TacoCat data has internal:source_file column."""
        tdf = ds_tacocat.data
        assert "internal:source_file" in tdf.columns


class TestLoadMetadata:
    """Test dataset metadata access."""

    def test_has_required_metadata(self, ds_zip_flat):
        """Dataset has required metadata fields."""
        assert ds_zip_flat.id is not None
        assert ds_zip_flat.version is not None
        assert ds_zip_flat.pit_schema is not None

    def test_extent_is_dict(self, ds_zip_flat):
        """Extent is a dictionary."""
        assert isinstance(ds_zip_flat.extent, dict)

    def test_collection_accessible(self, ds_zip_flat):
        """Full collection dict is accessible."""
        assert isinstance(ds_zip_flat.collection, dict)
        assert "taco:pit_schema" in ds_zip_flat.collection


class TestContextManager:
    """Test dataset context manager."""

    def test_closes_connection(self, zip_flat):
        """Context manager closes DuckDB connection."""
        with tacoreader.load(str(zip_flat)) as ds:
            _ = ds.data
        assert ds._duckdb is None

    def test_can_use_data_inside_context(self, zip_flat):
        """Can access data inside context."""
        with tacoreader.load(str(zip_flat)) as ds:
            tdf = ds.data
            assert len(tdf) > 0