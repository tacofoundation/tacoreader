"""Integration tests for TacoDataFrame base class navigation."""

import pytest

import tacoreader


class TestZipNavigation:
    """Navigation tests for ZIP format."""

    def test_zip_flat_navigation(self, zip_flat):
        """zip_flat: 1 level, 5 FILEs."""
        ds = tacoreader.load(str(zip_flat))

        assert len(ds.data) == 5

        # All are FILEs, read returns VSI path
        for i in range(len(ds.data)):
            path = ds.data.read(i)
            assert isinstance(path, str)
            assert "/vsisubfile/" in path

    def test_zip_nested_navigation(self, zip_nested):
        """zip_nested: 2 levels, 3 FOLDERs × 3 FILEs."""
        ds = tacoreader.load(str(zip_nested))

        # Level 0: 3 regions (FOLDER)
        assert len(ds.data) == 3

        for region_id in ["americas", "asia", "europe"]:
            region = ds.data.read(region_id)

            # Level 1: 3 FILEs per region
            assert len(region) == 3

            for j in range(len(region)):
                path = region.read(j)
                assert isinstance(path, str)
                assert "/vsisubfile/" in path

    def test_tacocat_deep_navigation(self, tacocat_deep):
        """tacocat_deep: 4 levels, consolidated from 2 parts."""
        ds = tacoreader.load(str(tacocat_deep))

        # Level 0: tiles
        assert len(ds.data) > 0

        # Navigate to first tile
        tile = ds.data.read(0)
        assert len(tile) > 0

        # Navigate to first sensor
        sensor = tile.read(0)
        assert len(sensor) > 0

        # Navigate to first band (FILE)
        band_path = sensor.read(0)
        assert isinstance(band_path, str)
        assert "/vsisubfile/" in band_path


class TestFolderNavigation:
    """Navigation tests for FOLDER format."""

    def test_folder_flat_navigation(self, folder_flat):
        """folder_flat: 1 level, 5 FILEs."""
        ds = tacoreader.load(str(folder_flat))

        assert len(ds.data) == 5

        # All are FILEs, read returns direct path
        for i in range(len(ds.data)):
            path = ds.data.read(i)
            assert isinstance(path, str)
            assert "/vsisubfile/" not in path  # FOLDER uses direct paths

    def test_folder_nested_navigation(self, folder_nested):
        """folder_nested: 2 levels, 3 FOLDERs × 3 FILEs."""
        ds = tacoreader.load(str(folder_nested))

        # Level 0: 3 regions (FOLDER)
        assert len(ds.data) == 3

        for region_id in ["americas", "asia", "europe"]:
            region = ds.data.read(region_id)

            # Level 1: 3 FILEs per region
            assert len(region) == 3

            for j in range(len(region)):
                path = region.read(j)
                assert isinstance(path, str)

    def test_folder_deep_navigation(self, folder_deep):
        """folder_deep: 4 levels, tiles × sensors × bands."""
        ds = tacoreader.load(str(folder_deep))

        # Level 0: tiles
        assert len(ds.data) > 0

        # Navigate all tiles
        for i in range(len(ds.data)):
            tile = ds.data.read(i)
            assert len(tile) > 0

            # Navigate all sensors
            for j in range(len(tile)):
                sensor = tile.read(j)
                assert len(sensor) > 0

                # Navigate all bands (FILE)
                for k in range(len(sensor)):
                    band_path = sensor.read(k)
                    assert isinstance(band_path, str)


class TestNavigationByIdAndIndex:
    """Test both navigation methods."""

    def test_navigation_by_id(self, zip_nested):
        """Navigate using string IDs."""
        ds = tacoreader.load(str(zip_nested))

        region = ds.data.read("europe")
        assert len(region) == 3

    def test_navigation_by_index(self, zip_nested):
        """Navigate using integer indices."""
        ds = tacoreader.load(str(zip_nested))

        region = ds.data.read(0)
        assert len(region) == 3

        file_path = region.read(0)
        assert isinstance(file_path, str)