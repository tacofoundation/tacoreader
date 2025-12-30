"""Integration tests for backend-specific functionality (polars, pandas)."""

import pytest

import tacoreader


class TestPolarsBackend:
    """Tests for Polars-specific methods."""

    @pytest.fixture(autouse=True)
    def setup_polars(self):
        """Setup polars backend for all tests in this class."""
        pytest.importorskip("polars")
        tacoreader.use("polars")
        yield
        tacoreader.use("pyarrow")

    def test_setitem_protected_column_raises(self, zip_flat):
        """Cannot modify protected columns."""
        ds = tacoreader.load(str(zip_flat))

        with pytest.raises(ValueError, match="Cannot modify protected column"):
            ds.data["id"] = ["X"] * len(ds.data)

        with pytest.raises(ValueError, match="Cannot modify protected column"):
            ds.data["type"] = ["FOLDER"] * len(ds.data)

        with pytest.raises(ValueError, match="Cannot modify protected column"):
            ds.data["internal:gdal_vsi"] = ["/a"] * len(ds.data)

    def test_setitem_regular_column(self, zip_flat):
        """Can modify non-protected columns."""
        ds = tacoreader.load(str(zip_flat))

        # Should not raise
        ds.data["cloud_cover"] = [99.0] * len(ds.data)

    def test_group_by_returns_native(self, folder_nested):
        """group_by returns Polars GroupBy, not TacoDataFrame."""
        import polars as pl

        ds = tacoreader.load(str(folder_nested))

        result = ds.data.group_by("type")

        assert isinstance(result, pl.dataframe.group_by.GroupBy)

    def test_filter_returns_taco(self, zip_flat):
        """filter() returns TacoDataFramePolars."""
        import polars as pl
        from tacoreader.dataframe.polars import TacoDataFramePolars

        ds = tacoreader.load(str(zip_flat))

        result = ds.data.filter(pl.col("type") == "FILE")

        assert isinstance(result, TacoDataFramePolars)

    def test_select_returns_taco(self, zip_flat):
        """select() returns TacoDataFramePolars."""
        from tacoreader.dataframe.polars import TacoDataFramePolars

        ds = tacoreader.load(str(zip_flat))

        result = ds.data.select("id", "type")

        assert isinstance(result, TacoDataFramePolars)

    def test_sort_returns_taco(self, zip_flat):
        """sort() returns TacoDataFramePolars."""
        from tacoreader.dataframe.polars import TacoDataFramePolars

        ds = tacoreader.load(str(zip_flat))

        result = ds.data.sort("id")

        assert isinstance(result, TacoDataFramePolars)


class TestPandasBackend:
    """Tests for Pandas-specific methods."""

    @pytest.fixture(autouse=True)
    def setup_pandas(self):
        """Setup pandas backend for all tests in this class."""
        pytest.importorskip("pandas")
        tacoreader.use("pandas")
        yield
        tacoreader.use("pyarrow")

    def test_setitem_protected_column_raises(self, zip_flat):
        """Cannot modify protected columns."""
        ds = tacoreader.load(str(zip_flat))

        with pytest.raises(ValueError, match="Cannot modify protected column"):
            ds.data["id"] = ["X"] * len(ds.data)

        with pytest.raises(ValueError, match="Cannot modify protected column"):
            ds.data["type"] = ["FOLDER"] * len(ds.data)

        with pytest.raises(ValueError, match="Cannot modify protected column"):
            ds.data["internal:gdal_vsi"] = ["/a"] * len(ds.data)

    def test_setitem_regular_column(self, zip_flat):
        """Can modify non-protected columns."""
        ds = tacoreader.load(str(zip_flat))

        # Should not raise
        ds.data["cloud_cover"] = [99.0] * len(ds.data)

    def test_groupby_returns_native(self, folder_nested):
        """groupby returns Pandas GroupBy, not TacoDataFrame."""
        import pandas as pd

        ds = tacoreader.load(str(folder_nested))

        result = ds.data.groupby("type")

        assert isinstance(result, pd.core.groupby.DataFrameGroupBy)

    def test_query_returns_taco(self, zip_flat):
        """query() returns TacoDataFramePandas."""
        from tacoreader.dataframe.pandas import TacoDataFramePandas

        ds = tacoreader.load(str(zip_flat))

        result = ds.data.query("type == 'FILE'")

        assert isinstance(result, TacoDataFramePandas)

    def test_sort_values_returns_taco(self, zip_flat):
        """sort_values() returns TacoDataFramePandas."""
        from tacoreader.dataframe.pandas import TacoDataFramePandas

        ds = tacoreader.load(str(zip_flat))

        result = ds.data.sort_values("id")

        assert isinstance(result, TacoDataFramePandas)

    def test_assign_protected_raises(self, zip_flat):
        """assign() with protected column raises."""
        ds = tacoreader.load(str(zip_flat))

        with pytest.raises(ValueError, match="Cannot modify protected column"):
            ds.data.assign(id=["X"] * len(ds.data))

    def test_assign_regular_returns_taco(self, zip_flat):
        """assign() with regular column returns TacoDataFramePandas."""
        from tacoreader.dataframe.pandas import TacoDataFramePandas

        ds = tacoreader.load(str(zip_flat))

        result = ds.data.assign(new_col=[1] * len(ds.data))

        assert isinstance(result, TacoDataFramePandas)