"""Tests for DataFrame backend operations across PyArrow, Pandas, and Polars."""

import pytest

import tacoreader


class TestBackendOperations:
    """Common operations that work across all backends."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        """Reset to pyarrow after each test."""
        yield
        tacoreader.use("pyarrow")

    @pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
    def test_basic_properties(self, zip_flat, backend):
        """Test len, columns, shape across all backends."""
        if backend == "pandas":
            pytest.importorskip("pandas")
        elif backend == "polars":
            pytest.importorskip("polars")

        tacoreader.use(backend)
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        # All backends support these
        assert len(df) > 0
        assert len(df.columns) > 0
        assert df.shape[0] == len(df)
        assert df.shape[1] == len(df.columns)

    @pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
    def test_head_tail(self, folder_nested, backend):
        """Test head() and tail() across all backends."""
        if backend == "pandas":
            pytest.importorskip("pandas")
        elif backend == "polars":
            pytest.importorskip("polars")

        tacoreader.use(backend)
        ds = tacoreader.load(str(folder_nested))
        df = ds.data

        head = df.head(2)
        tail = df.tail(2)

        # Head/tail should return smaller dataframes
        assert len(head) <= 2
        assert len(tail) <= 2

    @pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
    def test_repr_filters_padding(self, folder_nested, backend):
        """Test __repr__ doesn't show __TACOPAD__ samples."""
        if backend == "pandas":
            pytest.importorskip("pandas")
        elif backend == "polars":
            pytest.importorskip("polars")

        tacoreader.use(backend)
        ds = tacoreader.load(str(folder_nested))
        df = ds.data

        repr_str = repr(df)

        # Should contain format and backend info
        assert "TacoDataFrame" in repr_str
        assert backend in repr_str

        # Should NOT contain padding rows (if any exist)
        assert "__TACOPAD__" not in repr_str

    @pytest.mark.parametrize("backend", ["pyarrow", "pandas", "polars"])
    def test_column_access(self, zip_flat, backend):
        """Test accessing columns by name."""
        if backend == "pandas":
            pytest.importorskip("pandas")
        elif backend == "polars":
            pytest.importorskip("polars")

        tacoreader.use(backend)
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        # All backends should support column access
        id_col = df["id"]
        assert id_col is not None

    def test_pyarrow_to_arrow_export(self, zip_flat):
        """Test PyArrow backend .to_arrow() export."""
        tacoreader.use("pyarrow")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        arrow_table = df.to_arrow()

        # Should return PyArrow Table
        import pyarrow as pa

        assert isinstance(arrow_table, pa.Table)
        assert arrow_table.num_rows == len(df)

    @pytest.mark.pandas
    def test_pandas_to_pandas_export(self, zip_flat):
        """Test Pandas backend .to_pandas() export."""
        pd = pytest.importorskip("pandas")

        tacoreader.use("pandas")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        pandas_df = df.to_pandas()

        # Should return Pandas DataFrame
        assert isinstance(pandas_df, pd.DataFrame)
        assert len(pandas_df) == len(df)

    @pytest.mark.polars
    def test_polars_to_polars_export(self, zip_flat):
        """Test Polars backend .to_polars() export."""
        pl = pytest.importorskip("polars")

        tacoreader.use("polars")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        polars_df = df.to_polars()

        # Should return Polars DataFrame
        assert isinstance(polars_df, pl.DataFrame)
        assert len(polars_df) == len(df)


class TestPandasSpecific:
    """Tests for Pandas-specific operations."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        yield
        tacoreader.use("pyarrow")

    @pytest.mark.pandas
    def test_pandas_query(self, folder_nested):
        """Test .query() method for filtering."""
        pytest.importorskip("pandas")

        tacoreader.use("pandas")
        ds = tacoreader.load(str(folder_nested))
        df = ds.data

        # Filter using query string
        filtered = df.query("type == 'FOLDER'")

        assert len(filtered) <= len(df)
        # All remaining rows should be FOLDER type
        assert all(filtered["type"] == "FOLDER")

    @pytest.mark.pandas
    def test_pandas_sort_values(self, zip_flat):
        """Test .sort_values() method."""
        pytest.importorskip("pandas")

        tacoreader.use("pandas")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        # Sort by id column
        sorted_df = df.sort_values("id")

        assert len(sorted_df) == len(df)
        # Check it's actually sorted
        ids = sorted_df["id"].tolist()
        assert ids == sorted(ids)

    @pytest.mark.pandas
    def test_pandas_assign_new_column(self, zip_flat):
        """Test .assign() for adding columns."""
        pytest.importorskip("pandas")

        tacoreader.use("pandas")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        # Add new column
        df_with_new = df.assign(new_column=42)

        assert "new_column" in df_with_new.columns
        assert all(df_with_new["new_column"] == 42)
        # Original unchanged (immutable)
        assert "new_column" not in df.columns

    @pytest.mark.pandas
    def test_pandas_protected_columns_setitem(self, zip_flat):
        """Test error when modifying protected columns via __setitem__."""
        pytest.importorskip("pandas")

        tacoreader.use("pandas")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        # Should raise error for protected columns
        with pytest.raises(ValueError, match="Cannot modify protected column"):
            df["id"] = "hacked"

        with pytest.raises(ValueError, match="Cannot modify protected column"):
            df["type"] = "HACKED"

        with pytest.raises(ValueError, match="Cannot modify protected column"):
            df["internal:gdal_vsi"] = "/fake/path"

    @pytest.mark.pandas
    def test_pandas_groupby(self, folder_nested):
        """Test .groupby() aggregation."""
        pytest.importorskip("pandas")

        tacoreader.use("pandas")
        ds = tacoreader.load(str(folder_nested))
        df = ds.data

        # Group by type and count
        grouped = df.groupby("type").size()

        # Should return Series with counts
        assert len(grouped) > 0

    @pytest.mark.pandas
    def test_pandas_loc_iloc(self, zip_flat):
        """Test .loc and .iloc indexers."""
        pytest.importorskip("pandas")

        tacoreader.use("pandas")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        # iloc - integer position
        first_row = df.iloc[0]
        assert first_row is not None

        # loc - label-based (by index)
        first_by_loc = df.loc[0]
        assert first_by_loc is not None


class TestPolarsSpecific:
    """Tests for Polars-specific operations."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        yield
        tacoreader.use("pyarrow")

    @pytest.mark.polars
    def test_polars_filter(self, folder_nested):
        """Test .filter() with expressions."""
        pl = pytest.importorskip("polars")

        tacoreader.use("polars")
        ds = tacoreader.load(str(folder_nested))
        df = ds.data

        # Filter using Polars expression
        filtered = df.filter(pl.col("type") == "FOLDER")

        assert len(filtered) <= len(df)

    @pytest.mark.polars
    def test_polars_select(self, zip_flat):
        """Test .select() for column selection."""
        pl = pytest.importorskip("polars")

        tacoreader.use("polars")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        # Select specific columns
        selected = df.select(["id", "type"])

        assert len(selected.columns) == 2
        assert "id" in selected.columns
        assert "type" in selected.columns

    @pytest.mark.polars
    def test_polars_with_columns(self, zip_flat):
        """Test .with_columns() for adding/replacing columns."""
        pl = pytest.importorskip("polars")

        tacoreader.use("polars")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        # Add new column using expression
        df_with_new = df.with_columns(pl.lit(42).alias("new_column"))

        assert "new_column" in df_with_new.columns
        # Original unchanged
        assert "new_column" not in df.columns

    @pytest.mark.polars
    def test_polars_sort(self, zip_flat):
        """Test .sort() method."""
        pytest.importorskip("polars")

        tacoreader.use("polars")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        # Sort by id
        sorted_df = df.sort("id")

        assert len(sorted_df) == len(df)

    @pytest.mark.polars
    def test_polars_limit(self, folder_nested):
        """Test .limit() for row limiting."""
        pytest.importorskip("polars")

        tacoreader.use("polars")
        ds = tacoreader.load(str(folder_nested))
        df = ds.data

        # Limit to 2 rows
        limited = df.limit(2)

        assert len(limited) == 2

    @pytest.mark.polars
    def test_polars_protected_columns_setitem(self, zip_flat):
        """Test error when modifying protected columns via __setitem__."""
        pytest.importorskip("polars")

        tacoreader.use("polars")
        ds = tacoreader.load(str(zip_flat))
        df = ds.data

        # Should raise error for protected columns
        with pytest.raises(ValueError, match="Cannot modify protected column"):
            df["id"] = "hacked"

        with pytest.raises(ValueError, match="Cannot modify protected column"):
            df["internal:gdal_vsi"] = "/fake/path"