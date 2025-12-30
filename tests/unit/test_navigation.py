"""Unit tests for TacoDataFrame navigation (_get_position, read)."""

import pytest
import pyarrow as pa

from tacoreader._exceptions import TacoNavigationError


def get_dataframe_class(backend: str):
    """Get the TacoDataFrame class for a backend."""
    if backend == "pyarrow":
        from tacoreader.dataframe.pyarrow import TacoDataFrameArrow
        return TacoDataFrameArrow
    elif backend == "polars":
        from tacoreader.dataframe.polars import TacoDataFramePolars
        return TacoDataFramePolars
    elif backend == "pandas":
        from tacoreader.dataframe.pandas import TacoDataFramePandas
        return TacoDataFramePandas


@pytest.fixture(params=["pyarrow", "polars", "pandas"])
def backend_class(request):
    """Parametrize tests across all backends."""
    be = request.param
    if be == "polars":
        pytest.importorskip("polars")
    elif be == "pandas":
        pytest.importorskip("pandas")
    return get_dataframe_class(be)


class TestGetPosition:
    """Unit tests for _get_position() method."""

    def test_get_position_valid_int(self, backend_class):
        """Valid integer index returns same position."""
        table = pa.table({
            "id": ["A", "B", "C"],
            "type": ["FILE", "FILE", "FOLDER"],
        })
        df = backend_class.from_arrow(table, "folder")

        assert df._get_position(0) == 0
        assert df._get_position(1) == 1
        assert df._get_position(2) == 2

    def test_get_position_negative_index(self, backend_class):
        """Negative index raises TacoNavigationError."""
        table = pa.table({
            "id": ["A", "B", "C"],
            "type": ["FILE", "FILE", "FOLDER"],
        })
        df = backend_class.from_arrow(table, "folder")

        with pytest.raises(TacoNavigationError, match="Position -1 out of range"):
            df._get_position(-1)

        with pytest.raises(TacoNavigationError, match="Position -999 out of range"):
            df._get_position(-999)

    def test_get_position_index_too_high(self, backend_class):
        """Index >= len raises TacoNavigationError."""
        table = pa.table({
            "id": ["A", "B", "C"],
            "type": ["FILE", "FILE", "FOLDER"],
        })
        df = backend_class.from_arrow(table, "folder")

        with pytest.raises(TacoNavigationError, match="Position 3 out of range"):
            df._get_position(3)

        with pytest.raises(TacoNavigationError, match="Position 100 out of range"):
            df._get_position(100)

    def test_get_position_valid_id(self, backend_class):
        """Valid string ID returns correct position."""
        table = pa.table({
            "id": ["sample_0", "sample_1", "sample_2"],
            "type": ["FILE", "FILE", "FILE"],
        })
        df = backend_class.from_arrow(table, "zip")

        assert df._get_position("sample_0") == 0
        assert df._get_position("sample_1") == 1
        assert df._get_position("sample_2") == 2

    def test_get_position_id_not_found(self, backend_class):
        """Nonexistent ID raises TacoNavigationError."""
        table = pa.table({
            "id": ["sample_0", "sample_1", "sample_2"],
            "type": ["FILE", "FILE", "FILE"],
        })
        df = backend_class.from_arrow(table, "zip")

        with pytest.raises(TacoNavigationError, match="ID 'nonexistent' not found"):
            df._get_position("nonexistent")

        with pytest.raises(TacoNavigationError, match="ID 'sample_99' not found"):
            df._get_position("sample_99")

    def test_get_position_missing_id_column(self, backend_class):
        """Missing 'id' column raises TacoNavigationError when searching by ID."""
        table = pa.table({
            "name": ["A", "B", "C"],
            "type": ["FILE", "FILE", "FILE"],
        })
        df = backend_class.from_arrow(table, "folder")

        # Int index still works (no 'id' column needed)
        assert df._get_position(0) == 0

        # String search fails
        with pytest.raises(TacoNavigationError, match="Cannot search by ID: 'id' column not found"):
            df._get_position("A")

    def test_get_position_duplicate_ids_returns_first(self, backend_class):
        """Duplicate IDs return first occurrence."""
        table = pa.table({
            "id": ["A", "B", "A", "C"],
            "type": ["FILE", "FILE", "FILE", "FILE"],
        })
        df = backend_class.from_arrow(table, "zip")

        # Should return first "A" at position 0
        assert df._get_position("A") == 0


class TestReadMethod:
    """Unit tests for read() method."""

    def test_read_file_returns_string(self, backend_class):
        """Reading FILE sample returns GDAL VSI path string."""
        table = pa.table({
            "id": ["sample_0"],
            "type": ["FILE"],
            "internal:gdal_vsi": ["/vsisubfile/0_1000,/path/data.zip"],
        })
        df = backend_class.from_arrow(table, "zip")

        result = df.read(0)

        assert isinstance(result, str)
        assert result == "/vsisubfile/0_1000,/path/data.zip"

    def test_read_file_by_id_returns_string(self, backend_class):
        """Reading FILE sample by ID returns GDAL VSI path."""
        table = pa.table({
            "id": ["sample_0", "sample_1"],
            "type": ["FILE", "FILE"],
            "internal:gdal_vsi": ["/path/a", "/path/b"],
        })
        df = backend_class.from_arrow(table, "folder")

        result = df.read("sample_1")

        assert isinstance(result, str)
        assert result == "/path/b"

    def test_read_folder_calls_read_meta(self, backend_class, monkeypatch):
        """Reading FOLDER sample calls _read_meta()."""
        table = pa.table({
            "id": ["folder_0"],
            "type": ["FOLDER"],
            "internal:gdal_vsi": ["/path/to/__meta__"],
        })
        df = backend_class.from_arrow(table, "folder")

        # Mock _read_meta to avoid actual I/O
        mock_called = []

        def mock_read_meta(row):
            mock_called.append(row)
            # Return mock TacoDataFrame
            mock_table = pa.table({
                "id": ["child_0"],
                "type": ["FILE"],
                "internal:gdal_vsi": ["/path/child"],
            })
            return backend_class.from_arrow(mock_table, "folder")

        monkeypatch.setattr(df, "_read_meta", mock_read_meta)

        result = df.read(0)

        # Should have called _read_meta
        assert len(mock_called) == 1
        assert mock_called[0]["type"] == "FOLDER"

        # Should return TacoDataFrame (same backend)
        assert isinstance(result, backend_class)