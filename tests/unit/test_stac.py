"""Unit tests for STAC filtering logic - no I/O, pure functions only."""

from datetime import datetime, timedelta, timezone

import pytest

from tacoreader._exceptions import TacoQueryError
from tacoreader._stac import (
    _ensure_utc,
    _parse_iso_string,
    detect_geometry_column,
    detect_time_column,
    parse_datetime,
)


class TestParseDatetimeStringRange:

    def test_basic_range(self):
        start, end = parse_datetime("2023-01-01/2023-12-31")
        assert start < end
        assert datetime.fromtimestamp(start, tz=timezone.utc).year == 2023
        assert datetime.fromtimestamp(end, tz=timezone.utc).month == 12

    def test_range_with_time(self):
        start, end = parse_datetime("2023-01-01T10:00:00Z/2023-01-01T18:00:00Z")
        assert end - start == 8 * 3600

    def test_range_with_offset(self):
        start, end = parse_datetime("2023-01-01T00:00:00+02:00/2023-01-01T00:00:00-05:00")
        assert start < end  # +02:00 es antes que -05:00 en UTC

    def test_inverted_range_raises(self):
        with pytest.raises(TacoQueryError, match="start.*>.*end"):
            parse_datetime("2025-01-01/2020-01-01")


class TestParseDatetimeSingleString:

    def test_date_only(self):
        start, end = parse_datetime("2023-04-15")
        assert end is None
        dt = datetime.fromtimestamp(start, tz=timezone.utc)
        assert dt.month == 4 and dt.day == 15

    def test_with_time(self):
        start, end = parse_datetime("2023-04-15T14:30:00Z")
        assert end is None
        dt = datetime.fromtimestamp(start, tz=timezone.utc)
        assert dt.hour == 14 and dt.minute == 30


class TestParseDatetimeDatetimeObject:

    def test_aware_datetime(self):
        dt = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        start, end = parse_datetime(dt)
        assert end is None
        assert start == int(dt.timestamp())

    def test_naive_datetime_treated_as_utc(self):
        naive = datetime(2023, 6, 15, 12, 0, 0)
        aware = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        start_naive, _ = parse_datetime(naive)
        start_aware, _ = parse_datetime(aware)
        assert start_naive == start_aware

    def test_non_utc_timezone_converted(self):
        tz_tokyo = timezone(timedelta(hours=9))
        dt_tokyo = datetime(2023, 6, 15, 21, 0, 0, tzinfo=tz_tokyo)
        dt_utc = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        start_tokyo, _ = parse_datetime(dt_tokyo)
        start_utc, _ = parse_datetime(dt_utc)
        assert start_tokyo == start_utc


class TestParseDatetimeTuple:

    def test_basic_tuple(self):
        start_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_dt = datetime(2023, 12, 31, tzinfo=timezone.utc)
        start, end = parse_datetime((start_dt, end_dt))
        assert start == int(start_dt.timestamp())
        assert end == int(end_dt.timestamp())

    def test_inverted_tuple_raises(self):
        with pytest.raises(TacoQueryError, match="start > end"):
            parse_datetime((
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 1, tzinfo=timezone.utc),
            ))


class TestParseDatetimeInvalidInput:

    def test_integer_raises(self):
        with pytest.raises(TacoQueryError, match="Invalid datetime input"):
            parse_datetime(12345)

    def test_list_raises(self):
        with pytest.raises(TacoQueryError, match="Invalid datetime input"):
            parse_datetime([datetime.now(), datetime.now()])

    def test_single_element_tuple_raises(self):
        with pytest.raises(TacoQueryError, match="Invalid datetime input"):
            parse_datetime((datetime.now(),))

    def test_three_element_tuple_raises(self):
        with pytest.raises(TacoQueryError, match="Invalid datetime input"):
            parse_datetime((datetime.now(), datetime.now(), datetime.now()))


class TestParseIsoString:

    def test_z_suffix(self):
        dt = _parse_iso_string("2023-04-15T12:00:00Z")
        assert dt.tzinfo is not None
        assert dt.hour == 12

    def test_positive_offset(self):
        dt = _parse_iso_string("2023-04-15T14:00:00+02:00")
        utc = dt.astimezone(timezone.utc)
        assert utc.hour == 12

    def test_negative_offset(self):
        dt = _parse_iso_string("2023-04-15T07:00:00-05:00")
        utc = dt.astimezone(timezone.utc)
        assert utc.hour == 12

    def test_naive_assumed_utc(self):
        dt = _parse_iso_string("2023-04-15T12:00:00")
        assert dt.tzinfo == timezone.utc


class TestEnsureUtc:

    def test_naive_to_utc(self):
        naive = datetime(2023, 6, 15, 12, 0, 0)
        result = _ensure_utc(naive)
        assert result.tzinfo == timezone.utc
        assert result.hour == 12

    def test_aware_converted(self):
        tz_plus5 = timezone(timedelta(hours=5))
        aware = datetime(2023, 6, 15, 17, 0, 0, tzinfo=tz_plus5)
        result = _ensure_utc(aware)
        assert result.tzinfo == timezone.utc
        assert result.hour == 12

    def test_utc_unchanged(self):
        utc_dt = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _ensure_utc(utc_dt)
        assert result == utc_dt


class TestDetectGeometryColumn:

    def test_istac_geometry_priority(self):
        cols = ["other", "stac:centroid", "istac:geometry", "istac:centroid"]
        assert detect_geometry_column(cols) == "istac:geometry"

    def test_stac_centroid_fallback(self):
        cols = ["id", "type", "stac:centroid", "istac:centroid"]
        assert detect_geometry_column(cols) == "stac:centroid"

    def test_istac_centroid_last(self):
        cols = ["id", "istac:centroid", "other"]
        assert detect_geometry_column(cols) == "istac:centroid"

    def test_no_geometry_raises(self):
        cols = ["id", "type", "cloud_cover", "timestamp"]
        with pytest.raises(TacoQueryError, match="No geometry column found"):
            detect_geometry_column(cols)

    def test_empty_list_raises(self):
        with pytest.raises(TacoQueryError, match="No geometry column found"):
            detect_geometry_column([])


class TestDetectTimeColumn:

    def test_istac_priority(self):
        cols = ["stac:time_start", "istac:time_start", "other"]
        assert detect_time_column(cols) == "istac:time_start"

    def test_stac_fallback(self):
        cols = ["id", "stac:time_start", "type"]
        assert detect_time_column(cols) == "stac:time_start"

    def test_no_time_raises(self):
        cols = ["id", "type", "geometry"]
        with pytest.raises(TacoQueryError, match="No time column found"):
            detect_time_column(cols)

    def test_empty_list_raises(self):
        with pytest.raises(TacoQueryError, match="No time column found"):
            detect_time_column([])