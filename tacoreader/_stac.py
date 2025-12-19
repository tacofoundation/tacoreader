"""
Simple STAC filtering for level0 (no JOINs).

Provides spatial and temporal filtering on level0 without hierarchical JOINs.
Works for all formats: ZIP, FOLDER, TacoCat.

Functions:
    - apply_simple_bbox_filter: Spatial filtering on level0
    - apply_simple_datetime_filter: Temporal filtering on level0
    - detect_geometry_column: Auto-detect geometry column
    - detect_time_column: Auto-detect time column
    - parse_datetime: Parse datetime ranges to epochs
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tacoreader._constants import (
    DEFAULT_VIEW_NAME,
    STAC_GEOMETRY_COLUMN_PRIORITY,
    STAC_TIME_COLUMN_PRIORITY,
)
from tacoreader._exceptions import TacoQueryError

if TYPE_CHECKING:
    from tacoreader.dataset import TacoDataset


def apply_simple_bbox_filter(
    dataset: "TacoDataset",
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    geometry_col: str = "auto",
) -> "TacoDataset":
    """
    Filter level0 by bounding box.

    Uses DuckDB Spatial extension with ST_Intersects for bbox intersection.
    Works with WKB-encoded geometry columns.

    Args:
        dataset: TacoDataset to filter
        minx: Minimum X coordinate (longitude)
        miny: Minimum Y coordinate (latitude)
        maxx: Maximum X coordinate (longitude)
        maxy: Maximum Y coordinate (latitude)
        geometry_col: Geometry column name ("auto" for auto-detection)

    Returns:
        Filtered TacoDataset

    Example:
        # Filter samples within Pacific region
        filtered = dataset.filter_bbox(
            minx=120, miny=20, maxx=170, maxy=55,
            geometry_col="istac:geometry"
        )
    """
    # Auto-detect or validate geometry column
    if geometry_col == "auto":
        geometry_col = detect_geometry_column(dataset.data.columns)
    else:
        if geometry_col not in dataset.data.columns:
            raise TacoQueryError(
                f"Column '{geometry_col}' not found in level0.\n"
                f"Available columns: {dataset.data.columns}"
            )

    # Build spatial WHERE clause
    where_clause = (
        f"ST_Intersects("
        f'ST_GeomFromWKB("{geometry_col}"), '
        f"ST_MakeEnvelope({float(minx)}, {float(miny)}, {float(maxx)}, {float(maxy)})"
        f")"
    )

    # Execute simple query on level0
    return dataset.sql(f"SELECT * FROM {DEFAULT_VIEW_NAME} WHERE {where_clause}")


def apply_simple_datetime_filter(
    dataset: "TacoDataset",
    datetime_range: str | datetime | tuple[datetime, datetime],
    time_col: str = "auto",
) -> "TacoDataset":
    """
    Filter level0 by datetime.

    Automatically handles both TIMESTAMP and STRING date columns via TRY_CAST.
    Supports various datetime input formats.

    Args:
        dataset: TacoDataset to filter
        datetime_range: Temporal range as:
            - String range: "2023-01-01/2023-12-31"
            - Single datetime: datetime(2023, 1, 1)
            - Tuple: (start_dt, end_dt)
        time_col: Time column name ("auto" for auto-detection)

    Returns:
        Filtered TacoDataset

    Example:
        # Filter by date range
        filtered = dataset.filter_datetime(
            "2023-04-01/2023-04-30",
            time_col="timestamp"
        )

        # Filter by single date
        filtered = dataset.filter_datetime(
            datetime(2023, 4, 15),
            time_col="istac:time_start"
        )
    """
    # Auto-detect or validate time column
    if time_col == "auto":
        time_col = detect_time_column(dataset.data.columns)
    else:
        if time_col not in dataset.data.columns:
            raise TacoQueryError(
                f"Column '{time_col}' not found in level0.\n"
                f"Available columns: {dataset.data.columns}"
            )

    # Parse datetime range to epochs
    start, end = parse_datetime(datetime_range)

    # Build temporal WHERE clause with TRY_CAST
    # Handles both TIMESTAMP columns and STRING date columns
    col_cast = f'TRY_CAST("{time_col}" AS DATE)'

    start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
    start_str = start_dt.strftime("%Y-%m-%d")

    if end is None:
        # Point-in-time query
        where_clause = f"({col_cast} = DATE '{start_str}')"
    else:
        # Range query
        end_dt = datetime.fromtimestamp(end, tz=timezone.utc)
        end_str = end_dt.strftime("%Y-%m-%d")
        where_clause = f"({col_cast} BETWEEN DATE '{start_str}' AND DATE '{end_str}')"

    # Execute simple query on level0
    return dataset.sql(f"SELECT * FROM {DEFAULT_VIEW_NAME} WHERE {where_clause}")


def detect_geometry_column(columns: list[str]) -> str:
    """
    Auto-detect geometry column from available columns.

    Priority order:
        1. istac:geometry (full geometry, most precise)
        2. stac:centroid (point representation for STAC)
        3. istac:centroid (point representation for ISTAC)

    Args:
        columns: List of available column names

    Returns:
        First matching geometry column name

    Raises:
        TacoQueryError: If no geometry column found
    """
    for col in STAC_GEOMETRY_COLUMN_PRIORITY:
        if col in columns:
            return col

    raise TacoQueryError(
        "No geometry column found in level0.\n"
        f"Expected one of: {', '.join(STAC_GEOMETRY_COLUMN_PRIORITY)}\n"
        f"Available columns: {columns}"
    )


def detect_time_column(columns: list[str]) -> str:
    """
    Auto-detect time column from available columns.

    Always uses time_start (not middle/end) for temporal filtering.

    Priority order:
        1. istac:time_start
        2. stac:time_start

    Args:
        columns: List of available column names

    Returns:
        First matching time column name

    Raises:
        TacoQueryError: If no time column found
    """
    for col in STAC_TIME_COLUMN_PRIORITY:
        if col in columns:
            return col

    raise TacoQueryError(
        "No time column found in level0.\n"
        f"Expected one of: {', '.join(STAC_TIME_COLUMN_PRIORITY)}\n"
        f"Available columns: {columns}"
    )


def parse_datetime(
    dt_input: str | datetime | tuple[datetime, datetime]
) -> tuple[int, int | None]:
    """
    Parse datetime input to (start_epoch, end_epoch) tuple.

    Supports multiple input formats:
        - String range: "2023-01-01/2023-12-31"
        - Single datetime: datetime(2023, 1, 1)
        - Tuple: (start_dt, end_dt)

    Args:
        dt_input: Datetime specification in one of the supported formats

    Returns:
        Tuple of (start_epoch, end_epoch) where:
            - start_epoch: Unix timestamp (seconds) for start
            - end_epoch: Unix timestamp (seconds) for end, or None for point query

    Raises:
        TacoQueryError: If datetime format is invalid or range is invalid

    Examples:
        >>> parse_datetime("2023-01-01/2023-12-31")
        (1672531200, 1704067199)

        >>> parse_datetime(datetime(2023, 1, 1))
        (1672531200, None)

        >>> parse_datetime((datetime(2023, 1, 1), datetime(2023, 12, 31)))
        (1672531200, 1704067199)
    """
    # String range: "2023-01-01/2023-12-31"
    if isinstance(dt_input, str):
        if "/" in dt_input:
            start_str, end_str = dt_input.split("/", 1)
            start_dt = _parse_iso_string(start_str)
            end_dt = _parse_iso_string(end_str)

            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())

            if start_ts > end_ts:
                raise TacoQueryError(
                    f"Invalid datetime range: start ({start_str}) > end ({end_str})"
                )

            return start_ts, end_ts
        else:
            # Single date string
            start_dt = _parse_iso_string(dt_input)
            return int(start_dt.timestamp()), None

    # Single datetime object
    elif isinstance(dt_input, datetime):
        dt_utc = _ensure_utc(dt_input)
        return int(dt_utc.timestamp()), None

    # Tuple of datetime objects
    elif isinstance(dt_input, tuple) and len(dt_input) == 2:
        start_dt, end_dt = dt_input
        start_utc = _ensure_utc(start_dt)
        end_utc = _ensure_utc(end_dt)

        start_ts = int(start_utc.timestamp())
        end_ts = int(end_utc.timestamp())

        if start_ts > end_ts:
            raise TacoQueryError("Invalid datetime range: start > end")

        return start_ts, end_ts

    else:
        raise TacoQueryError(
            f"Invalid datetime input: {dt_input}\n"
            f"Expected: string range ('2023-01-01/2023-12-31'), "
            f"datetime object, or tuple of datetime objects"
        )


def _parse_iso_string(dt_str: str) -> datetime:
    """
    Parse ISO 8601 string to UTC datetime.

    Handles various ISO formats including 'Z' suffix.
    Assumes UTC if no timezone specified.

    Args:
        dt_str: ISO 8601 datetime string

    Returns:
        UTC-aware datetime object
    """
    # Replace 'Z' with UTC offset for fromisoformat()
    dt_str_normalized = dt_str.replace("Z", "+00:00")

    dt = datetime.fromisoformat(dt_str_normalized)

    # If naive (no timezone), assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to UTC if needed
    dt_utc = dt.astimezone(timezone.utc)

    return dt_utc


def _ensure_utc(dt: datetime) -> datetime:
    """
    Ensure datetime is UTC-aware.

    Args:
        dt: Datetime object (naive or aware)

    Returns:
        UTC-aware datetime object
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    else:
        return dt.astimezone(timezone.utc)
