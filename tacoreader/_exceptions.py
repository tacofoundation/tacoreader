"""
Exception hierarchy for tacoreader.

All TACO-specific exceptions inherit from TacoError.
Enables granular exception handling for different failure modes.

Usage:
    from tacoreader._exceptions import TacoError, TacoFormatError

    try:
        ds = tacoreader.load("data.taco")
    except TacoFormatError:
        # Handle corrupted files
        logger.warning("File corrupted, skipping")
    except TacoIOError:
        # Handle network/filesystem errors
        retry_with_backoff()
    except TacoError:
        # Catch-all for other TACO errors
        raise
"""


class TacoError(Exception):
    """Base exception for all TACO errors."""

    pass


class TacoFormatError(TacoError):
    """
    Invalid format or corrupted file.

    Raised when:
    - Magic number doesn't match expected format
    - COLLECTION.json is malformed or missing required fields
    - Parquet files are corrupted or unreadable
    - ZIP structure is invalid
    - TacoCat header is malformed

    Examples:
        - "Invalid TacoCat magic: b'CORRUPT\\x00'"
        - "Missing required field 'taco:pit_schema' in COLLECTION.json"
        - "TACO_HEADER missing in ZIP file"
    """

    pass


class TacoSchemaError(TacoError):
    """
    Schema incompatibility or validation error.

    Raised when:
    - PIT schemas are incompatible during concat
    - Required metadata columns are missing
    - Column types don't match expected schema
    - Hierarchy depth exceeds limits
    - Position-Isomorphic Tree validation fails

    Examples:
        - "Cannot concat: Dataset 1 has incompatible schema"
        - "Missing required column 'internal:gdal_vsi'"
        - "Invalid root type: must be 'FILE' or 'FOLDER'"
    """

    pass


class TacoIOError(TacoError):
    """
    I/O operation failed.

    Raised when:
    - File or directory not found
    - Permission denied (filesystem or cloud storage)
    - Network timeout or connection error
    - HTTP 403/404/500 errors
    - S3/GCS/Azure authentication failures

    Examples:
        - "File not found: /path/to/data.taco"
        - "Failed to download from s3://bucket/data.taco: 403 Forbidden"
        - "Network timeout reading from https://..."
    """

    pass


class TacoQueryError(TacoError):
    """
    Invalid query or operation.

    Raised when:
    - Level doesn't exist in dataset
    - Column not found in metadata
    - SQL query is malformed
    - Invalid filter parameters
    - Geometry/time column auto-detection fails

    Examples:
        - "Level 3 does not exist. Available levels: 0 to 2"
        - "Column 'cloud_cover' not found. Available: ['id', 'type', ...]"
        - "No geometry column found. Expected one of: istac:geometry, stac:centroid"
    """

    pass


class TacoNavigationError(TacoError):
    """
    Navigation error during hierarchical traversal.

    Raised when:
    - Sample ID not found during .read()
    - Position out of range
    - Missing metadata for FOLDER navigation
    - Corrupted __meta__ files

    Examples:
        - "ID 'sample_xyz' not found"
        - "Position 999 out of range [0, 100]"
        - "Missing required metadata: 'internal:offset' or 'internal:size'"
    """

    pass


class TacoBackendError(TacoError):
    """
    DataFrame backend error.

    Raised when:
    - Backend not registered or unavailable
    - Backend dependencies missing
    - Backend-specific conversion fails

    Examples:
        - "Backend 'polars' not registered. Install with: pip install polars"
        - "Unknown backend: 'pandas'. Available: ['pyarrow']"
    """

    pass
