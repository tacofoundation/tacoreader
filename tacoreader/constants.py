"""
Global constants for tacoreader.

Organization:
- SHARED_*      : Shared with tacotoolbox (format specs, metadata columns)
- TACOCAT_*     : TacoCat format specification (imported from tacotoolbox logic)
- METADATA_*    : Metadata column names (shared with tacotoolbox)
- PADDING_*     : Padding constants (shared with tacotoolbox)
- NETWORK_*     : Network and download configuration (tacoreader-specific)
- DUCKDB_*      : DuckDB configuration (tacoreader-specific)
- STAC_*        : STAC/spatial filtering (tacoreader-specific)
- DATAFRAME_*   : DataFrame operations (tacoreader-specific)
- CACHE_*       : Cache and temporary files (tacoreader-specific)
"""

import re

# =============================================================================
# METADATA COLUMNS (SHARED with tacotoolbox)
# =============================================================================

METADATA_PARENT_ID = "internal:parent_id"
"""Parent sample index in previous level DataFrame (enables relational queries)."""

METADATA_OFFSET = "internal:offset"
"""Byte offset in container file where data starts."""

METADATA_SIZE = "internal:size"
"""Size in bytes of the data."""

METADATA_RELATIVE_PATH = "internal:relative_path"
"""Relative path from DATA/ directory (for navigation)."""

METADATA_GDAL_VSI = "internal:gdal_vsi"
"""GDAL VSI path for reading files (tacoreader-specific)."""

METADATA_SOURCE_FILE = "internal:source_file"
"""Source file name for TacoCat consolidated datasets."""

# =============================================================================
# PROTECTED COLUMNS (tacoreader navigation)
# =============================================================================

PROTECTED_COLUMNS = frozenset({
    # Core columns (modifying breaks references and navigation)
    "id",
    "type",
    # Internal columns (modifying breaks hierarchical navigation)
    METADATA_PARENT_ID,
    METADATA_OFFSET,
    METADATA_SIZE,
    METADATA_GDAL_VSI,
    METADATA_SOURCE_FILE,
    METADATA_RELATIVE_PATH,
})
"""Columns that cannot be modified without breaking navigation in tacoreader."""

# =============================================================================
# HIERARCHY LIMITS (SHARED with tacotoolbox)
# =============================================================================

SHARED_MAX_DEPTH = 5
"""Maximum hierarchy depth (0-5 means 6 levels total)."""

SHARED_MAX_LEVELS = 6
"""Total number of possible levels (0 through 5)."""

# =============================================================================
# PADDING (SHARED with tacotoolbox)
# =============================================================================

PADDING_PREFIX = "__TACOPAD__"
"""Prefix for auto-generated padding sample IDs."""

# =============================================================================
# TACOCAT FORMAT SPECIFICATION (SHARED with tacotoolbox)
# =============================================================================

TACOCAT_MAGIC = b"TACOCAT\x00"
"""Magic number identifying TacoCat files (8 bytes)."""

TACOCAT_VERSION = 1
"""TacoCat format version (uint32)."""

TACOCAT_MAX_LEVELS = 6
"""
Fixed number of levels in TacoCat format (always 6 entries).
Structure: 5 metadata levels (level0-level5) + COLLECTION.json.
"""

TACOCAT_HEADER_SIZE = 16
"""TacoCat file header size: Magic(8) + Version(4) + MaxDepth(4)."""

TACOCAT_INDEX_ENTRY_SIZE = 16
"""Size of each index entry: Offset(8) + Size(8)."""

TACOCAT_INDEX_SIZE = (
    TACOCAT_MAX_LEVELS * TACOCAT_INDEX_ENTRY_SIZE + TACOCAT_INDEX_ENTRY_SIZE
)  # 112
"""Total index block size: 7 entries x 16 bytes."""

TACOCAT_TOTAL_HEADER_SIZE = TACOCAT_HEADER_SIZE + TACOCAT_INDEX_SIZE  # 128
"""Total header + index size (data starts at byte 128)."""

TACOCAT_FILENAME = "__TACOCAT__"
"""Fixed filename for TacoCat consolidated files."""

# =============================================================================
# AVRO SERIALIZATION (SHARED with tacotoolbox)
# =============================================================================

AVRO_COLON_REPLACEMENT = "_COLON_"
"""
Replacement string for colons in Avro column names.

CRITICAL: Avro specification does not allow colons (:) in field names.
We must replace them during serialization and restore during deserialization.

Serialization flow:
    Write: "internal:parent_id" → "internal_COLON_parent_id"
    Read:  "internal_COLON_parent_id" → "internal:parent_id"
"""

# =============================================================================
# NETWORK & DOWNLOAD CONFIGURATION (tacoreader-specific)
# =============================================================================

# Optimal chunk size for parallel downloads (empirically tested)
# Testing results (137MB file from S3, December 2024):
# - 2MB:  ~27s (5.0 MB/s) - too many requests, overhead dominates
# - 4MB:  ~15s (9.1 MB/s) - OPTIMAL ✓
# - 10MB: ~18s (7.6 MB/s) - fewer requests but larger buffers
# - 20MB: ~21s (6.5 MB/s) - too large, memory allocation overhead
NETWORK_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB
"""
Chunk size for parallel range requests when downloading TacoCat files.

Empirically tested as optimal for S3/GCS. Balances:
- Number of concurrent requests (parallelism)
- Buffer allocation overhead
- Network round-trip latency
"""

NETWORK_CLIENT_OPTIONS = None
"""
obstore client configuration.

Current testing shows default values are optimal. Custom options
like pool_idle_timeout and timeout provide <5% benefit while adding
complexity. Use None to rely on obstore's smart defaults.
"""

# =============================================================================
# CACHE & TEMPORARY FILES (tacoreader-specific)
# =============================================================================

CACHE_DIR_PREFIX = "tacoreader-"
"""Prefix for temporary cache directories."""

CACHE_CONCAT_PREFIX = "tacoreader-concat-"
"""Prefix for concat() operation cache directories."""

# =============================================================================
# DUCKDB CONFIGURATION (tacoreader-specific)
# =============================================================================

DUCKDB_MEMORY_LIMIT = None
"""DuckDB memory limit. None = unlimited (default)."""

DUCKDB_THREADS = None
"""DuckDB thread count. None = auto-detect (default)."""

# =============================================================================
# DATAFRAME LIMITS (tacoreader-specific)
# =============================================================================

DATAFRAME_DEFAULT_HEAD_ROWS = 5
"""Default number of rows for .head()"""

DATAFRAME_DEFAULT_TAIL_ROWS = 5
"""Default number of rows for .tail()"""

# =============================================================================
# PARALLEL LOADING (tacoreader-specific)
# =============================================================================

PARALLEL_DEFAULT_MAX_WORKERS = 8
"""Default max parallel workers for loading multiple files."""

PARALLEL_MIN_WORKERS = 1
"""Minimum parallel workers (sequential loading)."""

# =============================================================================
# STATISTICS AGGREGATION (tacoreader-specific)
# =============================================================================

STATS_CONTINUOUS_LENGTH = 9
"""Expected length for continuous stats: [min, max, mean, std, valid%, p25, p50, p75, p95]."""

# =============================================================================
# STAC / GEOMETRY (tacoreader-specific)
# =============================================================================

STAC_GEOMETRY_COLUMN_PRIORITY = [
    "istac:geometry",   # Most precise - full geometry
    "stac:centroid",    # Point representation for STAC
    "istac:centroid",   # Point representation for ISTAC
]
"""Priority order for auto-detecting geometry columns."""

STAC_TIME_COLUMN_PRIORITY = [
    "istac:time_start",  # ISTAC time
    "stac:time_start",   # STAC time
]
"""Priority order for auto-detecting time columns."""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def is_padding_id(sample_id: str) -> bool:
    """
    Check if a sample ID represents padding.

    Args:
        sample_id: Sample ID to check

    Returns:
        True if ID starts with padding prefix

    Example:
        >>> is_padding_id("__TACOPAD__0")
        True
        >>> is_padding_id("real_sample")
        False
    """
    return sample_id.startswith(PADDING_PREFIX)


def is_internal_column(column_name: str) -> bool:
    """
    Check if a column name is an internal metadata column.

    Args:
        column_name: Column name to check

    Returns:
        True if column starts with "internal:"

    Example:
        >>> is_internal_column("internal:parent_id")
        True
        >>> is_internal_column("custom_field")
        False
    """
    return column_name.startswith("internal:")


def is_protected_column(column_name: str) -> bool:
    """
    Check if a column is protected and cannot be modified.

    Protected columns are required for hierarchical navigation
    in tacoreader. Modifying these breaks .read() functionality.

    Args:
        column_name: Column name to check

    Returns:
        True if column is protected

    Example:
        >>> is_protected_column("id")
        True
        >>> is_protected_column("internal:offset")
        True
        >>> is_protected_column("cloud_cover")
        False
    """
    return column_name in PROTECTED_COLUMNS or column_name.startswith("internal:")


def validate_depth(depth: int, context: str = "operation") -> None:
    """
    Validate that depth is within allowed range.

    Args:
        depth: Depth value to validate
        context: Context string for error message

    Raises:
        ValueError: If depth is invalid

    Example:
        >>> validate_depth(3, "query")
        >>> validate_depth(6, "query")  # Raises ValueError
    """
    if depth < 0:
        raise ValueError(f"{context}: depth must be non-negative, got {depth}")

    if depth > SHARED_MAX_DEPTH:
        raise ValueError(
            f"{context}: depth {depth} exceeds maximum of {SHARED_MAX_DEPTH} "
            f"(levels 0-{SHARED_MAX_DEPTH})"
        )


def get_chunk_size() -> int:
    """
    Get optimal chunk size for downloads.
    
    Returns optimal chunk size in bytes. Can be extended later
    to adapt based on file size, network conditions, etc.
    
    Returns:
        Chunk size in bytes (currently 4MB)
    """
    return NETWORK_CHUNK_SIZE