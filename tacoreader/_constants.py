"""
Global constants for tacoreader.

Organization:
- Network/Performance: ZIP_MAX_GAP_SIZE
- Metadata columns: METADATA_*
- Protected columns: PROTECTED_COLUMNS, NAVIGATION_REQUIRED_COLUMNS
- Hierarchy limits: SHARED_MAX_DEPTH, SHARED_MAX_LEVELS
- TacoCat format: TACOCAT_*
- DuckDB config: DUCKDB_*
- DataFrame limits: DATAFRAME_*
- Statistics: STATS_CONTINUOUS_LENGTH
- STAC/Geometry: STAC_GEOMETRY_COLUMN_PRIORITY, STAC_TIME_COLUMN_PRIORITY
"""

# =============================================================================
# Network Constants (tacoreader-specific)
# =============================================================================

ZIP_MAX_GAP_SIZE = 4 * 1024 * 1024  # 4 MB
"""Maximum gap between files in ZIP before splitting into separate requests."""

# =============================================================================
# Metadata Columns (shared with tacotoolbox)
# =============================================================================

METADATA_PARENT_ID = "internal:parent_id"
"""Parent sample ID in previous level (enables relational queries)."""

METADATA_OFFSET = "internal:offset"
"""Byte offset in container file where data starts."""

METADATA_SIZE = "internal:size"
"""Size in bytes of the data."""

METADATA_RELATIVE_PATH = "internal:relative_path"
"""Relative path from DATA/ directory (for FOLDER navigation)."""

METADATA_GDAL_VSI = "internal:gdal_vsi"
"""GDAL VSI path for reading files (constructed by backends)."""

METADATA_SOURCE_FILE = "internal:source_file"
"""Source file name for TacoCat consolidated datasets."""

# =============================================================================
# Protected Columns (tacoreader navigation)
# =============================================================================

PROTECTED_COLUMNS = frozenset(
    {
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
    }
)
"""Columns that cannot be modified without breaking navigation."""

NAVIGATION_REQUIRED_COLUMNS = frozenset({"id", "type", "internal:gdal_vsi"})
"""Minimum columns required for .read() hierarchical navigation."""

# =============================================================================
# Hierarchy Limits (shared with tacotoolbox)
# =============================================================================

SHARED_MAX_DEPTH = 5
"""Maximum hierarchy depth (0-5 means 6 levels total)."""

SHARED_MAX_LEVELS = 6
"""Total number of possible levels (0 through 5)."""

# =============================================================================
# Padding (shared with tacotoolbox)
# =============================================================================

PADDING_PREFIX = "__TACOPAD__"
"""Prefix for auto-generated padding sample IDs."""

# =============================================================================
# TacoCat Format Specification (shared with tacotoolbox)
# =============================================================================

TACOCAT_MAGIC = b"TACOCAT\x00"
"""Magic number identifying TacoCat files (8 bytes)."""

TACOCAT_VERSION = 1
"""TacoCat format version (uint32)."""

TACOCAT_MAX_LEVELS = 6
"""Fixed number of level entries in TacoCat index (level0-5 + COLLECTION)."""

TACOCAT_HEADER_SIZE = 16
"""TacoCat header size: Magic(8) + Version(4) + MaxDepth(4)."""

TACOCAT_INDEX_ENTRY_SIZE = 16
"""Size of each index entry: Offset(8) + Size(8)."""

TACOCAT_INDEX_SIZE = (
    TACOCAT_MAX_LEVELS * TACOCAT_INDEX_ENTRY_SIZE + TACOCAT_INDEX_ENTRY_SIZE
)  # 112
"""Total index block size: 7 entries × 16 bytes."""

TACOCAT_TOTAL_HEADER_SIZE = TACOCAT_HEADER_SIZE + TACOCAT_INDEX_SIZE  # 128
"""Total header + index size (data starts at byte 128)."""

TACOCAT_FILENAME = "__TACOCAT__"
"""Fixed filename for TacoCat consolidated files."""

# =============================================================================
# DuckDB Configuration (tacoreader-specific)
# =============================================================================

DUCKDB_MEMORY_LIMIT = None
"""DuckDB memory limit. None = unlimited (default)."""

DUCKDB_THREADS = None
"""DuckDB thread count. None = auto-detect (default)."""

# =============================================================================
# DataFrame Limits (tacoreader-specific)
# =============================================================================

DATAFRAME_DEFAULT_HEAD_ROWS = 5
"""Default number of rows for .head()"""

DATAFRAME_DEFAULT_TAIL_ROWS = 5
"""Default number of rows for .tail()"""

# =============================================================================
# Statistics Aggregation (tacoreader-specific)
# =============================================================================

STATS_CONTINUOUS_LENGTH = 9
"""Expected length for continuous stats: [min, max, mean, std, valid%, p25, p50, p75, p95]."""

# =============================================================================
# STAC / Geometry (tacoreader-specific)
# =============================================================================

STAC_GEOMETRY_COLUMN_PRIORITY = [
    "istac:geometry",  # Most precise - full geometry
    "stac:centroid",  # Point representation for STAC
    "istac:centroid",  # Point representation for ISTAC
]
"""Priority order for auto-detecting geometry columns."""

STAC_TIME_COLUMN_PRIORITY = [
    "istac:time_start",  # ISTAC time
    "stac:time_start",  # STAC time
]
"""Priority order for auto-detecting time columns."""
