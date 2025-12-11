"""
Global constants for tacoreader.

Organized by: DataFrame Backend, Cloud Storage, File Extensions,
Sample Types, Metadata, Hierarchy, TacoCat, DuckDB, Statistics, STAC.
"""

from typing import Literal

# DataFrame Backend Configuration
DataFrameBackend = Literal["pyarrow", "polars", "pandas"]
"""Valid DataFrame backend types."""

DEFAULT_DATAFRAME_BACKEND: DataFrameBackend = "pyarrow"
"""Default DataFrame backend used by tacoreader."""

AVAILABLE_BACKENDS: tuple[DataFrameBackend, ...] = ("pyarrow", "polars", "pandas")
"""All supported DataFrame backends (registered or not)."""


# Cloud Storage & Protocols
PROTOCOL_MAPPINGS = {
    "s3": {"standard": "s3://", "vsi": "/vsis3/"},
    "gcs": {"standard": "gs://", "vsi": "/vsigs/"},
    "azure": {"standard": "az://", "vsi": "/vsiaz/", "alt": "azure://"},
    "oss": {"standard": "oss://", "vsi": "/vsioss/"},
    "swift": {"standard": "swift://", "vsi": "/vsiswift/"},
    "http": {"standard": "http://", "vsi": "/vsicurl/"},
    "https": {"standard": "https://", "vsi": "/vsicurl/"},
}
"""
Unified protocol mappings for cloud storage and GDAL VSI.

Maps storage protocols to their standard URL scheme and GDAL VSI prefix.
'alt' key provides alternative protocol names (e.g., azure:// vs az://).
"""

# Derived protocol lists for quick lookups
_all_standard = [p["standard"] for p in PROTOCOL_MAPPINGS.values()]
_all_alt = [p.get("alt") for p in PROTOCOL_MAPPINGS.values() if "alt" in p]
CLOUD_PROTOCOLS = tuple(_all_standard + _all_alt)
"""All valid cloud storage protocol prefixes (s3://, gs://, azure://, etc.)."""

VSI_PROTOCOLS = tuple(p["vsi"] for p in PROTOCOL_MAPPINGS.values() if "vsi" in p)
"""Cloud storage VSI prefixes (/vsis3/, /vsigs/, etc.)."""

VSI_SPECIAL = ("/vsizip/", "/vsisubfile/")
"""Special GDAL VSI prefixes for archive/subfile access."""

ALL_VSI_PREFIXES = VSI_PROTOCOLS + VSI_SPECIAL
"""All valid VSI prefixes (cloud + special)."""


# File Extensions
TACOZIP_EXTENSIONS = (".tacozip", ".zip")
"""Valid file extensions for ZIP format."""

PARQUET_EXTENSION = ".parquet"
"""Parquet metadata file extension."""

COLLECTION_JSON = "COLLECTION.json"
"""Standard filename for dataset metadata."""


# Sample Types (shared with tacotoolbox)
SAMPLE_TYPE_FILE = "FILE"
"""Sample type for file-based data (leaf nodes in hierarchy)."""

SAMPLE_TYPE_FOLDER = "FOLDER"
"""Sample type for folder-based data (containers with children)."""

VALID_SAMPLE_TYPES = frozenset({SAMPLE_TYPE_FILE, SAMPLE_TYPE_FOLDER})
"""All valid sample type values."""


# Metadata Columns (shared with tacotoolbox)
METADATA_PARENT_ID = "internal:parent_id"
"""Parent sample ID in previous level (enables relational queries)."""

METADATA_OFFSET = "internal:offset"
"""Byte offset in container file where data starts. Only relevant for ZIP/TacoCat."""

METADATA_SIZE = "internal:size"
"""Size in bytes of the data. Only relevant for ZIP/TacoCat."""

METADATA_RELATIVE_PATH = "internal:relative_path"
"""Relative path from DATA/ directory (for FOLDER navigation)."""

METADATA_GDAL_VSI = "internal:gdal_vsi"
"""GDAL VSI path for reading files (constructed by backends)."""

METADATA_SOURCE_FILE = "internal:source_file"
"""Source file name for TacoCat consolidated datasets."""


# Core Columns (shared with tacotoolbox)
COLUMN_ID = "id"
"""Unique sample identifier."""

COLUMN_TYPE = "type"
"""Sample type (FILE or FOLDER)."""


# Protected Columns (tacoreader navigation)
PROTECTED_COLUMNS = frozenset(
    {
        # Core columns (modifying breaks references and navigation)
        COLUMN_ID,
        COLUMN_TYPE,
        # Internal columns (modifying breaks hierarchical navigation)
        METADATA_PARENT_ID,
        METADATA_OFFSET,
        METADATA_SIZE,
        METADATA_GDAL_VSI,
        METADATA_SOURCE_FILE,
        METADATA_RELATIVE_PATH,
    }
)
"""
Columns that cannot be modified without breaking navigation.

These columns are critical for:
- Hierarchical traversal (.read() functionality)
- GDAL raster access
- Parent-child relationships
- Format-specific data location
"""

NAVIGATION_REQUIRED_COLUMNS = frozenset({COLUMN_ID, COLUMN_TYPE, METADATA_GDAL_VSI})
"""
Minimum columns required for .read() hierarchical navigation.

Without these columns, TacoDataFrame cannot:
- Navigate to children (.read())
- Load rasters from storage
- Distinguish between FILE and FOLDER samples
"""


# Hierarchy Limits (shared with tacotoolbox)
SHARED_MAX_DEPTH = 5
"""Maximum hierarchy depth"""

SHARED_MAX_LEVELS = 6
"""Total number of possible levels + COLLECTION (level0-5 + COLLECTION)."""


# Padding (shared with tacotoolbox)
PADDING_PREFIX = "__TACOPAD__"
"""
Prefix for auto-generated padding sample IDs.

Padding samples maintain Position-Isomorphic Tree (PIT) structure when
different parents have varying numbers of children. They are filtered
out from user-facing views.
"""


# TacoCat Format Specification (shared with tacotoolbox)
TACOCAT_FOLDER_NAME = ".tacocat"
"""
Fixed folder name for TacoCat consolidated format.

This name is RESERVED and cannot be used for regular FOLDER format datasets.
Any dataset folder named .tacocat will be interpreted as TacoCat format.
"""

TACOCAT_MAX_LEVELS = 6
"""Maximum number of hierarchy levels in TacoCat (level0-5)."""

RESERVED_FOLDER_NAMES = frozenset({TACOCAT_FOLDER_NAME, "DATA", "METADATA"})
"""
Folder names reserved by TACO specification.

These names cannot be used as dataset folder names:
- .tacocat: Reserved for TacoCat consolidated format
- DATA: Reserved for FOLDER format data directory
- METADATA: Reserved for FOLDER format metadata directory
"""


# DuckDB Configuration
DEFAULT_VIEW_NAME = "data"
"""
Default view name aliasing level0.

All datasets expose 'data' as the primary query interface, which
internally points to level0 metadata.
"""

LEVEL_VIEW_PREFIX = "level"
"""Prefix for hierarchical level views (level0, level1, level2, etc.)."""

LEVEL_TABLE_SUFFIX = "_table"
"""Suffix for raw DuckDB tables before view creation."""

UNION_VIEW_SUFFIX = "_union"
"""Suffix for intermediate union views in concat operations."""

SQL_JOIN_PATTERN = r"\b(?:JOIN|FROM)\s+(level[1-5])\b"
"""
Regex pattern for detecting JOINs with level1+ tables.

Used to track when queries involve multi-level relationships,
which affects certain optimizations and validations.
"""


# Network & Performance
ZIP_MAX_GAP_SIZE = 4 * 1024 * 1024  # 4 MB
"""
Maximum gap between files in ZIP before splitting into separate requests.

When downloading multiple files from remote ZIP, files separated by less
than this amount are fetched in a single HTTP range request to minimize
request count while avoiding excessive unused data transfer.
"""


# DataFrame Limits
DATAFRAME_DEFAULT_HEAD_ROWS = 5
"""Default number of rows for .head()"""

DATAFRAME_DEFAULT_TAIL_ROWS = 5
"""Default number of rows for .tail()"""

DATAFRAME_MAX_REPR_ROWS = 100
"""Maximum rows to display in TacoDataFrame.__repr__()"""


# Statistics Aggregation
STATS_CONTINUOUS_LENGTH = 9
"""
Expected length for continuous stats arrays.

Format: [min, max, mean, std, valid%, p25, p50, p75, p95]

Stats with different lengths are interpreted as categorical (class probabilities).
"""


# STAC Geometry & Time Columns
STAC_GEOMETRY_COLUMN_PRIORITY = [
    "istac:geometry",  # Most precise - full geometry
    "stac:centroid",  # Point representation for STAC
    "istac:centroid",  # Point representation for ISTAC
]
"""
Priority order for auto-detecting geometry columns.

When geometry_col='auto', search columns in this order.
First match is used for spatial filtering.
"""

STAC_TIME_COLUMN_PRIORITY = [
    "istac:time_start",  # ISTAC time
    "stac:time_start",  # STAC time
]
"""
Priority order for auto-detecting time columns.

When time_col='auto', search columns in this order.
Always use time_start (not middle/end) for temporal filtering.
"""
