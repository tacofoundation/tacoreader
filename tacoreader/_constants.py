"""Global constants for tacoreader.

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

METADATA_CURRENT_ID = "internal:current_id"
"""Current sample ID (int64) for parent-child relationships. Links to internal:parent_id in next level."""

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

METADATA_SOURCE_PATH = "internal:source_path"
"""Full path to source dataset (for concat across multiple sources)."""


# Core Columns (shared with tacotoolbox)
COLUMN_ID = "id"
"""Unique sample identifier."""

COLUMN_TYPE = "type"
"""Sample type (FILE or FOLDER)."""


# Protected Columns (tacoreader navigation)
PROTECTED_COLUMNS = frozenset(
    {
        COLUMN_ID,
        COLUMN_TYPE,
        METADATA_PARENT_ID,
        METADATA_OFFSET,
        METADATA_SIZE,
        METADATA_GDAL_VSI,
        METADATA_SOURCE_FILE,
        METADATA_SOURCE_PATH,
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


# Statistics Extensions
STATS_GEOTIFF_COLUMN = "geotiff:stats"
"""GeoTIFF statistics column name."""

STATS_SUPPORTED_COLUMNS = (STATS_GEOTIFF_COLUMN,)
"""Currently supported statistics columns."""

STATS_WEIGHT_COLUMN = "stac:tensor_shape"
"""Column used for weighted aggregation (pixel counts from shape)."""


# Navigation Columns (for .sql() column selection)
NAVIGATION_COLUMNS_ZIP = frozenset(
    {
        COLUMN_ID,
        COLUMN_TYPE,
        METADATA_CURRENT_ID,
        METADATA_OFFSET,
        METADATA_SIZE,
    }
)
"""Columns required for ZIP format navigation."""

NAVIGATION_COLUMNS_FOLDER = frozenset(
    {
        COLUMN_ID,
        COLUMN_TYPE,
        METADATA_CURRENT_ID,
    }
)
"""Columns required for FOLDER format navigation."""

NAVIGATION_COLUMNS_TACOCAT = frozenset(
    {
        COLUMN_ID,
        COLUMN_TYPE,
        METADATA_CURRENT_ID,
        METADATA_OFFSET,
        METADATA_SIZE,
        METADATA_SOURCE_FILE,
    }
)
"""Columns required for TacoCat format navigation."""

NAVIGATION_COLUMNS_BY_FORMAT = {
    "zip": NAVIGATION_COLUMNS_ZIP,
    "folder": NAVIGATION_COLUMNS_FOLDER,
    "tacocat": NAVIGATION_COLUMNS_TACOCAT,
}
"""Mapping of format type to required navigation columns."""

NAVIGATION_COLUMN_DESCRIPTIONS = {
    COLUMN_ID: "Unique sample identifier",
    COLUMN_TYPE: "Sample type (FILE or FOLDER)",
    METADATA_CURRENT_ID: "Row position (int64) for parent-child JOIN relationships",
    METADATA_OFFSET: "Byte offset in archive file",
    METADATA_SIZE: "Byte size of sample data",
    METADATA_SOURCE_FILE: "Source ZIP filename within TacoCat",
}
"""Human-readable descriptions for navigation columns."""


# DataFrame Limits
DATAFRAME_DEFAULT_HEAD_ROWS = 5
"""Default number of rows for .head()"""

DATAFRAME_DEFAULT_TAIL_ROWS = 5
"""Default number of rows for .tail()"""

DATAFRAME_MAX_REPR_ROWS = 100
"""Maximum rows to display in TacoDataFrame.__repr__()"""
