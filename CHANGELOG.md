# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.4] - 2025-01-05

### Fixed
- **TacoCat concat VSI paths**: Fixed incorrect GDAL VSI path construction when concatenating TacoCat datasets. Previously, `_root_path` included `.tacocat/` folder which caused invalid paths like `/vsisubfile/...,/bucket/.tacocat/file.tacozip`. Now uses `_vsi_base_path` which correctly points to parent directory containing `.tacozip` files.

### Changed
- **Internal**: Renamed `_root_path` to `_vsi_base_path` across all storage backends. Each backend now calculates the correct VSI base path once during `load()`, eliminating redundant transformations in `setup_duckdb_views()`.

### Technical Details
- `dataset.py`: `_root_path` → `_vsi_base_path`
- `storage/base.py`: `_finalize_dataset()` parameter renamed
- `storage/tacocat.py`: Calls `_extract_base_path()` before `_finalize_dataset()`, not inside `setup_duckdb_views()`
- `concat/_view_builder.py`: Uses `ds._vsi_base_path` for `METADATA_SOURCE_PATH`
- `concat/_orchestrator.py`: Updated attribute name
- `loader.py`: Updated attribute name for `base_path` override

## [2.4.3] - 2025-01-05

### Added

- **Disk cache for remote TacoCat datasets**: Remote `.tacocat` metadata is now cached locally for faster subsequent loads. Cache uses ETag validation to detect remote changes.
  - Cache location: `~/.cache/tacoreader/tacocat/` (Linux), `~/Library/Caches/tacoreader/tacocat/` (macOS), `%LOCALAPPDATA%\tacoreader\Cache\tacocat\` (Windows)
  - Override with `TACOREADER_CACHE_DIR` environment variable
  - Disable per-load with `tacoreader.load(url, cache=False)`

- New `_cache.py` module with cache management functions:
  - `get_cache_dir()` / `get_tacocat_cache_dir()` - platform-specific cache paths
  - `load_from_cache()` / `save_to_cache()` - cache I/O
  - `is_cache_valid()` - ETag/size validation
  - `clear_tacocat_cache()` - clear disk cache
  - `get_cache_stats()` - cache statistics

### Changed

- `tacoreader.load()` now accepts `cache: bool = True` parameter for TacoCat datasets
- `tacoreader.clear_cache()` now clears both in-memory LRU caches and disk cache

### Dependencies

- Added `platformdirs>=3.0` for cross-platform cache directory resolution

## [2.4.2] - 2025-01-04

### Added

- **Progress bars for multi-dataset operations** - Added tqdm progress bars when loading or concatenating 3+ datasets. Shows progress per-level during concat and per-path during multi-load.

## [2.4.1] - 2025-01-04

### Fixed

- **concat: VSI paths for remote datasets** - Fixed `internal:gdal_vsi` construction when concatenating remote datasets (HTTP/S3/GCS). Previously used raw URL (`https://...`) instead of VSI path (`/vsicurl/https://...`), causing GDAL to fail reading concatenated remote data. Changed `ds._path` → `ds._root_path` in `concat/_view_builder.py`.

### Changed

- **Replaced warnings with debug logging** - Removed verbose `warnings.warn()` calls throughout the codebase. Column compatibility messages, stats weight warnings, and parameter validation now use `logger.debug()` for cleaner output. Affected modules:
  - `concat/_columns.py` - column drop/fill messages
  - `dataframe/_stats.py` - weight column and percentile approximation messages  
  - `dataset.py` - stats parameter validation messages

## [2.4.0] - 2025-01-04

### Added
- `load()` now accepts `pathlib.Path` objects in addition to strings
  - Single paths: `tacoreader.load(Path("data.tacozip"))`
  - Lists: `tacoreader.load([Path("a.tacozip"), Path("b.tacozip")])`
  - Mixed: `tacoreader.load([str_path, Path_path])`
  - `base_path` parameter also accepts `Path` objects
- New test class `TestLoadPathTypes` with full coverage for Path support

### Fixed
- `AttributeError: 'PosixPath' object has no attribute 'endswith'` when passing Path objects to `load()`

## [2.3.3] - 2025-01-03

### Added
- `concat()` now respects pre-filtered datasets (filter → concat workflow)
  - Level0 reads from filtered view (`ds._view_name`) instead of raw table
  - Level1+ automatically filters children by valid `parent_ids` from previous level
  - Enables different filters per dataset before concatenation
- `METADATA_CURRENT_ID` constant for parent-child relationship tracking

### Fixed
- Filter propagation in hierarchical datasets during concat operations

## [2.3.2] - 2025-01-03

### Fixed
- `concat()` now correctly reads columns from raw tables instead of computed views
- TacoCat concat VSI paths now include `source_file` for correct file resolution

### Added
- `GENERATED_COLUMNS` constant documenting runtime-computed columns (`gdal_vsi`, `source_path`)