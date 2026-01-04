# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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