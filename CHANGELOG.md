## [2.3.2] - 2025-01-03

### Fixed
- `concat()` now correctly reads columns from raw tables instead of computed views
- TacoCat concat VSI paths now include `source_file` for correct file resolution

### Added
- `GENERATED_COLUMNS` constant documenting runtime-computed columns (`gdal_vsi`, `source_path`)