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