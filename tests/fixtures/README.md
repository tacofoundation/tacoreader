# Test Fixtures

Pre-generated TACO datasets for tacoreader tests.

## Structure

```
fixtures/
├── README.md
├── regenerate.py
│
├── zip/
│   ├── flat/flat.tacozip
│   ├── nested/nested.tacozip
│   ├── deep/
│   │   ├── deep_part0001.tacozip
│   │   ├── deep_part0002.tacozip
│   │   └── .tacocat/
│   └── cascade/
│       ├── cascade_part0001.tacozip
│       ├── cascade_part0002.tacozip
│       ├── ...
│       └── .tacocat/
│
└── folder/
    ├── flat/
    ├── nested/
    └── deep/
```

## Fixtures

| Name | Format | Levels | Purpose |
|------|--------|--------|---------|
| flat | ZIP/FOLDER | 1 | 5 FILEs, basic load test |
| nested | ZIP/FOLDER | 2 | 3 FOLDERs x 3 FILEs, navigation with .read() |
| deep | ZIP/FOLDER | 4 | 20 tiles, sensors, bands. TacoCat consolidation |
| cascade | ZIP only | 3 | 4 continents x 2 countries x 2 cities. Geometry at ALL levels for cascade JOIN testing |

## STAC Metadata

All fixtures include:
- `istac:geometry` - WKB polygon for filter_bbox()
- `istac:centroid` - WKB point
- `istac:time_start` - ISO date for filter_datetime()
- `cloud_cover` - float 0-100
- `stac:tensor_shape` - list for stats aggregation
- `geotiff:stats` - list[list[float32]] (flat, nested items, deep bands)

## Regenerating

```bash
python regenerate.py
```

Requires `tacotoolbox`. Run when:
- tacotoolbox updates the format
- Adding new test cases

## Notes

- TacoCat consolidation only for ZIP format (deep, cascade)
- Folder format ignores split_size parameter
- All fixtures follow RSUT (Root-Sibling Uniform Tree) constraints
- cascade fixture designed specifically for testing multi-level JOINs in _stac_cascade.py