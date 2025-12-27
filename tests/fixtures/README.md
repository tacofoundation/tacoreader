# Test Fixtures

Pre-generated TACO datasets for tacoreader tests.

## Structure

```
fixtures/
├── README.md
├── regenerate.py          # run to regenerate all fixtures
│
├── zip/
│   ├── flat/flat.tacozip       # 5 FILEs, 1 level
│   ├── nested/nested.tacozip   # 3 FOLDERs x 3 FILEs, 2 levels
│   └── deep/                   # 4 levels, mixed FILE/FOLDER
│       ├── deep_part0001.tacozip
│       ├── deep_part0002.tacozip
│       └── deep.tacocat/
│
└── folder/
    ├── flat/              # 5 FILEs, 1 level
    ├── nested/            # 3 FOLDERs x 3 FILEs, 2 levels
    └── deep/              # 4 levels, mixed FILE/FOLDER
```

## Fixtures

| Name | Format | Samples | Levels | Purpose |
|------|--------|---------|--------|---------|
| flat | ZIP/FOLDER | 5 FILE | 1 | Basic load test |
| nested | ZIP/FOLDER | 9 FILE | 2 | Navigation with .read() |
| deep | ZIP/FOLDER | 16+ FILE | 4 | Mixed hierarchy, TacoCat |

## Regenerating

```bash
python regenerate.py
```

Requires `tacotoolbox` installed. Run this when:
- tacotoolbox updates the format
- Adding new test cases

## Notes

- TacoCat consolidation only applies to ZIP (deep case)
- Folder format ignores split_size parameter
- All fixtures follow RSUT (Root-Sibling Uniform Tree) constraints