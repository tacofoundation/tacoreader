"""
Test fixtures for TACO datasets with STAC metadata.

Three complexity levels:
- flat: Single level, all FILEs
- nested: Two levels, simple hierarchy  
- deep: Four levels, mixed FILE/FOLDER

Output structure:

    tests/fixtures/
    ├── zip/
    │   ├── flat.tacozip           # 5 samples, 1 level
    │   ├── nested.tacozip         # 9 samples, 2 levels (3 groups x 3 items)
    │   └── deep/                  # 4 levels, split + consolidated
    │       ├── deep_part0001.tacozip
    │       ├── deep_part0002.tacozip
    │       └── deep.tacocat/
    │
    └── folder/
        ├── flat/                  # 5 samples, 1 level
        ├── nested/                # 9 samples, 2 levels
        └── deep/                  # 4 levels

STAC metadata fields:
    - istac:geometry     WKB polygon for filter_bbox()
    - istac:centroid     WKB point
    - istac:time_start   ISO date for filter_datetime()
    - cloud_cover        float 0-100 for SQL WHERE
    - stac:tensor_shape  list for stats aggregation

Note: TacoCat consolidation only applies to ZIP format.
"""

import pathlib
import struct
from datetime import datetime, timedelta

import tacotoolbox
from tacotoolbox.datamodel import Sample, Tortilla, Taco


COLLECTION_DEFAULTS = {
    "id": "test_dataset",
    "dataset_version": "1.0.0",
    "description": "Test fixture with STAC metadata",
    "licenses": ["CC-BY-4.0"],
    "providers": [{"name": "Test", "roles": ["producer"]}],
    "tasks": ["classification"],
}

# Locations for bbox/centroid testing
LOCATIONS = {
    "valencia": (-0.5, 39.3, -0.2, 39.6),
    "paris": (2.2, 48.7, 2.5, 49.0),
    "berlin": (13.2, 52.3, 13.6, 52.7),
    "nyc": (-74.2, 40.5, -73.8, 40.9),
    "tokyo": (139.5, 35.5, 139.9, 35.9),
    "lima": (-77.2, -12.2, -76.8, -11.8),
    "sydney": (150.9, -34.1, 151.5, -33.6),
    "shanghai": (121.2, 31.0, 121.7, 31.5),
    "rome": (12.3, 41.7, 12.7, 42.1),
}

BASE_DATE = datetime(2023, 1, 1)


def _point_wkb(lon: float, lat: float) -> bytes:
    """WKB Point (little-endian)."""
    return struct.pack('<bIdd', 1, 1, lon, lat)


def _polygon_wkb(minx: float, miny: float, maxx: float, maxy: float) -> bytes:
    """WKB Polygon from bbox (little-endian, closed ring)."""
    wkb = struct.pack('<bII', 1, 3, 1)  # polygon, 1 ring
    wkb += struct.pack('<I', 5)  # 5 points (closed)
    for x, y in [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]:
        wkb += struct.pack('<dd', x, y)
    return wkb


def _date(days: int) -> str:
    """ISO date offset from BASE_DATE."""
    return (BASE_DATE + timedelta(days=days)).strftime("%Y-%m-%d")


def _centroid(bbox: tuple) -> tuple[float, float]:
    """Center point of bbox."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def create_flat_taco(output: pathlib.Path) -> pathlib.Path:
    """
    5 FILEs with varied locations and times.
    
    Locations: valencia, paris, nyc, tokyo, lima
    Times: monthly from 2023-01-01
    Cloud: 0, 15, 30, 45, 60
    """
    locs = ["valencia", "paris", "nyc", "tokyo", "lima"]
    
    samples = []
    for i, name in enumerate(locs):
        bbox = LOCATIONS[name]
        cx, cy = _centroid(bbox)
        
        samples.append(Sample(
            id=f"sample_{i}",
            path=f"content_{name}".encode(),
            properties={
                "istac:geometry": _polygon_wkb(*bbox),
                "istac:centroid": _point_wkb(cx, cy),
                "istac:time_start": _date(i * 30),
                "cloud_cover": i * 15.0,
                "location": name,
                "stac:tensor_shape": [3, 256, 256],
            },
        ))
    
    taco = Taco(tortilla=Tortilla(samples), **COLLECTION_DEFAULTS)
    return tacotoolbox.create(taco, output)[0]


def create_nested_taco(output: pathlib.Path) -> pathlib.Path:
    """
    3 regional groups x 3 items each.
    
    europe/   -> valencia, paris, berlin
    americas/ -> nyc, lima, (nyc again for simplicity)
    asia/     -> tokyo, shanghai, sydney
    """
    regions = {
        "europe": ["valencia", "paris", "berlin"],
        "americas": ["nyc", "lima", "nyc"],
        "asia": ["tokyo", "shanghai", "sydney"],
    }
    
    groups = []
    for region, locs in regions.items():
        items = []
        for j, loc in enumerate(locs):
            bbox = LOCATIONS[loc]
            cx, cy = _centroid(bbox)
            
            items.append(Sample(
                id=f"item_{j}",
                path=f"data_{region}_{j}".encode(),
                properties={
                    "istac:geometry": _polygon_wkb(*bbox),
                    "istac:centroid": _point_wkb(cx, cy),
                    "istac:time_start": _date(j * 30),
                    "cloud_cover": j * 20.0 + 5,
                    "location": loc,
                    "stac:tensor_shape": [4, 512, 512],
                },
            ))
        
        # Group bbox = union of children
        all_bboxes = [LOCATIONS[l] for l in locs]
        group_bbox = (
            min(b[0] for b in all_bboxes),
            min(b[1] for b in all_bboxes),
            max(b[2] for b in all_bboxes),
            max(b[3] for b in all_bboxes),
        )
        
        groups.append(Sample(
            id=region,
            path=Tortilla(items),
            properties={
                "istac:geometry": _polygon_wkb(*group_bbox),
                "istac:time_start": _date(0),
                "region": region,
            },
        ))
    
    taco = Taco(tortilla=Tortilla(groups), **COLLECTION_DEFAULTS)
    return tacotoolbox.create(taco, output)[0]


def create_deep_taco(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    4-level satellite-like structure, 20 tiles.
    
    tile_N/
        sensor_A/
            band_R (FILE)
            band_G (FILE)
            band_B (FILE)
        sensor_B/
            band_R (FILE)
            band_G (FILE)
            band_B (FILE)
    
    Simplified structure - all FILEs at level 2 with consistent schema.
    """
    loc_names = list(LOCATIONS.keys())
    
    def make_sensor(name: str, idx: int) -> Sample:
        bands = []
        for band in ["band_R", "band_G", "band_B"]:
            bands.append(Sample(
                id=band,
                path=f"{band}_data".encode(),
                properties={
                    "wavelength_nm": {"band_R": 665, "band_G": 560, "band_B": 490}[band],
                    "stac:tensor_shape": [1, 256, 256],
                },
            ))
        
        return Sample(
            id=name,
            path=Tortilla(bands),
            properties={"resolution_m": 10 if name == "sensor_A" else 20},
        )
    
    def make_tile(idx: int) -> Sample:
        loc = loc_names[idx % len(loc_names)]
        bbox = LOCATIONS[loc]
        cx, cy = _centroid(bbox)
        
        return Sample(
            id=f"tile_{idx:02d}",
            path=Tortilla([
                make_sensor("sensor_A", idx),
                make_sensor("sensor_B", idx),
            ]),
            properties={
                "istac:geometry": _polygon_wkb(*bbox),
                "istac:centroid": _point_wkb(cx, cy),
                "istac:time_start": _date(idx),
                "cloud_cover": (idx * 13) % 100,
                "location": loc,
            },
        )
    
    tiles = [make_tile(i) for i in range(20)]
    taco = Taco(tortilla=Tortilla(tiles), **COLLECTION_DEFAULTS)
    
    return tacotoolbox.create(taco, output_dir, split_size="1KB", consolidate=True)


if __name__ == "__main__":
    import shutil
    
    fixtures = pathlib.Path("tests/fixtures")
    zip_dir = fixtures / "zip"
    folder_dir = fixtures / "folder"
    
    for d in [zip_dir, folder_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    
    # ZIP
    create_flat_taco(zip_dir / "flat/flat.tacozip")
    create_nested_taco(zip_dir / "nested/nested.tacozip")
    create_deep_taco(zip_dir / "deep/deep.tacozip")
    
    # FOLDER
    create_flat_taco(folder_dir / "flat")
    create_nested_taco(folder_dir / "nested")
    create_deep_taco(folder_dir / "deep")
    
    print(f"Fixtures: {fixtures}")