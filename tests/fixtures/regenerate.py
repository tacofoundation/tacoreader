"""
Test fixtures for TACO datasets with STAC metadata and geotiff:stats.

Three complexity levels:
- flat: Single level, all FILEs + stats
- nested: Two levels, simple hierarchy + stats on items
- deep: Four levels, mixed FILE/FOLDER + stats on band FILEs

Output structure:

    tests/fixtures/
    ├── zip/
    │   ├── flat.tacozip           # 5 samples, 1 level + stats
    │   ├── nested.tacozip         # 9 samples, 2 levels + stats on items
    │   └── deep/                  # 4 levels, split + consolidated + stats on bands
    │       ├── deep_part0001.tacozip
    │       ├── deep_part0002.tacozip
    │       └── deep.tacocat/
    │
    └── folder/
        ├── flat/                  # 5 samples, 1 level + stats
        ├── nested/                # 9 samples, 2 levels + stats on items
        └── deep/                  # 4 levels + stats on bands

STAC metadata fields:
    - istac:geometry     WKB polygon for filter_bbox()
    - istac:centroid     WKB point
    - istac:time_start   ISO date for filter_datetime()
    - cloud_cover        float 0-100 for SQL WHERE
    - stac:tensor_shape  list for stats aggregation
    - geotiff:stats      list[list[float32]] - categorical or continuous stats

Note: TacoCat consolidation only applies to ZIP format.
"""

import pathlib
import struct
from datetime import datetime, timedelta

import pyarrow as pa
import tacotoolbox
from tacotoolbox.datamodel import Sample, Tortilla, Taco
from tacotoolbox.sample.datamodel import SampleExtension


class FakeGeotiffStats(SampleExtension):
    """Fake stats extension for testing (no GDAL required)."""
    
    stats: list[list[float]]
    
    def get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("geotiff:stats", pa.list_(pa.list_(pa.float32()))),
        ])
    
    def get_field_descriptions(self) -> dict[str, str]:
        return {
            "geotiff:stats": "Fake statistics for testing"
        }
    
    def _compute(self, sample) -> pa.Table:
        data = {"geotiff:stats": [self.stats]}
        return pa.Table.from_pydict(data, schema=self.get_schema())


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


def _date(days: int) -> datetime:
    """Datetime offset from BASE_DATE."""
    return BASE_DATE + timedelta(days=days)


def _centroid(bbox: tuple) -> tuple[float, float]:
    """Center point of bbox."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def create_flat_taco(output: pathlib.Path) -> pathlib.Path:
    """
    5 FILEs with varied locations and times + geotiff:stats.
    
    Locations: valencia, paris, nyc, tokyo, lima
    Times: monthly from 2023-01-01
    Cloud: 0, 15, 30, 45, 60
    Stats: ALL continuous (3 bands x 9 values each)
    """
    locs = ["valencia", "paris", "nyc", "tokyo", "lima"]
    
    # Continuous: [min, max, mean, std, valid%, p25, p50, p75, p95]
    # All samples must have same shape for numpy aggregation
    continuous_stats = [
        [
            [0.0, 255.0, 128.5, 45.2, 98.5, 85.0, 125.0, 170.0, 220.0],
            [0.0, 255.0, 115.3, 38.7, 99.1, 75.0, 110.0, 155.0, 205.0],
            [0.0, 255.0, 95.8, 42.1, 97.8, 60.0, 92.0, 130.0, 185.0],
        ],
        [
            [5.0, 250.0, 135.2, 48.1, 97.2, 90.0, 132.0, 178.0, 225.0],
            [3.0, 248.0, 120.8, 41.5, 98.5, 80.0, 118.0, 162.0, 210.0],
            [2.0, 252.0, 102.3, 44.8, 96.9, 65.0, 100.0, 140.0, 192.0],
        ],
        [
            [10.0, 245.0, 142.0, 42.8, 99.0, 95.0, 138.0, 182.0, 218.0],
            [8.0, 240.0, 125.5, 39.2, 98.8, 82.0, 122.0, 165.0, 208.0],
            [5.0, 248.0, 108.7, 46.3, 97.5, 70.0, 105.0, 145.0, 195.0],
        ],
        [
            [0.0, 255.0, 150.8, 50.5, 96.8, 100.0, 148.0, 195.0, 235.0],
            [2.0, 252.0, 132.1, 43.8, 97.9, 88.0, 130.0, 175.0, 220.0],
            [1.0, 250.0, 112.5, 48.2, 95.5, 72.0, 110.0, 152.0, 200.0],
        ],
        [
            [3.0, 248.0, 138.3, 46.7, 98.2, 92.0, 135.0, 180.0, 222.0],
            [5.0, 245.0, 118.9, 40.1, 99.3, 78.0, 115.0, 158.0, 212.0],
            [0.0, 255.0, 99.2, 43.5, 97.1, 62.0, 96.0, 138.0, 188.0],
        ],
    ]
    
    samples = []
    for i, name in enumerate(locs):
        bbox = LOCATIONS[name]
        cx, cy = _centroid(bbox)
        
        stats = continuous_stats[i]
        
        sample = Sample(
            id=f"sample_{i}",
            path=f"content_{name}".encode(),
        )
        
        # Add metadata as direct fields (not in properties struct)
        sample.extend_with({
            "istac:geometry": _polygon_wkb(*bbox),
            "istac:centroid": _point_wkb(cx, cy),
            "istac:time_start": _date(i * 30),
            "cloud_cover": i * 15.0,
            "location": name,
            "stac:tensor_shape": [3, 256, 256],
        })
        
        # Add fake stats extension
        sample.extend_with(FakeGeotiffStats(stats=stats))
        samples.append(sample)
    
    taco = Taco(tortilla=Tortilla(samples), **COLLECTION_DEFAULTS)
    return tacotoolbox.create(taco, output)[0]


def create_nested_taco(output: pathlib.Path) -> pathlib.Path:
    """
    3 regional groups x 3 items each + geotiff:stats on items.
    
    europe/   -> valencia, paris, berlin (categorical, 3 bands each)
    americas/ -> nyc, lima, nyc (continuous, 3 bands each)
    asia/     -> tokyo, shanghai, sydney (mixed cat/cont/cat, 3 bands each)
    """
    regions = {
        "europe": ["valencia", "paris", "berlin"],
        "americas": ["nyc", "lima", "nyc"],
        "asia": ["tokyo", "shanghai", "sydney"],
    }
    
    # Different stats patterns per region
    stats_patterns = {
        "europe": [  # Categorical for all 3
            [[0.4, 0.4, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]],
            [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.4, 0.5]],
            [[0.3, 0.3, 0.4], [0.4, 0.4, 0.2], [0.2, 0.5, 0.3]],
        ],
        "americas": [  # Continuous for all 3 (3 bands each)
            [
                [0.0, 200.0, 98.5, 35.2, 98.1, 65.0, 95.0, 130.0, 175.0],
                [5.0, 180.0, 85.3, 28.5, 97.5, 55.0, 82.0, 115.0, 160.0],
                [10.0, 220.0, 115.8, 42.1, 99.0, 75.0, 110.0, 150.0, 200.0],
            ],
            [
                [-1.0, 1.0, 0.25, 0.38, 96.3, -0.20, 0.22, 0.55, 0.82],
                [-0.8, 0.95, 0.18, 0.32, 95.8, -0.15, 0.16, 0.48, 0.75],
                [-0.9, 0.98, 0.22, 0.35, 97.1, -0.18, 0.20, 0.52, 0.78],
            ],
            [
                [10.0, 250.0, 145.8, 48.6, 97.5, 95.0, 140.0, 195.0, 230.0],
                [15.0, 240.0, 138.2, 45.3, 98.2, 88.0, 135.0, 188.0, 220.0],
                [12.0, 255.0, 150.5, 50.1, 96.8, 98.0, 145.0, 200.0, 235.0],
            ],
        ],
        "asia": [  # Mixed: cat, cont, cat (3 bands each)
            [[0.2, 0.5, 0.3], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3]],
            [
                [0.0, 100.0, 52.3, 28.5, 99.0, 28.0, 50.0, 75.0, 92.0],
                [5.0, 95.0, 48.7, 25.2, 98.5, 25.0, 47.0, 70.0, 88.0],
                [2.0, 105.0, 55.8, 30.1, 99.5, 30.0, 53.0, 78.0, 95.0],
            ],
            [[0.6, 0.2, 0.2], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]],
        ],
    }
    
    groups = []
    for region, locs in regions.items():
        items = []
        for j, loc in enumerate(locs):
            bbox = LOCATIONS[loc]
            cx, cy = _centroid(bbox)
            
            sample = Sample(
                id=f"item_{j}",
                path=f"data_{region}_{j}".encode(),
            )
            
            # Add metadata as direct fields
            sample.extend_with({
                "istac:geometry": _polygon_wkb(*bbox),
                "istac:centroid": _point_wkb(cx, cy),
                "istac:time_start": _date(j * 30),
                "cloud_cover": j * 20.0 + 5,
                "location": loc,
                "stac:tensor_shape": [3, 512, 512],
            })
            
            # Add stats from pattern
            sample.extend_with(FakeGeotiffStats(stats=stats_patterns[region][j]))
            items.append(sample)
        
        # Group bbox = union of children (NO stats at parent level)
        all_bboxes = [LOCATIONS[l] for l in locs]
        group_bbox = (
            min(b[0] for b in all_bboxes),
            min(b[1] for b in all_bboxes),
            max(b[2] for b in all_bboxes),
            max(b[3] for b in all_bboxes),
        )
        
        group = Sample(
            id=region,
            path=Tortilla(items),
        )
        
        group.extend_with({
            "istac:geometry": _polygon_wkb(*group_bbox),
            "istac:time_start": _date(0),
            "region": region,
        })
        
        groups.append(group)
    
    taco = Taco(tortilla=Tortilla(groups), **COLLECTION_DEFAULTS)
    return tacotoolbox.create(taco, output)[0]


def create_deep_taco(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    4-level satellite-like structure, 20 tiles + geotiff:stats on band FILEs.
    
    tile_N/
        sensor_A/
            band_R (FILE + stats)
            band_G (FILE + stats)
            band_B (FILE + stats)
        sensor_B/
            band_R (FILE + stats)
            band_G (FILE + stats)
            band_B (FILE + stats)
    """
    loc_names = list(LOCATIONS.keys())
    
    # Stats for RGB bands (continuous mode, single band each)
    # [min, max, mean, std, valid%, p25, p50, p75, p95]
    band_stats = {
        "band_R": [[0.0, 255.0, 145.2, 48.3, 98.7, 95.0, 140.0, 185.0, 230.0]],
        "band_G": [[0.0, 255.0, 128.5, 42.1, 99.2, 85.0, 125.0, 170.0, 215.0]],
        "band_B": [[0.0, 255.0, 98.3, 38.6, 97.5, 60.0, 95.0, 135.0, 180.0]],
    }
    
    def make_sensor(name: str, idx: int) -> Sample:
        bands = []
        for band in ["band_R", "band_G", "band_B"]:
            sample = Sample(
                id=band,
                path=f"{band}_data".encode(),
            )
            
            sample.extend_with({
                "wavelength_nm": {"band_R": 665, "band_G": 560, "band_B": 490}[band],
                "stac:tensor_shape": [1, 256, 256],
            })
            
            # Add stats to band FILE
            sample.extend_with(FakeGeotiffStats(stats=band_stats[band]))
            bands.append(sample)
        
        sensor = Sample(
            id=name,
            path=Tortilla(bands),
        )
        
        sensor.extend_with({
            "resolution_m": 10 if name == "sensor_A" else 20,
        })
        
        return sensor
    
    def make_tile(idx: int) -> Sample:
        loc = loc_names[idx % len(loc_names)]
        bbox = LOCATIONS[loc]
        cx, cy = _centroid(bbox)
        
        tile = Sample(
            id=f"tile_{idx:02d}",
            path=Tortilla([
                make_sensor("sensor_A", idx),
                make_sensor("sensor_B", idx),
            ]),
        )
        
        tile.extend_with({
            "istac:geometry": _polygon_wkb(*bbox),
            "istac:centroid": _point_wkb(cx, cy),
            "istac:time_start": _date(idx),
            "cloud_cover": (idx * 13) % 100,
            "location": loc,
        })
        
        return tile
    
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
    
    print(f"Fixtures generated: {fixtures}")