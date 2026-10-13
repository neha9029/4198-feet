"""Optional AWS Terrain Tiles relief modulation for the 25k production mesh."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy import ndimage

from data import DATA_DIR, MEASUREMENT_CRS, SEED
from mesh import MESH_PATH


TERRAIN_ZOOM = 11
TERRAIN_TILE_SIZE = 256
TERRAIN_URL = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)
TERRAIN_REGISTRY_URL = "https://registry.opendata.aws/terrain-tiles/"
TERRAIN_FORMAT_URL = (
    "https://github.com/tilezen/joerd/blob/master/docs/formats.md"
)
TERRAIN_ATTRIBUTION_URL = (
    "https://github.com/tilezen/joerd/blob/master/docs/attribution.md"
)
TERRAIN_ACCESS_DATE = "2026-10-13"
SMOOTHING_METRES = 300.0
SUN_AZIMUTH_DEGREES = 315.0
SUN_ALTITUDE_DEGREES = 45.0
RELIEF_AMPLITUDE = 0.08
RAW_DIR = DATA_DIR / "raw" / "terrain_tiles" / f"z{TERRAIN_ZOOM}"
DEM_PATH = DATA_DIR / "terrain_dem_utm12.tif"
RELIEF_RASTER_PATH = DATA_DIR / "terrain_relief_factor_utm12.tif"
RELIEF_FACTORS_PATH = DATA_DIR / "triangle_relief_factors.npy"
REPORT_PATH = DATA_DIR.parent / "out" / "phase2" / "relief_report.json"


def _tile_xy(longitude: float, latitude: float, zoom: int) -> tuple[int, int]:
    count = 2**zoom
    x = int(math.floor((longitude + 180.0) / 360.0 * count))
    latitude_rad = math.radians(latitude)
    y = int(
        math.floor(
            (1.0 - math.asinh(math.tan(latitude_rad)) / math.pi)
            / 2.0
            * count
        )
    )
    return x, y


def _download_tile(tile: tuple[int, int]) -> tuple[int, int, Path, str | None]:
    x, y = tile
    path = RAW_DIR / str(x) / f"{y}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    imagery_source = None
    if not path.exists():
        url = TERRAIN_URL.format(z=TERRAIN_ZOOM, x=x, y=y)
        with urlopen(url, timeout=60) as response:
            payload = response.read()
            imagery_source = response.headers.get("x-amz-meta-x-imagery-sources")
        Image.open(io.BytesIO(payload)).verify()
        path.write_bytes(payload)
    return x, y, path, imagery_source


def _terrarium_height(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0


def _frame_bounds() -> tuple[tuple[float, float, float, float], rasterio.Affine, tuple[int, int]]:
    mesh = np.load(MESH_PATH)
    row0, row1, column0, column1 = (int(value) for value in mesh["crop_window"])
    frame_height, frame_width = (int(value) for value in mesh["raster_shape"])
    with rasterio.open(DATA_DIR / "gsl_classes_utm12.tif") as dataset:
        transform = dataset.transform
        left = transform.c + column0 * transform.a
        right = transform.c + column1 * transform.a
        top = transform.f + row0 * transform.e
        bottom = transform.f + row1 * transform.e
        frame_transform = from_origin(left, top, abs(transform.a), abs(transform.e))
    return (left, bottom, right, top), frame_transform, (frame_height, frame_width)


def _hillshade(elevation: np.ndarray, pixel_size_m: float) -> tuple[np.ndarray, dict]:
    sigma = SMOOTHING_METRES / pixel_size_m
    smoothed = ndimage.gaussian_filter(elevation, sigma=sigma, mode="nearest")
    dz_row, dz_east = np.gradient(smoothed, pixel_size_m, pixel_size_m)
    dz_north = -dz_row
    normal = np.stack((-dz_east, -dz_north, np.ones_like(smoothed)), axis=0)
    normal /= np.linalg.norm(normal, axis=0, keepdims=True)
    azimuth = math.radians(SUN_AZIMUTH_DEGREES)
    altitude = math.radians(SUN_ALTITUDE_DEGREES)
    sun = np.array(
        [
            math.cos(altitude) * math.sin(azimuth),
            math.cos(altitude) * math.cos(azimuth),
            math.sin(altitude),
        ]
    )
    raw = np.sum(normal * sun[:, None, None], axis=0)
    low, high = np.percentile(raw[np.isfinite(raw)], (2.0, 98.0))
    normalised = np.clip(2.0 * (raw - low) / max(high - low, 1e-9) - 1.0, -1.0, 1.0)
    factor = 1.0 + RELIEF_AMPLITUDE * normalised
    return factor.astype(np.float32), {
        "gaussian_sigma_pixels": float(sigma),
        "raw_p02_p98": [float(low), float(high)],
        "factor_min_max": [float(factor.min()), float(factor.max())],
        "factor_median": float(np.median(factor)),
    }


def build_relief(seed: int = SEED) -> dict:
    """Fetch Terrarium tiles and derive one factor per production triangle."""
    del seed
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bounds_utm, frame_transform, frame_shape = _frame_bounds()
    bounds_wgs84 = transform_bounds(
        MEASUREMENT_CRS, "EPSG:4326", *bounds_utm, densify_pts=21
    )
    west, south, east, north = bounds_wgs84
    x0, y0 = _tile_xy(west, north, TERRAIN_ZOOM)
    x1, y1 = _tile_xy(east, south, TERRAIN_ZOOM)
    tiles = [
        (x, y)
        for y in range(min(y0, y1), max(y0, y1) + 1)
        for x in range(min(x0, x1), max(x0, x1) + 1)
    ]
    with ThreadPoolExecutor(max_workers=12) as executor:
        downloaded = list(executor.map(_download_tile, tiles))

    x_values = sorted({item[0] for item in downloaded})
    y_values = sorted({item[1] for item in downloaded})
    mosaic = np.empty(
        (len(y_values) * TERRAIN_TILE_SIZE, len(x_values) * TERRAIN_TILE_SIZE),
        dtype=np.float32,
    )
    x_lookup = {value: index for index, value in enumerate(x_values)}
    y_lookup = {value: index for index, value in enumerate(y_values)}
    imagery_sources = set()
    for x, y, path, imagery_source in downloaded:
        row = y_lookup[y] * TERRAIN_TILE_SIZE
        column = x_lookup[x] * TERRAIN_TILE_SIZE
        mosaic[
            row : row + TERRAIN_TILE_SIZE,
            column : column + TERRAIN_TILE_SIZE,
        ] = _terrarium_height(path)
        if imagery_source:
            imagery_sources.add(imagery_source)

    web_mercator_half_world = 20037508.342789244
    tile_span = 2.0 * web_mercator_half_world / (2**TERRAIN_ZOOM)
    pixel_size_3857 = tile_span / TERRAIN_TILE_SIZE
    source_left = -web_mercator_half_world + x_values[0] * tile_span
    source_top = web_mercator_half_world - y_values[0] * tile_span
    source_transform = from_origin(
        source_left, source_top, pixel_size_3857, pixel_size_3857
    )
    elevation = np.full(frame_shape, np.nan, dtype=np.float32)
    reproject(
        source=mosaic,
        destination=elevation,
        src_transform=source_transform,
        src_crs="EPSG:3857",
        dst_transform=frame_transform,
        dst_crs=MEASUREMENT_CRS,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )
    if not np.isfinite(elevation).all():
        raise RuntimeError("Terrain tile mosaic does not cover the full film frame")

    profile = {
        "driver": "GTiff",
        "height": frame_shape[0],
        "width": frame_shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": MEASUREMENT_CRS,
        "transform": frame_transform,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(DEM_PATH, "w", **profile) as dataset:
        dataset.write(elevation, 1)

    relief, relief_stats = _hillshade(elevation, abs(frame_transform.a))
    with rasterio.open(RELIEF_RASTER_PATH, "w", **profile) as dataset:
        dataset.write(relief, 1)

    mesh = np.load(MESH_PATH)
    centroids = mesh["centroids"]
    columns = np.clip(
        np.rint(centroids[:, 0] * (frame_shape[1] - 1)),
        0,
        frame_shape[1] - 1,
    ).astype(int)
    rows = np.clip(
        np.rint(centroids[:, 1] * (frame_shape[0] - 1)),
        0,
        frame_shape[0] - 1,
    ).astype(int)
    factors = relief[rows, columns].astype(np.float64)
    factors[mesh["classes"] == 2] = 1.0
    np.save(RELIEF_FACTORS_PATH, factors)

    tile_hash = hashlib.sha256()
    for _, _, path, _ in sorted(downloaded):
        tile_hash.update(path.read_bytes())
    report = {
        "enabled": True,
        "source": {
            "name": "AWS Terrain Tiles / Mapzen Terrarium",
            "bucket": "s3://elevation-tiles-prod",
            "url_template": TERRAIN_URL,
            "registry_url": TERRAIN_REGISTRY_URL,
            "format_documentation_url": TERRAIN_FORMAT_URL,
            "attribution_url": TERRAIN_ATTRIBUTION_URL,
            "license": (
                "Underlying United States elevation is public-domain USGS 3DEP/NED; "
                "credit requested: 3DEP data courtesy of the U.S. Geological Survey"
            ),
            "accessed": TERRAIN_ACCESS_DATE,
            "zoom": TERRAIN_ZOOM,
            "tile_x_range": [x_values[0], x_values[-1]],
            "tile_y_range": [y_values[0], y_values[-1]],
            "tile_count": len(downloaded),
            "http_imagery_sources_observed": sorted(imagery_sources),
            "tile_payload_sha256": tile_hash.hexdigest(),
        },
        "frame_bounds_utm12": list(bounds_utm),
        "frame_bounds_wgs84": list(bounds_wgs84),
        "frame_shape": list(frame_shape),
        "dem_elevation_min_max_m": [
            float(np.min(elevation)),
            float(np.max(elevation)),
        ],
        "hillshade": {
            "sun_azimuth_degrees": SUN_AZIMUTH_DEGREES,
            "sun_altitude_degrees": SUN_ALTITUDE_DEGREES,
            "smoothing_metres": SMOOTHING_METRES,
            "modulation_amplitude": RELIEF_AMPLITUDE,
            **relief_stats,
        },
        "triangle_factors": {
            "count": int(len(factors)),
            "water_untouched": bool(np.all(factors[mesh["classes"] == 2] == 1.0)),
            "land_family_min_max": [
                float(np.min(factors[mesh["classes"] != 2])),
                float(np.max(factors[mesh["classes"] != 2])),
            ],
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_relief(), indent=2))
