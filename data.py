"""Phase 1 data preparation for *4,198 Feet*.

The artwork does not pretend that Landsat occurrence is surveyed elevation.
Instead, this module calibrates that monotonic proxy to two published
elevation-area targets, producing reproducible present and healthy shorelines.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.transform import rowcol
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from scipy.stats import rankdata


SEED = 1931
CURRENT_LEVEL_FT = 4189.6
HEALTHY_LEVEL_FT = 4198.0

GSW_VERSION = "1.5"
GSW_YEARS = "1984–2024"
GSW_TILE = "120W_50N"
GSW_LISTING_URL = "https://s3.waw4-1.cloudferro.com/swift/v1/global-surface-water/"
GSW_BASE_URL = (
    "https://s3.waw4-1.cloudferro.com/swift/v1/global-surface-water/"
    "download2024/Aggregated/VER1-5"
)
GSW_URLS = {
    "occurrence": f"{GSW_BASE_URL}/occurrence/occurrence_{GSW_TILE}_v1_5_2024.tif",
    "extent": f"{GSW_BASE_URL}/extent/extent_{GSW_TILE}_v1_5_2024.tif",
    "transitions": f"{GSW_BASE_URL}/transitions/transitions_{GSW_TILE}_v1_5_2024.tif",
}
GSW_DOWNLOAD_PAGE = "https://global-surface-water.appspot.com/download"
GSW_LICENSE = (
    "Copernicus Programme data: free of charge and without restriction of use; "
    "map attribution: Source: EC JRC/Google."
)

ACCOUNTING_PAGE = "https://gslaccounting.org/great-salt-lake-accounting-data-datasets"
BATHYMETRY_URL = (
    "https://assets.super.so/936849e5-ca7c-48b3-b907-91fc48c4b0f6/files/"
    "136f9cab-844d-45f7-bcfa-d28d500ffed3/bathymetry_casey_root_ngvd29.csv"
)
BATHYMETRY_METADATA_URL = (
    "https://assets.super.so/936849e5-ca7c-48b3-b907-91fc48c4b0f6/files/"
    "3d800dbb-7dd6-48dd-bfeb-081dba4a1622/bathymetry_metadata.csv"
)
VERTICAL_DATUM = "NGVD29"

# A 0.05-degree margin around the requested approximate lake bounds.
CROP_BOUNDS_WGS84 = (-113.20, 40.55, -111.75, 41.80)
MEASUREMENT_CRS = "EPSG:32612"
MEASUREMENT_RESOLUTION_M = 30.0
ACRE_TO_KM2 = 0.0040468564224
NODATA_BYTE = 255
NODATA_FLOAT = -9999.0

# Both arms are seeded so the railroad causeway does not make one disappear.
LAKE_SEEDS_WGS84 = {
    "north_arm": (-112.75, 41.35),
    "south_arm": (-112.45, 40.95),
}
PRIMARY_ARM_SEEDS = ("north_arm", "south_arm")
SEED_SEARCH_RADIUS_M = 7_500.0
CAUSEWAY_LAT = 41.223
CAUSEWAY_REFERENCE_LON = -112.50
# Edge connectivity avoids treating corner-touching 30 m water pixels as a
# hydrologic connection across a dike or causeway.
CONNECTIVITY = ndimage.generate_binary_structure(2, 1)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = ROOT / "out" / "phase1"


def download(url: str, path: Path) -> None:
    """Download a public source only when it is absent from the local cache."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(path)


def ensure_sources() -> dict[str, Path]:
    """Fetch the exact public inputs resolved from the two source pages."""
    sources = {
        layer: RAW_DIR / f"{layer}_{GSW_TILE}_v1_5_2024.tif"
        for layer in GSW_URLS
    }
    sources["bathymetry"] = RAW_DIR / "bathymetry_casey_root_ngvd29.csv"
    sources["bathymetry_metadata"] = RAW_DIR / "bathymetry_metadata.csv"
    for layer, url in GSW_URLS.items():
        download(url, sources[layer])
    download(BATHYMETRY_URL, sources["bathymetry"])
    download(BATHYMETRY_METADATA_URL, sources["bathymetry_metadata"])
    return sources


def sha256(path: Path) -> str:
    """Return a reproducibility checksum without loading a file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_native(source: Path, destination: Path) -> tuple[np.ndarray, dict]:
    """Crop a JRC tile in its native 0.00025-degree, roughly 30 m grid."""
    with rasterio.open(source) as dataset:
        requested = from_bounds(*CROP_BOUNDS_WGS84, transform=dataset.transform)
        full = Window(0, 0, dataset.width, dataset.height)
        window = requested.round_offsets().round_lengths().intersection(full)
        values = dataset.read(1, window=window)
        profile = dataset.profile.copy()
        profile.update(
            width=values.shape[1],
            height=values.shape[0],
            transform=dataset.window_transform(window),
            nodata=NODATA_BYTE,
            compress="deflate",
            tiled=True,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as output:
        output.write(values, 1)
        output.update_tags(
            source=GSW_DOWNLOAD_PAGE,
            version=GSW_VERSION,
            years=GSW_YEARS,
            tile=GSW_TILE,
            license=GSW_LICENSE,
        )
    return values, profile


def measurement_grid(profile: dict) -> tuple[rasterio.Affine, int, int]:
    """Build a north-up UTM 12N grid with exact 30 m square pixels."""
    bounds = rasterio.transform.array_bounds(
        profile["height"], profile["width"], profile["transform"]
    )
    return calculate_default_transform(
        profile["crs"],
        MEASUREMENT_CRS,
        profile["width"],
        profile["height"],
        *bounds,
        resolution=MEASUREMENT_RESOLUTION_M,
    )


def reproject_occurrence(
    values: np.ndarray,
    source_profile: dict,
    destination_transform: rasterio.Affine,
    width: int,
    height: int,
) -> np.ndarray:
    """Reproject continuous occurrence frequency using bilinear interpolation."""
    destination = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=values,
        destination=destination,
        src_transform=source_profile["transform"],
        src_crs=source_profile["crs"],
        src_nodata=NODATA_BYTE,
        dst_transform=destination_transform,
        dst_crs=MEASUREMENT_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return destination


def reproject_categories(
    values: np.ndarray,
    source_profile: dict,
    destination_transform: rasterio.Affine,
    width: int,
    height: int,
) -> np.ndarray:
    """Reproject diagnostic categorical layers without inventing classes."""
    destination = np.full((height, width), NODATA_BYTE, dtype=np.uint8)
    reproject(
        source=values,
        destination=destination,
        src_transform=source_profile["transform"],
        src_crs=source_profile["crs"],
        src_nodata=NODATA_BYTE,
        dst_transform=destination_transform,
        dst_crs=MEASUREMENT_CRS,
        dst_nodata=NODATA_BYTE,
        resampling=Resampling.nearest,
    )
    return destination


def read_bathymetry_rows(path: Path) -> dict[float, dict[str, float]]:
    """Read the two exact NGVD29 elevation rows used for calibration."""
    wanted = {CURRENT_LEVEL_FT, HEALTHY_LEVEL_FT}
    found: dict[float, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            elevation = float(row["elevation_ft_ngvd29"])
            if elevation in wanted:
                found[elevation] = {key: float(value) for key, value in row.items()}
    if set(found) != wanted:
        raise RuntimeError(f"Missing exact bathymetry rows: {sorted(wanted - set(found))}")
    return found


def choose_seed_pixels(
    occurrence: np.ndarray, grid_transform: rasterio.Affine
) -> tuple[dict[str, tuple[int, int]], dict[str, dict[str, float]]]:
    """Anchor each arm at its highest-occurrence pixel near a named seed."""
    xs, ys = zip(*LAKE_SEEDS_WGS84.values())
    eastings, northings = transform("EPSG:4326", MEASUREMENT_CRS, xs, ys)
    radius = int(round(SEED_SEARCH_RADIUS_M / MEASUREMENT_RESOLUTION_M))
    seed_pixels: dict[str, tuple[int, int]] = {}
    report: dict[str, dict[str, float]] = {}
    for name, easting, northing in zip(LAKE_SEEDS_WGS84, eastings, northings):
        center_row, center_col = rowcol(grid_transform, easting, northing)
        row0, row1 = max(0, center_row - radius), min(occurrence.shape[0], center_row + radius + 1)
        col0, col1 = max(0, center_col - radius), min(occurrence.shape[1], center_col + radius + 1)
        local = occurrence[row0:row1, col0:col1]
        candidates = np.where(np.isfinite(local), local, -np.inf)
        if not np.isfinite(candidates).any():
            raise RuntimeError(f"No valid occurrence pixels near {name}")
        local_row, local_col = np.unravel_index(np.argmax(candidates), candidates.shape)
        selected_row, selected_col = row0 + local_row, col0 + local_col
        seed_pixels[name] = (selected_row, selected_col)
        report[name] = {
            "row": int(selected_row),
            "column": int(selected_col),
            "occurrence": float(occurrence[selected_row, selected_col]),
        }
    return seed_pixels, report


def combine_seed_masks(
    shape: tuple[int, int], seed_pixels: dict[str, tuple[int, int]], names
) -> np.ndarray:
    """Combine named one-pixel anchors without relying on array ordering."""
    combined = np.zeros(shape, dtype=bool)
    for name in names:
        combined[seed_pixels[name]] = True
    return combined


def connected_lake_mask(
    occurrence: np.ndarray, threshold: float, seed_mask: np.ndarray
) -> np.ndarray:
    """Return thresholded water connected to either seeded lake arm."""
    eligible = np.isfinite(occurrence) & (occurrence >= threshold)
    active_seeds = seed_mask & eligible
    if not active_seeds.any():
        return np.zeros(occurrence.shape, dtype=bool)
    return ndimage.binary_propagation(
        active_seeds, structure=CONNECTIVITY, mask=eligible
    )


def calibrate_threshold(
    occurrence: np.ndarray,
    seed_mask: np.ndarray,
    target_km2: float,
    iterations: int = 24,
) -> tuple[float, np.ndarray, float]:
    """Binary-search a continuous threshold against connected pixel area."""
    pixel_km2 = MEASUREMENT_RESOLUTION_M**2 / 1_000_000.0
    low, high = 0.0, 100.0
    candidates: list[tuple[float, float]] = []
    for _ in range(iterations):
        threshold = (low + high) / 2.0
        mask = connected_lake_mask(occurrence, threshold, seed_mask)
        area_km2 = float(mask.sum()) * pixel_km2
        candidates.append((threshold, area_km2))
        if area_km2 >= target_km2:
            low = threshold
        else:
            high = threshold
    threshold, _ = min(candidates, key=lambda item: abs(item[1] - target_km2))
    mask = connected_lake_mask(occurrence, threshold, seed_mask)
    achieved_km2 = float(mask.sum()) * pixel_km2
    return threshold, mask, achieved_km2


def rank_depth(occurrence: np.ndarray, lost_water: np.ndarray) -> np.ndarray:
    """Map occurrence ranks in the lost band to 0–8.4 feet of pseudo-depth."""
    output = np.zeros(occurrence.shape, dtype=np.float32)
    values = occurrence[lost_water]
    if values.size < 2:
        return output
    ranks = rankdata(values, method="average")
    ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    output[lost_water] = ranks.astype(np.float32) * (HEALTHY_LEVEL_FT - CURRENT_LEVEL_FT)
    return output


def threshold_for_area(values: np.ndarray, target_km2: float) -> float:
    """Estimate a separate-arm threshold without adopting a separate fit."""
    pixel_km2 = MEASUREMENT_RESOLUTION_M**2 / 1_000_000.0
    pixels = int(round(target_km2 / pixel_km2))
    if pixels <= 0:
        return 100.0
    if pixels >= values.size:
        return float(np.nanmin(values))
    index = values.size - pixels
    return float(np.partition(values, index)[index])


def arm_sensitivity(
    occurrence: np.ndarray,
    seed_mask: np.ndarray,
    grid_transform: rasterio.Affine,
    bathymetry: dict[float, dict[str, float]],
    joint_masks: dict[float, np.ndarray],
) -> dict:
    """Estimate how much north/south thresholds would alter the joint masks."""
    _, causeway_y = transform(
        "EPSG:4326", MEASUREMENT_CRS, [CAUSEWAY_REFERENCE_LON], [CAUSEWAY_LAT]
    )
    row_y = grid_transform.f + (np.arange(occurrence.shape[0]) + 0.5) * grid_transform.e
    north = row_y[:, None] >= causeway_y[0]
    south = ~north
    basin = connected_lake_mask(occurrence, 0.01, seed_mask)
    report = {}
    for level, joint in joint_masks.items():
        row = bathymetry[level]
        north_target = (
            row["north_arm_area_ac"] + row["bear_river_bay_area_ac"]
        ) * ACRE_TO_KM2
        south_target = (
            row["south_arm_area_ac"] + row["farmington_bay_area_ac"]
        ) * ACRE_TO_KM2
        north_values = occurrence[basin & north]
        south_values = occurrence[basin & south]
        north_threshold = threshold_for_area(north_values, north_target)
        south_threshold = threshold_for_area(south_values, south_target)
        hypothetical = (
            basin
            & north
            & (occurrence >= north_threshold)
        ) | (
            basin
            & south
            & (occurrence >= south_threshold)
        )
        union = np.count_nonzero(hypothetical | joint)
        changed = np.count_nonzero(hypothetical ^ joint)
        report[str(level)] = {
            "north_target_km2": north_target,
            "south_target_km2": south_target,
            "estimated_north_threshold": north_threshold,
            "estimated_south_threshold": south_threshold,
            "symmetric_difference_percent_of_union": 100.0 * changed / max(union, 1),
            "noticeable": (100.0 * changed / max(union, 1)) >= 5.0,
        }
    return report


def write_geotiff(path: Path, values: np.ndarray, grid_transform, nodata, dtype) -> None:
    """Write a compressed single-band Phase 1 working raster."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": values.shape[1],
        "height": values.shape[0],
        "count": 1,
        "dtype": dtype,
        "crs": MEASUREMENT_CRS,
        "transform": grid_transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as output:
        output.write(values.astype(dtype, copy=False), 1)


def write_class_diagnostic(classes: np.ndarray, path: Path) -> None:
    """Write a legible three-class diagnostic with nearest-neighbour scaling."""
    palette = {
        0: (194, 178, 142),
        1: (228, 218, 194),
        2: (31, 93, 112),
        255: (248, 247, 243),
    }
    indexed = Image.fromarray(classes, mode="L")
    scale = 1800 / max(indexed.size)
    indexed = indexed.resize(
        (round(indexed.width * scale), round(indexed.height * scale)),
        Image.Resampling.NEAREST,
    )
    small_classes = np.asarray(indexed)
    rgb = np.empty((*small_classes.shape, 3), dtype=np.uint8)
    for value, color in palette.items():
        rgb[small_classes == value] = color
    image = Image.fromarray(rgb, mode="RGB")
    canvas = Image.new("RGB", (image.width, image.height + 100), "#f8f7f3")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, image.height + 12), "4,198 Feet · Phase 1 classes", fill="#282724")
    labels = [("water now", 2), ("lost water", 1), ("land", 0)]
    x = 16
    for label, value in labels:
        draw.rectangle((x, image.height + 48, x + 24, image.height + 72), fill=palette[value])
        draw.text((x + 32, image.height + 52), label, fill="#282724")
        x += 170
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def write_histogram(
    occurrence: np.ndarray, lake_basin_mask: np.ndarray, thresholds: dict[str, float], path: Path
) -> None:
    """Draw the lake-basin occurrence histogram and calibrated thresholds."""
    values = occurrence[lake_basin_mask]
    counts, edges = np.histogram(values, bins=100, range=(0, 100))
    width, height = 1400, 800
    left, right, top, bottom = 90, 40, 70, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "#f8f7f3")
    draw = ImageDraw.Draw(image)
    scaled = np.log1p(counts)
    peak = max(float(scaled.max()), 1.0)
    for index, value in enumerate(scaled):
        x0 = left + index * plot_w / 100
        x1 = left + (index + 1) * plot_w / 100
        y0 = top + plot_h * (1.0 - value / peak)
        draw.rectangle((x0, y0, x1, top + plot_h), fill="#8b9b91")
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#282724", width=2)
    draw.line((left, top, left, top + plot_h), fill="#282724", width=2)
    colors = {"healthy": "#c17d37", "current": "#245a70"}
    for name, threshold in thresholds.items():
        x = left + threshold / 100.0 * plot_w
        draw.line((x, top, x, top + plot_h), fill=colors[name], width=4)
    for tick in range(0, 101, 10):
        x = left + tick / 100.0 * plot_w
        draw.line((x, top + plot_h, x, top + plot_h + 8), fill="#282724", width=2)
        draw.text((x - 8, top + plot_h + 14), str(tick), fill="#282724")
    draw.text((left, 24), "JRC water-occurrence frequency inside connected lake basin", fill="#282724")
    draw.text(
        (left + 420, 24),
        f"healthy {thresholds['healthy']:.3f}",
        fill=colors["healthy"],
    )
    draw.text(
        (left + 575, 24),
        f"current {thresholds['current']:.3f}",
        fill=colors["current"],
    )
    draw.text((left + plot_w // 2 - 70, height - 42), "occurrence (%)", fill="#282724")
    draw.text((12, top), "log pixel count", fill="#282724")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def run_phase1() -> dict:
    """Run the complete deterministic Phase 1 pipeline and return its report."""
    np.random.seed(SEED)
    sources = ensure_sources()
    native_paths = {
        layer: DATA_DIR / f"gsw_{'maximum_extent' if layer == 'extent' else layer}_gsl.tif"
        for layer in GSW_URLS
    }
    occurrence_native, occurrence_profile = crop_native(
        sources["occurrence"], native_paths["occurrence"]
    )
    grid_transform, width, height = measurement_grid(occurrence_profile)
    occurrence = reproject_occurrence(
        occurrence_native, occurrence_profile, grid_transform, width, height
    )
    del occurrence_native
    for layer in ("extent", "transitions"):
        native_values, native_profile = crop_native(sources[layer], native_paths[layer])
        projected = reproject_categories(
            native_values, native_profile, grid_transform, width, height
        )
        output_name = "maximum_extent" if layer == "extent" else layer
        write_geotiff(
            DATA_DIR / f"gsw_{output_name}_gsl_utm12.tif",
            projected,
            grid_transform,
            NODATA_BYTE,
            "uint8",
        )
        del native_values, projected
    seed_pixels, seed_report = choose_seed_pixels(occurrence, grid_transform)
    primary_seed_mask = combine_seed_masks(occurrence.shape, seed_pixels, PRIMARY_ARM_SEEDS)
    healthy_seed_mask = primary_seed_mask
    bathymetry = read_bathymetry_rows(sources["bathymetry"])

    targets = {
        level: bathymetry[level]["basin_plan_total_area_ac"] * ACRE_TO_KM2
        for level in (HEALTHY_LEVEL_FT, CURRENT_LEVEL_FT)
    }
    healthy_threshold, healthy_mask, healthy_area = calibrate_threshold(
        occurrence, healthy_seed_mask, targets[HEALTHY_LEVEL_FT]
    )
    current_threshold, current_mask, current_area = calibrate_threshold(
        occurrence, primary_seed_mask, targets[CURRENT_LEVEL_FT]
    )
    if not healthy_threshold < current_threshold:
        raise RuntimeError("Calibrated thresholds are not monotonic")
    if np.any(current_mask & ~healthy_mask):
        raise RuntimeError("Current water is not nested within healthy water")

    errors = {
        HEALTHY_LEVEL_FT: 100.0 * (healthy_area - targets[HEALTHY_LEVEL_FT]) / targets[HEALTHY_LEVEL_FT],
        CURRENT_LEVEL_FT: 100.0 * (current_area - targets[CURRENT_LEVEL_FT]) / targets[CURRENT_LEVEL_FT],
    }
    if any(abs(error) > 1.0 for error in errors.values()):
        raise RuntimeError(f"Calibration missed the 1% tolerance: {errors}")

    valid = np.isfinite(occurrence)
    lost_water = healthy_mask & ~current_mask
    land = valid & ~healthy_mask
    lost_depth = rank_depth(occurrence, lost_water)
    classes = np.full(occurrence.shape, NODATA_BYTE, dtype=np.uint8)
    classes[land] = 0
    classes[lost_water] = 1
    classes[current_mask] = 2

    lost_depth[~valid] = np.nan
    write_geotiff(DATA_DIR / "gsw_occurrence_gsl_utm12.tif", occurrence, grid_transform, np.nan, "float32")
    write_geotiff(DATA_DIR / "gsl_classes_utm12.tif", classes, grid_transform, NODATA_BYTE, "uint8")
    write_geotiff(DATA_DIR / "gsl_lost_depth_ft_utm12.tif", lost_depth, grid_transform, np.nan, "float32")

    write_class_diagnostic(classes, OUT_DIR / "phase1_classes.png")
    histogram_mask = connected_lake_mask(occurrence, 0.01, primary_seed_mask)
    write_histogram(
        occurrence,
        histogram_mask,
        {"healthy": healthy_threshold, "current": current_threshold},
        OUT_DIR / "occurrence_histogram.png",
    )
    del histogram_mask

    pixel_km2 = MEASUREMENT_RESOLUTION_M**2 / 1_000_000.0
    sensitivity = arm_sensitivity(
        occurrence,
        healthy_seed_mask,
        grid_transform,
        bathymetry,
        {HEALTHY_LEVEL_FT: healthy_mask, CURRENT_LEVEL_FT: current_mask},
    )
    report = {
        "seed": SEED,
        "source": {
            "jrc_download_page": GSW_DOWNLOAD_PAGE,
            "jrc_listing": GSW_LISTING_URL,
            "jrc_version": GSW_VERSION,
            "jrc_years": GSW_YEARS,
            "jrc_tile": GSW_TILE,
            "jrc_tile_bounds_wgs84": [-120.0, 40.0, -110.0, 50.0],
            "jrc_urls": GSW_URLS,
            "jrc_license": GSW_LICENSE,
            "accounting_page": ACCOUNTING_PAGE,
            "bathymetry_url": BATHYMETRY_URL,
            "bathymetry_metadata_url": BATHYMETRY_METADATA_URL,
            "vertical_datum": VERTICAL_DATUM,
            "checksums_sha256": {name: sha256(path) for name, path in sources.items()},
        },
        "native_crop": {
            "bounds_wgs84": list(CROP_BOUNDS_WGS84),
            "crs": str(occurrence_profile["crs"]),
            "resolution_degrees": [
                occurrence_profile["transform"].a,
                abs(occurrence_profile["transform"].e),
            ],
            "shape": [occurrence_profile["height"], occurrence_profile["width"]],
        },
        "measurement_grid": {
            "crs": MEASUREMENT_CRS,
            "resolution_m": MEASUREMENT_RESOLUTION_M,
            "pixel_area_km2": pixel_km2,
            "shape": [height, width],
            "transform": list(grid_transform)[:6],
            "occurrence_resampling": "bilinear",
        },
        "lake_seeds": seed_report,
        "calibration": {
            "healthy": {
                "level_ft_ngvd29": HEALTHY_LEVEL_FT,
                "threshold": healthy_threshold,
                "target_km2": targets[HEALTHY_LEVEL_FT],
                "achieved_km2": healthy_area,
                "error_percent": errors[HEALTHY_LEVEL_FT],
            },
            "current": {
                "level_ft_ngvd29": CURRENT_LEVEL_FT,
                "threshold": current_threshold,
                "target_km2": targets[CURRENT_LEVEL_FT],
                "achieved_km2": current_area,
                "error_percent": errors[CURRENT_LEVEL_FT],
            },
        },
        "class_area_km2": {
            "water_now": float(current_mask.sum()) * pixel_km2,
            "lost_water": float(lost_water.sum()) * pixel_km2,
            "land": float(land.sum()) * pixel_km2,
            "nodata": float((~valid).sum()) * pixel_km2,
        },
        "lost_depth_ft": {
            "minimum": float(lost_depth[lost_water].min()),
            "maximum": float(lost_depth[lost_water].max()),
            "method": "empirical average-rank of occurrence within lost-water band, scaled linearly to 0–8.4 ft",
        },
        "north_south_separate_fit_diagnostic": sensitivity,
    }
    metadata_path = DATA_DIR / "gsw_occurrence_gsl.metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "phase1_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_phase1(), indent=2))
