"""Apply approved managed-pond exclusions and recalibrate Phase 1.

Only the seven explicitly approved regions are assigned to the persistent
``managed`` class.  A looser Compass Minerals screen is reported and drawn but
is never folded into the exclusion mask without a later user decision.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.transform import rowcol, xy
from rasterio.warp import transform
from scipy import ndimage

from data import (
    CONNECTIVITY,
    CURRENT_LEVEL_FT,
    DATA_DIR,
    HEALTHY_LEVEL_FT,
    MEASUREMENT_CRS,
    MEASUREMENT_RESOLUTION_M,
    NODATA_BYTE,
    OUT_DIR,
    rank_depth,
    write_geotiff,
)
from phase1_checks import (
    EROSION_PIXELS,
    LABEL_POINTS_WGS84,
    PIXEL_KM2,
    SQUARE_CONNECTIVITY,
    _candidate_regions,
    _draw_label,
    _draw_legend,
    _font,
    _map_size,
    _occurrence_rgb,
    _resize_float,
    _resize_nearest,
    eroded_connected_mask,
)


ROOT = Path(__file__).resolve().parent
REPORT_PATH = OUT_DIR / "phase1_report.json"
PRE_MANAGED_CLASSES = DATA_DIR / "gsl_classes_pre_managed_utm12.tif"
MANAGED_MASK_PATH = DATA_DIR / "gsl_managed_mask_utm12.tif"
COMPASS_CANDIDATES_PATH = DATA_DIR / "gsl_compass_candidates_utm12.tif"
MANAGED_REPORT_PATH = OUT_DIR / "phase1_managed_report.json"

UGS_RI_171_URL = (
    "https://ugspub.nr.utah.gov/publications/reports_of_investigations/RI-171.pdf"
)
USACE_NOTICE_URL = (
    "https://www.spk.usace.army.mil/Portals/12/documents/regulatory/eis/"
    "200700121-eis/200700121-PN-posted-07-09-2013.pdf"
)

# Broad review windows around the documented operating areas.  They are used
# only to restrict the diagnostic screen, never as exclusion polygons.
COMPASS_AOIS_UTM12 = {
    "west_ponds": (318_000.0, 355_000.0, 4_560_000.0, 4_595_000.0),
    "east_ponds": (385_000.0, 410_000.0, 4_570_000.0, 4_600_000.0),
}
COMPASS_REFERENCE_NAD83_UTM12 = (332_885.0, 4_578_003.0)

# The looser screen samples multiple occurrence levels.  Nested detections are
# merged before areas are reported.
LOOSE_THRESHOLDS = (50.0, 65.0, 80.0, 90.0, 95.0, 98.0)
LOOSE_MIN_AREA_KM2 = 0.5
LOOSE_RECTANGULARITY = 0.20
LOOSE_HIGH_MEAN = 80.0
LOOSE_HIGH_STD = 15.0


def _seed_mask(shape: tuple[int, int], report: dict) -> np.ndarray:
    """Recreate the two accepted lake anchors from the Phase 1 report."""
    output = np.zeros(shape, dtype=bool)
    for seed in report["lake_seeds"].values():
        output[int(seed["row"]), int(seed["column"])] = True
    return output


def _connected_without_managed(
    occurrence: np.ndarray,
    threshold: float,
    seed_mask: np.ndarray,
    managed: np.ndarray,
) -> np.ndarray:
    """Use original four-neighbour connectivity while forbidding managed pixels."""
    eligible = np.isfinite(occurrence) & ~managed & (occurrence >= threshold)
    seeds = seed_mask & eligible
    if not seeds.any():
        return np.zeros(occurrence.shape, dtype=bool)
    return ndimage.binary_propagation(seeds, structure=CONNECTIVITY, mask=eligible)


def _calibrate_without_managed(
    occurrence: np.ndarray,
    seed_mask: np.ndarray,
    managed: np.ndarray,
    target_km2: float,
    iterations: int = 26,
) -> tuple[float, np.ndarray, float]:
    """Fit a continuous occurrence threshold after removing managed pixels."""
    low, high = 0.0, 100.0
    candidates: list[tuple[float, float]] = []
    for _ in range(iterations):
        threshold = (low + high) / 2.0
        mask = _connected_without_managed(occurrence, threshold, seed_mask, managed)
        area = float(mask.sum()) * PIXEL_KM2
        candidates.append((threshold, area))
        if area >= target_km2:
            low = threshold
        else:
            high = threshold
    threshold, _ = min(candidates, key=lambda item: abs(item[1] - target_km2))
    mask = _connected_without_managed(occurrence, threshold, seed_mask, managed)
    return threshold, mask, float(mask.sum()) * PIXEL_KM2


def _approved_candidate_mask(
    occurrence: np.ndarray,
    original_classes: np.ndarray,
    seed_mask: np.ndarray,
    grid_transform: rasterio.Affine,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Recreate the exact C1-C7 regions approved from the prior diagnostic."""
    healthy = np.isin(original_classes, (1, 2))
    current = original_classes == 2
    robust_healthy, detached_healthy = eroded_connected_mask(healthy, seed_mask)
    robust_current, detached_current = eroded_connected_mask(current, seed_mask)
    candidate_ids, candidates = _candidate_regions(
        occurrence,
        healthy,
        current,
        detached_healthy,
        detached_current,
        robust_healthy,
        robust_current,
        grid_transform,
    )
    return candidate_ids > 0, candidates, candidate_ids


def _confirm_c1(
    original_classes: np.ndarray,
    seed_mask: np.ndarray,
    candidate_ids: np.ndarray,
) -> dict:
    """Test whether C1 detaches after closing one-pixel water connections."""
    current = original_classes == 2
    eroded = ndimage.binary_erosion(
        current,
        structure=SQUARE_CONNECTIVITY,
        iterations=1,
        border_value=0,
    )
    connected_core = ndimage.binary_propagation(
        seed_mask & eroded,
        structure=CONNECTIVITY,
        mask=eroded,
    )
    restored = ndimage.binary_dilation(
        connected_core,
        structure=SQUARE_CONNECTIVITY,
        iterations=1,
        mask=current,
    )
    c1 = candidate_ids == 1
    pixels = int(c1.sum())
    detached = int(np.count_nonzero(c1 & ~restored))
    return {
        "test": "one-pixel (30 m) square erosion, seeded connectivity, then one-pixel dilation within original current-water mask",
        "c1_area_km2": pixels * PIXEL_KM2,
        "detached_area_km2": detached * PIXEL_KM2,
        "detached_percent": 100.0 * detached / max(pixels, 1),
        "confirmed_diked_off": detached == pixels,
    }


def _coordinate_grids(
    shape: tuple[int, int], grid_transform: rasterio.Affine
) -> tuple[np.ndarray, np.ndarray]:
    """Return UTM centre coordinates as broadcastable one-dimensional arrays."""
    rows = np.arange(shape[0])
    columns = np.arange(shape[1])
    ys = grid_transform.f + (rows + 0.5) * grid_transform.e
    xs = grid_transform.c + (columns + 0.5) * grid_transform.a
    return xs, ys


def _touches_aoi_edge(
    component: np.ndarray,
    bounds: tuple[slice, slice],
    aoi_bounds: tuple[int, int, int, int],
) -> bool:
    """Reject components clipped by a review-window edge."""
    row0, row1, col0, col1 = aoi_bounds
    return (
        bounds[0].start <= row0
        or bounds[0].stop >= row1
        or bounds[1].start <= col0
        or bounds[1].stop >= col1
    )


def _screen_compass_candidates(
    occurrence: np.ndarray,
    already_managed: np.ndarray,
    grid_transform: rasterio.Affine,
) -> tuple[np.ndarray, list[dict], dict]:
    """Run a loose, AOI-restricted shape/uniformity screen for missed ponds."""
    xs, ys = _coordinate_grids(occurrence.shape, grid_transform)
    union = np.zeros(occurrence.shape, dtype=bool)
    aoi_report = {}
    for aoi_name, (xmin, xmax, ymin, ymax) in COMPASS_AOIS_UTM12.items():
        columns = np.flatnonzero((xs >= xmin) & (xs <= xmax))
        rows = np.flatnonzero((ys >= ymin) & (ys <= ymax))
        if not len(rows) or not len(columns):
            raise RuntimeError(f"Compass review area {aoi_name} falls outside the crop")
        row0, row1 = int(rows.min()), int(rows.max()) + 1
        col0, col1 = int(columns.min()), int(columns.max()) + 1
        local_occurrence = occurrence[row0:row1, col0:col1]
        local_managed = already_managed[row0:row1, col0:col1]
        local_union = np.zeros(local_occurrence.shape, dtype=bool)
        for threshold in LOOSE_THRESHOLDS:
            eligible = (
                np.isfinite(local_occurrence)
                & ~local_managed
                & (local_occurrence >= threshold)
            )
            opened = ndimage.binary_opening(
                eligible,
                structure=SQUARE_CONNECTIVITY,
                iterations=1,
            )
            labels, count = ndimage.label(opened, structure=CONNECTIVITY)
            for label_id, bounds in enumerate(
                ndimage.find_objects(labels, max_label=count), start=1
            ):
                if bounds is None:
                    continue
                component = labels[bounds] == label_id
                pixels = int(component.sum())
                area_km2 = pixels * PIXEL_KM2
                if area_km2 < LOOSE_MIN_AREA_KM2:
                    continue
                global_bounds = (
                    slice(bounds[0].start + row0, bounds[0].stop + row0),
                    slice(bounds[1].start + col0, bounds[1].stop + col0),
                )
                if _touches_aoi_edge(
                    component, global_bounds, (row0, row1, col0, col1)
                ):
                    continue
                rectangularity = pixels / max(component.size, 1)
                values = local_occurrence[bounds][component]
                mean = float(np.nanmean(values))
                std = float(np.nanstd(values))
                rectilinear = rectangularity >= LOOSE_RECTANGULARITY
                uniform_high = mean >= LOOSE_HIGH_MEAN and std <= LOOSE_HIGH_STD
                if rectilinear or uniform_high:
                    local_union[bounds] |= component
        union[row0:row1, col0:col1] |= local_union
        aoi_report[aoi_name] = {
            "bounds_utm12": [xmin, xmax, ymin, ymax],
            "area_km2": float((row1 - row0) * (col1 - col0)) * PIXEL_KM2,
        }

    # Subtract approved exclusions, merge nested multi-threshold detections, and
    # report only substantial connected regions.
    union &= ~already_managed
    union = ndimage.binary_closing(union, structure=SQUARE_CONNECTIVITY, iterations=1)
    labels, count = ndimage.label(union, structure=CONNECTIVITY)
    objects = ndimage.find_objects(labels, max_label=count)
    candidate_ids = np.zeros(union.shape, dtype=np.uint16)
    candidates = []
    provisional = []
    for label_id, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        component = labels[bounds] == label_id
        pixels = int(component.sum())
        area_km2 = pixels * PIXEL_KM2
        if area_km2 < LOOSE_MIN_AREA_KM2:
            continue
        local_values = occurrence[bounds][component]
        rows, columns = component.shape
        rectangularity = pixels / max(rows * columns, 1)
        local_rows, local_columns = np.nonzero(component)
        center_row = int(round(bounds[0].start + float(local_rows.mean())))
        center_col = int(round(bounds[1].start + float(local_columns.mean())))
        easting, northing = xy(grid_transform, center_row, center_col)
        longitude, latitude = transform(
            MEASUREMENT_CRS, "EPSG:4326", [easting], [northing]
        )
        aoi_name = next(
            name
            for name, (xmin, xmax, ymin, ymax) in COMPASS_AOIS_UTM12.items()
            if xmin <= easting <= xmax and ymin <= northing <= ymax
        )
        provisional.append(
            {
                "source_label": label_id,
                "bounds": bounds,
                "component": component,
                "area_km2": area_km2,
                "centroid_wgs84": [longitude[0], latitude[0]],
                "centroid_utm12": [easting, northing],
                "review_area": aoi_name,
                "mean_occurrence": float(np.nanmean(local_values)),
                "occurrence_std": float(np.nanstd(local_values)),
                "bounding_box_fill_ratio": rectangularity,
            }
        )
    provisional.sort(key=lambda item: item["area_km2"], reverse=True)
    for output_id, item in enumerate(provisional, start=1):
        bounds = item.pop("bounds")
        component = item.pop("component")
        item.pop("source_label")
        item["id"] = f"N{output_id}"
        item["status"] = "candidate_only_not_excluded"
        local = candidate_ids[bounds]
        local[component] = output_id
        candidates.append(item)
    return candidate_ids, candidates, aoi_report


def _write_class_diagnostic(classes: np.ndarray, path: Path) -> None:
    """Render the four persistent Phase 1 classes."""
    palette = {
        0: (194, 178, 142),
        1: (228, 190, 94),
        2: (31, 93, 112),
        3: (93, 89, 80),
        255: (248, 247, 243),
    }
    size = _map_size(classes.shape)
    small = _resize_nearest(classes, size)
    rgb = np.empty((*small.shape, 3), dtype=np.uint8)
    for value, color in palette.items():
        rgb[small == value] = color
    canvas = Image.new("RGB", (size[0], size[1] + 104), "#f8f7f3")
    canvas.paste(Image.fromarray(rgb, "RGB"), (0, 0))
    _draw_legend(
        canvas,
        "Recalibrated Phase 1 classes",
        [
            ("land", palette[0]),
            ("lost water", palette[1]),
            ("water now", palette[2]),
            ("managed (renders as land)", palette[3]),
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def _write_updated_overlay(
    occurrence: np.ndarray,
    classes: np.ndarray,
    compass_candidates: np.ndarray,
    candidate_rows: list[dict],
    grid_transform: rasterio.Affine,
    path: Path,
) -> None:
    """Overlay recalibrated classes, managed exclusions, and review candidates."""
    size = _map_size(occurrence.shape)
    small_occurrence = _resize_float(occurrence, size)
    small_classes = _resize_nearest(classes, size)
    small_candidates = _resize_nearest(compass_candidates.astype(np.uint16), size)
    image = Image.fromarray(_occurrence_rgb(small_occurrence), "RGB").convert("RGBA")
    overlay = np.zeros((*small_classes.shape, 4), dtype=np.uint8)
    colors = {
        0: (170, 139, 82, 66),
        1: (236, 175, 54, 126),
        2: (20, 105, 145, 112),
        3: (78, 73, 66, 210),
    }
    for value, color in colors.items():
        overlay[small_classes == value] = color
    image = Image.alpha_composite(image, Image.fromarray(overlay, "RGBA"))
    draw = ImageDraw.Draw(image)
    font = _font(17)

    # Outline, rather than fill, candidates awaiting a decision.
    for numeric_id, row in enumerate(candidate_rows, start=1):
        mask = small_candidates == numeric_id
        edge = mask & ~ndimage.binary_erosion(mask, structure=SQUARE_CONNECTIVITY)
        ys, xs = np.nonzero(edge)
        if len(xs):
            for x_value, y_value in zip(xs, ys):
                draw.point((int(x_value), int(y_value)), fill="#d44435")
        inside_y, inside_x = np.nonzero(mask)
        if len(inside_x):
            x_center = float(inside_x.mean())
            y_center = float(inside_y.mean())
            draw.text(
                (x_center + 7, y_center - 8),
                row["id"],
                fill="#242927",
                font=font,
                stroke_width=3,
                stroke_fill="#f8f7f3",
            )

    longitudes = [point[0] for point in LABEL_POINTS_WGS84.values()]
    latitudes = [point[1] for point in LABEL_POINTS_WGS84.values()]
    eastings, northings = transform("EPSG:4326", MEASUREMENT_CRS, longitudes, latitudes)
    scale_x = size[0] / occurrence.shape[1]
    scale_y = size[1] / occurrence.shape[0]
    for (label, (_, _, offset)), easting, northing in zip(
        LABEL_POINTS_WGS84.items(), eastings, northings
    ):
        row_index, column = rowcol(grid_transform, easting, northing)
        point = ((column + 0.5) * scale_x, (row_index + 0.5) * scale_y)
        _draw_label(draw, point, label, offset, font)

    canvas = Image.new("RGB", (size[0], size[1] + 104), "#f8f7f3")
    canvas.paste(image.convert("RGB"), (0, 0))
    _draw_legend(
        canvas,
        "Recalibrated classes and Compass review candidates",
        [
            ("land", (194, 166, 111)),
            ("lost water", (236, 175, 54)),
            ("water now", (20, 105, 145)),
            ("managed excluded", (78, 73, 66)),
            ("new candidate outline", (212, 68, 53)),
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def run_phase1_managed() -> dict:
    """Confirm C1, apply approved exclusions, recalibrate, and screen Compass."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    with rasterio.open(DATA_DIR / "gsw_occurrence_gsl_utm12.tif") as dataset:
        occurrence = dataset.read(1)
        grid_transform = dataset.transform

    classes_path = DATA_DIR / "gsl_classes_utm12.tif"
    if not PRE_MANAGED_CLASSES.exists():
        shutil.copy2(classes_path, PRE_MANAGED_CLASSES)
    with rasterio.open(PRE_MANAGED_CLASSES) as dataset:
        original_classes = dataset.read(1)

    seed_mask = _seed_mask(original_classes.shape, report)
    managed, approved_candidates, approved_ids = _approved_candidate_mask(
        occurrence, original_classes, seed_mask, grid_transform
    )
    c1_confirmation = _confirm_c1(original_classes, seed_mask, approved_ids)
    if not c1_confirmation["confirmed_diked_off"]:
        raise RuntimeError("C1 remained connected after the required one-pixel gap test")

    compass_ids, compass_candidates, aoi_report = _screen_compass_candidates(
        occurrence, managed, grid_transform
    )

    old_calibration = report.get("calibration_before_managed_exclusion", report["calibration"])
    old_class_area = report.get("class_area_before_managed_exclusion", report["class_area_km2"])
    targets = {
        "healthy": float(old_calibration["healthy"]["target_km2"]),
        "current": float(old_calibration["current"]["target_km2"]),
    }
    healthy_threshold, healthy, healthy_area = _calibrate_without_managed(
        occurrence, seed_mask, managed, targets["healthy"]
    )
    current_threshold, current, current_area = _calibrate_without_managed(
        occurrence, seed_mask, managed, targets["current"]
    )
    if not healthy_threshold < current_threshold:
        raise RuntimeError("Managed-exclusion thresholds are not monotonic")
    if np.any(current & ~healthy):
        raise RuntimeError("Recalibrated current mask is not inside healthy mask")

    valid = np.isfinite(occurrence)
    lost = healthy & ~current
    land = valid & ~managed & ~healthy
    classes = np.full(occurrence.shape, NODATA_BYTE, dtype=np.uint8)
    classes[land] = 0
    classes[lost] = 1
    classes[current] = 2
    classes[managed] = 3
    lost_depth = rank_depth(occurrence, lost)
    lost_depth[~valid | managed] = np.nan

    write_geotiff(classes_path, classes, grid_transform, NODATA_BYTE, "uint8")
    with rasterio.open(classes_path, "r+") as dataset:
        dataset.update_tags(
            class_0="land",
            class_1="lost_water",
            class_2="water_now",
            class_3="managed_renders_as_land",
            class_255="nodata",
        )
    write_geotiff(MANAGED_MASK_PATH, managed.astype(np.uint8), grid_transform, 0, "uint8")
    write_geotiff(
        COMPASS_CANDIDATES_PATH,
        compass_ids,
        grid_transform,
        0,
        "uint16",
    )
    write_geotiff(
        DATA_DIR / "gsl_lost_depth_ft_utm12.tif",
        lost_depth,
        grid_transform,
        np.nan,
        "float32",
    )
    _write_class_diagnostic(classes, OUT_DIR / "phase1_classes.png")
    _write_updated_overlay(
        occurrence,
        classes,
        compass_ids,
        compass_candidates,
        grid_transform,
        OUT_DIR / "overlay_labeled.png",
    )

    new_calibration = {
        "healthy": {
            "level_ft_ngvd29": HEALTHY_LEVEL_FT,
            "threshold": healthy_threshold,
            "target_km2": targets["healthy"],
            "achieved_km2": healthy_area,
            "error_percent": 100.0 * (healthy_area - targets["healthy"]) / targets["healthy"],
        },
        "current": {
            "level_ft_ngvd29": CURRENT_LEVEL_FT,
            "threshold": current_threshold,
            "target_km2": targets["current"],
            "achieved_km2": current_area,
            "error_percent": 100.0 * (current_area - targets["current"]) / targets["current"],
        },
    }
    class_area = {
        "water_now": float(current.sum()) * PIXEL_KM2,
        "lost_water": float(lost.sum()) * PIXEL_KM2,
        "land": float(land.sum()) * PIXEL_KM2,
        "managed": float(managed.sum()) * PIXEL_KM2,
        "nodata": float((~valid).sum()) * PIXEL_KM2,
    }
    if any(abs(row["error_percent"]) > 1.0 for row in new_calibration.values()):
        raise RuntimeError(f"Managed recalibration missed tolerance: {new_calibration}")

    reference_x, reference_y = COMPASS_REFERENCE_NAD83_UTM12
    crop_bounds = rasterio.transform.array_bounds(
        occurrence.shape[0], occurrence.shape[1], grid_transform
    )
    reference_inside = (
        crop_bounds[0] <= reference_x <= crop_bounds[2]
        and crop_bounds[1] <= reference_y <= crop_bounds[3]
    )
    managed_report = {
        "sources": {
            "us_magnesium_stansbury_basin": UGS_RI_171_URL,
            "compass_minerals_public_notice": USACE_NOTICE_URL,
        },
        "c1_connectivity_confirmation": c1_confirmation,
        "approved_managed_exclusions": {
            "candidate_ids": [f"C{candidate['id']}" for candidate in approved_candidates],
            "area_km2": float(managed.sum()) * PIXEL_KM2,
            "automatic_additions": 0,
        },
        "compass_reference": {
            "crs": "NAD83 / UTM zone 12N",
            "easting": reference_x,
            "northing": reference_y,
            "inside_crop": reference_inside,
            "measurement_crop_bounds_utm12": list(crop_bounds),
        },
        "loose_compass_screen": {
            "review_areas": aoi_report,
            "occurrence_thresholds": list(LOOSE_THRESHOLDS),
            "minimum_area_km2": LOOSE_MIN_AREA_KM2,
            "rectilinearity_minimum": LOOSE_RECTANGULARITY,
            "uniform_high_rule": {
                "mean_minimum": LOOSE_HIGH_MEAN,
                "standard_deviation_maximum": LOOSE_HIGH_STD,
            },
            "candidates": compass_candidates,
            "excluded_pending_decision": 0,
        },
        "recalibration": {
            "connectivity": "original four-neighbour seeded connectivity; no global erosion",
            "before": old_calibration,
            "after": new_calibration,
            "class_area_before_km2": old_class_area,
            "class_area_after_km2": class_area,
        },
        "outputs": {
            "updated_overlay": str(OUT_DIR / "overlay_labeled.png"),
            "managed_mask": str(MANAGED_MASK_PATH),
            "compass_candidate_ids": str(COMPASS_CANDIDATES_PATH),
            "classes": str(classes_path),
        },
    }
    MANAGED_REPORT_PATH.write_text(
        json.dumps(managed_report, indent=2) + "\n", encoding="utf-8"
    )

    report["calibration_before_managed_exclusion"] = old_calibration
    report["class_area_before_managed_exclusion"] = old_class_area
    report["calibration"] = new_calibration
    report["class_area_km2"] = class_area
    report["managed_exclusion"] = managed_report
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (DATA_DIR / "gsw_occurrence_gsl.metadata.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return managed_report


if __name__ == "__main__":
    print(json.dumps(run_phase1_managed(), indent=2))
