"""Post-calibration diagnostics for the accepted Phase 1 masks.

These checks never modify or replace the accepted class raster.  They expose
possible managed-water leakage, source no-data, and north/south sensitivity so
that exclusions remain an explicit editorial decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.transform import rowcol, xy
from rasterio.warp import transform
from scipy import ndimage

from data import (
    CAUSEWAY_LAT,
    CAUSEWAY_REFERENCE_LON,
    CONNECTIVITY,
    CURRENT_LEVEL_FT,
    DATA_DIR,
    HEALTHY_LEVEL_FT,
    MEASUREMENT_CRS,
    MEASUREMENT_RESOLUTION_M,
    NODATA_BYTE,
    OUT_DIR,
)


ROOT = Path(__file__).resolve().parent
REPORT_PATH = OUT_DIR / "phase1_report.json"
CHECKS_REPORT_PATH = OUT_DIR / "phase1_checks_report.json"
PIXEL_KM2 = MEASUREMENT_RESOLUTION_M**2 / 1_000_000.0
EROSION_PIXELS = 2
ONE_KM_PIXELS = 1_000.0 / MEASUREMENT_RESOLUTION_M
SQUARE_CONNECTIVITY = np.ones((3, 3), dtype=bool)

# Marker coordinates identify named geographic features, not calibration seeds.
LABEL_POINTS_WGS84 = {
    "Gilbert Bay (south arm)": (-112.52, 40.94, (22, 20)),
    "Gunnison Bay (north arm)": (-112.72, 41.37, (-238, -34)),
    "Railroad causeway": (-112.54, 41.223, (22, -38)),
    "Antelope Island": (-112.23, 41.06, (22, 18)),
    "Bear River Bay": (-112.15, 41.43, (22, -34)),
    "Farmington Bay": (-112.10, 40.96, (22, 18)),
    "West-shore mineral evaporation ponds": (-112.68, 40.86, (22, 18)),
}


def _font(size: int = 18) -> ImageFont.ImageFont:
    """Load a readable bundled/system font with a safe fallback."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _resize_nearest(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a categorical array without blending class identities."""
    return np.asarray(Image.fromarray(values).resize(size, Image.Resampling.NEAREST))


def _resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize continuous values while preserving NaN through a validity band."""
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0).astype(np.float32)
    resized = np.array(
        Image.fromarray(filled, mode="F").resize(size, Image.Resampling.BILINEAR)
    )
    valid_resized = _resize_nearest(valid.astype(np.uint8), size).astype(bool)
    resized[~valid_resized] = np.nan
    return resized


def _map_size(shape: tuple[int, int], max_dimension: int = 1700) -> tuple[int, int]:
    """Return a display size that keeps the entire diagnostic grid visible."""
    height, width = shape
    scale = max_dimension / max(height, width)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _occurrence_rgb(occurrence: np.ndarray) -> np.ndarray:
    """Render occurrence with a neutral-to-blue continuous ramp."""
    normalized = np.clip(np.nan_to_num(occurrence, nan=0.0) / 100.0, 0.0, 1.0)
    low = np.array([238, 232, 216], dtype=np.float32)
    high = np.array([38, 93, 111], dtype=np.float32)
    rgb = low + normalized[..., None] * (high - low)
    rgb[~np.isfinite(occurrence)] = (244, 242, 236)
    return rgb.astype(np.uint8)


def _class_overlay(base: Image.Image, classes: np.ndarray) -> Image.Image:
    """Place all three accepted classes semi-transparently over occurrence."""
    colors = {
        0: (170, 139, 82, 72),
        1: (236, 175, 54, 126),
        2: (20, 105, 145, 112),
    }
    rgba = np.zeros((*classes.shape, 4), dtype=np.uint8)
    for value, color in colors.items():
        rgba[classes == value] = color
    return Image.alpha_composite(base.convert("RGBA"), Image.fromarray(rgba, "RGBA"))


def _draw_label(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    label: str,
    offset: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    """Draw one high-contrast marker, leader, and collision-tuned label."""
    x, y = point
    label_x, label_y = x + offset[0], y + offset[1]
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#f7f2e7", outline="#242927", width=3)
    draw.line((x, y, label_x, label_y), fill="#242927", width=2)
    box = draw.textbbox((label_x, label_y), label, font=font, stroke_width=2)
    padded = (box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3)
    draw.rectangle(padded, fill=(248, 247, 243, 226))
    draw.text(
        (label_x, label_y),
        label,
        fill="#242927",
        font=font,
        stroke_width=1,
        stroke_fill="#f8f7f3",
    )


def _draw_legend(
    canvas: Image.Image,
    title: str,
    entries: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """Draw a compact title and categorical legend below a map."""
    draw = ImageDraw.Draw(canvas)
    title_font = _font(21)
    font = _font(17)
    map_height = canvas.height - 104
    draw.text((18, map_height + 12), title, fill="#242927", font=title_font)
    x = 18
    for label, color in entries:
        draw.rectangle((x, map_height + 56, x + 24, map_height + 80), fill=color, outline="#585b57")
        draw.text((x + 32, map_height + 58), label, fill="#242927", font=font)
        x += 42 + int(draw.textlength(label, font=font))


def write_labeled_overlay(
    occurrence: np.ndarray,
    classes: np.ndarray,
    grid_transform: rasterio.Affine,
    path: Path,
) -> None:
    """Render accepted classes over occurrence with requested place labels."""
    size = _map_size(occurrence.shape)
    small_occurrence = _resize_float(occurrence, size)
    small_classes = _resize_nearest(classes, size)
    map_image = _class_overlay(Image.fromarray(_occurrence_rgb(small_occurrence), "RGB"), small_classes)
    draw = ImageDraw.Draw(map_image)
    font = _font(18)
    longitudes = [point[0] for point in LABEL_POINTS_WGS84.values()]
    latitudes = [point[1] for point in LABEL_POINTS_WGS84.values()]
    eastings, northings = transform("EPSG:4326", MEASUREMENT_CRS, longitudes, latitudes)
    scale_x = size[0] / occurrence.shape[1]
    scale_y = size[1] / occurrence.shape[0]
    for (label, (_, _, offset)), easting, northing in zip(
        LABEL_POINTS_WGS84.items(), eastings, northings
    ):
        row, column = rowcol(grid_transform, easting, northing)
        point = ((column + 0.5) * scale_x, (row + 0.5) * scale_y)
        _draw_label(draw, point, label, offset, font)
    canvas = Image.new("RGB", (size[0], size[1] + 104), "#f8f7f3")
    canvas.paste(map_image.convert("RGB"), (0, 0))
    _draw_legend(
        canvas,
        "Accepted Phase 1 classes over JRC occurrence",
        [
            ("land", (194, 166, 111)),
            ("lost water", (236, 175, 54)),
            ("water now", (20, 105, 145)),
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def eroded_connected_mask(
    original_mask: np.ndarray,
    seed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Break two-pixel necks, reconnect from lake seeds, then dilate in-mask."""
    eroded = ndimage.binary_erosion(
        original_mask,
        structure=SQUARE_CONNECTIVITY,
        iterations=EROSION_PIXELS,
        border_value=0,
    )
    active_seeds = eroded & seed_mask
    if not active_seeds.any():
        # The selected seeds lie in stable interior water, but keep failure plain.
        raise RuntimeError("Erosion removed every accepted lake seed")
    connected_core = ndimage.binary_propagation(
        active_seeds,
        structure=CONNECTIVITY,
        mask=eroded,
    )
    restored = ndimage.binary_dilation(
        connected_core,
        structure=SQUARE_CONNECTIVITY,
        iterations=EROSION_PIXELS,
        mask=original_mask,
    )
    return restored, eroded & ~connected_core


def class_transition_report(original: np.ndarray, revised: np.ndarray) -> dict:
    """Return exact old-to-eroded-connectivity class movements in km²."""
    names = {0: "land", 1: "lost_water", 2: "water_now"}
    movements = {}
    for old in range(3):
        for new in range(3):
            if old == new:
                continue
            area = float(np.count_nonzero((original == old) & (revised == new))) * PIXEL_KM2
            if area:
                movements[f"{names[old]}_to_{names[new]}"] = area
    changed = np.isin(original, (0, 1, 2)) & np.isin(revised, (0, 1, 2)) & (original != revised)
    return {
        "total_changed_km2": float(changed.sum()) * PIXEL_KM2,
        "movements_km2": movements,
    }


def _candidate_regions(
    occurrence: np.ndarray,
    original_healthy: np.ndarray,
    original_current: np.ndarray,
    disconnected_healthy_core: np.ndarray,
    disconnected_current_core: np.ndarray,
    robust_healthy: np.ndarray,
    robust_current: np.ndarray,
    grid_transform: rasterio.Affine,
) -> tuple[np.ndarray, list[dict]]:
    """Flag detached regions with rectangular shape or uniform high water."""
    detached_healthy = ndimage.binary_dilation(
        disconnected_healthy_core,
        structure=SQUARE_CONNECTIVITY,
        iterations=EROSION_PIXELS,
        mask=original_healthy,
    ) & ~robust_healthy
    detached_current = ndimage.binary_dilation(
        disconnected_current_core,
        structure=SQUARE_CONNECTIVITY,
        iterations=EROSION_PIXELS,
        mask=original_current,
    ) & ~robust_current
    detached = detached_healthy | detached_current
    labels, count = ndimage.label(detached, structure=CONNECTIVITY)
    objects = ndimage.find_objects(labels, max_label=count)
    flagged = np.zeros(detached.shape, dtype=np.int32)
    candidates: list[dict] = []
    output_id = 0
    for label_id, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        component = labels[bounds] == label_id
        pixels = int(component.sum())
        area_km2 = pixels * PIXEL_KM2
        if area_km2 < 0.25:
            continue
        rows, columns = component.shape
        rectangularity = pixels / max(rows * columns, 1)
        local_occurrence = occurrence[bounds][component]
        mean_occurrence = float(np.nanmean(local_occurrence))
        occurrence_std = float(np.nanstd(local_occurrence))
        rectilinear = rectangularity >= 0.42 and area_km2 >= 0.5
        near_constant_high = (
            mean_occurrence >= 95.0 and occurrence_std <= 5.0 and area_km2 >= 0.5
        )
        if not (rectilinear or near_constant_high):
            continue
        output_id += 1
        local_rows, local_columns = np.nonzero(component)
        center_row = int(round(bounds[0].start + float(local_rows.mean())))
        center_column = int(round(bounds[1].start + float(local_columns.mean())))
        easting, northing = xy(grid_transform, center_row, center_column)
        longitude, latitude = transform(
            MEASUREMENT_CRS, "EPSG:4326", [easting], [northing]
        )
        global_component = np.zeros(detached.shape, dtype=bool)
        global_component[bounds] = component
        flagged[global_component] = output_id
        candidates.append(
            {
                "id": output_id,
                "area_km2": area_km2,
                "centroid_wgs84": [longitude[0], latitude[0]],
                "mean_occurrence": mean_occurrence,
                "occurrence_std": occurrence_std,
                "bounding_box_fill_ratio": rectangularity,
                "flags": [
                    flag
                    for flag, enabled in (
                        ("rectilinear", rectilinear),
                        ("near_constant_high_occurrence", near_constant_high),
                    )
                    if enabled
                ],
            }
        )
    candidates.sort(key=lambda item: item["area_km2"], reverse=True)
    return flagged, candidates


def write_managed_candidates(
    occurrence: np.ndarray,
    candidates: np.ndarray,
    candidate_rows: list[dict],
    path: Path,
) -> None:
    """Map candidate regions and put each candidate's area on the image."""
    size = _map_size(occurrence.shape)
    small_occurrence = _resize_float(occurrence, size)
    small_candidates = _resize_nearest(candidates.astype(np.int32), size)
    rgb = _occurrence_rgb(small_occurrence)
    rgb = (rgb.astype(np.float32) * 0.62 + np.array([248, 247, 243]) * 0.38).astype(np.uint8)
    image = Image.fromarray(rgb, "RGB").convert("RGBA")
    layer = np.zeros((*small_candidates.shape, 4), dtype=np.uint8)
    layer[small_candidates > 0] = (206, 84, 38, 185)
    image = Image.alpha_composite(image, Image.fromarray(layer, "RGBA"))
    draw = ImageDraw.Draw(image)
    font = _font(17)
    scale_x = size[0] / occurrence.shape[1]
    scale_y = size[1] / occurrence.shape[0]
    lookup = {row["id"]: row for row in candidate_rows}
    for candidate_id in np.unique(small_candidates):
        if candidate_id == 0 or candidate_id not in lookup:
            continue
        ys, xs = np.nonzero(small_candidates == candidate_id)
        if not len(xs):
            continue
        x = float(xs.mean())
        y = float(ys.mean())
        text = f"C{candidate_id} · {lookup[candidate_id]['area_km2']:.2f} km²"
        draw.text(
            (x + 8, y - 10),
            text,
            fill="#242927",
            font=font,
            stroke_width=3,
            stroke_fill="#f8f7f3",
        )
    canvas = Image.new("RGB", (size[0], size[1] + 104), "#f8f7f3")
    canvas.paste(image.convert("RGB"), (0, 0))
    _draw_legend(
        canvas,
        "Managed-water candidates after 60 m erosion test",
        [("flagged candidate (no automatic exclusion)", (206, 84, 38))],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def nodata_report(
    valid: np.ndarray,
    healthy: np.ndarray,
    current: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Measure source no-data enclosed by healthy water or near a shoreline."""
    nodata = ~valid
    enclosed_by_healthy = nodata & ndimage.binary_fill_holes(healthy)
    healthy_edge = healthy ^ ndimage.binary_erosion(healthy, structure=CONNECTIVITY)
    current_edge = current ^ ndimage.binary_erosion(current, structure=CONNECTIVITY)
    shorelines = healthy_edge | current_edge
    distance_pixels = ndimage.distance_transform_edt(~shorelines)
    within_one_km = nodata & (distance_pixels <= ONE_KM_PIXELS)
    relevant = enclosed_by_healthy | within_one_km
    categories = np.zeros(valid.shape, dtype=np.uint8)
    categories[nodata] = 1
    categories[within_one_km] = 2
    categories[enclosed_by_healthy] = 3
    categories[within_one_km & enclosed_by_healthy] = 4
    report = {
        "total_nodata_km2": float(nodata.sum()) * PIXEL_KM2,
        "inside_healthy_envelope_km2": float(enclosed_by_healthy.sum()) * PIXEL_KM2,
        "within_1km_of_either_shoreline_km2": float(within_one_km.sum()) * PIXEL_KM2,
        "inside_or_within_1km_km2": float(relevant.sum()) * PIXEL_KM2,
        "overlap_km2": float((within_one_km & enclosed_by_healthy).sum()) * PIXEL_KM2,
        "inside_definition": "no-data enclosed as a hole by the accepted healthy mask",
        "shoreline_distance_method": "Euclidean pixel distance to the interior boundary of either accepted mask",
    }
    return report, categories


def write_nodata_map(
    occurrence: np.ndarray,
    categories: np.ndarray,
    path: Path,
) -> None:
    """Map all source no-data and emphasize pixels relevant to the masks."""
    size = _map_size(occurrence.shape)
    small_occurrence = _resize_float(occurrence, size)
    small_categories = _resize_nearest(categories, size)
    rgb = _occurrence_rgb(small_occurrence)
    rgb = (rgb.astype(np.float32) * 0.45 + np.array([248, 247, 243]) * 0.55).astype(np.uint8)
    colors = {
        1: (147, 147, 142),
        2: (220, 145, 43),
        3: (177, 49, 56),
        4: (121, 45, 117),
    }
    for value, color in colors.items():
        rgb[small_categories == value] = color
    canvas = Image.new("RGB", (size[0], size[1] + 104), "#f8f7f3")
    canvas.paste(Image.fromarray(rgb, "RGB"), (0, 0))
    _draw_legend(
        canvas,
        "JRC no-data proximity to accepted shorelines",
        [
            ("other no-data", colors[1]),
            ("within 1 km", colors[2]),
            ("inside healthy", colors[3]),
            ("both", colors[4]),
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def per_arm_masks(
    occurrence: np.ndarray,
    accepted_classes: np.ndarray,
    seed_mask: np.ndarray,
    grid_transform: rasterio.Affine,
    report: dict,
) -> dict[str, np.ndarray]:
    """Recreate the prior non-adopted per-arm diagnostic masks."""
    healthy = np.isin(accepted_classes, (1, 2))
    eligible_basin = np.isfinite(occurrence) & (occurrence >= 0.01)
    basin = ndimage.binary_propagation(
        seed_mask & eligible_basin,
        structure=CONNECTIVITY,
        mask=eligible_basin,
    )
    _, causeway_y = transform(
        "EPSG:4326", MEASUREMENT_CRS, [CAUSEWAY_REFERENCE_LON], [CAUSEWAY_LAT]
    )
    row_y = grid_transform.f + (np.arange(occurrence.shape[0]) + 0.5) * grid_transform.e
    north = row_y[:, None] >= causeway_y[0]
    south = ~north
    output = {}
    diagnostics = report["north_south_separate_fit_diagnostic"]
    for name, level in (("healthy", HEALTHY_LEVEL_FT), ("current", CURRENT_LEVEL_FT)):
        row = diagnostics[str(level)]
        output[name] = basin & (
            (north & (occurrence >= row["estimated_north_threshold"]))
            | (south & (occurrence >= row["estimated_south_threshold"]))
        )
    return output


def _difference_panel(
    occurrence: np.ndarray,
    joint: np.ndarray,
    separate: np.ndarray,
    size: tuple[int, int],
) -> Image.Image:
    """Render joint-only and separate-only areas for one elevation."""
    small_occurrence = _resize_float(occurrence, size)
    joint_only = _resize_nearest((joint & ~separate).astype(np.uint8), size).astype(bool)
    separate_only = _resize_nearest((separate & ~joint).astype(np.uint8), size).astype(bool)
    rgb = _occurrence_rgb(small_occurrence)
    rgb = (rgb.astype(np.float32) * 0.42 + np.array([248, 247, 243]) * 0.58).astype(np.uint8)
    rgb[joint_only] = (192, 65, 45)
    rgb[separate_only] = (51, 126, 143)
    return Image.fromarray(rgb, "RGB")


def write_arm_difference(
    occurrence: np.ndarray,
    accepted_classes: np.ndarray,
    arm_masks: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Save side-by-side joint-versus-per-arm disagreement maps."""
    single = _map_size(occurrence.shape, max_dimension=1050)
    healthy_joint = np.isin(accepted_classes, (1, 2))
    current_joint = accepted_classes == 2
    panels = [
        _difference_panel(occurrence, healthy_joint, arm_masks["healthy"], single),
        _difference_panel(occurrence, current_joint, arm_masks["current"], single),
    ]
    gutter = 28
    header = 54
    footer = 104
    canvas = Image.new(
        "RGB", (single[0] * 2 + gutter, single[1] + header + footer), "#f8f7f3"
    )
    canvas.paste(panels[0], (0, header))
    canvas.paste(panels[1], (single[0] + gutter, header))
    draw = ImageDraw.Draw(canvas)
    heading = _font(20)
    draw.text((16, 14), "Healthy · 4,198.0 ft", fill="#242927", font=heading)
    draw.text((single[0] + gutter + 16, 14), "Current · 4,189.6 ft", fill="#242927", font=heading)
    _draw_legend(
        canvas,
        "Joint-fit versus diagnostic per-arm masks",
        [
            ("joint fit only", (192, 65, 45)),
            ("per-arm fit only", (51, 126, 143)),
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def run_phase1_checks() -> dict:
    """Run all requested pre-Phase-2 diagnostics without altering Phase 1."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    with rasterio.open(DATA_DIR / "gsw_occurrence_gsl_utm12.tif") as dataset:
        occurrence = dataset.read(1)
        grid_transform = dataset.transform
    with rasterio.open(DATA_DIR / "gsl_classes_utm12.tif") as dataset:
        classes = dataset.read(1)

    valid = np.isfinite(occurrence)
    healthy = np.isin(classes, (1, 2))
    current = classes == 2
    seed_mask = np.zeros(classes.shape, dtype=bool)
    for seed in report["lake_seeds"].values():
        seed_mask[int(seed["row"]), int(seed["column"])] = True

    robust_healthy, detached_healthy_core = eroded_connected_mask(healthy, seed_mask)
    robust_current, detached_current_core = eroded_connected_mask(current, seed_mask)
    robust_classes = np.full(classes.shape, NODATA_BYTE, dtype=np.uint8)
    robust_classes[valid] = 0
    robust_classes[robust_healthy & ~robust_current] = 1
    robust_classes[robust_current] = 2
    leakage = class_transition_report(classes, robust_classes)
    leakage["erosion_pixels"] = EROSION_PIXELS
    leakage["erosion_distance_m"] = EROSION_PIXELS * MEASUREMENT_RESOLUTION_M
    leakage["accepted_masks_changed"] = False

    candidate_mask, candidates = _candidate_regions(
        occurrence,
        healthy,
        current,
        detached_healthy_core,
        detached_current_core,
        robust_healthy,
        robust_current,
        grid_transform,
    )
    no_data, no_data_categories = nodata_report(valid, healthy, current)
    arm_masks = per_arm_masks(occurrence, classes, seed_mask, grid_transform, report)
    arm_summary = {}
    for name, level, joint in (
        ("healthy", HEALTHY_LEVEL_FT, healthy),
        ("current", CURRENT_LEVEL_FT, current),
    ):
        separate = arm_masks[name]
        union = joint | separate
        changed = joint ^ separate
        arm_summary[name] = {
            "level_ft_ngvd29": level,
            "joint_only_km2": float((joint & ~separate).sum()) * PIXEL_KM2,
            "per_arm_only_km2": float((separate & ~joint).sum()) * PIXEL_KM2,
            "symmetric_difference_km2": float(changed.sum()) * PIXEL_KM2,
            "symmetric_difference_percent_of_union": 100.0
            * float(changed.sum())
            / max(float(union.sum()), 1.0),
        }

    write_labeled_overlay(
        occurrence, classes, grid_transform, OUT_DIR / "overlay_labeled.png"
    )
    write_managed_candidates(
        occurrence, candidate_mask, candidates, OUT_DIR / "managed_candidates.png"
    )
    write_nodata_map(
        occurrence, no_data_categories, OUT_DIR / "nodata_proximity.png"
    )
    write_arm_difference(
        occurrence, classes, arm_masks, OUT_DIR / "arm_difference.png"
    )

    output = {
        "managed_water_leakage_test": leakage,
        "managed_candidates": candidates,
        "managed_candidate_total_km2": float(
            sum(candidate["area_km2"] for candidate in candidates)
        ),
        "managed_candidate_rule": {
            "source": "parts of an accepted lake mask disconnected by two-pixel erosion",
            "minimum_area_km2": 0.5,
            "rectilinear": "axis-aligned bounding-box fill ratio >= 0.42",
            "near_constant_high_occurrence": "mean >= 95 and standard deviation <= 5 percentage points",
            "automatic_exclusions": 0,
        },
        "nodata": no_data,
        "north_south": {
            "thresholds_and_targets": report["north_south_separate_fit_diagnostic"],
            "mask_difference": arm_summary,
        },
        "outputs": {
            "overlay_labeled": str(OUT_DIR / "overlay_labeled.png"),
            "managed_candidates": str(OUT_DIR / "managed_candidates.png"),
            "nodata_proximity": str(OUT_DIR / "nodata_proximity.png"),
            "arm_difference": str(OUT_DIR / "arm_difference.png"),
        },
    }
    CHECKS_REPORT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(json.dumps(run_phase1_checks(), indent=2))
