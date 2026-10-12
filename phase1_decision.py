"""Apply the uniform N1-N19 managed-water decision rule.

The rule is intentionally mechanical: a screened region is managed only when
at least 90 percent detaches after removal of one-pixel water connections and
the region is rectilinear or spectrally near-constant.  Willard Reservoir is
checked against an official Census areal-hydrography polygon.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from scipy import ndimage

from data import (
    CONNECTIVITY,
    CURRENT_LEVEL_FT,
    DATA_DIR,
    HEALTHY_LEVEL_FT,
    NODATA_BYTE,
    OUT_DIR,
    rank_depth,
    write_geotiff,
)
from phase1_checks import PIXEL_KM2, SQUARE_CONNECTIVITY
from phase1_managed import (
    COMPASS_CANDIDATES_PATH,
    LOOSE_HIGH_MEAN,
    LOOSE_HIGH_STD,
    LOOSE_RECTANGULARITY,
    MANAGED_MASK_PATH,
    MANAGED_REPORT_PATH,
    REPORT_PATH,
    _calibrate_without_managed,
    _seed_mask,
    _write_class_diagnostic,
)


DECISION_REPORT_PATH = OUT_DIR / "managed_decision_report.json"
WILLARD_GEOJSON = DATA_DIR / "raw" / "willard_reservoir.geojson"
WILLARD_OID = "110480750164"
DETACHMENT_MINIMUM = 0.90


def _one_pixel_connected(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """Remove one-pixel water necks, seed both arms, and restore in-mask."""
    eroded = ndimage.binary_erosion(
        mask, structure=SQUARE_CONNECTIVITY, iterations=1, border_value=0
    )
    connected = ndimage.binary_propagation(
        seed_mask & eroded, structure=CONNECTIVITY, mask=eroded
    )
    return ndimage.binary_dilation(
        connected,
        structure=SQUARE_CONNECTIVITY,
        iterations=1,
        mask=mask,
    )


def _willard_mask(
    shape: tuple[int, int], grid_transform: rasterio.Affine
) -> tuple[np.ndarray, dict]:
    """Rasterize the named 2026 Census Willard Reservoir polygon."""
    collection = json.loads(WILLARD_GEOJSON.read_text(encoding="utf-8"))
    feature = next(
        item
        for item in collection["features"]
        if item["properties"].get("OID") == WILLARD_OID
    )
    geometry = transform_geom(
        "EPSG:4326", "EPSG:32612", feature["geometry"], precision=1
    )
    mask = rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=grid_transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    return mask, feature["properties"]


def _reclassify(
    occurrence: np.ndarray,
    managed: np.ndarray,
    seed_mask: np.ndarray,
    old_calibration: dict,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Recalibrate both levels and return classes, depth, fit, and areas."""
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
    if not healthy_threshold < current_threshold or np.any(current & ~healthy):
        raise RuntimeError("Second managed recalibration is not nested and monotonic")
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
    calibration = {
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
    area = {
        "water_now": float(current.sum()) * PIXEL_KM2,
        "lost_water": float(lost.sum()) * PIXEL_KM2,
        "land": float(land.sum()) * PIXEL_KM2,
        "managed": float(managed.sum()) * PIXEL_KM2,
        "nodata": float((~valid).sum()) * PIXEL_KM2,
    }
    return classes, lost_depth, calibration, area


def run_managed_decision() -> dict:
    """Apply the rule, update Phase 1 rasters, and return the audit report."""
    phase_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    managed_report = json.loads(MANAGED_REPORT_PATH.read_text(encoding="utf-8"))
    with rasterio.open(DATA_DIR / "gsw_occurrence_gsl_utm12.tif") as dataset:
        occurrence = dataset.read(1)
        grid_transform = dataset.transform
    with rasterio.open(DATA_DIR / "gsl_classes_utm12.tif") as dataset:
        old_classes = dataset.read(1)
    with rasterio.open(MANAGED_MASK_PATH) as dataset:
        old_managed = dataset.read(1).astype(bool)
    with rasterio.open(COMPASS_CANDIDATES_PATH) as dataset:
        candidate_ids = dataset.read(1)

    seed_mask = _seed_mask(old_classes.shape, phase_report)
    healthy_connected = _one_pixel_connected(np.isin(old_classes, (1, 2)), seed_mask)
    current_connected = _one_pixel_connected(old_classes == 2, seed_mask)
    willard, willard_properties = _willard_mask(old_classes.shape, grid_transform)

    rows = []
    added_managed = np.zeros(old_classes.shape, dtype=bool)
    for numeric_id, candidate in enumerate(
        managed_report["loose_compass_screen"]["candidates"], start=1
    ):
        mask = candidate_ids == numeric_id
        if not mask.any() or candidate["id"] != f"N{numeric_id}":
            raise RuntimeError(f"Candidate raster/report mismatch at N{numeric_id}")
        retained = (
            (mask & (old_classes == 2) & current_connected)
            | (mask & (old_classes == 1) & healthy_connected)
        )
        detached_fraction = 1.0 - float(retained.sum()) / float(mask.sum())
        rectilinear = (
            float(candidate["bounding_box_fill_ratio"]) >= LOOSE_RECTANGULARITY
        )
        near_constant = (
            float(candidate["mean_occurrence"]) >= LOOSE_HIGH_MEAN
            and float(candidate["occurrence_std"]) <= LOOSE_HIGH_STD
        )
        willard_overlap_km2 = float((mask & willard).sum()) * PIXEL_KM2
        willard_overlap = willard_overlap_km2 > 0.0
        managed_by_rule = detached_fraction >= DETACHMENT_MINIMUM and (
            rectilinear or near_constant
        )
        decision = "managed" if managed_by_rule or willard_overlap else "keep"
        if decision == "managed":
            added_managed |= mask
        rows.append(
            {
                "id": candidate["id"],
                "area_km2": float(mask.sum()) * PIXEL_KM2,
                "region": candidate["review_area"].replace("_ponds", ""),
                "centroid_wgs84": candidate["centroid_wgs84"],
                "detached_fraction": detached_fraction,
                "rectilinear": rectilinear,
                "near_constant_occurrence": near_constant,
                "willard_overlap_km2": willard_overlap_km2,
                "willard_override": willard_overlap,
                "decision": decision,
            }
        )

    managed = old_managed | added_managed
    old_calibration = phase_report["calibration"]
    old_area = phase_report["class_area_km2"]
    classes, lost_depth, calibration, area = _reclassify(
        occurrence, managed, seed_mask, old_calibration
    )
    if any(abs(row["error_percent"]) >= 1.0 for row in calibration.values()):
        raise RuntimeError(f"Second recalibration missed one-percent tolerance: {calibration}")

    classes_path = DATA_DIR / "gsl_classes_utm12.tif"
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
        DATA_DIR / "gsl_lost_depth_ft_utm12.tif",
        lost_depth,
        grid_transform,
        np.nan,
        "float32",
    )
    _write_class_diagnostic(classes, OUT_DIR / "phase1_classes.png")

    report = {
        "rule": {
            "detached_fraction_minimum": DETACHMENT_MINIMUM,
            "shape_requirement": "rectilinear OR near-constant occurrence",
            "detachment_method": "one-pixel square erosion, four-neighbour seeded propagation from both arms, one-pixel dilation within the accepted class mask",
            "land_pixels_in_a_candidate": "counted as detached because they are outside seeded accepted-water masks",
        },
        "willard_reservoir": {
            "source": "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Hydro/MapServer/1",
            "properties": willard_properties,
            "n4_overlap_km2": next(row["willard_overlap_km2"] for row in rows if row["id"] == "N4"),
            "n4_override_applied": next(row["willard_override"] for row in rows if row["id"] == "N4"),
        },
        "candidates": rows,
        "excluded_ids": [row["id"] for row in rows if row["decision"] == "managed"],
        "kept_ids": [row["id"] for row in rows if row["decision"] == "keep"],
        "recalibration": {
            "before": old_calibration,
            "after": calibration,
            "class_area_before_km2": old_area,
            "class_area_after_km2": area,
        },
    }
    DECISION_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    phase_report["calibration_before_candidate_rule"] = old_calibration
    phase_report["class_area_before_candidate_rule"] = old_area
    phase_report["calibration"] = calibration
    phase_report["class_area_km2"] = area
    phase_report["managed_candidate_rule"] = report
    REPORT_PATH.write_text(json.dumps(phase_report, indent=2) + "\n", encoding="utf-8")
    (DATA_DIR / "gsw_occurrence_gsl.metadata.json").write_text(
        json.dumps(phase_report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_managed_decision(), indent=2))
