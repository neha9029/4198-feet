"""Apply whole-complex managed-water AOIs and recalibrate Phase 1.

The earlier candidate screen remains an audit trail.  This step converts the
screened pond complexes into stable AOIs, moves every accepted-water pixel in
those AOIs to ``managed``, and iterates calibration until no water class can
re-enter an AOI.
"""

from __future__ import annotations

import json

import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from scipy.spatial import ConvexHull

from data import DATA_DIR, NODATA_BYTE, OUT_DIR, write_geotiff
from phase1_checks import PIXEL_KM2, SQUARE_CONNECTIVITY
from phase1_decision import WILLARD_GEOJSON, _reclassify, _willard_mask
from phase1_managed import (
    COMPASS_CANDIDATES_PATH,
    MANAGED_MASK_PATH,
    MANAGED_REPORT_PATH,
    REPORT_PATH,
    _seed_mask,
    _write_class_diagnostic,
)


AOI_REPORT_PATH = OUT_DIR / "managed_aoi_report.json"
AOI_MASK_PATH = DATA_DIR / "gsl_managed_aoi_utm12.tif"
WEST_CLUSTER_NORTHING_MAX = 4_580_000.0
EAST_CLOSING_PIXELS = 10  # 300 m: closes dikes/gaps without spanning Bear River Bay.
MAX_RECALIBRATION_PASSES = 12

USACE_NOTICE_URL = (
    "https://www.spk.usace.army.mil/Portals/12/documents/regulatory/eis/"
    "200700121-eis/200700121-PN-posted-07-09-2013.pdf"
)
SEC_TECHNICAL_REPORT_URL = (
    "https://www.sec.gov/Archives/edgar/data/1227654/000122765421000300/"
    "cmp-20210930xxex961ogdenpo.htm"
)
TIGERWEB_HYDRO_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Hydro/"
    "MapServer/1"
)


def _convex_hull_mask(mask: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    """Rasterize the outer hull of a documented contiguous pond cluster."""
    rows, columns = np.nonzero(mask)
    if len(rows) < 3:
        return mask.copy()
    points = np.column_stack([columns, rows])
    hull = ConvexHull(points)
    ring = [
        (
            transform.c + (column + 0.5) * transform.a,
            transform.f + (row + 0.5) * transform.e,
        )
        for column, row in points[hull.vertices]
    ]
    return rasterize(
        [({"type": "Polygon", "coordinates": [ring]}, 1)],
        out_shape=mask.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def _compass_aoi_masks(
    candidate_ids: np.ndarray,
    managed_report: dict,
    transform: rasterio.Affine,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build east/west whole-complex masks from the audited loose screen."""
    candidates = managed_report["loose_compass_screen"]["candidates"]
    east_ids = [
        index
        for index, candidate in enumerate(candidates, start=1)
        if candidate["review_area"] == "east_ponds"
    ]
    west_ids = [
        index
        for index, candidate in enumerate(candidates, start=1)
        if candidate["review_area"] == "west_ponds"
        and float(candidate["centroid_utm12"][1]) <= WEST_CLUSTER_NORTHING_MAX
    ]
    west_outliers = [
        candidate["id"]
        for candidate in candidates
        if candidate["review_area"] == "west_ponds"
        and float(candidate["centroid_utm12"][1]) > WEST_CLUSTER_NORTHING_MAX
    ]

    east_seed = np.isin(candidate_ids, east_ids)
    east = ndimage.binary_closing(
        east_seed,
        structure=SQUARE_CONNECTIVITY,
        iterations=EAST_CLOSING_PIXELS,
    )
    east = ndimage.binary_fill_holes(east)
    west_seed = np.isin(candidate_ids, west_ids)
    west = _convex_hull_mask(west_seed, transform)
    return {"compass_east": east, "compass_west": west}, {
        "compass_east": {
            "candidate_ids": [f"N{value}" for value in east_ids],
            "construction": (
                "union of all east-screen pond regions, 300 m binary closing, "
                "then internal-hole fill"
            ),
        },
        "compass_west": {
            "candidate_ids": [f"N{value}" for value in west_ids],
            "construction": (
                "convex outer hull of the contiguous Clyman/Gunnison west-pond "
                "screen cluster"
            ),
            "excluded_noncluster_outliers": west_outliers,
        },
    }


def run_managed_aoi() -> dict:
    """Apply all whole-complex AOIs, recalibrate, and write an audit report."""
    phase_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    managed_report = json.loads(MANAGED_REPORT_PATH.read_text(encoding="utf-8"))
    with rasterio.open(DATA_DIR / "gsw_occurrence_gsl_utm12.tif") as dataset:
        occurrence = dataset.read(1)
        grid_transform = dataset.transform
    with rasterio.open(DATA_DIR / "gsl_classes_utm12.tif") as dataset:
        classes = dataset.read(1)
    with rasterio.open(MANAGED_MASK_PATH) as dataset:
        managed = dataset.read(1).astype(bool)
    with rasterio.open(COMPASS_CANDIDATES_PATH) as dataset:
        candidate_ids = dataset.read(1)

    seed_mask = _seed_mask(classes.shape, phase_report)
    old_calibration = phase_report["calibration"]
    old_area = phase_report["class_area_km2"]
    compass_masks, methods = _compass_aoi_masks(
        candidate_ids, managed_report, grid_transform
    )
    willard, willard_properties = _willard_mask(classes.shape, grid_transform)
    aois = {**compass_masks, "willard_bay_reservoir": willard}
    union = np.logical_or.reduce(list(aois.values()))

    initial_managed = managed.copy()
    additions_by_pass: list[dict] = []
    calibration = old_calibration
    area = old_area
    lost_depth = np.full(occurrence.shape, np.nan, dtype=np.float32)
    for pass_index in range(1, MAX_RECALIBRATION_PASSES + 1):
        accepted = np.isin(classes, (1, 2))
        newly_managed = accepted & union & ~managed
        if not newly_managed.any():
            break
        additions_by_pass.append(
            {
                "pass": pass_index,
                "added_km2": float(newly_managed.sum()) * PIXEL_KM2,
                "by_aoi_km2": {
                    name: float((newly_managed & mask).sum()) * PIXEL_KM2
                    for name, mask in aois.items()
                },
            }
        )
        managed |= newly_managed
        classes, lost_depth, calibration, area = _reclassify(
            occurrence, managed, seed_mask, old_calibration
        )
    else:
        raise RuntimeError("Whole-complex AOI recalibration did not converge")

    remaining = np.isin(classes, (1, 2)) & union
    if remaining.any():
        raise RuntimeError("Accepted water remains inside a managed-water AOI")
    if any(abs(row["error_percent"]) >= 1.0 for row in calibration.values()):
        raise RuntimeError(f"Whole-complex recalibration missed tolerance: {calibration}")

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
    aoi_ids = np.zeros(classes.shape, dtype=np.uint8)
    for value, name in enumerate(
        ("compass_west", "compass_east", "willard_bay_reservoir"), start=1
    ):
        aoi_ids[aois[name]] = value
    write_geotiff(AOI_MASK_PATH, aoi_ids, grid_transform, 0, "uint8")
    _write_class_diagnostic(classes, OUT_DIR / "phase1_classes.png")

    report = {
        "rule": (
            "Every water_now or lost_water pixel inside the Compass east, "
            "Compass west, or TIGERweb Willard Bay AOI is managed; calibration "
            "repeats until no accepted-water pixel can re-enter an AOI."
        ),
        "sources": {
            "compass_usace_public_notice": USACE_NOTICE_URL,
            "compass_sec_technical_report_figures_13_1_to_13_3": SEC_TECHNICAL_REPORT_URL,
            "willard_tigerweb_areal_hydrography": TIGERWEB_HYDRO_URL,
        },
        "aoi_construction": {
            **methods,
            "willard_bay_reservoir": {
                "construction": "official TIGERweb Willard Reservoir polygon",
                "properties": willard_properties,
            },
        },
        "aoi_area_km2": {
            name: float(mask.sum()) * PIXEL_KM2 for name, mask in aois.items()
        },
        "new_managed_area_km2": float((managed & ~initial_managed).sum()) * PIXEL_KM2,
        "new_managed_by_aoi_km2": {
            name: float(((managed & ~initial_managed) & mask).sum()) * PIXEL_KM2
            for name, mask in aois.items()
        },
        "passes": additions_by_pass,
        "remaining_water_pixels_inside_aois": int(remaining.sum()),
        "recalibration": {
            "before": old_calibration,
            "after": calibration,
            "class_area_before_km2": old_area,
            "class_area_after_km2": area,
        },
        "outputs": {
            "aoi_mask": str(AOI_MASK_PATH),
            "managed_mask": str(MANAGED_MASK_PATH),
            "classes": str(classes_path),
        },
    }
    AOI_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    phase_report["calibration_before_whole_complex_aois"] = old_calibration
    phase_report["class_area_before_whole_complex_aois"] = old_area
    phase_report["calibration"] = calibration
    phase_report["class_area_km2"] = area
    phase_report["managed_whole_complex_aois"] = report
    REPORT_PATH.write_text(json.dumps(phase_report, indent=2) + "\n", encoding="utf-8")
    (DATA_DIR / "gsw_occurrence_gsl.metadata.json").write_text(
        json.dumps(phase_report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_managed_aoi(), indent=2))
