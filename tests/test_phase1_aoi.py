"""Checks for whole-complex pond AOIs and their final recalibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "out" / "phase1" / "managed_aoi_report.json"


def result() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_all_accepted_water_is_removed_from_whole_complex_aois() -> None:
    report = result()
    assert report["remaining_water_pixels_inside_aois"] == 0
    with rasterio.open(ROOT / "data" / "gsl_managed_aoi_utm12.tif") as dataset:
        aois = dataset.read(1) > 0
    with rasterio.open(ROOT / "data" / "gsl_classes_utm12.tif") as dataset:
        classes = dataset.read(1)
    assert not np.any(aois & np.isin(classes, (1, 2)))


def test_aoi_recalibration_is_within_one_percent() -> None:
    report = result()
    assert report["new_managed_area_km2"] > 0.0
    for row in report["recalibration"]["after"].values():
        assert abs(row["error_percent"]) < 1.0


def test_willard_uses_official_polygon_and_does_not_overlap_water() -> None:
    report = result()
    willard = report["aoi_construction"]["willard_bay_reservoir"]
    assert willard["properties"]["NAME"] == "Willard Reservoir"
    assert report["new_managed_by_aoi_km2"]["willard_bay_reservoir"] == 0.0
