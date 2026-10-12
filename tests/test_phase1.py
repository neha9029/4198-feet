"""Checks for the generated Phase 1 calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "gsw_occurrence_gsl.metadata.json"


def report() -> dict:
    return json.loads(METADATA.read_text(encoding="utf-8"))


def test_calibration_is_within_one_percent() -> None:
    calibration = report()["calibration"]
    assert abs(calibration["healthy"]["error_percent"]) < 1.0
    assert abs(calibration["current"]["error_percent"]) < 1.0
    assert calibration["healthy"]["threshold"] < calibration["current"]["threshold"]


def test_class_areas_match_raster_counts() -> None:
    metadata = report()
    with rasterio.open(ROOT / "data" / "gsl_classes_utm12.tif") as dataset:
        classes = dataset.read(1)
        pixel_km2 = abs(dataset.transform.a * dataset.transform.e) / 1_000_000.0
    expected = metadata["class_area_km2"]
    assert np.isclose(np.count_nonzero(classes == 2) * pixel_km2, expected["water_now"])
    assert np.isclose(np.count_nonzero(classes == 1) * pixel_km2, expected["lost_water"])
    assert np.isclose(np.count_nonzero(classes == 0) * pixel_km2, expected["land"])
    assert np.isclose(np.count_nonzero(classes == 3) * pixel_km2, expected["managed"])


def test_pseudo_depth_bounds_and_datum() -> None:
    metadata = report()
    assert metadata["source"]["vertical_datum"] == "NGVD29"
    with rasterio.open(ROOT / "data" / "gsl_lost_depth_ft_utm12.tif") as dataset:
        depth = dataset.read(1)
    finite = depth[np.isfinite(depth)]
    assert finite.min() >= 0.0
    assert finite.max() <= 8.4
