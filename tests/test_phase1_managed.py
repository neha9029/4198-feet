"""Checks for approved managed exclusions and post-exclusion calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "out" / "phase1" / "phase1_managed_report.json"


def report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_c1_is_confirmed_and_only_approved_candidates_are_managed() -> None:
    result = report()
    assert result["c1_connectivity_confirmation"]["confirmed_diked_off"] is True
    assert result["c1_connectivity_confirmation"]["detached_percent"] == 100.0
    assert result["approved_managed_exclusions"]["automatic_additions"] == 0
    assert set(result["approved_managed_exclusions"]["candidate_ids"]) == {
        "C1", "C2", "C3", "C4", "C5", "C6", "C7"
    }


def test_managed_class_is_distinct_and_recalibration_is_within_tolerance() -> None:
    result = report()
    final = json.loads(
        (ROOT / "out" / "phase1" / "managed_decision_report.json").read_text(
            encoding="utf-8"
        )
    )
    with rasterio.open(ROOT / "data" / "gsl_classes_utm12.tif") as dataset:
        classes = dataset.read(1)
        pixel_km2 = abs(dataset.transform.a * dataset.transform.e) / 1_000_000.0
    managed_area = np.count_nonzero(classes == 3) * pixel_km2
    assert managed_area >= result["approved_managed_exclusions"]["area_km2"]
    assert np.isclose(
        managed_area, final["recalibration"]["class_area_after_km2"]["managed"]
    )
    for row in result["recalibration"]["after"].values():
        assert abs(row["error_percent"]) < 1.0


def test_compass_screen_is_non_destructive_and_reference_is_in_crop() -> None:
    result = report()
    assert result["compass_reference"]["inside_crop"] is True
    assert result["loose_compass_screen"]["excluded_pending_decision"] == 0
    assert len(result["loose_compass_screen"]["candidates"]) > 0
