"""Checks for the post-calibration Phase 1 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "out" / "phase1" / "phase1_checks_report.json"


def checks() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_checks_do_not_automatically_change_accepted_masks() -> None:
    report = checks()
    assert report["managed_water_leakage_test"]["accepted_masks_changed"] is False
    assert report["managed_candidate_rule"]["automatic_exclusions"] == 0
    assert len(report["managed_candidates"]) == 7


def test_all_requested_diagnostic_images_exist() -> None:
    for output in checks()["outputs"].values():
        path = Path(output)
        assert path.exists()
        with Image.open(path) as image:
            assert image.width >= 1000
            assert image.height >= 1000


def test_reported_arm_differences_match_phase1_diagnostic() -> None:
    report = checks()["north_south"]
    thresholds = report["thresholds_and_targets"]
    masks = report["mask_difference"]
    assert masks["healthy"]["symmetric_difference_percent_of_union"] == thresholds["4198.0"]["symmetric_difference_percent_of_union"]
    assert masks["current"]["symmetric_difference_percent_of_union"] == thresholds["4189.6"]["symmetric_difference_percent_of_union"]
