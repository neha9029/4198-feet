"""Checks for the uniform N1-N19 managed-water decision."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "out" / "phase1" / "managed_decision_report.json"


def result() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_uniform_rule_is_applied_to_all_candidates() -> None:
    report = result()
    assert len(report["candidates"]) == 19
    for candidate in report["candidates"]:
        rule_managed = candidate["detached_fraction"] >= 0.90 and (
            candidate["rectilinear"] or candidate["near_constant_occurrence"]
        )
        expected = "managed" if rule_managed or candidate["willard_override"] else "keep"
        assert candidate["decision"] == expected


def test_willard_overlap_and_exclusions() -> None:
    report = result()
    assert report["willard_reservoir"]["n4_overlap_km2"] == 0.0
    assert report["willard_reservoir"]["n4_override_applied"] is False
    assert report["excluded_ids"] == [
        "N1", "N4", "N6", "N7", "N9", "N10", "N15", "N16", "N17", "N19"
    ]


def test_second_recalibration_is_within_one_percent() -> None:
    for row in result()["recalibration"]["after"].values():
        assert abs(row["error_percent"]) < 1.0
