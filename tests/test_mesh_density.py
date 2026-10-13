"""High-density mesh comparison tests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "out" / "phase2" / "mesh_density"


def test_all_density_trials_pass_and_production_is_25k() -> None:
    report = json.loads(
        (OUTPUT / "mesh_density_comparison_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["production_target_points"] == 25_000
    assert report["all_density_tests_pass"] is True
    assert [item["actual_points"] for item in report["variants"]] == [
        15_000,
        25_000,
        30_000,
    ]
    for item in report["variants"]:
        assert item["density_camouflage"]["relative_difference"] <= 0.10
        assert item["spectral_camouflage"]["passes_target"] is True


def test_density_comparison_is_1600_px() -> None:
    image = Image.open(OUTPUT / "mesh_density_comparison_1600.png")
    assert image.width == 1600
