"""Phase 2 spectral, rendering, and deterministic-seed tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "out" / "phase2" / "phase2_report.json"


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_d65_rendered_identity_is_exact() -> None:
    baseline = np.asarray(Image.open(ROOT / "out" / "phase2" / "d65_baseline.png"))
    final = np.asarray(Image.open(ROOT / "out" / "phase2" / "preview_d65.png"))
    assert np.max(np.abs(baseline.astype(int) - final.astype(int))) == 0
    assert report()["render"]["d65_identity"] is True


def test_spectra_are_bounded_and_d65_xyz_is_preserved() -> None:
    spectra = np.load(ROOT / "data" / "phase2_final_spectra.npy")
    assert spectra.min() >= 0.015
    assert spectra.max() <= 0.96
    assert report()["metamers"]["final"]["d65_xyz_max_abs_difference"] < 1e-10


def test_seeded_solver_is_deterministic_and_complete() -> None:
    result = report()
    assert result["seed"] == 1931
    assert result["metamers"]["deterministic_repeat_equal"] is True
    assert result["metamers"]["repeat_success_pattern_equal"] is True
    assert result["metamers"]["final"]["failure_rate"] == 0.0
    assert result["tests_pass"] is True


def test_daylight_camouflage_and_six_reveal_trials() -> None:
    metamers = report()["metamers"]
    camouflage = metamers["d65_camouflage"]
    assert camouflage["passes_target"] is True
    assert camouflage["median_delta_e76_d65_lost_to_land"] < 3.0
    assert len(metamers["option_sweep"]) == 6
    assert {row["smoothness_weight"] for row in metamers["option_sweep"]} == {
        50.0,
        100.0,
        200.0,
    }


def test_render_crop_has_no_nodata_triangles() -> None:
    mesh = report()["mesh"]
    assert mesh["valid_data_crop"]["contains_nodata"] is False
    assert mesh["class_triangle_counts"]["nodata"] == 0
