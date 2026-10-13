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


def test_daylight_camouflage_and_per_arm_reveal_trials() -> None:
    metamers = report()["metamers"]
    camouflage = metamers["d65_camouflage"]
    assert camouflage["passes_target"] is True
    assert camouflage["median_delta_e76_d65_lost_to_land"] < 3.0
    assert len(metamers["option_sweep"]) == 18
    assert {row["smoothness_weight"] for row in metamers["option_sweep"]} == {
        20.0,
        50.0,
        200.0,
    }
    assert {row["angle_degrees"] for row in metamers["option_sweep"] if row["arm"] == "north"} == {0.0, 15.0, -15.0}
    assert {row["angle_degrees"] for row in metamers["option_sweep"] if row["arm"] == "south"} == {-180.0, -165.0, -150.0}
    for arm in metamers["arms"].values():
        assert arm["final"]["median_delta_e76_a"] >= 15.0
        assert arm["final"]["mean_coherence"] >= 0.90
        assert arm["final"]["failure_rate"] == 0.0
        assert [row["tercile"] for row in arm["depth_terciles"]] == [
            "shallow",
            "middle",
            "deep",
        ]


def test_render_crop_has_no_nodata_triangles() -> None:
    mesh = report()["mesh"]
    assert mesh["valid_data_crop"]["contains_nodata"] is False
    assert mesh["class_triangle_counts"]["nodata"] == 0


def test_causeway_arm_split_and_current_water_colours() -> None:
    result = report()
    mesh_report = result["mesh"]
    north = mesh_report["lost_water_by_arm"]["north"]
    south = mesh_report["lost_water_by_arm"]["south"]
    assert north["area_km2"] > 0.0 and south["area_km2"] > 0.0
    assert north["triangle_count"] + south["triangle_count"] == mesh_report["class_triangle_counts"]["lost_water"]

    mesh = np.load(ROOT / "data" / "gsl_mesh_phase2.npz")
    spectra = np.load(ROOT / "data" / "phase2_base_spectra.npy")
    from render import spectral_colours

    colours = spectral_colours(spectra, "D65")
    north_rgb = np.median(colours[(mesh["classes"] == 2) & (mesh["arms"] == 1)], axis=0)
    south_rgb = np.median(colours[(mesh["classes"] == 2) & (mesh["arms"] == 0)], axis=0)
    assert north_rgb[0] > north_rgb[2] > north_rgb[1]
    assert south_rgb[1] > south_rgb[0] and south_rgb[2] > south_rgb[0]


def test_midpoint_preview_exists() -> None:
    midpoint = Image.open(ROOT / "out" / "phase2" / "preview_midpoint.png")
    assert max(midpoint.size) == 1600
    assert report()["render"]["midpoint_illuminant"] == "50% D65 SPD + 50% 2856 K SPD"
