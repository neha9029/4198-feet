"""Orchestrate Phase 2 geometry, spectral solving, and previews."""

from __future__ import annotations

import json

from data import DATA_DIR, SEED
from mesh import MESH_REPORT_PATH, build_mesh
from metamers import REPORT_PATH as METAMER_REPORT_PATH, solve_metamers
from render import render_previews


REPORT_PATH = DATA_DIR.parent / "out" / "phase2" / "phase2_report.json"


def _write_report(seed: int, mesh: dict, metamers: dict, render: dict) -> dict:
    arm_targets_pass = all(
        arm["final"]["median_delta_e76_a"] >= 15.0
        and arm["final"]["mean_coherence"] >= 0.90
        and arm["final"]["failure_rate"] == 0.0
        for arm in metamers["arms"].values()
    )
    tests_pass = bool(
        metamers["deterministic_repeat_equal"]
        and metamers["repeat_success_pattern_equal"]
        and metamers["final"]["failure_rate"] == 0.0
        and metamers["final"]["spectra_min"] >= metamers["bounds"][0] - 1e-8
        and metamers["final"]["spectra_max"] <= metamers["bounds"][1] + 1e-8
        and metamers["d65_camouflage"]["passes_target"]
        and mesh["valid_data_crop"]["contains_nodata"] is False
        and mesh["class_triangle_counts"]["nodata"] == 0
        and render["d65_identity"]
        and arm_targets_pass
    )
    report = {
        "seed": seed,
        "mesh": mesh,
        "metamers": metamers,
        "render": render,
        "arm_reveal_targets_pass": arm_targets_pass,
        "tests_pass": tests_pass,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_phase2(seed: int = SEED) -> dict:
    """Build and verify all Phase 2 deliverables deterministically."""
    mesh = build_mesh(seed)
    metamers = solve_metamers(seed)
    render = render_previews(long_side=1600)
    return _write_report(seed, mesh, metamers, render)


def refresh_phase2_report(seed: int = SEED) -> dict:
    """Refresh previews and the combined report from verified cached solves."""
    mesh = json.loads(MESH_REPORT_PATH.read_text(encoding="utf-8"))
    metamers = json.loads(METAMER_REPORT_PATH.read_text(encoding="utf-8"))
    render = render_previews(long_side=1600)
    return _write_report(seed, mesh, metamers, render)


if __name__ == "__main__":
    print(json.dumps(run_phase2(), indent=2))
