"""Reflectance families and coherent D65 metamer solver for Phase 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from data import DATA_DIR, OUT_DIR, SEED
from engine import (
    ADAPTATION,
    D65_SPD,
    WAVELENGTHS,
    XBAR,
    YBAR,
    ZBAR,
    adapt,
    illuminant,
    smooth_field,
    spectra_to_xyz,
)
from mesh import MESH_PATH


LOW = 0.015
HIGH = 0.96
ILLUMINANT_A_K = 2856
SWEEP_ANGLES_DEG = (-180, -135, -90, -45, 0, 45, 90, 135)
SWEEP_SAMPLE_SIZE = 96
NULL_BASIS_SIZE = 9
SPECTRA_BASE_PATH = DATA_DIR / "phase2_base_spectra.npy"
SPECTRA_FINAL_PATH = DATA_DIR / "phase2_final_spectra.npy"
REPORT_PATH = DATA_DIR.parent / "out" / "phase2" / "metamer_report.json"


def _gaussian(centre: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((WAVELENGTHS - centre) / sigma) ** 2)


def build_base_spectra(
    centroids: np.ndarray, classes: np.ndarray, occurrence: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """Build smooth, bounded land, water, and pale-salt spectral families."""
    rng = np.random.default_rng(seed)
    field_a = smooth_field(centroids, rng, waves=4, scale=1.15)
    field_b = smooth_field(centroids, rng, waves=3, scale=2.05)
    field_c = smooth_field(centroids, rng, waves=3, scale=0.72)
    grit = rng.normal(0.0, 0.008, size=len(centroids))

    # Earth/sage land: broad, low-chroma mineral reflectance with a restrained
    # green-red rise. Managed cells use this exact family.
    land = (
        0.22
        + 0.13 * _gaussian(585, 135)[None, :]
        + 0.055 * _gaussian(545, 55)[None, :]
        - 0.035 * _gaussian(435, 42)[None, :]
    )
    land = land * (0.82 + 0.20 * ((field_a + 1.0) / 2.0))[:, None]
    land += (0.025 * field_b + grit)[:, None]

    # The exposed-lakebed family is intentionally pale salt/playa. Lost-water
    # triangles start from this same family so daylight gives no class cue.
    salt = (
        0.61
        + 0.105 * _gaussian(590, 150)[None, :]
        + 0.025 * _gaussian(455, 58)[None, :]
        - 0.018 * _gaussian(690, 62)[None, :]
    )
    salt = salt * (0.91 + 0.10 * ((field_b + 1.0) / 2.0))[:, None]
    salt += (0.017 * field_c + grit * 0.5)[:, None]

    # Dark blue-green water: low albedo, with relatively more short/midwave
    # return and strong red absorption.
    water = (
        0.032
        + 0.115 * _gaussian(500, 62)[None, :]
        + 0.032 * _gaussian(440, 45)[None, :]
        - 0.017 * _gaussian(650, 75)[None, :]
    )
    water = water * (0.77 + 0.24 * ((field_c + 1.0) / 2.0))[:, None]
    water += (0.006 * field_a + grit * 0.25)[:, None]

    spectra = land.copy()
    salt_like_land = (classes == 0) & (occurrence >= 16.0)
    salt_mix = np.clip((occurrence - 16.0) / 45.0, 0.0, 1.0)
    spectra[salt_like_land] = (
        (1.0 - salt_mix[salt_like_land, None]) * land[salt_like_land]
        + salt_mix[salt_like_land, None] * salt[salt_like_land]
    )
    spectra[classes == 1] = salt[classes == 1]
    spectra[classes == 2] = water[classes == 2]
    spectra[classes == 3] = land[classes == 3]
    spectra[classes == 255] = 0.18
    return np.clip(spectra, LOW, HIGH)


def _xyz_constraint() -> np.ndarray:
    weight = 100.0 / np.sum(D65_SPD * YBAR)
    return weight * np.stack(
        [D65_SPD * XBAR, D65_SPD * YBAR, D65_SPD * ZBAR], axis=0
    )


def _null_basis() -> np.ndarray:
    """Return a smooth orthonormal basis lying in the D65 XYZ nullspace."""
    constraint = _xyz_constraint()
    projector = np.eye(len(WAVELENGTHS)) - constraint.T @ np.linalg.pinv(
        constraint @ constraint.T
    ) @ constraint
    x = np.linspace(0.0, 1.0, len(WAVELENGTHS))
    candidates = np.column_stack(
        [
            np.cos(np.pi * mode * x)
            for mode in range(1, NULL_BASIS_SIZE + 8)
        ]
    )
    projected = projector @ candidates
    q, _ = np.linalg.qr(projected)
    basis = q[:, :NULL_BASIS_SIZE]
    # Remove the last few floating-point ulps of D65 tristimulus leakage.
    basis = projector @ basis
    basis, _ = np.linalg.qr(basis)
    return basis[:, :NULL_BASIS_SIZE]


def _lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    ratio = xyz / white[None, :]
    f = np.where(
        ratio > delta**3,
        np.cbrt(ratio),
        ratio / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.column_stack(
        [116.0 * f[:, 1] - 16.0, 500.0 * (f[:, 0] - f[:, 1]), 200.0 * (f[:, 1] - f[:, 2])]
    )


def _lab_under_a(spectra: np.ndarray) -> np.ndarray:
    a_spd = illuminant(ILLUMINANT_A_K)
    source_white = spectra_to_xyz(np.ones((1, len(WAVELENGTHS))), a_spd)[0]
    d65_white = spectra_to_xyz(np.ones((1, len(WAVELENGTHS))), D65_SPD)[0]
    xyz = spectra_to_xyz(spectra, a_spd)
    adapted = adapt(xyz, source_white, d65_white, ADAPTATION)
    return _lab(adapted, d65_white)


def _direction(angle_deg: float) -> np.ndarray:
    radians = np.deg2rad(angle_deg)
    vector = np.array([-0.12, np.cos(radians), np.sin(radians)], dtype=float)
    return vector / np.linalg.norm(vector)


def _feasible_start(
    base: np.ndarray, basis: np.ndarray, rng: np.random.Generator, scale: float
) -> np.ndarray:
    if scale == 0.0:
        return np.zeros(basis.shape[1])
    for _ in range(8):
        z = rng.normal(0.0, scale, size=basis.shape[1])
        candidate = base + basis @ z
        if candidate.min() >= LOW and candidate.max() <= HIGH:
            return z
        scale *= 0.45
    return np.zeros(basis.shape[1])


def _solve_one(
    base: np.ndarray,
    depth_fraction: float,
    direction: np.ndarray,
    basis: np.ndarray,
    seed: int,
    restarts: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    baseline_lab = _lab_under_a(base[None, :])[0]
    magnitude = 3.2 + 13.8 * np.clip(depth_fraction, 0.0, 1.0) ** 0.72
    target = baseline_lab + magnitude * direction
    d2 = np.diff(np.eye(len(WAVELENGTHS)), n=2, axis=0)

    def objective(z: np.ndarray) -> float:
        spectrum = base + basis @ z
        delta = _lab_under_a(spectrum[None, :])[0] - baseline_lab
        along = float(delta @ direction)
        orthogonal = delta - along * direction
        smooth = d2 @ spectrum
        return (
            1.0 * (along - magnitude) ** 2
            + 2.8 * float(orthogonal @ orthogonal)
            + 7.5 * float(smooth @ smooth)
            + 0.015 * float(z @ z)
        )

    def bounds_constraint(z: np.ndarray) -> np.ndarray:
        spectrum = base + basis @ z
        return np.concatenate([spectrum - LOW, HIGH - spectrum])

    starts = [np.zeros(basis.shape[1])]
    for candidate_index in range(48):
        scale = 0.08 + 0.37 * (candidate_index / 47.0)
        z0 = _feasible_start(base, basis, rng, scale)
        if np.any(z0):
            starts.append(z0)
    starts = sorted(starts, key=objective)[:restarts]

    best = None
    best_z = starts[0]
    best_fun = objective(best_z)
    optimizer_succeeded = False
    best_iterations = 0
    for z0 in starts:
        result = minimize(
            objective,
            z0,
            method="SLSQP",
            constraints=({"type": "ineq", "fun": bounds_constraint},),
            options={"maxiter": 90, "ftol": 2e-9, "disp": False},
        )
        optimizer_succeeded = optimizer_succeeded or bool(result.success)
        if best is None or result.fun < best.fun:
            best = result
        if result.fun < best_fun:
            best_fun = float(result.fun)
            best_z = result.x
            best_iterations = int(result.nit)
    assert best is not None
    delta_spectrum = basis @ best_z
    alpha = 1.0
    falling = delta_spectrum < 0.0
    rising = delta_spectrum > 0.0
    if np.any(falling):
        alpha = min(alpha, float(np.min((base[falling] - LOW) / -delta_spectrum[falling])))
    if np.any(rising):
        alpha = min(alpha, float(np.min((HIGH - base[rising]) / delta_spectrum[rising])))
    if alpha < 1.0:
        alpha = max(0.0, alpha * (1.0 - 1e-10))
    solved = base + alpha * delta_spectrum
    solved_lab = _lab_under_a(solved[None, :])[0]
    shift = solved_lab - baseline_lab
    shift_norm = float(np.linalg.norm(shift))
    cosine = float(shift @ direction / max(shift_norm, 1e-12))
    xyz_error = float(
        np.max(np.abs(spectra_to_xyz(solved[None, :], D65_SPD) - spectra_to_xyz(base[None, :], D65_SPD)))
    )
    ok = bool(
        optimizer_succeeded
        and solved.min() >= LOW - 1e-8
        and solved.max() <= HIGH + 1e-8
        and xyz_error <= 1e-8
    )
    return solved, {
        "success": ok,
        "optimizer_success": optimizer_succeeded,
        "delta_e76": shift_norm,
        "coherence": cosine,
        "d65_xyz_max_error": xyz_error,
        "iterations": best_iterations,
    }


def _sample_lost_indices(lost_indices: np.ndarray) -> np.ndarray:
    if len(lost_indices) <= SWEEP_SAMPLE_SIZE:
        return lost_indices
    locations = np.rint(np.linspace(0, len(lost_indices) - 1, SWEEP_SAMPLE_SIZE)).astype(int)
    return lost_indices[locations]


def _run_direction(
    angle: float,
    sample_indices: np.ndarray,
    bases: np.ndarray,
    depths: np.ndarray,
    basis: np.ndarray,
) -> dict:
    direction = _direction(angle)
    diagnostics = []
    for order, triangle_index in enumerate(sample_indices):
        _, diagnostic = _solve_one(
            bases[triangle_index],
            depths[triangle_index],
            direction,
            basis,
            SEED + int(angle + 360) * 1000 + order,
            restarts=2,
        )
        diagnostics.append(diagnostic)
    delta_e = np.array([item["delta_e76"] for item in diagnostics])
    coherence = np.array([item["coherence"] for item in diagnostics])
    success = np.array([item["success"] for item in diagnostics])
    return {
        "angle_deg_ab": float(angle),
        "direction_lab": direction.tolist(),
        "sample_count": int(len(sample_indices)),
        "median_delta_e76_a": float(np.median(delta_e)),
        "min_delta_e76_a": float(np.min(delta_e)),
        "mean_coherence": float(np.mean(coherence)),
        "failure_rate": float(1.0 - np.mean(success)),
    }


def _choose_direction(sweep: list[dict]) -> dict:
    candidates = [item for item in sweep if item["direction_lab"][2] < -0.25]
    passing = [item for item in candidates if item["failure_rate"] <= 0.05]
    pool = passing or candidates
    return max(
        pool,
        key=lambda item: item["median_delta_e76_a"]
        + 0.30 * item["min_delta_e76_a"]
        + 4.0 * item["mean_coherence"]
        - 25.0 * item["failure_rate"],
    )


def _solve_all_lost(
    bases: np.ndarray,
    classes: np.ndarray,
    depths: np.ndarray,
    direction: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    final = bases.copy()
    diagnostics: list[dict] = []
    for order, triangle_index in enumerate(np.flatnonzero(classes == 1)):
        final[triangle_index], diagnostic = _solve_one(
            bases[triangle_index],
            depths[triangle_index],
            direction,
            basis,
            SEED + 900_000 + order,
            restarts=3,
        )
        diagnostic["triangle_index"] = int(triangle_index)
        diagnostics.append(diagnostic)
    return final, diagnostics


def solve_metamers(seed: int = SEED) -> dict:
    """Sweep coherent directions, solve every lost cell, and verify determinism."""
    mesh = np.load(MESH_PATH)
    centroids = mesh["centroids"]
    classes = mesh["classes"]
    depths_ft = np.nan_to_num(mesh["lost_depth"], nan=0.0)
    occurrence = np.nan_to_num(mesh["occurrence"], nan=0.0)
    depth_fraction = np.clip(depths_ft / 8.4, 0.0, 1.0)
    bases = build_base_spectra(centroids, classes, occurrence, seed)
    basis = _null_basis()
    lost_indices = np.flatnonzero(classes == 1)
    sample_indices = _sample_lost_indices(lost_indices)

    sweep = [
        _run_direction(angle, sample_indices, bases, depth_fraction, basis)
        for angle in SWEEP_ANGLES_DEG
    ]
    chosen = _choose_direction(sweep)
    direction = np.asarray(chosen["direction_lab"], dtype=float)
    final, diagnostics = _solve_all_lost(
        bases, classes, depth_fraction, direction, basis
    )
    repeated, repeated_diagnostics = _solve_all_lost(
        bases, classes, depth_fraction, direction, basis
    )
    deterministic = bool(np.array_equal(final, repeated))
    successes = np.array([item["success"] for item in diagnostics])
    delta_e = np.array([item["delta_e76"] for item in diagnostics])
    coherence = np.array([item["coherence"] for item in diagnostics])
    xyz_errors = np.array([item["d65_xyz_max_error"] for item in diagnostics])
    d65_xyz_difference = np.max(
        np.abs(spectra_to_xyz(final, D65_SPD) - spectra_to_xyz(bases, D65_SPD))
    )

    np.save(SPECTRA_BASE_PATH, bases)
    np.save(SPECTRA_FINAL_PATH, final)
    report = {
        "seed": seed,
        "bounds": [LOW, HIGH],
        "illuminant_a_kelvin": ILLUMINANT_A_K,
        "chromatic_adaptation_degree": ADAPTATION,
        "d65_constraint": "exact linear XYZ equality in a projected spectral nullspace",
        "direction_sweep": sweep,
        "chosen_direction": chosen,
        "lost_triangle_count": int(len(lost_indices)),
        "final": {
            "median_delta_e76_a": float(np.median(delta_e)),
            "min_delta_e76_a": float(np.min(delta_e)),
            "mean_coherence": float(np.mean(coherence)),
            "failure_rate": float(1.0 - np.mean(successes)),
            "d65_xyz_max_abs_difference": float(d65_xyz_difference),
            "maximum_per_cell_d65_xyz_error": float(np.max(xyz_errors)),
            "spectra_min": float(final.min()),
            "spectra_max": float(final.max()),
        },
        "deterministic_repeat_equal": deterministic,
        "base_spectra_sha256": hashlib.sha256(bases.tobytes()).hexdigest(),
        "final_spectra_sha256": hashlib.sha256(final.tobytes()).hexdigest(),
        "repeat_success_pattern_equal": [item["success"] for item in diagnostics]
        == [item["success"] for item in repeated_diagnostics],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(solve_metamers(), indent=2))
