"""Shared daylight spectra and coherent D65-metamer reveal for Phase 2."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import rasterio
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from data import DATA_DIR, SEED
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
SMOOTHNESS_WEIGHTS = (50.0, 100.0, 200.0)
DIRECTION_OPTIONS = ("fixed_blue_green", "toward_water")
SWEEP_SAMPLE_SIZE = 64
NULL_BASIS_SIZE = 12
PROJECTION_GAIN = 450.0
ORTHOGONAL_WEIGHT = 35.0
COEFFICIENT_WEIGHT = 0.001
DEPTH_STRENGTH_MIN = 0.60
TARGET_MEDIAN_DELTA_E = 15.0
TARGET_COHERENCE = 0.85
SPECTRA_BASE_PATH = DATA_DIR / "phase2_base_spectra.npy"
SPECTRA_FINAL_PATH = DATA_DIR / "phase2_final_spectra.npy"
REPORT_PATH = DATA_DIR.parent / "out" / "phase2" / "metamer_report.json"


def _gaussian(centre: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((WAVELENGTHS - centre) / sigma) ** 2)


def build_spectral_families(
    centroids: np.ndarray, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous earth/sage land and dark blue-green water families."""
    rng = np.random.default_rng(seed)
    field_a = smooth_field(centroids, rng, waves=4, scale=1.15)
    field_b = smooth_field(centroids, rng, waves=3, scale=2.05)
    field_c = smooth_field(centroids, rng, waves=3, scale=0.72)

    land = (
        0.275
        + 0.105 * _gaussian(585, 140)[None, :]
        + 0.045 * _gaussian(540, 58)[None, :]
        - 0.028 * _gaussian(435, 46)[None, :]
    )
    land = land * (0.86 + 0.22 * ((field_a + 1.0) / 2.0))[:, None]
    land += (0.020 * field_b + 0.010 * field_c)[:, None]

    water = (
        0.030
        + 0.118 * _gaussian(500, 64)[None, :]
        + 0.030 * _gaussian(438, 46)[None, :]
        - 0.016 * _gaussian(650, 78)[None, :]
    )
    water = water * (0.78 + 0.23 * ((field_c + 1.0) / 2.0))[:, None]
    water += (0.006 * field_a)[:, None]
    return np.clip(land, LOW, HIGH), np.clip(water, LOW, HIGH)


def build_base_spectra(
    centroids: np.ndarray,
    classes: np.ndarray,
    occurrence: np.ndarray,
    seed: int = SEED,
    *,
    return_water_reference: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Assign lost water, land, and managed cells to one continuous land family."""
    del occurrence  # Class-independent land continuity is deliberate.
    land, water = build_spectral_families(centroids, seed)
    spectra = land.copy()
    spectra[classes == 2] = water[classes == 2]
    spectra[classes == 255] = 0.18
    if return_water_reference:
        return spectra, water
    return spectra


def _xyz_constraint() -> np.ndarray:
    weight = 100.0 / np.sum(D65_SPD * YBAR)
    return weight * np.stack(
        [D65_SPD * XBAR, D65_SPD * YBAR, D65_SPD * ZBAR], axis=0
    )


def _null_basis() -> np.ndarray:
    """Return a smooth basis lying in the exact D65 XYZ nullspace."""
    constraint = _xyz_constraint()
    projector = np.eye(len(WAVELENGTHS)) - constraint.T @ np.linalg.pinv(
        constraint @ constraint.T
    ) @ constraint
    x = np.linspace(0.0, 1.0, len(WAVELENGTHS))
    candidates = np.column_stack(
        [
            np.cos(np.pi * mode * x)
            for mode in range(1, NULL_BASIS_SIZE + 12)
        ]
    )
    projected = projector @ candidates
    q, _ = np.linalg.qr(projected)
    basis = projector @ q[:, :NULL_BASIS_SIZE]
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
        [
            116.0 * f[:, 1] - 16.0,
            500.0 * (f[:, 0] - f[:, 1]),
            200.0 * (f[:, 1] - f[:, 2]),
        ]
    )


def _lab_under_d65(spectra: np.ndarray) -> np.ndarray:
    white = spectra_to_xyz(np.ones((1, len(WAVELENGTHS))), D65_SPD)[0]
    return _lab(spectra_to_xyz(spectra, D65_SPD), white)


def _lab_under_a(spectra: np.ndarray) -> np.ndarray:
    a_spd = illuminant(ILLUMINANT_A_K)
    source_white = spectra_to_xyz(np.ones((1, len(WAVELENGTHS))), a_spd)[0]
    d65_white = spectra_to_xyz(np.ones((1, len(WAVELENGTHS))), D65_SPD)[0]
    xyz = spectra_to_xyz(spectra, a_spd)
    return _lab(adapt(xyz, source_white, d65_white, ADAPTATION), d65_white)


def _fixed_blue_green_direction() -> np.ndarray:
    radians = np.deg2rad(-135.0)
    vector = np.array([-0.12, np.cos(radians), np.sin(radians)], dtype=float)
    return vector / np.linalg.norm(vector)


def _directions(
    option: str,
    indices: np.ndarray,
    bases_lab_a: np.ndarray,
    water_lab_a: np.ndarray,
) -> np.ndarray:
    if option == "fixed_blue_green":
        return np.repeat(_fixed_blue_green_direction()[None, :], len(indices), axis=0)
    delta = water_lab_a[indices] - bases_lab_a[indices]
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    fallback = _fixed_blue_green_direction()
    return np.where(norm > 1e-10, delta / np.maximum(norm, 1e-10), fallback)


def _feasible_start(
    base: np.ndarray, basis: np.ndarray, rng: np.random.Generator, scale: float
) -> np.ndarray:
    for _ in range(10):
        z = rng.normal(0.0, scale, size=basis.shape[1])
        spectrum = base + basis @ z
        if spectrum.min() >= LOW and spectrum.max() <= HIGH:
            return z
        scale *= 0.62
    return np.zeros(basis.shape[1])


def _solve_one(
    base: np.ndarray,
    depth_fraction: float,
    direction: np.ndarray,
    smoothness_weight: float,
    basis: np.ndarray,
    seed: int,
    restarts: int,
) -> tuple[np.ndarray, dict]:
    """Maximise the desired A-light shift subject to exact D65 metamerism."""
    rng = np.random.default_rng(seed)
    baseline_lab = _lab_under_a(base[None, :])[0]
    strength = DEPTH_STRENGTH_MIN + (1.0 - DEPTH_STRENGTH_MIN) * np.clip(
        depth_fraction, 0.0, 1.0
    )
    d2 = np.diff(np.eye(len(WAVELENGTHS)), n=2, axis=0)

    def objective(z: np.ndarray) -> float:
        spectrum = base + basis @ z
        shift = _lab_under_a(spectrum[None, :])[0] - baseline_lab
        projection = float(shift @ direction)
        orthogonal = shift - projection * direction
        roughness = d2 @ spectrum
        return (
            -PROJECTION_GAIN * strength * projection
            + ORTHOGONAL_WEIGHT * float(orthogonal @ orthogonal)
            + smoothness_weight * float(roughness @ roughness)
            + COEFFICIENT_WEIGHT * float(z @ z)
        )

    def bounds_constraint(z: np.ndarray) -> np.ndarray:
        spectrum = base + basis @ z
        return np.concatenate([spectrum - LOW, HIGH - spectrum])

    starts = [np.zeros(basis.shape[1])]
    for candidate_index in range(40):
        scale = 0.08 + 0.72 * (candidate_index / 39.0)
        z0 = _feasible_start(base, basis, rng, scale)
        if np.any(z0):
            starts.append(z0)
    starts = sorted(starts, key=objective)[:restarts]

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
        if result.fun < best_fun:
            best_fun = float(result.fun)
            best_z = result.x
            best_iterations = int(result.nit)

    delta_spectrum = basis @ best_z
    alpha = 1.0
    falling = delta_spectrum < 0.0
    rising = delta_spectrum > 0.0
    if np.any(falling):
        alpha = min(
            alpha, float(np.min((base[falling] - LOW) / -delta_spectrum[falling]))
        )
    if np.any(rising):
        alpha = min(
            alpha, float(np.min((HIGH - base[rising]) / delta_spectrum[rising]))
        )
    if alpha < 1.0:
        alpha = max(0.0, alpha * (1.0 - 1e-10))
    solved = base + alpha * delta_spectrum

    shift = _lab_under_a(solved[None, :])[0] - baseline_lab
    delta_e = float(np.linalg.norm(shift))
    projection = float(shift @ direction)
    coherence = float(projection / max(delta_e, 1e-12))
    xyz_error = float(
        np.max(
            np.abs(
                spectra_to_xyz(solved[None, :], D65_SPD)
                - spectra_to_xyz(base[None, :], D65_SPD)
            )
        )
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
        "delta_e76": delta_e,
        "projection": projection,
        "coherence": coherence,
        "d65_xyz_max_error": xyz_error,
        "iterations": best_iterations,
        "depth_strength": float(strength),
    }


def _sample_lost_indices(lost_indices: np.ndarray) -> np.ndarray:
    if len(lost_indices) <= SWEEP_SAMPLE_SIZE:
        return lost_indices
    locations = np.rint(
        np.linspace(0, len(lost_indices) - 1, SWEEP_SAMPLE_SIZE)
    ).astype(int)
    return lost_indices[locations]


def _summarise(diagnostics: list[dict]) -> dict:
    delta_e = np.array([item["delta_e76"] for item in diagnostics])
    projection = np.array([item["projection"] for item in diagnostics])
    coherence = np.array([item["coherence"] for item in diagnostics])
    success = np.array([item["success"] for item in diagnostics])
    return {
        "median_delta_e76_a": float(np.median(delta_e)),
        "min_delta_e76_a": float(np.min(delta_e)),
        "median_projection_a": float(np.median(projection)),
        "mean_coherence": float(np.mean(coherence)),
        "failure_rate": float(1.0 - np.mean(success)),
    }


def _run_option(
    option: str,
    smoothness_weight: float,
    sample_indices: np.ndarray,
    bases: np.ndarray,
    depths: np.ndarray,
    directions: np.ndarray,
    basis: np.ndarray,
) -> dict:
    diagnostics = []
    for order, triangle_index in enumerate(sample_indices):
        _, diagnostic = _solve_one(
            bases[triangle_index],
            depths[triangle_index],
            directions[order],
            smoothness_weight,
            basis,
            SEED + DIRECTION_OPTIONS.index(option) * 100_000 + int(smoothness_weight) * 100 + order,
            restarts=2,
        )
        diagnostics.append(diagnostic)
    return {
        "option": option,
        "smoothness_weight": smoothness_weight,
        "sample_count": int(len(sample_indices)),
        **_summarise(diagnostics),
    }


def _choose_option(sweep: list[dict]) -> dict:
    passing = [
        item
        for item in sweep
        if item["median_delta_e76_a"] >= TARGET_MEDIAN_DELTA_E
        and item["mean_coherence"] >= TARGET_COHERENCE
        and item["failure_rate"] <= 0.05
    ]
    pool = passing or [item for item in sweep if item["failure_rate"] <= 0.05] or sweep
    chosen = max(
        pool,
        key=lambda item: item["median_delta_e76_a"]
        + 0.25 * item["min_delta_e76_a"]
        + 5.0 * item["mean_coherence"]
        - 30.0 * item["failure_rate"],
    )
    return {**chosen, "meets_target": chosen in passing}


def _solve_all_lost(
    bases: np.ndarray,
    classes: np.ndarray,
    depths: np.ndarray,
    directions: np.ndarray,
    smoothness_weight: float,
    basis: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    final = bases.copy()
    lost_indices = np.flatnonzero(classes == 1)
    diagnostics: list[dict] = []
    for order, triangle_index in enumerate(lost_indices):
        final[triangle_index], diagnostic = _solve_one(
            bases[triangle_index],
            depths[triangle_index],
            directions[order],
            smoothness_weight,
            basis,
            SEED + 900_000 + order,
            restarts=2,
        )
        diagnostic["triangle_index"] = int(triangle_index)
        diagnostics.append(diagnostic)
    return final, diagnostics


def _shoreline_camouflage(
    mesh: np.lib.npyio.NpzFile, bases: np.ndarray
) -> dict:
    """Compare lost-to-land D65 differences with local land variation."""
    classes = mesh["classes"]
    centroids = mesh["centroids"]
    raster_h, raster_w = (int(value) for value in mesh["raster_shape"])
    row0, row1, col0, col1 = (int(value) for value in mesh["crop_window"])
    with rasterio.open(DATA_DIR / "gsl_classes_utm12.tif") as dataset:
        raster_classes = dataset.read(1)[row0:row1, col0:col1]
        resolution = abs(float(dataset.transform.a))
    healthy = np.isin(raster_classes, (1, 2))
    shoreline = healthy ^ ndimage.binary_erosion(healthy)
    distance_m = ndimage.distance_transform_edt(~shoreline) * resolution
    rows = np.clip(
        np.rint(centroids[:, 1] * (raster_h - 1)), 0, raster_h - 1
    ).astype(int)
    columns = np.clip(
        np.rint(centroids[:, 0] * (raster_w - 1)), 0, raster_w - 1
    ).astype(int)
    near = distance_m[rows, columns] <= 5_000.0
    lost_indices = np.flatnonzero((classes == 1) & near)
    land_indices = np.flatnonzero((classes == 0) & near)
    xy_m = np.column_stack(
        [centroids[:, 0] * raster_w * resolution, centroids[:, 1] * raster_h * resolution]
    )
    lab = _lab_under_d65(bases)
    land_tree = cKDTree(xy_m[land_indices])
    lost_distance, lost_neighbor = land_tree.query(xy_m[lost_indices], k=1)
    lost_land_delta = np.linalg.norm(
        lab[lost_indices] - lab[land_indices[lost_neighbor]], axis=1
    )
    land_distance, land_neighbor = land_tree.query(xy_m[land_indices], k=2)
    land_land_delta = np.linalg.norm(
        lab[land_indices] - lab[land_indices[land_neighbor[:, 1]]], axis=1
    )
    return {
        "shoreline_distance_limit_km": 5.0,
        "lost_land_pair_count": int(len(lost_land_delta)),
        "land_land_pair_count": int(len(land_land_delta)),
        "median_delta_e76_d65_lost_to_land": float(np.median(lost_land_delta)),
        "p90_delta_e76_d65_lost_to_land": float(np.percentile(lost_land_delta, 90)),
        "median_delta_e76_d65_land_to_land": float(np.median(land_land_delta)),
        "p90_delta_e76_d65_land_to_land": float(np.percentile(land_land_delta, 90)),
        "median_lost_to_land_distance_m": float(np.median(lost_distance)),
        "median_land_to_land_distance_m": float(np.median(land_distance[:, 1])),
        "target_lost_to_land_delta_e76": 3.0,
        "passes_target": bool(np.median(lost_land_delta) < 3.0),
    }


def solve_metamers(seed: int = SEED) -> dict:
    """Run the six reveal trials, solve all lost cells, and verify identity."""
    mesh = np.load(MESH_PATH)
    centroids = mesh["centroids"]
    classes = mesh["classes"]
    depths_ft = np.nan_to_num(mesh["lost_depth"], nan=0.0)
    occurrence = np.nan_to_num(mesh["occurrence"], nan=0.0)
    depth_fraction = np.clip(depths_ft / 8.4, 0.0, 1.0)
    bases, water_reference = build_base_spectra(
        centroids,
        classes,
        occurrence,
        seed,
        return_water_reference=True,
    )
    basis = _null_basis()
    lost_indices = np.flatnonzero(classes == 1)
    sample_indices = _sample_lost_indices(lost_indices)
    bases_lab_a = _lab_under_a(bases)
    water_lab_a = _lab_under_a(water_reference)

    sweep = []
    for option in DIRECTION_OPTIONS:
        sample_directions = _directions(
            option, sample_indices, bases_lab_a, water_lab_a
        )
        for smoothness_weight in SMOOTHNESS_WEIGHTS:
            sweep.append(
                _run_option(
                    option,
                    smoothness_weight,
                    sample_indices,
                    bases,
                    depth_fraction,
                    sample_directions,
                    basis,
                )
            )
    chosen = _choose_option(sweep)
    final_directions = _directions(
        chosen["option"], lost_indices, bases_lab_a, water_lab_a
    )
    final, diagnostics = _solve_all_lost(
        bases,
        classes,
        depth_fraction,
        final_directions,
        float(chosen["smoothness_weight"]),
        basis,
    )
    repeated, repeated_diagnostics = _solve_all_lost(
        bases,
        classes,
        depth_fraction,
        final_directions,
        float(chosen["smoothness_weight"]),
        basis,
    )
    deterministic = bool(np.array_equal(final, repeated))
    final_summary = _summarise(diagnostics)
    xyz_errors = np.array([item["d65_xyz_max_error"] for item in diagnostics])
    d65_xyz_difference = np.max(
        np.abs(spectra_to_xyz(final, D65_SPD) - spectra_to_xyz(bases, D65_SPD))
    )
    camouflage = _shoreline_camouflage(mesh, bases)
    base_reflectance = {
        label: {
            "sample_median": float(np.median(bases[classes == class_value])),
            "sample_min": float(np.min(bases[classes == class_value])),
            "sample_max": float(np.max(bases[classes == class_value])),
        }
        for label, class_value in (("land", 0), ("lost_water", 1), ("managed", 3))
    }

    np.save(SPECTRA_BASE_PATH, bases)
    np.save(SPECTRA_FINAL_PATH, final)
    report = {
        "seed": seed,
        "bounds": [LOW, HIGH],
        "illuminant_a_kelvin": ILLUMINANT_A_K,
        "chromatic_adaptation_degree": ADAPTATION,
        "base_spectra": (
            "lost_water, land, and managed cells share one continuous earth/sage "
            "smooth_field family"
        ),
        "base_reflectance_by_class": base_reflectance,
        "d65_camouflage": camouflage,
        "d65_constraint": "exact linear XYZ equality in a projected spectral nullspace",
        "objective": {
            "form": (
                "maximise depth-scaled projection of delta Lab_A onto the target "
                "direction, with orthogonal, second-difference, and coefficient penalties"
            ),
            "orthogonal_weight": ORTHOGONAL_WEIGHT,
            "projection_gain": PROJECTION_GAIN,
            "coefficient_weight": COEFFICIENT_WEIGHT,
            "smoothness_weights_tested": list(SMOOTHNESS_WEIGHTS),
            "depth_strength": "0.6 + 0.4 * normalised_depth",
        },
        "option_sweep": sweep,
        "chosen_option": chosen,
        "target": {
            "median_delta_e76_a_minimum": TARGET_MEDIAN_DELTA_E,
            "coherence_minimum": TARGET_COHERENCE,
            "achieved_in_sweep": bool(chosen["meets_target"]),
        },
        "lost_triangle_count": int(len(lost_indices)),
        "final": {
            **final_summary,
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
