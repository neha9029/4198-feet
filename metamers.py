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
SMOOTHNESS_WEIGHTS = (20.0, 50.0, 200.0)
ARM_ANGLES = {
    "north": (0.0, 15.0, -15.0),
    "south": (-180.0, -165.0, -150.0),
}
SWEEP_SAMPLE_SIZE_PER_ARM = 64
NULL_BASIS_SIZE = 12
PROJECTION_GAIN = 10.0
ORTHOGONAL_WEIGHT = 2.0
SEEDED_RESTARTS = 5
DEPTH_STRENGTH_MIN = 0.60
TARGET_MEDIAN_DELTA_E = 15.0
TARGET_COHERENCE = 0.90
SPECTRA_BASE_PATH = DATA_DIR / "phase2_base_spectra.npy"
SPECTRA_FINAL_PATH = DATA_DIR / "phase2_final_spectra.npy"
REPORT_PATH = DATA_DIR.parent / "out" / "phase2" / "metamer_report.json"


def _gaussian(centre: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((WAVELENGTHS - centre) / sigma) ** 2)


def build_spectral_families(
    centroids: np.ndarray, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return continuous land plus fixed north/south current-water spectra."""
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

    north_water = (
        0.018
        + 0.070 * _gaussian(430, 45)
        + 0.090 * _gaussian(660, 60)
        - 0.015 * _gaussian(530, 55)
    )
    south_water = (
        0.018
        + 0.110 * _gaussian(490, 55)
        + 0.050 * _gaussian(535, 65)
        - 0.020 * _gaussian(650, 75)
    )
    return (
        np.clip(land, LOW, HIGH),
        np.clip(north_water, LOW, HIGH),
        np.clip(south_water, LOW, HIGH),
    )


def build_base_spectra(
    centroids: np.ndarray,
    classes: np.ndarray,
    arms: np.ndarray,
    occurrence: np.ndarray,
    seed: int = SEED,
    *,
    return_water_reference: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Assign lost water, land, and managed cells to one continuous land family."""
    del occurrence  # Class-independent land continuity is deliberate.
    land, north_water, south_water = build_spectral_families(centroids, seed)
    spectra = land.copy()
    spectra[(classes == 2) & (arms == 1)] = north_water
    spectra[(classes == 2) & (arms == 0)] = south_water
    spectra[classes == 255] = 0.18
    if return_water_reference:
        return spectra, {"north": north_water, "south": south_water}
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


def _angle_direction(angle_degrees: float) -> np.ndarray:
    """Return a unit Lab direction with zero lightness component."""
    radians = np.deg2rad(angle_degrees)
    return np.array([0.0, np.cos(radians), np.sin(radians)], dtype=float)


def _bounded_scale(base: np.ndarray, delta: np.ndarray) -> float:
    """Largest scale in [0, 1] that keeps base + scale * delta in bounds."""
    alpha = 1.0
    falling = delta < 0.0
    rising = delta > 0.0
    if np.any(falling):
        alpha = min(alpha, float(np.min((base[falling] - LOW) / -delta[falling])))
    if np.any(rising):
        alpha = min(alpha, float(np.min((HIGH - base[rising]) / delta[rising])))
    return max(0.0, min(1.0, alpha) * (1.0 - 1e-10))


def _seeded_restart(
    base: np.ndarray, basis: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Project clip(r0 + N(0, 0.15)) into the exact D65 nullspace."""
    perturbed = np.clip(base + rng.normal(0.0, 0.15, size=base.shape), LOW, HIGH)
    z = basis.T @ (perturbed - base)
    delta = basis @ z
    return z * _bounded_scale(base, delta)


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
        )

    def bounds_constraint(z: np.ndarray) -> np.ndarray:
        spectrum = base + basis @ z
        return np.concatenate([spectrum - LOW, HIGH - spectrum])

    starts = [_seeded_restart(base, basis, rng) for _ in range(restarts)]

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
            options={"maxiter": 140, "ftol": 2e-9, "disp": False},
        )
        optimizer_succeeded = optimizer_succeeded or bool(result.success)
        if result.fun < best_fun:
            best_fun = float(result.fun)
            best_z = result.x
            best_iterations = int(result.nit)

    delta_spectrum = basis @ best_z
    alpha = _bounded_scale(base, delta_spectrum)
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
    if len(lost_indices) <= SWEEP_SAMPLE_SIZE_PER_ARM:
        return lost_indices
    locations = np.rint(
        np.linspace(0, len(lost_indices) - 1, SWEEP_SAMPLE_SIZE_PER_ARM)
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


def _run_trial(
    arm: str,
    angle_degrees: float,
    smoothness_weight: float,
    sample_indices: np.ndarray,
    bases: np.ndarray,
    depths: np.ndarray,
    basis: np.ndarray,
) -> dict:
    direction = _angle_direction(angle_degrees)
    diagnostics = []
    for order, triangle_index in enumerate(sample_indices):
        _, diagnostic = _solve_one(
            bases[triangle_index],
            depths[triangle_index],
            direction,
            smoothness_weight,
            basis,
            (
                SEED
                + (0 if arm == "south" else 1_000_000)
                + int(round(angle_degrees + 180.0)) * 10_000
                + int(smoothness_weight) * 100
                + order
            ),
            restarts=SEEDED_RESTARTS,
        )
        diagnostics.append(diagnostic)
    return {
        "arm": arm,
        "angle_degrees": angle_degrees,
        "smoothness_weight": smoothness_weight,
        "sample_count": int(len(sample_indices)),
        **_summarise(diagnostics),
    }


def _choose_arm_trial(arm: str, sweep: list[dict]) -> dict:
    coherent = [
        item
        for item in sweep
        if item["mean_coherence"] >= TARGET_COHERENCE
        and item["failure_rate"] == 0.0
    ]
    passing = [
        item for item in coherent if item["median_delta_e76_a"] >= TARGET_MEDIAN_DELTA_E
    ]
    if arm == "north":
        pool = coherent or [item for item in sweep if item["failure_rate"] == 0.0] or sweep
        chosen = max(pool, key=lambda item: item["median_delta_e76_a"])
    else:
        # In CIELAB, moving from -180 toward -150 adds negative b* and is the
        # requested progression from green toward blue-green.
        if passing:
            bluest_angle = max(item["angle_degrees"] for item in passing)
            pool = [item for item in passing if item["angle_degrees"] == bluest_angle]
            chosen = max(pool, key=lambda item: item["median_delta_e76_a"])
        else:
            pool = coherent or [item for item in sweep if item["failure_rate"] == 0.0] or sweep
            chosen = max(pool, key=lambda item: item["median_delta_e76_a"])
    return {**chosen, "meets_target": chosen in passing}


def _solve_indices(
    bases: np.ndarray,
    indices: np.ndarray,
    depths: np.ndarray,
    direction: np.ndarray,
    smoothness_weight: float,
    basis: np.ndarray,
    seed_offset: int,
) -> tuple[np.ndarray, list[dict]]:
    solved = bases.copy()
    diagnostics: list[dict] = []
    for triangle_index in indices:
        solved[triangle_index], diagnostic = _solve_one(
            bases[triangle_index],
            depths[triangle_index],
            direction,
            smoothness_weight,
            basis,
            SEED + seed_offset + int(triangle_index),
            restarts=SEEDED_RESTARTS,
        )
        diagnostic["triangle_index"] = int(triangle_index)
        diagnostics.append(diagnostic)
    return solved, diagnostics


def _depth_terciles(diagnostics: list[dict], depths: np.ndarray) -> list[dict]:
    """Summarise reveal strength from shallow edge through deepest lost cells."""
    if not diagnostics:
        return []
    ordered = sorted(diagnostics, key=lambda item: depths[item["triangle_index"]])
    rows = []
    for label, group in zip(("shallow", "middle", "deep"), np.array_split(ordered, 3)):
        items = list(group)
        group_depth = np.array([depths[item["triangle_index"]] for item in items])
        delta_e = np.array([item["delta_e76"] for item in items])
        rows.append(
            {
                "tercile": label,
                "triangle_count": len(items),
                "depth_min_ft": float(group_depth.min()),
                "depth_max_ft": float(group_depth.max()),
                "median_delta_e76_a": float(np.median(delta_e)),
            }
        )
    return rows


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
    """Sweep per-arm reveal directions, solve all lost cells, and verify identity."""
    mesh = np.load(MESH_PATH)
    centroids = mesh["centroids"]
    classes = mesh["classes"]
    arms = mesh["arms"]
    depths_ft = np.nan_to_num(mesh["lost_depth"], nan=0.0)
    occurrence = np.nan_to_num(mesh["occurrence"], nan=0.0)
    depth_fraction = np.clip(depths_ft / 8.4, 0.0, 1.0)
    bases = build_base_spectra(
        centroids,
        classes,
        arms,
        occurrence,
        seed,
    )
    basis = _null_basis()
    lost_indices = np.flatnonzero(classes == 1)
    sweep: list[dict] = []
    chosen_by_arm: dict[str, dict] = {}
    arm_indices: dict[str, np.ndarray] = {}
    for arm, arm_value in (("north", 1), ("south", 0)):
        indices = np.flatnonzero((classes == 1) & (arms == arm_value))
        arm_indices[arm] = indices
        sample_indices = _sample_lost_indices(indices)
        arm_sweep = []
        for angle_degrees in ARM_ANGLES[arm]:
            for smoothness_weight in SMOOTHNESS_WEIGHTS:
                row = _run_trial(
                    arm,
                    angle_degrees,
                    smoothness_weight,
                    sample_indices,
                    bases,
                    depth_fraction,
                    basis,
                )
                arm_sweep.append(row)
                sweep.append(row)
        chosen_by_arm[arm] = _choose_arm_trial(arm, arm_sweep)

    final = bases.copy()
    repeated = bases.copy()
    diagnostics_by_arm: dict[str, list[dict]] = {}
    repeated_by_arm: dict[str, list[dict]] = {}
    for arm, seed_offset in (("north", 2_000_000), ("south", 3_000_000)):
        chosen = chosen_by_arm[arm]
        indices = arm_indices[arm]
        solved, diagnostics = _solve_indices(
            bases,
            indices,
            depth_fraction,
            _angle_direction(float(chosen["angle_degrees"])),
            float(chosen["smoothness_weight"]),
            basis,
            seed_offset,
        )
        repeated_solved, repeated_diagnostics = _solve_indices(
            bases,
            indices,
            depth_fraction,
            _angle_direction(float(chosen["angle_degrees"])),
            float(chosen["smoothness_weight"]),
            basis,
            seed_offset,
        )
        final[indices] = solved[indices]
        repeated[indices] = repeated_solved[indices]
        diagnostics_by_arm[arm] = diagnostics
        repeated_by_arm[arm] = repeated_diagnostics

    diagnostics = diagnostics_by_arm["north"] + diagnostics_by_arm["south"]
    repeated_diagnostics = repeated_by_arm["north"] + repeated_by_arm["south"]
    deterministic = bool(np.array_equal(final, repeated))
    final_summary = _summarise(diagnostics)
    arm_results = {
        arm: {
            "chosen": chosen_by_arm[arm],
            "triangle_count": int(len(arm_indices[arm])),
            "final": _summarise(diagnostics_by_arm[arm]),
            "depth_terciles": _depth_terciles(
                diagnostics_by_arm[arm], depths_ft
            ),
        }
        for arm in ("north", "south")
    }
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
                "-10 * depth_strength * projection + 2 * orthogonal_norm_squared "
                "+ smoothness_weight * second_difference_norm_squared"
            ),
            "orthogonal_weight": ORTHOGONAL_WEIGHT,
            "projection_gain": PROJECTION_GAIN,
            "smoothness_weights_tested": list(SMOOTHNESS_WEIGHTS),
            "depth_strength": "0.6 + 0.4 * normalised_depth, applied to projection",
            "seeded_restarts": SEEDED_RESTARTS,
            "restart_initialisation": "project clip(r0 + N(0, 0.15)) into the D65 nullspace",
            "lab_lightness_weight": 0.0,
        },
        "option_sweep": sweep,
        "chosen_by_arm": chosen_by_arm,
        "arms": arm_results,
        "target": {
            "median_delta_e76_a_minimum": TARGET_MEDIAN_DELTA_E,
            "coherence_minimum": TARGET_COHERENCE,
            "achieved_in_sweep_by_arm": {
                arm: bool(chosen_by_arm[arm]["meets_target"])
                for arm in ("north", "south")
            },
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
