"""Deterministic, shoreline-aware tessellation for *4,198 Feet*."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.spatial import Delaunay

from data import DATA_DIR, OUT_DIR, SEED
from engine import lloyd_relax


INTERIOR_POINTS = 2_100
SHORELINE_ANCHORS = 800
FRAME_SAMPLES_PER_EDGE = 34
LLOYD_ITERATIONS = 2
DENSITY_STRIDE = 8
MESH_PATH = DATA_DIR / "gsl_mesh_phase2.npz"
MESH_REPORT_PATH = DATA_DIR.parent / "out" / "phase2" / "mesh_report.json"


def _edge(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel interior edge for a Boolean region."""
    return mask ^ ndimage.binary_erosion(mask)


def _sample_weighted_points(
    occurrence: np.ndarray,
    classes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points from occurrence gradient with strong shoreline boosts."""
    small_occurrence = occurrence[::DENSITY_STRIDE, ::DENSITY_STRIDE]
    small_classes = classes[::DENSITY_STRIDE, ::DENSITY_STRIDE]
    valid = np.isfinite(small_occurrence) & (small_classes != 255)
    filled = np.nan_to_num(small_occurrence, nan=0.0)
    gy, gx = np.gradient(filled)
    gradient = np.hypot(gx, gy)
    if np.any(valid):
        scale = np.percentile(gradient[valid], 98)
        gradient = np.clip(gradient / max(scale, 1e-6), 0.0, 1.0)
    healthy_edge = _edge(np.isin(small_classes, (1, 2)))
    current_edge = _edge(small_classes == 2)
    shoreline = ndimage.gaussian_filter(
        (healthy_edge | current_edge).astype(np.float32), sigma=2.0
    )
    if shoreline.max() > 0:
        shoreline /= shoreline.max()
    density = valid * (0.04 + 1.35 * gradient + 5.5 * shoreline)
    probability = density.ravel()
    probability /= probability.sum()
    indices = rng.choice(
        probability.size,
        size=INTERIOR_POINTS,
        replace=False,
        p=probability,
    )
    rows, columns = np.unravel_index(indices, density.shape)
    jitter = rng.uniform(-0.42, 0.42, size=(INTERIOR_POINTS, 2))
    x = (columns + 0.5 + jitter[:, 0]) / density.shape[1]
    y = (rows + 0.5 + jitter[:, 1]) / density.shape[0]
    return np.clip(np.column_stack([x, y]), 0.0, 1.0)


def _shoreline_points(
    classes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Keep fixed anchors on both accepted shorelines after Lloyd relaxation."""
    healthy = _edge(np.isin(classes, (1, 2)))
    current = _edge(classes == 2)
    candidates = np.argwhere(healthy | current)
    if len(candidates) <= SHORELINE_ANCHORS:
        selected = candidates
    else:
        selected = candidates[
            rng.choice(len(candidates), SHORELINE_ANCHORS, replace=False)
        ]
    jitter = rng.uniform(-0.2, 0.2, size=(len(selected), 2))
    y = (selected[:, 0] + 0.5 + jitter[:, 0]) / classes.shape[0]
    x = (selected[:, 1] + 0.5 + jitter[:, 1]) / classes.shape[1]
    return np.clip(np.column_stack([x, y]), 0.0, 1.0)


def _frame_points() -> np.ndarray:
    """Close the triangulation cleanly on the rectangular crop boundary."""
    edge = np.linspace(0.0, 1.0, FRAME_SAMPLES_PER_EDGE)
    return np.vstack(
        [
            np.column_stack([edge, np.zeros_like(edge)]),
            np.column_stack([edge, np.ones_like(edge)]),
            np.column_stack([np.zeros_like(edge), edge]),
            np.column_stack([np.ones_like(edge), edge]),
        ]
    )


def _sample_raster(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-sample a raster at normalized image-coordinate points."""
    columns = np.clip(np.rint(points[:, 0] * (values.shape[1] - 1)), 0, values.shape[1] - 1).astype(int)
    rows = np.clip(np.rint(points[:, 1] * (values.shape[0] - 1)), 0, values.shape[0] - 1).astype(int)
    return values[rows, columns]


def _mixed_fraction(
    points: np.ndarray, simplices: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Flag triangles whose vertex/edge samples disagree with their centroid."""
    vertices = points[simplices]
    centroids = vertices.mean(axis=1)
    samples = [vertices[:, index] for index in range(3)]
    samples.extend(
        [
            (vertices[:, 0] + vertices[:, 1]) / 2.0,
            (vertices[:, 1] + vertices[:, 2]) / 2.0,
            (vertices[:, 2] + vertices[:, 0]) / 2.0,
        ]
    )
    centroid_class = _sample_raster(classes, centroids)
    sampled = np.stack([_sample_raster(classes, sample) for sample in samples], axis=1)
    mixed = np.any(sampled != centroid_class[:, None], axis=1)
    valid = centroid_class != 255
    healthy_relevant = valid & np.any(
        np.isin(sampled, (0, 1, 2)) & (sampled != centroid_class[:, None]), axis=1
    )
    return (
        mixed,
        float(mixed[valid].mean()) if valid.any() else 0.0,
        float(healthy_relevant.sum() / max(valid.sum(), 1)),
    )


def build_mesh(seed: int = SEED) -> dict:
    """Build and cache the approximately 3,000-point Phase 2 mesh."""
    rng = np.random.default_rng(seed)
    with rasterio.open(DATA_DIR / "gsw_occurrence_gsl_utm12.tif") as dataset:
        occurrence = dataset.read(1)
    with rasterio.open(DATA_DIR / "gsl_classes_utm12.tif") as dataset:
        classes = dataset.read(1)
    with rasterio.open(DATA_DIR / "gsl_lost_depth_ft_utm12.tif") as dataset:
        lost_depth = dataset.read(1)

    interior = _sample_weighted_points(occurrence, classes, rng)
    relaxed = lloyd_relax(interior, LLOYD_ITERATIONS)
    anchors = _shoreline_points(classes, rng)
    points = np.vstack([relaxed, anchors, _frame_points()])
    points = np.unique(np.round(points, 9), axis=0)
    triangulation = Delaunay(points)
    simplices = triangulation.simplices.astype(np.int32)
    centroids = points[simplices].mean(axis=1)
    triangle_classes = _sample_raster(classes, centroids).astype(np.uint8)
    triangle_depth = _sample_raster(lost_depth, centroids).astype(np.float32)
    triangle_occurrence = _sample_raster(occurrence, centroids).astype(np.float32)
    mixed, mixed_fraction, contour_fraction = _mixed_fraction(
        points, simplices, classes
    )
    np.savez_compressed(
        MESH_PATH,
        points=points.astype(np.float64),
        simplices=simplices,
        centroids=centroids.astype(np.float64),
        classes=triangle_classes,
        lost_depth=triangle_depth,
        occurrence=triangle_occurrence,
        mixed=mixed,
        raster_shape=np.array(classes.shape, dtype=np.int32),
    )
    report = {
        "seed": seed,
        "points": int(len(points)),
        "triangles": int(len(simplices)),
        "lloyd_iterations": LLOYD_ITERATIONS,
        "shoreline_anchors": int(len(anchors)),
        "mixed_class_triangles": int(mixed.sum()),
        "mixed_class_fraction_of_valid": mixed_fraction,
        "class_boundary_mixed_fraction_of_valid": contour_fraction,
        "class_triangle_counts": {
            "land": int(np.count_nonzero(triangle_classes == 0)),
            "lost_water": int(np.count_nonzero(triangle_classes == 1)),
            "water_now": int(np.count_nonzero(triangle_classes == 2)),
            "managed": int(np.count_nonzero(triangle_classes == 3)),
            "nodata": int(np.count_nonzero(triangle_classes == 255)),
        },
        "density": "JRC occurrence-gradient magnitude with boosts at both accepted shoreline masks",
    }
    MESH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MESH_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_mesh(), indent=2))
