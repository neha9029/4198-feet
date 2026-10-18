"""Opt-in three-dimensional terrain geometry for the Phase 4 film."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from scipy import ndimage

from data import DATA_DIR, SEED
from mesh import MESH_PATH
from relief import (
    DEM_PATH,
    SUN_ALTITUDE_DEGREES,
    SUN_AZIMUTH_DEGREES,
)


LAKE_LEVEL_FT = 4_198.0
FEET_PER_METRE = 3.280839895013123
DEFAULT_EXAGGERATION = 2.0
SHORELINE_RAMP_METRES = 1_000.0
PIXEL_SIZE_METRES = 30.0
DEFAULT_SHADE_STRENGTH = 0.6
DEFAULT_TERRAIN_SMOOTHING_METRES = 600.0
RADIAL_TILE_MARGIN_FRACTION = 0.03
RADIAL_INNER_FRACTION = 0.45
DEFAULT_RADIAL_NOISE_AMPLITUDE = 0.12


@dataclass(frozen=True)
class TerrainGeometry:
    """Per-vertex terrain heights and the measurements used to derive them."""

    heights_world: np.ndarray
    elevations_metres: np.ndarray
    flat_vertex_mask: np.ndarray
    radial_falloff: np.ndarray
    radial_distance: np.ndarray
    radial_outer_radius: np.ndarray
    healthy_centroid: np.ndarray
    radial_semiaxes: np.ndarray
    radial_full_radius: float
    radial_zero_radius: float
    radial_noise_amplitude: float
    shoreline_ramp: np.ndarray
    frame_height_metres: float
    exaggeration: float
    smoothing_metres: float

    @property
    def max_height_world(self) -> float:
        return float(np.max(self.heights_world))

    @property
    def max_height_metres_exaggerated(self) -> float:
        return self.max_height_world * self.frame_height_metres


def facet_lambert_shades(
    points_world: np.ndarray,
    simplices: np.ndarray,
    classes: np.ndarray,
    *,
    strength: float = DEFAULT_SHADE_STRENGTH,
) -> np.ndarray:
    """Return one source-facet Lambert scalar from the real 3D normals."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Terrain shade strength must be within [0, 1]")
    triangles = points_world[simplices]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals[normals[:, 2] < 0.0] *= -1.0
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], np.finfo(np.float64).eps)

    azimuth = np.radians(SUN_AZIMUTH_DEGREES)
    altitude = np.radians(SUN_ALTITUDE_DEGREES)
    light = np.array(
        [
            np.cos(altitude) * np.sin(azimuth),
            np.cos(altitude) * np.cos(azimuth),
            np.sin(altitude),
        ],
        dtype=np.float64,
    )
    flat_dot = float(light[2])
    lambert = np.maximum(0.0, normals @ light)
    shades = 1.0 - strength * (1.0 - lambert / flat_dot)

    flat_geometry = np.ptp(triangles[:, :, 2], axis=1) == 0.0
    flat_water = np.isin(classes, (1, 2, 3))
    shades[flat_geometry | flat_water] = 1.0
    return shades.astype(np.float64)


def _smootherstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * values * (
        values * (values * 6.0 - 15.0) + 10.0
    )


def _radial_falloff(
    points: np.ndarray,
    centroid: np.ndarray,
    *,
    seed: int,
    noise_amplitude: float,
    tile_margin_fraction: float = RADIAL_TILE_MARGIN_FRACTION,
    inner_fraction: float = RADIAL_INNER_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a broad, lobed elliptical height multiplier.

    Radius 1 is the largest centroid-centred, axis-aligned ellipse inside the
    source tile's three-percent inset. A seeded mix of two- and three-lobe
    waves perturbs that radius, while an exact per-angle rectangle limit keeps
    every lobe inside the inset boundary.
    """
    if not 0.0 <= noise_amplitude < 1.0:
        raise ValueError("Terrain radial noise amplitude must be within [0, 1)")
    if not 0.0 < tile_margin_fraction < 0.5:
        raise ValueError("Terrain tile margin must be within (0, 0.5)")
    if not 0.0 < inner_fraction < 1.0:
        raise ValueError("Terrain radial inner fraction must be within (0, 1)")

    lower = tile_margin_fraction
    upper = 1.0 - tile_margin_fraction
    semiaxes = np.array(
        [
            min(centroid[0] - lower, upper - centroid[0]),
            min(centroid[1] - lower, upper - centroid[1]),
        ],
        dtype=np.float64,
    )
    if np.any(semiaxes <= 0.0):
        raise ValueError("Healthy-lake centroid lies outside terrain fade inset")

    normalized = (points - centroid) / semiaxes
    radial_distance = np.linalg.norm(normalized, axis=1)
    angles = np.arctan2(normalized[:, 1], normalized[:, 0])
    cos_angle = np.cos(angles)
    sin_angle = np.sin(angles)

    # Exact ray-to-inset-rectangle limit in ellipse-radius units. This cap is
    # important where a positive lobe would otherwise approach a tile edge.
    tiny = np.finfo(np.float64).eps
    x_limit = np.where(
        cos_angle >= 0.0,
        (upper - centroid[0]) / np.maximum(semiaxes[0] * cos_angle, tiny),
        (centroid[0] - lower) / np.maximum(-semiaxes[0] * cos_angle, tiny),
    )
    y_limit = np.where(
        sin_angle >= 0.0,
        (upper - centroid[1]) / np.maximum(semiaxes[1] * sin_angle, tiny),
        (centroid[1] - lower) / np.maximum(-semiaxes[1] * sin_angle, tiny),
    )
    inset_boundary_radius = np.minimum(x_limit, y_limit)

    rng = np.random.default_rng(seed + 4_198)
    phase_two, phase_three = rng.uniform(0.0, 2.0 * np.pi, size=2)
    low_frequency_noise = (
        0.58 * np.cos(2.0 * angles + phase_two)
        + 0.42 * np.cos(3.0 * angles + phase_three)
    )
    perturbed_radius = 1.0 + noise_amplitude * low_frequency_noise
    outer_radius = np.minimum(perturbed_radius, inset_boundary_radius)
    inner_radius = inner_fraction * outer_radius
    progress = (radial_distance - inner_radius) / (
        outer_radius - inner_radius
    )
    falloff = 1.0 - _smootherstep(progress)
    falloff[radial_distance >= outer_radius] = 0.0
    falloff[radial_distance <= inner_radius] = 1.0
    return falloff, radial_distance, outer_radius, semiaxes


def _film_class_raster() -> np.ndarray:
    """Reconstruct the north-up 16:9 class frame stored by ``mesh.py``."""
    mesh = np.load(MESH_PATH)
    frame_height, frame_width = (int(value) for value in mesh["raster_shape"])
    row0, row1, column0, column1 = (
        int(value) for value in mesh["crop_window"]
    )
    frame = np.zeros((frame_height, frame_width), dtype=np.uint8)
    with rasterio.open(DATA_DIR / "gsl_classes_utm12.tif") as dataset:
        source = dataset.read(1)
    source_row0, source_row1 = max(0, row0), min(source.shape[0], row1)
    source_col0, source_col1 = max(0, column0), min(source.shape[1], column1)
    target_row0, target_row1 = source_row0 - row0, source_row1 - row0
    target_col0, target_col1 = source_col0 - column0, source_col1 - column0
    values = source[source_row0:source_row1, source_col0:source_col1]
    frame[target_row0:target_row1, target_col0:target_col1] = np.where(
        values == 255,
        0,
        values,
    )
    return frame


def _sample(array: np.ndarray, points: np.ndarray, *, order: int) -> np.ndarray:
    rows = points[:, 1] * (array.shape[0] - 1)
    columns = points[:, 0] * (array.shape[1] - 1)
    return ndimage.map_coordinates(
        array,
        np.vstack((rows, columns)),
        order=order,
        mode="nearest",
        prefilter=False,
    )


def build_vertex_terrain(
    points: np.ndarray,
    *,
    exaggeration: float = DEFAULT_EXAGGERATION,
    smoothing_metres: float = DEFAULT_TERRAIN_SMOOTHING_METRES,
    shoreline_ramp_metres: float = SHORELINE_RAMP_METRES,
    radial_noise_amplitude: float = DEFAULT_RADIAL_NOISE_AMPLITUDE,
    seed: int = SEED,
) -> TerrainGeometry:
    """Sample the DEM and return deterministic film-space vertex heights.

    The lake, managed impoundments, and reconstructed 4,198 ft extent are an
    exactly flat z=0 plane. Terrain begins with a smooth shoreline ramp and
    returns to z=0 on an elliptical radial falloff before the source boundary.
    """
    if exaggeration < 0.0:
        raise ValueError("Terrain exaggeration must be non-negative")
    if shoreline_ramp_metres <= 0.0 or smoothing_metres <= 0.0:
        raise ValueError("Terrain smoothing and shoreline ramp must be positive")

    with rasterio.open(DEM_PATH) as dataset:
        elevation = dataset.read(1).astype(np.float32)
        pixel_size = float(abs(dataset.transform.a))
    classes = _film_class_raster()
    if elevation.shape != classes.shape:
        raise RuntimeError(
            f"DEM {elevation.shape} is not aligned to film frame {classes.shape}"
        )

    valid = np.isfinite(elevation)
    if not np.all(valid):
        nearest = ndimage.distance_transform_edt(
            ~valid,
            return_distances=False,
            return_indices=True,
        )
        elevation = elevation[tuple(nearest)]
    sigma = smoothing_metres / pixel_size
    smoothed = ndimage.gaussian_filter(elevation, sigma=sigma, mode="nearest")

    vertex_elevation = _sample(smoothed, points, order=1).astype(np.float64)
    vertex_classes = _sample(classes, points, order=0).astype(np.uint8)
    flat_vertex_mask = np.isin(vertex_classes, (1, 2, 3))
    mesh = np.load(MESH_PATH)
    if len(points) != len(mesh["points"]):
        raise ValueError("Terrain points must match the production mesh")
    flat_triangles = np.isin(mesh["classes"], (1, 2, 3))
    flat_triangle_vertices = np.unique(mesh["simplices"][flat_triangles])
    flat_vertex_mask[flat_triangle_vertices] = True

    healthy = np.isin(classes, (1, 2))
    exterior_distance = ndimage.distance_transform_edt(~healthy) * pixel_size
    shoreline_ramp_raster = _smootherstep(
        exterior_distance / shoreline_ramp_metres
    ).astype(np.float32)
    shoreline_ramp = _sample(shoreline_ramp_raster, points, order=1)

    mesh_triangles = mesh["points"][mesh["simplices"]]
    cross = (
        (mesh_triangles[:, 1, 0] - mesh_triangles[:, 0, 0])
        * (mesh_triangles[:, 2, 1] - mesh_triangles[:, 0, 1])
        - (mesh_triangles[:, 1, 1] - mesh_triangles[:, 0, 1])
        * (mesh_triangles[:, 2, 0] - mesh_triangles[:, 0, 0])
    )
    triangle_areas = np.abs(cross) / 2.0
    healthy_triangles = np.isin(mesh["classes"], (1, 2))
    healthy_centroid = np.average(
        mesh["centroids"][healthy_triangles],
        axis=0,
        weights=triangle_areas[healthy_triangles],
    )
    (
        radial_falloff,
        radial_distance,
        radial_outer_radius,
        radial_semiaxes,
    ) = _radial_falloff(
        points,
        healthy_centroid,
        seed=seed,
        noise_amplitude=radial_noise_amplitude,
    )

    frame_height_metres = (classes.shape[0] - 1) * pixel_size
    above_lake_metres = np.maximum(
        vertex_elevation - LAKE_LEVEL_FT / FEET_PER_METRE,
        0.0,
    )
    heights_world = (
        above_lake_metres
        * exaggeration
        * shoreline_ramp
        * radial_falloff
        / frame_height_metres
    )
    heights_world[flat_vertex_mask] = 0.0
    boundary = np.any((points == 0.0) | (points == 1.0), axis=1)
    heights_world[boundary] = 0.0

    return TerrainGeometry(
        heights_world=heights_world.astype(np.float64),
        elevations_metres=vertex_elevation,
        flat_vertex_mask=flat_vertex_mask,
        radial_falloff=radial_falloff.astype(np.float64),
        radial_distance=radial_distance.astype(np.float64),
        radial_outer_radius=radial_outer_radius.astype(np.float64),
        healthy_centroid=healthy_centroid.astype(np.float64),
        radial_semiaxes=radial_semiaxes.astype(np.float64),
        radial_full_radius=float(RADIAL_INNER_FRACTION),
        radial_zero_radius=1.0,
        radial_noise_amplitude=float(radial_noise_amplitude),
        shoreline_ramp=shoreline_ramp.astype(np.float64),
        frame_height_metres=float(frame_height_metres),
        exaggeration=float(exaggeration),
        smoothing_metres=float(smoothing_metres),
    )
