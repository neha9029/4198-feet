"""Opt-in three-dimensional terrain geometry for the Phase 4 film."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from scipy import ndimage

from data import DATA_DIR
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
DEFAULT_TERRAIN_SMOOTHING_METRES = 300.0
RADIAL_FULL_RADIUS = 0.78
RADIAL_ZERO_RADIUS = 0.90


@dataclass(frozen=True)
class TerrainGeometry:
    """Per-vertex terrain heights and the measurements used to derive them."""

    heights_world: np.ndarray
    elevations_metres: np.ndarray
    flat_vertex_mask: np.ndarray
    radial_falloff: np.ndarray
    radial_distance: np.ndarray
    healthy_centroid: np.ndarray
    radial_full_radius: float
    radial_zero_radius: float
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
    radial_full_radius: float = RADIAL_FULL_RADIUS,
    radial_zero_radius: float = RADIAL_ZERO_RADIUS,
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
    if not 0.0 < radial_full_radius < radial_zero_radius:
        raise ValueError("Terrain radial radii must satisfy 0 < r0 < r1")

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
    radial_distance = np.sqrt(
        ((points[:, 0] - healthy_centroid[0]) / 0.5) ** 2
        + ((points[:, 1] - healthy_centroid[1]) / 0.5) ** 2
    )
    radial_progress = (
        (radial_distance - radial_full_radius)
        / (radial_zero_radius - radial_full_radius)
    )
    radial_falloff = 1.0 - _smootherstep(radial_progress)
    radial_falloff[radial_distance >= radial_zero_radius] = 0.0
    radial_falloff[radial_distance <= radial_full_radius] = 1.0

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
        healthy_centroid=healthy_centroid.astype(np.float64),
        radial_full_radius=float(radial_full_radius),
        radial_zero_radius=float(radial_zero_radius),
        shoreline_ramp=shoreline_ramp.astype(np.float64),
        frame_height_metres=float(frame_height_metres),
        exaggeration=float(exaggeration),
        smoothing_metres=float(smoothing_metres),
    )
