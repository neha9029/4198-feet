"""Opt-in three-dimensional terrain geometry for the Phase 4 film."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from scipy import ndimage

from data import DATA_DIR
from mesh import MESH_PATH
from relief import DEM_PATH, SMOOTHING_METRES


LAKE_LEVEL_FT = 4_198.0
FEET_PER_METRE = 3.280839895013123
DEFAULT_EXAGGERATION = 2.0
SHORELINE_RAMP_METRES = 1_000.0
EDGE_FADE_FRACTION = 0.08
PIXEL_SIZE_METRES = 30.0


@dataclass(frozen=True)
class TerrainGeometry:
    """Per-vertex terrain heights and the measurements used to derive them."""

    heights_world: np.ndarray
    elevations_metres: np.ndarray
    flat_vertex_mask: np.ndarray
    edge_fade: np.ndarray
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
    smoothing_metres: float = SMOOTHING_METRES,
    shoreline_ramp_metres: float = SHORELINE_RAMP_METRES,
    edge_fade_fraction: float = EDGE_FADE_FRACTION,
) -> TerrainGeometry:
    """Sample the DEM and return deterministic film-space vertex heights.

    The lake, managed impoundments, and reconstructed 4,198 ft extent are an
    exactly flat z=0 plane. Terrain begins with a smooth exterior ramp and
    returns to z=0 at every outer edge of the repeatable source tile.
    """
    if exaggeration < 0.0:
        raise ValueError("Terrain exaggeration must be non-negative")
    if shoreline_ramp_metres <= 0.0 or not 0.0 < edge_fade_fraction < 0.5:
        raise ValueError("Terrain ramp and edge fade must be positive")

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

    healthy = np.isin(classes, (1, 2))
    exterior_distance = ndimage.distance_transform_edt(~healthy) * pixel_size
    shoreline_ramp_raster = _smootherstep(
        exterior_distance / shoreline_ramp_metres
    ).astype(np.float32)
    shoreline_ramp = _sample(shoreline_ramp_raster, points, order=1)

    edge_distance = np.minimum.reduce(
        (points[:, 0], 1.0 - points[:, 0], points[:, 1], 1.0 - points[:, 1])
    )
    edge_fade = _smootherstep(edge_distance / edge_fade_fraction)

    frame_height_metres = (classes.shape[0] - 1) * pixel_size
    above_lake_metres = np.maximum(
        vertex_elevation - LAKE_LEVEL_FT / FEET_PER_METRE,
        0.0,
    )
    heights_world = (
        above_lake_metres
        * exaggeration
        * shoreline_ramp
        * edge_fade
        / frame_height_metres
    )
    heights_world[flat_vertex_mask] = 0.0
    heights_world[edge_distance <= np.finfo(np.float64).eps] = 0.0

    return TerrainGeometry(
        heights_world=heights_world.astype(np.float64),
        elevations_metres=vertex_elevation,
        flat_vertex_mask=flat_vertex_mask,
        edge_fade=edge_fade.astype(np.float64),
        shoreline_ramp=shoreline_ramp.astype(np.float64),
        frame_height_metres=float(frame_height_metres),
        exaggeration=float(exaggeration),
        smoothing_metres=float(smoothing_metres),
    )
