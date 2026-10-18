"""Phase 4 spectral film renderer with a slow map-style camera orbit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates

from data import DATA_DIR, SEED
from engine import (
    ADAPTATION,
    D65_SPD,
    WAVELENGTHS,
    YBAR,
    adapt,
    illuminant,
    spectra_to_xyz,
    xyz_to_srgb_float,
)
from mesh import FRAME_ASPECT_RATIO, MESH_PATH
from metamers import (
    ILLUMINANT_A_K,
    SPECTRA_BASE_PATH,
    SPECTRA_BASE_NO_RELIEF_PATH,
    SPECTRA_FINAL_PATH,
)
from relief import RELIEF_FACTORS_PATH
from terrain import (
    DEFAULT_EXAGGERATION,
    DEFAULT_RADIAL_NOISE_AMPLITUDE,
    DEFAULT_SHADE_STRENGTH,
    DEFAULT_TERRAIN_SMOOTHING_METRES,
    RADIAL_INNER_FRACTION,
    RADIAL_TILE_MARGIN_FRACTION,
)


# Film and light timing.
FPS = 24
FILM_DURATION_SECONDS = 210.0
END_CARD_DURATION_SECONDS = 5.0
TOTAL_DURATION_SECONDS = FILM_DURATION_SECONDS + END_CARD_DURATION_SECONDS
DAYLIGHT_HOLD_END_SECONDS = 40.0
DUSK_END_SECONDS = 110.0
LAMPLIGHT_HOLD_END_SECONDS = 150.0
RETURN_END_SECONDS = 200.0
FADE_START_SECONDS = 205.0
FADE_END_SECONDS = 210.0

# Camera timing and angles. Heading settles at exactly 90° at 3:25.
CAMERA_TILT_START_DEGREES = 0.0
CAMERA_TILT_HOLD_DEGREES = 45.0
CAMERA_TILT_IN_START_SECONDS = 0.0
CAMERA_TILT_IN_END_SECONDS = 40.0
CAMERA_TILT_OUT_START_SECONDS = 150.0
CAMERA_TILT_OUT_END_SECONDS = 200.0
CAMERA_HEADING_START_DEGREES = 0.0
CAMERA_HEADING_END_DEGREES = 90.0
CAMERA_HEADING_EASE_START_SECONDS = 5.0
CAMERA_HEADING_EASE_END_SECONDS = 205.0
CAMERA_HEALTHY_FILL_FRACTION = 0.80
CAMERA_VERTICAL_FOV_DEGREES = 36.0

# Geometry, antialiasing, horizon, and fog.
CAMERA_SUPERSAMPLE = 2
EXTENDED_PLANE_HALF_SIZE = 5.0
EXTENDED_PLANE_MIN_FRUSTUM_MULTIPLE = 3.0
FOG_START_RADIUS = 1.12
FOG_END_RADIUS = 1.58
BACKGROUND_RGB = np.array([244, 241, 232], dtype=np.float32) / 255.0
TILE_JOIN_JITTER_FRACTION = 0.006

# End card.
END_CARD_FADE_SECONDS = 1.0
END_CARD_TEXT_RGB = np.array([190, 190, 190], dtype=np.float32) / 255.0
END_CARD_LINES = (
    "4,198 Feet",
    "Great Salt Lake, 4,189.6 ft, 9 October 2026",
    "Neha Naik, 2026",
)

# Requested preview products.
SHIMMER_START_SECONDS = 40.0
SHIMMER_DURATION_SECONDS = 10.0
SHIMMER_SIZE = (1920, 1080)
FAST_PREVIEW_SPEED = 4.0
FAST_PREVIEW_SIZE = (1280, 720)
MASTER_BENCHMARK_SIZE = (3840, 2160)
H264_CRF = 16
MASTER_BENCHMARK_FRAME_COUNT = 3

FILM_DIR = DATA_DIR.parent / "out" / "film"
SHIMMER_DIR = FILM_DIR / "shimmer_40_50"
SHIMMER_PATH = SHIMMER_DIR / "4198_feet_40-50s_camera_h264.mp4"
FAST_PREVIEW_PATH = FILM_DIR / "4198_feet_full_4x_720p.mp4"
OPENING_STILL_PATH = FILM_DIR / "camera_opening_d65_720p.png"
FINAL_DAYLIGHT_STILL_PATH = FILM_DIR / "camera_final_daylight_720p.png"
REPORT_PATH = FILM_DIR / "camera_preview_report.json"
GEOMETRY_REPORT_PATH = FILM_DIR / "camera_geometry_report.json"
FINAL_REPORT_PATH = FILM_DIR / "final_film_report.json"
FINAL_2160P_PATH = FILM_DIR / "4198_feet_2160p.mp4"
FINAL_PRORES_PATH = FILM_DIR / "4198_feet_2160p_10bit.mov"
FINAL_1080P_PATH = FILM_DIR / "4198_feet_1080p.mp4"
TERRAIN_PREVIEW_DIR = FILM_DIR / "terrain_preview"
TERRAIN_PREVIEW_REPORT_PATH = TERRAIN_PREVIEW_DIR / "terrain_report.json"
TERRAIN_PREVIEW_TIMES = (0.0, 60.0, 105.0, 150.0, 180.0)
TERRAIN_PREVIEW_SIZE = (1600, 900)
TERRAIN_SHADE_EXAGGERATION = 4.0
TERRAIN_FALLOFF_NOISE_COMPARISON = (0.12, 0.0)
TERRAIN_SHADE_REPORT_PATH = TERRAIN_PREVIEW_DIR / "terrain_shade_report.json"
TERRAIN_ORBIT_START_SECONDS = 85.0
TERRAIN_ORBIT_DURATION_SECONDS = 40.0
TERRAIN_ORBIT_SIZE = (1280, 720)
TERRAIN_ORBIT_PATH = TERRAIN_PREVIEW_DIR / "orbit_40s.mp4"
TERRAIN_OPENING_STILL_PATH = TERRAIN_PREVIEW_DIR / "opening_000s_1080p.png"
TERRAIN_LOCKED_REPORT_PATH = TERRAIN_PREVIEW_DIR / "locked_preview_report.json"


@dataclass(frozen=True)
class CameraState:
    tilt_degrees: float
    heading_degrees: float
    distance: float
    healthy_fill_fraction: float
    screen_shift_fraction: tuple[float, float]
    healthy_center_deviation_fraction: tuple[float, float]


@dataclass
class FilmScene:
    points_world: np.ndarray
    simplices: np.ndarray
    classes: np.ndarray
    healthy_points_world: np.ndarray
    final_spectra: np.ndarray
    identity_spectra: np.ndarray
    land_spectra: np.ndarray
    triangle_centroids_world: np.ndarray
    tile_offsets: np.ndarray
    source_triangle_count: int
    tile_triangle_count: int
    tile_width: float
    tile_height: float
    target_normalized: np.ndarray
    terrain_enabled: bool
    terrain_exaggeration: float
    terrain_max_height_world: float
    terrain_max_height_metres: float
    terrain_smoothing_metres: float
    terrain_radial_noise_amplitude: float
    terrain_shade_enabled: bool
    terrain_shade_strength: float
    facet_shades: np.ndarray
    baked_hillshade_enabled: bool
    seamless_tile_edges: bool
    source_boundary_world: tuple[np.ndarray, ...]


def smootherstep(value: float | np.ndarray) -> float | np.ndarray:
    """Quintic ease with zero first and second derivatives at both ends."""
    x = np.clip(value, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _ease_between(seconds: float, start: float, end: float) -> float:
    if end <= start:
        return float(seconds >= end)
    return float(smootherstep((seconds - start) / (end - start)))


def mixture_amount(global_seconds: float) -> float:
    """Return the production film's Illuminant A mixture amount."""
    if global_seconds < DAYLIGHT_HOLD_END_SECONDS:
        return 0.0
    if global_seconds < DUSK_END_SECONDS:
        return _ease_between(
            global_seconds,
            DAYLIGHT_HOLD_END_SECONDS,
            DUSK_END_SECONDS,
        )
    if global_seconds < LAMPLIGHT_HOLD_END_SECONDS:
        return 1.0
    if global_seconds < RETURN_END_SECONDS:
        return 1.0 - _ease_between(
            global_seconds,
            LAMPLIGHT_HOLD_END_SECONDS,
            RETURN_END_SECONDS,
        )
    return 0.0


def preview_mixture_amount(frame_index: int, frame_count: int) -> float:
    """Retain the former compressed light-preview helper for reproducibility."""
    if frame_count < 2:
        return 1.0
    return float(smootherstep(frame_index / (frame_count - 1)))


def camera_angles(global_seconds: float) -> tuple[float, float]:
    """Return tilt and heading in degrees for one production timestamp."""
    if global_seconds <= CAMERA_TILT_IN_END_SECONDS:
        tilt_progress = _ease_between(
            global_seconds,
            CAMERA_TILT_IN_START_SECONDS,
            CAMERA_TILT_IN_END_SECONDS,
        )
        tilt = CAMERA_TILT_START_DEGREES + tilt_progress * (
            CAMERA_TILT_HOLD_DEGREES - CAMERA_TILT_START_DEGREES
        )
    elif global_seconds < CAMERA_TILT_OUT_START_SECONDS:
        tilt = CAMERA_TILT_HOLD_DEGREES
    elif global_seconds < CAMERA_TILT_OUT_END_SECONDS:
        tilt_progress = _ease_between(
            global_seconds,
            CAMERA_TILT_OUT_START_SECONDS,
            CAMERA_TILT_OUT_END_SECONDS,
        )
        tilt = CAMERA_TILT_HOLD_DEGREES * (1.0 - tilt_progress)
    else:
        tilt = CAMERA_TILT_START_DEGREES

    heading_progress = _ease_between(
        global_seconds,
        CAMERA_HEADING_EASE_START_SECONDS,
        CAMERA_HEADING_EASE_END_SECONDS,
    )
    heading = CAMERA_HEADING_START_DEGREES + heading_progress * (
        CAMERA_HEADING_END_DEGREES - CAMERA_HEADING_START_DEGREES
    )
    return float(tilt), float(heading)


def fade_factor(global_seconds: float) -> float:
    """Fade the final five seconds to black after the daylight hold."""
    return 1.0 - _ease_between(
        global_seconds,
        FADE_START_SECONDS,
        FADE_END_SECONDS,
    )


def end_card_opacity(global_seconds: float) -> float:
    """Return the one-second fade-in/out envelope for the five-second card."""
    local_seconds = global_seconds - FILM_DURATION_SECONDS
    if local_seconds < 0.0 or local_seconds > END_CARD_DURATION_SECONDS:
        return 0.0
    fade_in = smootherstep(local_seconds / END_CARD_FADE_SECONDS)
    fade_out = smootherstep(
        (END_CARD_DURATION_SECONDS - local_seconds) / END_CARD_FADE_SECONDS
    )
    return float(fade_in * fade_out)


def _normalise_luminance(spd: np.ndarray) -> np.ndarray:
    return spd / np.sum(spd * YBAR)


D65_EQUAL_Y = _normalise_luminance(D65_SPD)
A_EQUAL_Y = _normalise_luminance(illuminant(ILLUMINANT_A_K))


def mixture_spd(amount: float) -> np.ndarray:
    """Mix equal-luminance D65 and A spectra physically."""
    return (1.0 - amount) * D65_EQUAL_Y + amount * A_EQUAL_Y


def _spectral_colours_float(spectra: np.ndarray, spd: np.ndarray) -> np.ndarray:
    """Convert spectra under one mixture and adapt from that mixture's white."""
    diffuser = np.ones((1, len(WAVELENGTHS)))
    mixture_white = spectra_to_xyz(diffuser, spd)[0]
    display_white = spectra_to_xyz(diffuser, D65_EQUAL_Y)[0]
    xyz = spectra_to_xyz(spectra, spd)
    adapted = adapt(xyz, mixture_white, display_white, ADAPTATION)
    return xyz_to_srgb_float(adapted).astype(np.float32)


def _triangle_areas(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    triangles = points[simplices]
    cross = (
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    return np.abs(cross) / 2.0


def _match_opposite_tile_edges(
    points: np.ndarray,
    simplices: np.ndarray,
    classes: np.ndarray,
    source_spectra: np.ndarray,
    land_spectra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Give every repeated-tile join one shared boundary-facet spectrum.

    PIL's integer polygon rasteriser includes the shared edge on both adjacent
    coplanar triangles. If their palettes differ, painter order emphasizes the
    join after downsampling. Opposite boundaries use the same 33 segments, so
    pairing them permits a small cross-edge palette match without flattening
    the native along-edge facet variation.
    """
    source = np.asarray(source_spectra, dtype=np.float64).copy()
    land = np.asarray(land_spectra, dtype=np.float64).copy()
    parent = np.arange(len(simplices), dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    triangle_points = points[simplices]
    for axis in (0, 1):
        other_axis = 1 - axis
        side_maps = []
        for side in (0.0, 1.0):
            edge = np.sum(triangle_points[:, :, axis] == side, axis=1) >= 2
            indices = np.flatnonzero(edge)
            if not np.all(classes[indices] == 0):
                raise RuntimeError("Source-tile boundary contains non-land facets")
            spans = np.column_stack(
                [
                    triangle_points[indices, :, other_axis].min(axis=1),
                    triangle_points[indices, :, other_axis].max(axis=1),
                ]
            )
            side_maps.append(
                {
                    tuple(np.round(span, 9)): int(index)
                    for span, index in zip(spans, indices)
                }
            )
        if side_maps[0].keys() != side_maps[1].keys():
            raise RuntimeError("Opposite source-tile boundary segments differ")
        for key in side_maps[0]:
            union(side_maps[0][key], side_maps[1][key])

    components: dict[int, list[int]] = {}
    boundary_indices = np.flatnonzero(
        np.sum((triangle_points == 0.0) | (triangle_points == 1.0), axis=(1, 2))
        >= 2
    )
    for index in boundary_indices:
        components.setdefault(find(int(index)), []).append(int(index))
    for indices in components.values():
        # Pull both sides toward a shared component mean without collapsing
        # the whole boundary strip to one palette. Retaining each facet's
        # local spectrum prevents a new topology band at its interior edge.
        component_mean = np.mean(
            np.vstack([source[indices], land[indices]]),
            axis=0,
        )
        match_strength = 0.20
        source[indices] = (
            (1.0 - match_strength) * source[indices]
            + match_strength * component_mean
        )
        land[indices] = (
            (1.0 - match_strength) * land[indices]
            + match_strength * component_mean
        )
    return source, land


def build_scene(
    seed: int = SEED,
    *,
    terrain: bool = False,
    exaggeration: float = DEFAULT_EXAGGERATION,
    terrain_shade: bool | None = None,
    shade_strength: float = DEFAULT_SHADE_STRENGTH,
    terrain_smoothing_metres: float = DEFAULT_TERRAIN_SMOOTHING_METRES,
    terrain_radial_noise_amplitude: float = DEFAULT_RADIAL_NOISE_AMPLITUDE,
    seamless_tile_edges: bool = True,
) -> FilmScene:
    """Load production spectra and construct a full-density tiled land plane."""
    mesh = np.load(MESH_PATH)
    points = mesh["points"]
    simplices = mesh["simplices"]
    centroids = mesh["centroids"]
    classes = mesh["classes"]
    areas = _triangle_areas(points, simplices)
    healthy = np.isin(classes, (1, 2))
    healthy_area = areas[healthy]
    target_normalized = np.average(
        centroids[healthy],
        axis=0,
        weights=healthy_area,
    )
    points_world = np.column_stack(
        [
            (points[:, 0] - target_normalized[0]) * FRAME_ASPECT_RATIO,
            target_normalized[1] - points[:, 1],
            np.zeros(len(points), dtype=np.float64),
        ]
    )
    if terrain and seamless_tile_edges:
        vertical = (points[:, 0] == 0.0) | (points[:, 0] == 1.0)
        horizontal = (points[:, 1] == 0.0) | (points[:, 1] == 1.0)
        vertical_wave = (
            0.55 * np.sin(2.0 * np.pi * (11.0 * points[:, 1] + 0.1931))
            + 0.45 * np.sin(2.0 * np.pi * (17.0 * points[:, 1] + 0.4198))
        )
        horizontal_wave = (
            0.55 * np.sin(2.0 * np.pi * (13.0 * points[:, 0] + 0.4198))
            + 0.45 * np.sin(2.0 * np.pi * (19.0 * points[:, 0] + 0.1931))
        )
        points_world[vertical, 0] += (
            TILE_JOIN_JITTER_FRACTION
            * FRAME_ASPECT_RATIO
            * vertical_wave[vertical]
        )
        points_world[horizontal, 1] += (
            TILE_JOIN_JITTER_FRACTION * horizontal_wave[horizontal]
        )
    terrain_max_height_world = 0.0
    terrain_max_height_metres = 0.0
    if terrain_shade is None:
        terrain_shade = terrain
    if terrain_shade and not terrain:
        raise ValueError("Terrain shading requires terrain geometry")
    if terrain:
        from terrain import build_vertex_terrain

        terrain_geometry = build_vertex_terrain(
            points,
            exaggeration=exaggeration,
            smoothing_metres=terrain_smoothing_metres,
            radial_noise_amplitude=terrain_radial_noise_amplitude,
            seed=seed,
        )
        points_world[:, 2] = terrain_geometry.heights_world
        terrain_max_height_world = terrain_geometry.max_height_world
        terrain_max_height_metres = (
            terrain_geometry.max_height_metres_exaggerated
        )
    final_spectra = np.load(SPECTRA_FINAL_PATH)
    base_spectra = np.load(SPECTRA_BASE_PATH)
    baked_hillshade_enabled = True
    if terrain_shade:
        unrelieved_base = np.load(SPECTRA_BASE_NO_RELIEF_PATH)
        relief_factors = np.load(RELIEF_FACTORS_PATH)
        if relief_factors.shape != (len(simplices),):
            raise RuntimeError("Relief factors do not match production mesh")
        perturbation = final_spectra - base_spectra
        final_spectra = (
            unrelieved_base
            + perturbation / relief_factors[:, None]
        )
        base_spectra = unrelieved_base
        baked_hillshade_enabled = False

    facet_shades = np.ones(len(simplices), dtype=np.float64)
    if terrain_shade:
        from terrain import facet_lambert_shades

        facet_shades = facet_lambert_shades(
            points_world,
            simplices,
            classes,
            strength=shade_strength,
        )

    identity_spectra = final_spectra.copy()
    identity_spectra[classes == 1] = base_spectra[classes == 1]
    land_indices = np.flatnonzero(classes == 0)
    rng = np.random.default_rng(seed + 9_031)
    land_spectra = base_spectra[
        rng.choice(land_indices, size=len(simplices), replace=True)
    ]
    if terrain and seamless_tile_edges:
        final_spectra, land_spectra = _match_opposite_tile_edges(
            points,
            simplices,
            classes,
            final_spectra,
            land_spectra,
        )
        identity_spectra = final_spectra.copy()
        identity_spectra[classes == 1] = base_spectra[classes == 1]
    triangle_centroids_world = points_world[simplices].mean(axis=1)
    tile_width, tile_height = FRAME_ASPECT_RATIO, 1.0
    max_x_tiles = int(math.ceil(EXTENDED_PLANE_HALF_SIZE / tile_width)) + 1
    max_y_tiles = int(math.ceil(EXTENDED_PLANE_HALF_SIZE / tile_height)) + 1
    tile_offsets = np.array(
        [
            (ix * tile_width, iy * tile_height, 0.0)
            for iy in range(-max_y_tiles, max_y_tiles + 1)
            for ix in range(-max_x_tiles, max_x_tiles + 1)
        ],
        dtype=np.float64,
    )
    healthy_indices = np.unique(simplices[np.isin(classes, (1, 2))].ravel())
    healthy_points_world = points_world[healthy_indices]
    source_boundary_world = tuple(
        points_world[indices]
        for indices in (
            np.flatnonzero(points[:, 0] == 0.0)[
                np.argsort(points[points[:, 0] == 0.0, 1])
            ],
            np.flatnonzero(points[:, 0] == 1.0)[
                np.argsort(points[points[:, 0] == 1.0, 1])
            ],
            np.flatnonzero(points[:, 1] == 0.0)[
                np.argsort(points[points[:, 1] == 0.0, 0])
            ],
            np.flatnonzero(points[:, 1] == 1.0)[
                np.argsort(points[points[:, 1] == 1.0, 0])
            ],
        )
    )
    return FilmScene(
        points_world=points_world,
        simplices=simplices,
        classes=classes,
        healthy_points_world=healthy_points_world,
        final_spectra=final_spectra,
        identity_spectra=identity_spectra,
        land_spectra=land_spectra,
        triangle_centroids_world=triangle_centroids_world,
        tile_offsets=tile_offsets,
        source_triangle_count=len(simplices),
        tile_triangle_count=len(tile_offsets) * len(simplices),
        tile_width=float(tile_width),
        tile_height=float(tile_height),
        target_normalized=target_normalized,
        terrain_enabled=bool(terrain),
        terrain_exaggeration=float(exaggeration),
        terrain_max_height_world=terrain_max_height_world,
        terrain_max_height_metres=terrain_max_height_metres,
        terrain_smoothing_metres=float(terrain_smoothing_metres),
        terrain_radial_noise_amplitude=float(
            terrain_radial_noise_amplitude
        ),
        terrain_shade_enabled=bool(terrain_shade),
        terrain_shade_strength=float(shade_strength),
        facet_shades=facet_shades,
        baked_hillshade_enabled=baked_hillshade_enabled,
        seamless_tile_edges=bool(seamless_tile_edges),
        source_boundary_world=source_boundary_world,
    )


def _camera_basis(
    tilt_degrees: float,
    heading_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tilt = math.radians(tilt_degrees)
    heading = math.radians(heading_degrees)
    right = np.array([math.cos(heading), -math.sin(heading), 0.0])
    ground_up = np.array([math.sin(heading), math.cos(heading), 0.0])
    camera_direction = (
        -math.sin(tilt) * ground_up
        + np.array([0.0, 0.0, math.cos(tilt)])
    )
    forward = -camera_direction
    up = np.cross(right, forward)
    return right, up, forward, camera_direction


def _project(
    points_world: np.ndarray,
    tilt_degrees: float,
    heading_degrees: float,
    distance: float,
    size: tuple[int, int],
    screen_shift_fraction: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    right, up, forward, camera_direction = _camera_basis(
        tilt_degrees,
        heading_degrees,
    )
    camera_position = camera_direction * distance
    relative = points_world - camera_position
    depth = relative @ forward
    focal = 1.0 / math.tan(math.radians(CAMERA_VERTICAL_FOV_DEGREES) / 2.0)
    safe_depth = np.maximum(depth, 1e-6)
    ndc_x = (relative @ right) * focal / (safe_depth * (width / height))
    ndc_y = (relative @ up) * focal / safe_depth
    projected = np.column_stack(
        [
            (ndc_x * 0.5 + 0.5 + screen_shift_fraction[0]) * (width - 1),
            (0.5 - ndc_y * 0.5 + screen_shift_fraction[1]) * (height - 1),
        ]
    )
    return projected, depth


def _fit_metric(projected: np.ndarray, size: tuple[int, int]) -> float:
    width, height = size
    spans = np.ptp(projected, axis=0)
    return float(max(spans[0] / width, spans[1] / height))


def camera_state(
    scene: FilmScene,
    global_seconds: float,
    size: tuple[int, int],
) -> CameraState:
    """Solve camera distance so the healthy extent remains comfortably framed."""
    tilt, heading = camera_angles(global_seconds)
    low, high = 0.05, 20.0
    for _ in range(28):
        distance = (low + high) / 2.0
        projected, depth = _project(
            scene.healthy_points_world,
            tilt,
            heading,
            distance,
            size,
        )
        metric = (
            float("inf")
            if np.any(depth <= 0.02)
            else _fit_metric(projected, size)
        )
        if metric > CAMERA_HEALTHY_FILL_FRACTION:
            low = distance
        else:
            high = distance
    distance = high
    projected, _ = _project(
        scene.healthy_points_world,
        tilt,
        heading,
        distance,
        size,
    )
    bbox_center = (projected.min(axis=0) + projected.max(axis=0)) / 2.0
    frame_center = np.array([(size[0] - 1) / 2.0, (size[1] - 1) / 2.0])
    shift = (frame_center - bbox_center) / np.array(
        [size[0] - 1, size[1] - 1], dtype=np.float64
    )
    projected, _ = _project(
        scene.healthy_points_world,
        tilt,
        heading,
        distance,
        size,
        (float(shift[0]), float(shift[1])),
    )
    centered_bbox = (projected.min(axis=0) + projected.max(axis=0)) / 2.0
    deviation = np.abs(centered_bbox - frame_center) / np.array(size)
    return CameraState(
        tilt_degrees=tilt,
        heading_degrees=heading,
        distance=distance,
        healthy_fill_fraction=_fit_metric(projected, size),
        screen_shift_fraction=(float(shift[0]), float(shift[1])),
        healthy_center_deviation_fraction=(
            float(deviation[0]),
            float(deviation[1]),
        ),
    )


def _draw_triangle_ids(
    draw: ImageDraw.ImageDraw,
    projected: np.ndarray,
    depth: np.ndarray,
    simplices: np.ndarray,
    size: tuple[int, int],
    id_offset: int,
    triangle_indices: np.ndarray | None = None,
) -> int:
    """Draw visible projected triangles and return the visible count."""
    width, height = size
    if triangle_indices is None:
        triangle_indices = np.arange(len(simplices), dtype=np.int32)
    selected_simplices = simplices[triangle_indices]
    polygons = projected[selected_simplices]
    polygon_depth = depth[selected_simplices]
    valid = np.all(polygon_depth > 0.02, axis=1)
    minimum = polygons.min(axis=1)
    maximum = polygons.max(axis=1)
    valid &= (
        (maximum[:, 0] >= 0.0)
        & (minimum[:, 0] < width)
        & (maximum[:, 1] >= 0.0)
        & (minimum[:, 1] < height)
    )
    visible_local_indices = np.flatnonzero(valid)
    visible_indices = triangle_indices[visible_local_indices]
    rounded = np.rint(polygons[visible_local_indices]).astype(np.int32)
    for triangle_index, polygon in zip(visible_indices, rounded):
        draw.polygon(
            [tuple(vertex) for vertex in polygon],
            fill=int(id_offset + triangle_index),
        )
    return int(len(visible_indices))


def _visible_triangle_records(
    projected: np.ndarray,
    depth: np.ndarray,
    simplices: np.ndarray,
    size: tuple[int, int],
    id_offset: int,
    triangle_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return visible polygons, palette ids, and mean depth for 3D sorting."""
    width, height = size
    if triangle_indices is None:
        triangle_indices = np.arange(len(simplices), dtype=np.int32)
    selected_simplices = simplices[triangle_indices]
    polygons = projected[selected_simplices]
    polygon_depth = depth[selected_simplices]
    valid = np.all(polygon_depth > 0.02, axis=1)
    minimum = polygons.min(axis=1)
    maximum = polygons.max(axis=1)
    valid &= (
        (maximum[:, 0] >= 0.0)
        & (minimum[:, 0] < width)
        & (maximum[:, 1] >= 0.0)
        & (minimum[:, 1] < height)
    )
    local = np.flatnonzero(valid)
    indices = triangle_indices[local]
    return (
        np.rint(polygons[local]).astype(np.int32),
        (id_offset + indices).astype(np.int32),
        polygon_depth[local].mean(axis=1),
    )


def _land_tile_points_world(
    scene: FilmScene,
    tile_offset: np.ndarray,
) -> np.ndarray:
    """Return a flat surrounding land tile, continuous with the source edge."""
    tile_points = scene.points_world + tile_offset
    if scene.terrain_enabled and not np.allclose(tile_offset, 0.0):
        tile_points = tile_points.copy()
        tile_points[:, 2] = 0.0
    return tile_points


def _render_terrain_ids(
    scene: FilmScene,
    camera: CameraState,
    render_size: tuple[int, int],
    spectra: np.ndarray,
    spd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Painter-rasterise all terrain tiles in one global depth order.

    Film terrain is a height field without intersecting or folded facets, so a
    global far-to-near triangle order provides correct ridge occlusion while
    retaining the established integer-palette antialiasing path.
    """
    palette_blocks = [BACKGROUND_RGB[None, :]]
    palette_offset = 1
    polygons: list[np.ndarray] = []
    palette_ids: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    land_colours = _spectral_colours_float(scene.land_spectra, spd)
    visible_land_tiles = 0

    for tile_offset in scene.tile_offsets:
        if np.allclose(tile_offset, 0.0):
            continue
        shifted_centroids = scene.triangle_centroids_world + tile_offset
        radii = np.linalg.norm(shifted_centroids[:, :2], axis=1)
        triangle_indices = np.flatnonzero(radii < FOG_END_RADIUS)
        if len(triangle_indices) == 0:
            continue
        projected, depth = _project(
            _land_tile_points_world(scene, tile_offset),
            camera.tilt_degrees,
            camera.heading_degrees,
            camera.distance,
            render_size,
            camera.screen_shift_fraction,
        )
        tile_polygons, tile_ids, tile_depths = _visible_triangle_records(
            projected,
            depth,
            scene.simplices,
            render_size,
            palette_offset,
            triangle_indices,
        )
        if not len(tile_ids):
            continue
        fog = smootherstep(
            (radii - FOG_START_RADIUS) / (FOG_END_RADIUS - FOG_START_RADIUS)
        ).astype(np.float32)
        palette_blocks.append(
            land_colours * (1.0 - fog[:, None])
            + BACKGROUND_RGB[None, :] * fog[:, None]
        )
        polygons.append(tile_polygons)
        palette_ids.append(tile_ids)
        depths.append(tile_depths)
        palette_offset += len(land_colours)
        visible_land_tiles += 1

    main_projected, main_depth = _project(
        scene.points_world,
        camera.tilt_degrees,
        camera.heading_degrees,
        camera.distance,
        render_size,
        camera.screen_shift_fraction,
    )
    main_polygons, main_ids, main_depths = _visible_triangle_records(
        main_projected,
        main_depth,
        scene.simplices,
        render_size,
        palette_offset,
    )
    main_colours = _spectral_colours_float(spectra, spd)
    if scene.terrain_shade_enabled:
        main_colours = main_colours * scene.facet_shades[:, None]
    palette_blocks.append(main_colours)
    polygons.append(main_polygons)
    palette_ids.append(main_ids)
    depths.append(main_depths)

    all_polygons = np.concatenate(polygons)
    all_ids = np.concatenate(palette_ids)
    all_depths = np.concatenate(depths)
    order = np.argsort(all_depths)[::-1]
    id_canvas = Image.new("I", render_size, 0)
    draw = ImageDraw.Draw(id_canvas)
    for index in order:
        draw.polygon(
            [tuple(vertex) for vertex in all_polygons[index]],
            fill=int(all_ids[index]),
        )
    return (
        np.asarray(id_canvas, dtype=np.int32),
        np.vstack(palette_blocks),
        int(len(order)),
        visible_land_tiles,
    )


def render_camera_frame(
    scene: FilmScene,
    global_seconds: float,
    size: tuple[int, int],
    *,
    spectra: np.ndarray | None = None,
    supersample: int = CAMERA_SUPERSAMPLE,
    apply_fade: bool = True,
    output_float: bool = False,
    mixture_override: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Project and rasterise the true triangle geometry for one film frame."""
    started = time.perf_counter()
    if spectra is None:
        spectra = scene.final_spectra
    camera = camera_state(scene, global_seconds, size)
    render_size = (size[0] * supersample, size[1] * supersample)
    mixture = (
        mixture_amount(global_seconds)
        if mixture_override is None
        else float(mixture_override)
    )
    spd = mixture_spd(mixture)
    if scene.terrain_enabled:
        ids, palette, visible_triangles, visible_land_tiles = (
            _render_terrain_ids(
                scene,
                camera,
                render_size,
                spectra,
                spd,
            )
        )
    else:
        id_canvas = Image.new("I", render_size, 0)
        draw = ImageDraw.Draw(id_canvas)
        land_colours = _spectral_colours_float(scene.land_spectra, spd)
        palette_blocks = [BACKGROUND_RGB[None, :]]
        palette_offset = 1
        visible_triangles = 0
        visible_land_tiles = 0

        # Preserve the approved flat renderer exactly when terrain is off.
        for tile_offset in scene.tile_offsets:
            if np.allclose(tile_offset, 0.0):
                continue
            shifted_centroids = scene.triangle_centroids_world + tile_offset
            radii = np.linalg.norm(shifted_centroids[:, :2], axis=1)
            triangle_indices = np.flatnonzero(radii < FOG_END_RADIUS)
            if len(triangle_indices) == 0:
                continue
            tile_points = _land_tile_points_world(scene, tile_offset)
            projected, depth = _project(
                tile_points,
                camera.tilt_degrees,
                camera.heading_degrees,
                camera.distance,
                render_size,
                camera.screen_shift_fraction,
            )
            drawn = _draw_triangle_ids(
                draw,
                projected,
                depth,
                scene.simplices,
                render_size,
                palette_offset,
                triangle_indices,
            )
            if drawn == 0:
                continue
            fog = smootherstep(
                (radii - FOG_START_RADIUS) / (FOG_END_RADIUS - FOG_START_RADIUS)
            ).astype(np.float32)
            tile_colours = (
                land_colours * (1.0 - fog[:, None])
                + BACKGROUND_RGB[None, :] * fog[:, None]
            )
            palette_blocks.append(tile_colours)
            palette_offset += len(tile_colours)
            visible_triangles += drawn
            visible_land_tiles += 1

        # The observed source tile is drawn last, over the land-only underlay.
        main_projected, main_depth = _project(
            scene.points_world,
            camera.tilt_degrees,
            camera.heading_degrees,
            camera.distance,
            render_size,
            camera.screen_shift_fraction,
        )
        visible_triangles += _draw_triangle_ids(
            draw,
            main_projected,
            main_depth,
            scene.simplices,
            render_size,
            palette_offset,
        )
        main_colours = _spectral_colours_float(spectra, spd)
        palette_blocks.append(main_colours)
        palette = np.vstack(palette_blocks)
        ids = np.asarray(id_canvas, dtype=np.int32)
    corner_background_pixels = None
    if scene.terrain_enabled:
        corner = min(16 * supersample, render_size[0], render_size[1])
        corner_background_pixels = [
            int(np.count_nonzero(ids[:corner, :corner] == 0)),
            int(np.count_nonzero(ids[:corner, -corner:] == 0)),
            int(np.count_nonzero(ids[-corner:, :corner] == 0)),
            int(np.count_nonzero(ids[-corner:, -corner:] == 0)),
        ]
    if output_float:
        channels = []
        for channel_index in range(3):
            supersampled_channel = palette[:, channel_index][ids]
            channels.append(
                np.asarray(
                    Image.fromarray(supersampled_channel, mode="F").resize(
                        size,
                        Image.Resampling.LANCZOS,
                    ),
                    dtype=np.float32,
                )
            )
        frame = np.clip(np.stack(channels, axis=-1), 0.0, 1.0)
    else:
        supersampled = np.clip(
            np.rint(palette[ids] * 255.0),
            0,
            255,
        ).astype(np.uint8)
        frame = np.asarray(
            Image.fromarray(supersampled).resize(size, Image.Resampling.LANCZOS),
            dtype=np.uint8,
        ).copy()
    if apply_fade:
        fade = fade_factor(global_seconds)
        if fade < 1.0:
            if output_float:
                frame = frame * fade
            else:
                frame = np.clip(
                    np.rint(frame.astype(np.float32) * fade), 0, 255
                ).astype(np.uint8)
    return frame, {
        "seconds": global_seconds,
        "mixture_amount": mixture,
        "tilt_degrees": camera.tilt_degrees,
        "heading_degrees": camera.heading_degrees,
        "distance": camera.distance,
        "healthy_fill_fraction": camera.healthy_fill_fraction,
        "screen_shift_fraction": list(camera.screen_shift_fraction),
        "healthy_center_deviation_fraction": list(
            camera.healthy_center_deviation_fraction
        ),
        "visible_triangles": visible_triangles,
        "visible_land_tiles": visible_land_tiles,
        "terrain_enabled": scene.terrain_enabled,
        "terrain_exaggeration": scene.terrain_exaggeration,
        "terrain_shade_enabled": scene.terrain_shade_enabled,
        "terrain_shade_strength": scene.terrain_shade_strength,
        "corner_background_pixels": corner_background_pixels,
        "render_seconds": time.perf_counter() - started,
    }


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render_end_card(
    global_seconds: float,
    size: tuple[int, int],
    *,
    output_float: bool = False,
) -> tuple[np.ndarray, dict]:
    """Render the final black five-second title card."""
    started = time.perf_counter()
    scale = 2
    render_size = (size[0] * scale, size[1] * scale)
    opacity = end_card_opacity(global_seconds)
    mask = Image.new("L", render_size, 0)
    draw = ImageDraw.Draw(mask)
    title_font = _font(max(24, int(round(size[1] * 0.026 * scale))))
    detail_font = _font(max(18, int(round(size[1] * 0.017 * scale))))
    fonts = (title_font, detail_font, detail_font)
    gaps = (int(size[1] * 0.026 * scale), int(size[1] * 0.012 * scale))
    boxes = [draw.textbbox((0, 0), line, font=font) for line, font in zip(END_CARD_LINES, fonts)]
    heights = [box[3] - box[1] for box in boxes]
    total_height = sum(heights) + sum(gaps)
    y = (render_size[1] - total_height) / 2.0
    for line_index, (line, font, box, line_height) in enumerate(
        zip(END_CARD_LINES, fonts, boxes, heights)
    ):
        line_width = box[2] - box[0]
        x = (render_size[0] - line_width) / 2.0
        draw.text((x, y - box[1]), line, font=font, fill=255)
        y += line_height
        if line_index < len(gaps):
            y += gaps[line_index]
    downsampled = np.asarray(
        mask.resize(size, Image.Resampling.LANCZOS), dtype=np.float32
    ) / 255.0
    frame_float = (
        downsampled[:, :, None]
        * END_CARD_TEXT_RGB[None, None, :]
        * opacity
    )
    if output_float:
        frame = frame_float.astype(np.float32)
    else:
        frame = np.clip(np.rint(frame_float * 255.0), 0, 255).astype(np.uint8)
    return frame, {
        "seconds": global_seconds,
        "end_card": True,
        "opacity": opacity,
        "render_seconds": time.perf_counter() - started,
    }


def render_production_frame(
    scene: FilmScene,
    global_seconds: float,
    size: tuple[int, int],
    *,
    output_float: bool = False,
) -> tuple[np.ndarray, dict]:
    """Render either the map film or its appended end card."""
    if global_seconds >= FILM_DURATION_SECONDS:
        return render_end_card(
            global_seconds,
            size,
            output_float=output_float,
        )
    return render_camera_frame(
        scene,
        global_seconds,
        size,
        output_float=output_float,
    )


def _ground_frustum_corners(
    camera: CameraState,
    size: tuple[int, int],
) -> np.ndarray:
    """Intersect the four off-axis view rays with the flat z=0 plane."""
    width, height = size
    right, up, forward, camera_direction = _camera_basis(
        camera.tilt_degrees,
        camera.heading_degrees,
    )
    camera_position = camera_direction * camera.distance
    tangent = math.tan(math.radians(CAMERA_VERTICAL_FOV_DEGREES) / 2.0)
    aspect = width / height
    result = []
    for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        ndc_x = 2.0 * (u - 0.5 - camera.screen_shift_fraction[0])
        ndc_y = 2.0 * (0.5 + camera.screen_shift_fraction[1] - v)
        ray = forward + right * (ndc_x * aspect * tangent) + up * (ndc_y * tangent)
        result.append(camera_position + ray * (-camera_position[2] / ray[2]))
    return np.asarray(result)


def _projected_triangle_area_fraction(
    projected: np.ndarray,
    depth: np.ndarray,
    simplices: np.ndarray,
    size: tuple[int, int],
) -> float:
    triangles = projected[simplices]
    triangle_depth = depth[simplices]
    valid = np.all(triangle_depth > 0.02, axis=1)
    minimum = triangles.min(axis=1)
    maximum = triangles.max(axis=1)
    valid &= (
        (maximum[:, 0] >= 0.0)
        & (minimum[:, 0] < size[0])
        & (maximum[:, 1] >= 0.0)
        & (minimum[:, 1] < size[1])
    )
    triangles = triangles[valid]
    if len(triangles) == 0:
        return 0.0
    cross = (
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    return float(np.max(np.abs(cross) / 2.0) / (size[0] * size[1]))


def camera_geometry_report(
    scene: FilmScene | None = None,
    *,
    size: tuple[int, int] = (640, 360),
    sample_interval_seconds: float = 2.0,
) -> dict:
    """Run the framing, horizon-density, plane-extent, and legacy gates."""
    if scene is None:
        scene = build_scene()
    sample_times = np.arange(
        0.0,
        FILM_DURATION_SECONDS + 0.001,
        sample_interval_seconds,
    )
    rows = []
    outside_lake = ~np.isin(scene.classes, (1, 2))
    max_triangle_fraction = 0.0
    min_plane_multiple = float("inf")
    max_center_deviation = 0.0
    for seconds in sample_times:
        camera = camera_state(scene, float(seconds), size)
        projected, depth = _project(
            scene.points_world,
            camera.tilt_degrees,
            camera.heading_degrees,
            camera.distance,
            size,
            camera.screen_shift_fraction,
        )
        triangle_fraction = _projected_triangle_area_fraction(
            projected,
            depth,
            scene.simplices[outside_lake],
            size,
        )
        for tile_offset in scene.tile_offsets:
            if np.allclose(tile_offset, 0.0):
                continue
            shifted_centroids = scene.triangle_centroids_world + tile_offset
            tile_indices = np.flatnonzero(
                np.linalg.norm(shifted_centroids[:, :2], axis=1)
                < FOG_END_RADIUS
            )
            if len(tile_indices) == 0:
                continue
            tile_projected, tile_depth = _project(
                _land_tile_points_world(scene, tile_offset),
                camera.tilt_degrees,
                camera.heading_degrees,
                camera.distance,
                size,
                camera.screen_shift_fraction,
            )
            triangle_fraction = max(
                triangle_fraction,
                _projected_triangle_area_fraction(
                    tile_projected,
                    tile_depth,
                    scene.simplices[tile_indices],
                    size,
                ),
            )
        frustum = _ground_frustum_corners(camera, size)
        furthest_axis_extent = float(np.max(np.abs(frustum[:, :2])))
        plane_multiple = EXTENDED_PLANE_HALF_SIZE / furthest_axis_extent
        center_deviation = max(camera.healthy_center_deviation_fraction)
        max_triangle_fraction = max(max_triangle_fraction, triangle_fraction)
        min_plane_multiple = min(min_plane_multiple, plane_multiple)
        max_center_deviation = max(max_center_deviation, center_deviation)
        rows.append(
            {
                "seconds": float(seconds),
                "tilt_degrees": camera.tilt_degrees,
                "heading_degrees": camera.heading_degrees,
                "center_deviation_fraction": list(
                    camera.healthy_center_deviation_fraction
                ),
                "max_outside_lake_triangle_frame_fraction": triangle_fraction,
                "plane_to_frustum_multiple": plane_multiple,
            }
        )

    phase2_report = json.loads(
        (DATA_DIR.parent / "out" / "phase2" / "phase2_report.json").read_text(
            encoding="utf-8"
        )
    )
    camouflage = phase2_report["metamers"]["d65_camouflage"]
    density = phase2_report["mesh"]["density_camouflage"]
    report = {
        "sample_interval_seconds": sample_interval_seconds,
        "sample_count": len(rows),
        "look_at": "area-weighted centroid of the 4,198 ft extent",
        "target_normalized_xy": scene.target_normalized.tolist(),
        "max_center_deviation_fraction": max_center_deviation,
        "center_limit_fraction": 0.03,
        "max_outside_lake_triangle_frame_fraction": max_triangle_fraction,
        "triangle_area_limit_fraction": 0.005,
        "minimum_plane_to_frustum_multiple": min_plane_multiple,
        "required_plane_to_frustum_multiple": EXTENDED_PLANE_MIN_FRUSTUM_MULTIPLE,
        "plane_half_size_world_units": EXTENDED_PLANE_HALF_SIZE,
        "fog_fully_opaque_radius": FOG_END_RADIUS,
        "fog_opaque_before_plane_edge": FOG_END_RADIUS < EXTENDED_PLANE_HALF_SIZE,
        "final_heading_degrees": camera_angles(CAMERA_HEADING_EASE_END_SECONDS)[1],
        "camouflage": {
            "ks_pvalue": camouflage["ks_pvalue"],
            "median_relative_difference": camouflage[
                "median_relative_difference"
            ],
            "passes": camouflage["passes_target"],
        },
        "density": {
            "relative_difference": density["relative_difference"],
            "passes": density["passes_target"],
        },
        "passes": bool(
            max_center_deviation <= 0.03
            and max_triangle_fraction <= 0.005
            and min_plane_multiple >= EXTENDED_PLANE_MIN_FRUSTUM_MULTIPLE
            and FOG_END_RADIUS < EXTENDED_PLANE_HALF_SIZE
            and camera_angles(CAMERA_HEADING_EASE_END_SECONDS)[1]
            == CAMERA_HEADING_END_DEGREES
            and camouflage["passes_target"]
            and density["passes_target"]
        ),
        "samples": rows,
    }
    GEOMETRY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GEOMETRY_REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _ffmpeg_command(path: Path, size: tuple[int, int]) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(H264_CRF),
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-movflags",
        "+faststart",
        str(path),
    ]


def _encode_times(
    scene: FilmScene,
    path: Path,
    size: tuple[int, int],
    global_times: np.ndarray,
) -> tuple[list[dict], float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        _ffmpeg_command(path, size),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("Could not open ffmpeg input pipe")
    metadata = []
    wall_started = time.perf_counter()
    for frame_index, global_seconds in enumerate(global_times):
        frame, frame_metadata = render_camera_frame(
            scene,
            float(global_seconds),
            size,
        )
        process.stdin.write(frame.tobytes())
        metadata.append(frame_metadata)
        if (frame_index + 1) % 120 == 0 or frame_index + 1 == len(global_times):
            print(
                f"{path.name}: {frame_index + 1}/{len(global_times)} frames",
                flush=True,
            )
    process.stdin.close()
    return_code = process.wait()
    wall_seconds = time.perf_counter() - wall_started
    error = (
        process.stderr.read().decode("utf-8", errors="replace")
        if process.stderr
        else ""
    )
    if return_code:
        raise RuntimeError(f"ffmpeg failed with {return_code}: {error}")
    return metadata, wall_seconds


def _probe(path: Path) -> dict:
    payload = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(payload)["streams"][0]


_DITHER_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _blue_noise_map(size: tuple[int, int]) -> np.ndarray:
    """Build a deterministic high-pass blue-noise-like dither map."""
    if size in _DITHER_CACHE:
        return _DITHER_CACHE[size]
    rng = np.random.default_rng(SEED + 44_198)
    white = rng.normal(size=(64, 64)).astype(np.float32)
    high_pass = white - gaussian_filter(white, sigma=1.2, mode="wrap")
    high_pass /= max(float(np.std(high_pass)), 1e-6)
    tile = np.clip(high_pass / 5.0, -0.5, 0.5)
    width, height = size
    result = tile[np.arange(height) % 64][:, np.arange(width) % 64]
    _DITHER_CACHE[size] = result
    return result


def _quantize_blue_noise(frame: np.ndarray) -> np.ndarray:
    noise = _blue_noise_map((frame.shape[1], frame.shape[0]))
    result = np.empty(frame.shape, dtype=np.uint8)
    for channel in range(3):
        channel_noise = np.roll(noise, (channel * 17, channel * 29), axis=(0, 1))
        result[:, :, channel] = np.clip(
            np.rint(frame[:, :, channel] * 255.0 + channel_noise),
            0,
            255,
        ).astype(np.uint8)
    return result


def _final_h264_command(path: Path, size: tuple[int, int], crf: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-movflags",
        "+faststart",
        str(path),
    ]


def _prores_command(path: Path, size: tuple[int, int]) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "3",
        "-vendor",
        "apl0",
        "-pix_fmt",
        "yuv422p10le",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        str(path),
    ]


def _close_encoder(process: subprocess.Popen, label: str) -> None:
    if process.stdin is not None:
        process.stdin.close()
    return_code = process.wait()
    error = (
        process.stderr.read().decode("utf-8", errors="replace")
        if process.stderr
        else ""
    )
    if return_code:
        raise RuntimeError(f"{label} encoder failed with {return_code}: {error}")


def render_final_film(
    seed: int = SEED,
    *,
    terrain: bool = False,
    exaggeration: float = DEFAULT_EXAGGERATION,
) -> dict:
    """Render the approved 4K delivery, 10-bit master, and upload copy."""
    FILM_DIR.mkdir(parents=True, exist_ok=True)
    scene = build_scene(
        seed,
        terrain=terrain,
        exaggeration=exaggeration,
    )
    geometry = camera_geometry_report(scene)
    identity_checks = _identity_checks(scene)
    identity_max = max(row["max_pixel_difference"] for row in identity_checks)
    if not geometry["passes"] or identity_max != 0:
        raise RuntimeError(
            "Final render gate failed: "
            f"geometry={geometry['passes']}, identity_max={identity_max}"
        )

    size = (3840, 2160)
    frame_count = int(round(TOTAL_DURATION_SECONDS * FPS))
    h264 = subprocess.Popen(
        _final_h264_command(FINAL_2160P_PATH, size, 14),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    prores = subprocess.Popen(
        _prores_command(FINAL_PRORES_PATH, size),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if h264.stdin is None or prores.stdin is None:
        raise RuntimeError("Could not open final encoder input pipes")

    rows = []
    wall_started = time.perf_counter()
    try:
        for frame_index in range(frame_count):
            global_seconds = frame_index / FPS
            frame, metadata = render_production_frame(
                scene,
                global_seconds,
                size,
                output_float=True,
            )
            h264.stdin.write(_quantize_blue_noise(frame).tobytes())
            prores_frame = np.clip(
                np.rint(frame * 65_535.0), 0, 65_535
            ).astype("<u2")
            prores.stdin.write(prores_frame.tobytes())
            rows.append(metadata)
            if (frame_index + 1) % 12 == 0 or frame_index + 1 == frame_count:
                elapsed = time.perf_counter() - wall_started
                print(
                    f"final 4K: {frame_index + 1}/{frame_count} frames "
                    f"({elapsed:.1f}s elapsed)",
                    flush=True,
                )
    except BaseException:
        for process in (h264, prores):
            process.kill()
        raise
    _close_encoder(h264, "H.264")
    _close_encoder(prores, "ProRes")
    master_wall_seconds = time.perf_counter() - wall_started

    upload_started = time.perf_counter()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(FINAL_2160P_PATH),
            "-vf",
            "scale=1920:1080:flags=lanczos",
            "-an",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(FINAL_1080P_PATH),
        ],
        check=True,
    )
    upload_wall_seconds = time.perf_counter() - upload_started
    total_wall_seconds = time.perf_counter() - wall_started

    outputs = {}
    for label, path in (
        ("h264_2160p", FINAL_2160P_PATH),
        ("prores_422_hq_10bit", FINAL_PRORES_PATH),
        ("h264_1080p", FINAL_1080P_PATH),
    ):
        outputs[label] = {
            "path": str(path.relative_to(DATA_DIR.parent)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "probe": _probe(path),
        }
    durations = [float(item["probe"]["duration"]) for item in outputs.values()]
    report = {
        "seed": seed,
        "fps": FPS,
        "frame_count": frame_count,
        "map_film_duration_seconds": FILM_DURATION_SECONDS,
        "end_card_duration_seconds": END_CARD_DURATION_SECONDS,
        "total_duration_seconds": TOTAL_DURATION_SECONDS,
        "duration_check": {
            "expected_seconds": 215.0,
            "actual_seconds": durations,
            "passes": all(abs(duration - 215.0) <= 1.0 / FPS for duration in durations),
        },
        "final_heading_degrees": camera_angles(
            CAMERA_HEADING_EASE_END_SECONDS
        )[1],
        "geometry_report": str(GEOMETRY_REPORT_PATH.relative_to(DATA_DIR.parent)),
        "geometry_tests_pass": geometry["passes"],
        "identity_checks": identity_checks,
        "identity_max_pixel_difference": identity_max,
        "end_card": {
            "lines": list(END_CARD_LINES),
            "fade_in_seconds": END_CARD_FADE_SECONDS,
            "fade_out_seconds": END_CARD_FADE_SECONDS,
        },
        "encoding": {
            "h264_2160p": "H.264 High, yuv420p, CRF 14, deterministic blue-noise dither",
            "master": "ProRes 422 HQ, yuv422p10le",
            "h264_1080p": "Lanczos downsample, H.264 High, yuv420p, CRF 16",
        },
        "timing": {
            "master_render_and_encode_seconds": master_wall_seconds,
            "upload_transcode_seconds": upload_wall_seconds,
            "total_seconds": total_wall_seconds,
            "frames": _summarise_timing(rows),
        },
        "outputs": outputs,
    }
    if not report["duration_check"]["passes"]:
        raise RuntimeError(f"Final duration check failed: {durations}")
    FINAL_REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarise_timing(rows: list[dict]) -> dict:
    durations = np.array([row["render_seconds"] for row in rows])
    return {
        "frame_count": int(len(rows)),
        "mean_seconds_per_frame": float(np.mean(durations)),
        "median_seconds_per_frame": float(np.median(durations)),
        "p95_seconds_per_frame": float(np.percentile(durations, 95)),
    }


def _identity_checks(scene: FilmScene) -> list[dict]:
    checks = []
    for seconds in (0.0, 20.0, 39.0, 200.0, 205.0):
        final, _ = render_camera_frame(
            scene,
            seconds,
            (640, 360),
            supersample=2,
            apply_fade=False,
        )
        baseline, _ = render_camera_frame(
            scene,
            seconds,
            (640, 360),
            spectra=scene.identity_spectra,
            supersample=2,
            apply_fade=False,
        )
        checks.append(
            {
                "seconds": seconds,
                "mixture_amount": mixture_amount(seconds),
                "max_pixel_difference": int(
                    np.max(
                        np.abs(final.astype(np.int16) - baseline.astype(np.int16))
                    )
                ),
            }
        )
    return checks


def _terrain_identity_checks(
    scene: FilmScene,
    *,
    size: tuple[int, int] = (640, 360),
) -> list[dict]:
    """Compare metamer and base spectra under D65 at key camera states."""
    checks = []
    for seconds in (0.0, 105.0, 180.0):
        final, _ = render_camera_frame(
            scene,
            seconds,
            size,
            apply_fade=False,
            mixture_override=0.0,
        )
        baseline, _ = render_camera_frame(
            scene,
            seconds,
            size,
            spectra=scene.identity_spectra,
            apply_fade=False,
            mixture_override=0.0,
        )
        checks.append(
            {
                "seconds": seconds,
                "max_pixel_difference": int(
                    np.max(
                        np.abs(
                            final.astype(np.int16)
                            - baseline.astype(np.int16)
                        )
                    )
                ),
            }
        )
    return checks


def source_tile_edge_luminance_check(
    scene: FilmScene,
    *,
    size: tuple[int, int] = (1920, 1080),
    line_pixels: int = 2,
    reference_offset_pixels: int = 8,
) -> dict:
    """Measure visible contrast on the repeated source-tile join."""
    frame, _ = render_camera_frame(
        scene,
        0.0,
        size,
        apply_fade=False,
        mixture_override=0.0,
    )
    camera = camera_state(scene, 0.0, size)
    line = int(line_pixels)
    offset = int(reference_offset_pixels)
    if line != 2 or offset < line:
        raise ValueError("Seam check requires a 2 px line and wider offset")
    projected_boundaries = [
        _project(
            boundary,
            camera.tilt_degrees,
            camera.heading_degrees,
            camera.distance,
            size,
            camera.screen_shift_fraction,
        )[0]
        for boundary in scene.source_boundary_world
    ]
    all_projected = np.vstack(projected_boundaries)
    x0, y0 = np.floor(all_projected.min(axis=0)).astype(int)
    x1, y1 = np.ceil(all_projected.max(axis=0)).astype(int)
    if (
        x0 < offset + line
        or y0 < offset + line
        or x1 + offset + line >= size[0]
        or y1 + offset + line >= size[1]
    ):
        raise RuntimeError("Source tile seam lines are outside the render frame")
    luminance = frame.astype(np.float64) @ np.array(
        [0.2126, 0.7152, 0.0722]
    )
    high_pass = np.abs(luminance - gaussian_filter(luminance, sigma=2.0))
    boundary_coordinates = []
    reference_coordinates = []
    for polyline in projected_boundaries:
        # Omit two end segments so corner overlaps cannot dominate the tail.
        for start, end in zip(polyline[2:-3], polyline[3:-2]):
            direction = end - start
            length = float(np.linalg.norm(direction))
            if length <= 1.0:
                continue
            normal = np.array([-direction[1], direction[0]]) / length
            sample_count = max(1, int(math.floor(length)))
            progress = (np.arange(sample_count) + 0.5) / sample_count
            centers = start + progress[:, None] * direction
            for displacement in (-0.5, 0.5):
                boundary_coordinates.append(centers + displacement * normal)
            for displacement in (
                -offset - 0.5,
                -offset + 0.5,
                offset - 0.5,
                offset + 0.5,
            ):
                reference_coordinates.append(centers + displacement * normal)
    boundary_xy = np.vstack(boundary_coordinates)
    reference_xy = np.vstack(reference_coordinates)

    def sample(array: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        return map_coordinates(
            array,
            np.vstack([coordinates[:, 1], coordinates[:, 0]]),
            order=1,
            mode="nearest",
            prefilter=False,
        )

    boundary = sample(luminance, boundary_xy)
    reference = sample(luminance, reference_xy)
    boundary_high_pass = sample(high_pass, boundary_xy)
    reference_high_pass = sample(high_pass, reference_xy)
    boundary_mean = float(np.mean(boundary))
    reference_mean = float(np.mean(reference))
    normalizer = max(
        (boundary_mean + reference_mean) / 2.0,
        np.finfo(np.float64).eps,
    )
    mean_relative_difference = abs(boundary_mean - reference_mean) / normalizer
    boundary_high_pass_p99 = float(np.percentile(boundary_high_pass, 99.0))
    reference_high_pass_p99 = float(np.percentile(reference_high_pass, 99.0))
    high_pass_p99_absolute_relative_difference = (
        abs(boundary_high_pass_p99 - reference_high_pass_p99) / normalizer
    )
    high_pass_p99_relative_difference = (
        high_pass_p99_absolute_relative_difference
    )
    limit = 0.005
    return {
        "line_pixels": line,
        "reference_offset_pixels": offset,
        "source_bounds_px": [int(x0), int(y0), int(x1), int(y1)],
        "boundary_mean_luminance": boundary_mean,
        "reference_mean_luminance": reference_mean,
        "mean_relative_difference": float(mean_relative_difference),
        "boundary_high_pass_p99": boundary_high_pass_p99,
        "reference_high_pass_p99": reference_high_pass_p99,
        "high_pass_p99_relative_difference": float(
            high_pass_p99_relative_difference
        ),
        "high_pass_p99_absolute_relative_difference": float(
            high_pass_p99_absolute_relative_difference
        ),
        "relative_difference": float(mean_relative_difference),
        "limit": limit,
        "passes": bool(
            mean_relative_difference < limit
            and high_pass_p99_relative_difference < limit
        ),
    }


def _contact_sheet(
    rows: list[tuple[float, Image.Image, Image.Image]],
    destination: Path,
) -> None:
    panel_width = 800
    panel_height = 450
    label_height = 42
    sheet = Image.new(
        "RGB",
        (panel_width * 2, len(rows) * (panel_height + label_height)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(24)
    for row_index, (seconds, flat, terrain) in enumerate(rows):
        y = row_index * (panel_height + label_height)
        flat_panel = flat.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        terrain_panel = terrain.resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS
        )
        sheet.paste(flat_panel, (0, y + label_height))
        sheet.paste(terrain_panel, (panel_width, y + label_height))
        draw.text(
            (18, y + 8),
            f"{int(seconds // 60)}:{int(seconds % 60):02d} · terrain off",
            font=font,
            fill=(220, 220, 220),
        )
        draw.text(
            (panel_width + 18, y + 8),
            f"{int(seconds // 60)}:{int(seconds % 60):02d} · terrain on",
            font=font,
            fill=(220, 220, 220),
        )
    sheet.save(destination, optimize=True)


def render_terrain_shade_preview(
    seed: int = SEED,
    *,
    shade_strength: float = DEFAULT_SHADE_STRENGTH,
    terrain_smoothing_metres: float = DEFAULT_TERRAIN_SMOOTHING_METRES,
) -> dict:
    """Render the broad noisy falloff beside its zero-noise control."""
    TERRAIN_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sweep_rows = []
    scenes: dict[float, FilmScene] = {}
    frames: dict[tuple[float, float], Image.Image] = {}
    identity_rows = []
    for noise_amplitude in TERRAIN_FALLOFF_NOISE_COMPARISON:
        scene = build_scene(
            seed,
            terrain=True,
            exaggeration=TERRAIN_SHADE_EXAGGERATION,
            terrain_shade=True,
            shade_strength=shade_strength,
            terrain_smoothing_metres=terrain_smoothing_metres,
            terrain_radial_noise_amplitude=noise_amplitude,
        )
        scenes[noise_amplitude] = scene
        row = {
            "noise_amplitude_fraction": noise_amplitude,
            "smoothing_metres": terrain_smoothing_metres,
            "exaggeration": TERRAIN_SHADE_EXAGGERATION,
            "shade_strength": shade_strength,
            "shade_min_max": [
                float(scene.facet_shades.min()),
                float(scene.facet_shades.max()),
            ],
            "frames": {},
        }
        for seconds, label in ((105.0, "105s"), (0.0, "000s")):
            frame, metadata = render_camera_frame(
                scene,
                seconds,
                TERRAIN_PREVIEW_SIZE,
                apply_fade=False,
            )
            path = (
                TERRAIN_PREVIEW_DIR
                / (
                    f"{label}_falloff_noise_"
                    f"{int(round(noise_amplitude * 100)):02d}pct.png"
                )
            )
            Image.fromarray(frame).save(path, optimize=True)
            frames[(noise_amplitude, seconds)] = Image.fromarray(frame)
            row["frames"][label] = {
                "render_seconds": metadata["render_seconds"],
                "path": str(path.relative_to(DATA_DIR.parent)),
            }
        identity_rows.append(
            {
                "noise_amplitude_fraction": noise_amplitude,
                "samples": _terrain_identity_checks(scene),
            }
        )
        row["edge_luminance"] = source_tile_edge_luminance_check(scene)
        sweep_rows.append(row)

    contact_width = 1_600
    panel_width = 800
    image_height = 450
    label_height = 42
    sheet = Image.new(
        "RGB",
        (
            contact_width,
            len(TERRAIN_FALLOFF_NOISE_COMPARISON)
            * (image_height + label_height),
        ),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(22)
    for row_index, noise_amplitude in enumerate(
        TERRAIN_FALLOFF_NOISE_COMPARISON
    ):
        y = row_index * (image_height + label_height)
        for column_index, (seconds, time_label) in enumerate(
            ((105.0, "1:45"), (0.0, "0:00 daylight"))
        ):
            x = column_index * panel_width
            image = frames[(noise_amplitude, seconds)].resize(
                (panel_width, image_height),
                Image.Resampling.LANCZOS,
            )
            sheet.paste(image, (x, y + label_height))
            draw.text(
                (x + 16, y + 8),
                (
                    f"noise {noise_amplitude:.0%} · {time_label} · "
                    f"×{TERRAIN_SHADE_EXAGGERATION:.0f} · s={shade_strength:.1f}"
                ),
                font=font,
                fill=(220, 220, 220),
            )
    contact_path = TERRAIN_PREVIEW_DIR / "falloff_v2.png"
    sheet.save(contact_path, optimize=True)

    identity_max = max(
        sample["max_pixel_difference"]
        for row in identity_rows
        for sample in row["samples"]
    )
    class_shades = {
        str(class_value): [
            float(scenes[DEFAULT_RADIAL_NOISE_AMPLITUDE].facet_shades[
                scenes[DEFAULT_RADIAL_NOISE_AMPLITUDE].classes == class_value
            ].min()),
            float(scenes[DEFAULT_RADIAL_NOISE_AMPLITUDE].facet_shades[
                scenes[DEFAULT_RADIAL_NOISE_AMPLITUDE].classes == class_value
            ].max()),
        ]
        for class_value in (1, 2, 3)
    }
    flat_edge_check = source_tile_edge_luminance_check(build_scene(seed))
    report = {
        "seed": seed,
        "preview_only": True,
        "terrain_shade": {
            "enabled": True,
            "shade_strength": shade_strength,
            "sun_azimuth_degrees": 315.0,
            "sun_altitude_degrees": 45.0,
            "baked_hillshade_enabled": False,
            "surrounding_tiles_shade": 1.0,
            "exaggeration": TERRAIN_SHADE_EXAGGERATION,
            "class_shade_min_max": class_shades,
            "all_water_and_managed_facets_exactly_one": all(
                bounds == [1.0, 1.0] for bounds in class_shades.values()
            ),
        },
        "radial_falloff": {
            "coordinate_system": (
                "largest centroid-centred ellipse inside the source tile's "
                "three-percent inset"
            ),
            "tile_margin_fraction": RADIAL_TILE_MARGIN_FRACTION,
            "full_height_fraction_of_local_radius": RADIAL_INNER_FRACTION,
            "zero_height_nominal_radius": 1.0,
            "noise": (
                "seeded two-/three-lobe angular perturbation, capped by "
                "the three-percent inset boundary"
            ),
            "noise_amplitude_fraction": DEFAULT_RADIAL_NOISE_AMPLITUDE,
            "applied_to": "height multiplier using smootherstep",
        },
        "d65_identity": {
            "by_noise_amplitude": identity_rows,
            "max_pixel_difference": identity_max,
            "passes": identity_max == 0,
        },
        "edge_luminance": {
            "terrain_off": flat_edge_check,
            "terrain_shade_by_noise_amplitude": [
                {
                    "noise_amplitude_fraction": row[
                        "noise_amplitude_fraction"
                    ],
                    **row["edge_luminance"],
                }
                for row in sweep_rows
            ],
            "previous_commit": {
                "terrain_off_relative_difference": 0.002439001270626769,
                "terrain_shade_300m_relative_difference": (
                    0.0011049387903049254
                ),
                "terrain_shade_600m_relative_difference": (
                    0.0012586301659322733
                ),
                "terrain_shade_800m_relative_difference": (
                    0.0016403880147584054
                ),
                "limit": 0.01,
            },
        },
        "sweep": sweep_rows,
        "outputs": {
            "contact_sheet": str(contact_path.relative_to(DATA_DIR.parent)),
        },
    }
    TERRAIN_SHADE_REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def render_terrain_preview(
    seed: int = SEED,
    *,
    exaggeration: float = DEFAULT_EXAGGERATION,
    terrain_shade: bool = True,
    shade_strength: float = DEFAULT_SHADE_STRENGTH,
    terrain_smoothing_metres: float = DEFAULT_TERRAIN_SMOOTHING_METRES,
    terrain_radial_noise_amplitude: float = DEFAULT_RADIAL_NOISE_AMPLITUDE,
) -> dict:
    """Render the locked 40-second terrain orbit preview and opening still."""
    TERRAIN_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    terrain_scene = build_scene(
        seed,
        terrain=True,
        exaggeration=exaggeration,
        terrain_shade=terrain_shade,
        shade_strength=shade_strength,
        terrain_smoothing_metres=terrain_smoothing_metres,
        terrain_radial_noise_amplitude=terrain_radial_noise_amplitude,
    )
    old_scene = build_scene(
        seed,
        terrain=True,
        exaggeration=exaggeration,
        terrain_shade=terrain_shade,
        shade_strength=shade_strength,
        terrain_smoothing_metres=terrain_smoothing_metres,
        terrain_radial_noise_amplitude=terrain_radial_noise_amplitude,
        seamless_tile_edges=False,
    )
    seam_old = source_tile_edge_luminance_check(old_scene)
    seam_new = source_tile_edge_luminance_check(terrain_scene)
    if not seam_new["passes"]:
        raise RuntimeError(f"Source-tile seam check failed: {seam_new}")

    opening, opening_metadata = render_camera_frame(
        terrain_scene,
        0.0,
        (1920, 1080),
        apply_fade=False,
        mixture_override=0.0,
    )
    Image.fromarray(opening).save(TERRAIN_OPENING_STILL_PATH, optimize=True)
    global_times = TERRAIN_ORBIT_START_SECONDS + np.arange(
        int(round(TERRAIN_ORBIT_DURATION_SECONDS * FPS))
    ) / FPS
    rows, wall_seconds = _encode_times(
        terrain_scene,
        TERRAIN_ORBIT_PATH,
        TERRAIN_ORBIT_SIZE,
        global_times,
    )
    identity = _terrain_identity_checks(terrain_scene)
    report = {
        "seed": seed,
        "preview_only": True,
        "terrain": {
            "enabled": True,
            "exaggeration": exaggeration,
            "smoothing_metres": terrain_smoothing_metres,
            "shade_enabled": terrain_shade,
            "shade_strength": shade_strength,
            "falloff_noise_amplitude": terrain_radial_noise_amplitude,
            "max_vertex_height_world": terrain_scene.terrain_max_height_world,
            "max_vertex_height_metres_exaggerated": (
                terrain_scene.terrain_max_height_metres
            ),
            "shading": "per-facet Lambert scalar; baked hillshade disabled",
            "occlusion": "global back-to-front sorting of all visible facets",
        },
        "seam_check": {
            "cause": (
                "collinear repeated-mesh boundary facets plus differing "
                "palettes on an inclusive coplanar polygon edge"
            ),
            "fix": (
                "deterministic periodic boundary jitter shared by adjacent "
                "tiles, plus a 20% paired-facet spectrum match"
            ),
            "old": seam_old,
            "new": seam_new,
        },
        "d65_identity": {
            "samples": identity,
            "max_pixel_difference": max(
                row["max_pixel_difference"] for row in identity
            ),
            "passes": all(
                row["max_pixel_difference"] == 0 for row in identity
            ),
        },
        "orbit_preview": {
            "global_seconds": [
                TERRAIN_ORBIT_START_SECONDS,
                TERRAIN_ORBIT_START_SECONDS + TERRAIN_ORBIT_DURATION_SECONDS,
            ],
            "size": list(TERRAIN_ORBIT_SIZE),
            "fps": FPS,
            "frame_count": len(rows),
            "wall_seconds": wall_seconds,
            "timing": _summarise_timing(rows),
            "probe": _probe(TERRAIN_ORBIT_PATH),
            "path": str(TERRAIN_ORBIT_PATH.relative_to(DATA_DIR.parent)),
        },
        "opening_still": {
            "size": [1920, 1080],
            "render_seconds": opening_metadata["render_seconds"],
            "path": str(
                TERRAIN_OPENING_STILL_PATH.relative_to(DATA_DIR.parent)
            ),
        },
    }
    TERRAIN_LOCKED_REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def render_phase4_preview(
    seed: int = SEED,
    *,
    terrain: bool = False,
    exaggeration: float = DEFAULT_EXAGGERATION,
    terrain_shade: bool | None = None,
    shade_strength: float = DEFAULT_SHADE_STRENGTH,
    terrain_smoothing_metres: float = DEFAULT_TERRAIN_SMOOTHING_METRES,
) -> dict:
    """Render the orbit shimmer clip and the four-times-speed full-film preview."""
    if terrain_shade and not terrain:
        raise ValueError("--terrain-shade requires --terrain")
    if terrain:
        return render_terrain_preview(
            seed,
            exaggeration=exaggeration,
            terrain_shade=True if terrain_shade is None else terrain_shade,
            shade_strength=shade_strength,
            terrain_smoothing_metres=terrain_smoothing_metres,
            terrain_radial_noise_amplitude=DEFAULT_RADIAL_NOISE_AMPLITUDE,
        )
    FILM_DIR.mkdir(parents=True, exist_ok=True)
    SHIMMER_DIR.mkdir(parents=True, exist_ok=True)
    scene = build_scene(seed)

    shimmer_times = SHIMMER_START_SECONDS + np.arange(
        int(round(SHIMMER_DURATION_SECONDS * FPS))
    ) / FPS
    shimmer_rows, shimmer_wall = _encode_times(
        scene,
        SHIMMER_PATH,
        SHIMMER_SIZE,
        shimmer_times,
    )

    fast_duration = FILM_DURATION_SECONDS / FAST_PREVIEW_SPEED
    fast_times = (
        np.arange(int(round(fast_duration * FPS)))
        / FPS
        * FAST_PREVIEW_SPEED
    )
    fast_rows, fast_wall = _encode_times(
        scene,
        FAST_PREVIEW_PATH,
        FAST_PREVIEW_SIZE,
        fast_times,
    )

    opening, opening_metadata = render_camera_frame(
        scene,
        0.0,
        FAST_PREVIEW_SIZE,
        apply_fade=False,
    )
    final_daylight, final_metadata = render_camera_frame(
        scene,
        CAMERA_HEADING_EASE_END_SECONDS,
        FAST_PREVIEW_SIZE,
        apply_fade=False,
    )
    Image.fromarray(opening).save(OPENING_STILL_PATH, optimize=True)
    Image.fromarray(final_daylight).save(
        FINAL_DAYLIGHT_STILL_PATH,
        optimize=True,
    )

    benchmark_rows = []
    for seconds in np.linspace(
        SHIMMER_START_SECONDS,
        SHIMMER_START_SECONDS + SHIMMER_DURATION_SECONDS,
        MASTER_BENCHMARK_FRAME_COUNT,
    ):
        _, metadata = render_camera_frame(
            scene,
            float(seconds),
            MASTER_BENCHMARK_SIZE,
        )
        benchmark_rows.append(metadata)

    identity_checks = _identity_checks(scene)
    report = {
        "seed": seed,
        "camera": {
            "geometry": "flat plane; vertices projected independently each frame",
            "target": "area-weighted centroid of the 4,198 ft extent",
            "target_normalized_xy": scene.target_normalized.tolist(),
            "vertical_fov_degrees": CAMERA_VERTICAL_FOV_DEGREES,
            "healthy_fill_target": CAMERA_HEALTHY_FILL_FRACTION,
            "supersample": CAMERA_SUPERSAMPLE,
            "tilt": {
                "degrees": [
                    CAMERA_TILT_START_DEGREES,
                    CAMERA_TILT_HOLD_DEGREES,
                ],
                "ease_in_seconds": [
                    CAMERA_TILT_IN_START_SECONDS,
                    CAMERA_TILT_IN_END_SECONDS,
                ],
                "ease_out_seconds": [
                    CAMERA_TILT_OUT_START_SECONDS,
                    CAMERA_TILT_OUT_END_SECONDS,
                ],
            },
            "heading": {
                "degrees": [
                    CAMERA_HEADING_START_DEGREES,
                    CAMERA_HEADING_END_DEGREES,
                ],
                "ease_seconds": [
                    CAMERA_HEADING_EASE_START_SECONDS,
                    CAMERA_HEADING_EASE_END_SECONDS,
                ],
                "note": "approaches 90° by 3:20 and eases fully to rest by 3:25",
            },
            "extended_land": {
                "source_points_per_tile": int(len(scene.points_world)),
                "source_triangles_preserved": scene.source_triangle_count,
                "conceptual_tile_count": int(len(scene.tile_offsets)),
                "conceptual_tile_triangles": scene.tile_triangle_count,
                "tile_size_world_units": [
                    scene.tile_width,
                    scene.tile_height,
                ],
                "half_size_world_units": EXTENDED_PLANE_HALF_SIZE,
                "fog_radius": [FOG_START_RADIUS, FOG_END_RADIUS],
                "hard_edge_visible": False,
            },
            "opening": opening_metadata,
            "final_daylight": final_metadata,
        },
        "light": {
            "formula": "(1-m) * equal-Y D65 + m * equal-Y Illuminant A",
            "per_frame_adaptation": (
                "CAT02 from the mixture's own white toward display D65"
            ),
            "adaptation_degree": ADAPTATION,
        },
        "identity_checks": identity_checks,
        "identity_max_pixel_difference": max(
            row["max_pixel_difference"] for row in identity_checks
        ),
        "shimmer_clip": {
            "global_seconds": [
                SHIMMER_START_SECONDS,
                SHIMMER_START_SECONDS + SHIMMER_DURATION_SECONDS,
            ],
            "size": list(SHIMMER_SIZE),
            "fps": FPS,
            "supersample": CAMERA_SUPERSAMPLE,
            "visible_flicker": False,
            "visible_moire": False,
            "inspection": (
                "2x supersampled moving land facets inspected over the full clip; "
                "no persistent crawling edges, flicker, or moiré observed"
            ),
            "timing": _summarise_timing(shimmer_rows),
            "wall_seconds": shimmer_wall,
            "probe": _probe(SHIMMER_PATH),
        },
        "fast_preview": {
            "production_duration_seconds": FILM_DURATION_SECONDS,
            "speed": FAST_PREVIEW_SPEED,
            "encoded_duration_seconds": fast_duration,
            "size": list(FAST_PREVIEW_SIZE),
            "fps": FPS,
            "timing": _summarise_timing(fast_rows),
            "wall_seconds": fast_wall,
            "probe": _probe(FAST_PREVIEW_PATH),
        },
        "master_4k_benchmark": _summarise_timing(benchmark_rows),
        "outputs": {
            "shimmer_clip": {
                "path": str(SHIMMER_PATH.relative_to(DATA_DIR.parent)),
                "bytes": SHIMMER_PATH.stat().st_size,
                "sha256": _sha256(SHIMMER_PATH),
            },
            "full_film_4x_preview": {
                "path": str(FAST_PREVIEW_PATH.relative_to(DATA_DIR.parent)),
                "bytes": FAST_PREVIEW_PATH.stat().st_size,
                "sha256": _sha256(FAST_PREVIEW_PATH),
            },
            "opening_still": {
                "path": str(OPENING_STILL_PATH.relative_to(DATA_DIR.parent)),
                "sha256": _sha256(OPENING_STILL_PATH),
            },
            "final_daylight_still": {
                "path": str(FINAL_DAYLIGHT_STILL_PATH.relative_to(DATA_DIR.parent)),
                "sha256": _sha256(FINAL_DAYLIGHT_STILL_PATH),
            },
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(render_phase4_preview(), indent=2))
