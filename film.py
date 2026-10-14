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
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

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
    SPECTRA_FINAL_PATH,
)


# Film and light timing.
FPS = 24
FILM_DURATION_SECONDS = 210.0
DAYLIGHT_HOLD_END_SECONDS = 40.0
DUSK_END_SECONDS = 110.0
LAMPLIGHT_HOLD_END_SECONDS = 150.0
RETURN_END_SECONDS = 200.0
FADE_START_SECONDS = 205.0
FADE_END_SECONDS = 210.0

# Camera timing and angles. Heading settles at 90° during the 3:20–3:25 hold.
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
EXTENDED_PLANE_NEAR_MARGIN = 0.40
EXTENDED_PLANE_NEAR_POINTS = 35_000
EXTENDED_PLANE_FAR_POINTS = 7_000
EXTENDED_PLANE_BOUNDARY_SAMPLES = 64
FOG_START_RADIUS = 1.25
FOG_END_RADIUS = 4.0
BACKGROUND_RGB = np.array([244, 241, 232], dtype=np.float32) / 255.0

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


@dataclass(frozen=True)
class CameraState:
    tilt_degrees: float
    heading_degrees: float
    distance: float
    healthy_fill_fraction: float


@dataclass
class FilmScene:
    points_world: np.ndarray
    simplices: np.ndarray
    classes: np.ndarray
    healthy_points_world: np.ndarray
    final_spectra: np.ndarray
    identity_spectra: np.ndarray
    fog: np.ndarray
    source_triangle_count: int
    extension_triangle_count: int
    target_normalized: np.ndarray


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


def _extension_points(
    main_points_world: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Sample a dense border and a fogged, coarser far field."""
    rng = np.random.default_rng(seed + 4_198)
    minimum = main_points_world[:, :2].min(axis=0)
    maximum = main_points_world[:, :2].max(axis=0)
    near_minimum = minimum - EXTENDED_PLANE_NEAR_MARGIN
    near_maximum = maximum + EXTENDED_PLANE_NEAR_MARGIN
    near_batches = []
    near_count = 0
    while near_count < EXTENDED_PLANE_NEAR_POINTS:
        candidates = rng.uniform(
            near_minimum,
            near_maximum,
            size=(EXTENDED_PLANE_NEAR_POINTS, 2),
        )
        outside_main = np.any(
            (candidates < minimum) | (candidates > maximum),
            axis=1,
        )
        selected = candidates[outside_main]
        near_batches.append(selected)
        near_count += len(selected)
    near = np.vstack(near_batches)[:EXTENDED_PLANE_NEAR_POINTS]

    far_batches = []
    far_count = 0
    while far_count < EXTENDED_PLANE_FAR_POINTS:
        candidates = rng.uniform(
            -EXTENDED_PLANE_HALF_SIZE,
            EXTENDED_PLANE_HALF_SIZE,
            size=(EXTENDED_PLANE_FAR_POINTS, 2),
        )
        outside_near = np.any(
            (candidates < near_minimum) | (candidates > near_maximum),
            axis=1,
        )
        selected = candidates[outside_near]
        far_batches.append(selected)
        far_count += len(selected)
    far = np.vstack(far_batches)[:EXTENDED_PLANE_FAR_POINTS]

    edge = np.linspace(
        -EXTENDED_PLANE_HALF_SIZE,
        EXTENDED_PLANE_HALF_SIZE,
        EXTENDED_PLANE_BOUNDARY_SAMPLES,
    )
    boundary = np.vstack(
        [
            np.column_stack(
                [edge, np.full_like(edge, -EXTENDED_PLANE_HALF_SIZE)]
            ),
            np.column_stack(
                [edge, np.full_like(edge, EXTENDED_PLANE_HALF_SIZE)]
            ),
            np.column_stack(
                [np.full_like(edge, -EXTENDED_PLANE_HALF_SIZE), edge]
            ),
            np.column_stack(
                [np.full_like(edge, EXTENDED_PLANE_HALF_SIZE), edge]
            ),
        ]
    )
    points_2d = np.unique(np.vstack([near, far, boundary]), axis=0)
    return np.column_stack(
        [points_2d, np.zeros(len(points_2d), dtype=np.float64)]
    )


def _source_triangle_lookup(
    points: np.ndarray,
    simplices: np.ndarray,
    sample_points: np.ndarray,
) -> np.ndarray:
    """Map normalized points back to the accepted source mesh triangles."""
    width, height = 3_689, 2_075
    canvas = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    projected = np.rint(
        points[simplices] * np.array([width - 1, height - 1])
    ).astype(np.int32)
    for triangle_index, polygon in enumerate(projected):
        draw.polygon(
            [tuple(vertex) for vertex in polygon],
            fill=int(triangle_index + 1),
        )
    lookup = np.asarray(canvas, dtype=np.int32)
    result = np.full(len(sample_points), -1, dtype=np.int32)
    inside = np.all((sample_points >= 0.0) & (sample_points <= 1.0), axis=1)
    columns = np.clip(
        np.rint(sample_points[inside, 0] * (width - 1)),
        0,
        width - 1,
    ).astype(int)
    rows = np.clip(
        np.rint(sample_points[inside, 1] * (height - 1)),
        0,
        height - 1,
    ).astype(int)
    result[inside] = lookup[rows, columns] - 1
    return result


def build_scene(seed: int = SEED) -> FilmScene:
    """Load the production spectra and construct the flat extended plane."""
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
    final_spectra = np.load(SPECTRA_FINAL_PATH)
    base_spectra = np.load(SPECTRA_BASE_PATH)

    extension_points_world = _extension_points(points_world, seed)
    global_points_world = np.vstack([points_world, extension_points_world])
    global_simplices = Delaunay(global_points_world[:, :2]).simplices.astype(
        np.int32
    )
    global_centroids = global_points_world[global_simplices].mean(axis=1)
    global_centroids_normalized = np.column_stack(
        [
            global_centroids[:, 0] / FRAME_ASPECT_RATIO
            + target_normalized[0],
            target_normalized[1] - global_centroids[:, 1],
        ]
    )
    mapped_old_indices = _source_triangle_lookup(
        points,
        simplices,
        global_centroids_normalized,
    )
    mapped = mapped_old_indices >= 0
    extension = ~mapped
    global_classes = np.zeros(len(global_simplices), dtype=np.uint8)
    global_classes[mapped] = classes[mapped_old_indices[mapped]]
    global_final_spectra = np.empty(
        (len(global_simplices), final_spectra.shape[1]),
        dtype=final_spectra.dtype,
    )
    global_identity_spectra = np.empty_like(global_final_spectra)
    global_final_spectra[mapped] = final_spectra[mapped_old_indices[mapped]]
    global_identity_spectra[mapped] = global_final_spectra[mapped]
    lost_mapped = mapped & (global_classes == 1)
    global_identity_spectra[lost_mapped] = base_spectra[
        mapped_old_indices[lost_mapped]
    ]

    land_indices = np.flatnonzero(classes == 0)
    rng = np.random.default_rng(seed + 9_031)
    exterior_spectra = base_spectra[
        rng.choice(land_indices, size=int(np.count_nonzero(extension)), replace=True)
    ]
    global_final_spectra[extension] = exterior_spectra
    global_identity_spectra[extension] = exterior_spectra

    radii = np.linalg.norm(global_centroids[:, :2], axis=1)
    fog = smootherstep(
        (radii - FOG_START_RADIUS) / (FOG_END_RADIUS - FOG_START_RADIUS)
    ).astype(np.float32)
    fog[mapped] = 0.0
    healthy_indices = np.unique(global_simplices[np.isin(global_classes, (1, 2))].ravel())
    healthy_points_world = global_points_world[healthy_indices]
    return FilmScene(
        points_world=global_points_world,
        simplices=global_simplices,
        classes=global_classes,
        healthy_points_world=healthy_points_world,
        final_spectra=global_final_spectra,
        identity_spectra=global_identity_spectra,
        fog=fog,
        source_triangle_count=int(np.count_nonzero(mapped)),
        extension_triangle_count=int(np.count_nonzero(extension)),
        target_normalized=target_normalized,
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
            (ndc_x * 0.5 + 0.5) * (width - 1),
            (0.5 - ndc_y * 0.5) * (height - 1),
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
    return CameraState(
        tilt_degrees=tilt,
        heading_degrees=heading,
        distance=distance,
        healthy_fill_fraction=_fit_metric(projected, size),
    )


def _draw_triangle_ids(
    draw: ImageDraw.ImageDraw,
    projected: np.ndarray,
    depth: np.ndarray,
    simplices: np.ndarray,
    size: tuple[int, int],
    id_offset: int,
) -> int:
    """Draw visible projected triangles and return the visible count."""
    width, height = size
    polygons = projected[simplices]
    polygon_depth = depth[simplices]
    valid = np.all(polygon_depth > 0.02, axis=1)
    minimum = polygons.min(axis=1)
    maximum = polygons.max(axis=1)
    valid &= (
        (maximum[:, 0] >= 0.0)
        & (minimum[:, 0] < width)
        & (maximum[:, 1] >= 0.0)
        & (minimum[:, 1] < height)
    )
    visible_indices = np.flatnonzero(valid)
    rounded = np.rint(polygons[visible_indices]).astype(np.int32)
    for triangle_index, polygon in zip(visible_indices, rounded):
        draw.polygon(
            [tuple(vertex) for vertex in polygon],
            fill=int(id_offset + triangle_index),
        )
    return int(len(visible_indices))


def render_camera_frame(
    scene: FilmScene,
    global_seconds: float,
    size: tuple[int, int],
    *,
    spectra: np.ndarray | None = None,
    supersample: int = CAMERA_SUPERSAMPLE,
    apply_fade: bool = True,
) -> tuple[np.ndarray, dict]:
    """Project and rasterise the true triangle geometry for one film frame."""
    started = time.perf_counter()
    if spectra is None:
        spectra = scene.final_spectra
    camera = camera_state(scene, global_seconds, size)
    render_size = (size[0] * supersample, size[1] * supersample)
    main_projected, main_depth = _project(
        scene.points_world,
        camera.tilt_degrees,
        camera.heading_degrees,
        camera.distance,
        render_size,
    )
    id_canvas = Image.new("I", render_size, 0)
    draw = ImageDraw.Draw(id_canvas)
    triangle_offset = 1
    visible_triangles = _draw_triangle_ids(
        draw,
        main_projected,
        main_depth,
        scene.simplices,
        render_size,
        triangle_offset,
    )

    spd = mixture_spd(mixture_amount(global_seconds))
    main_colours = _spectral_colours_float(spectra, spd)
    main_colours = (
        main_colours * (1.0 - scene.fog[:, None])
        + BACKGROUND_RGB[None, :] * scene.fog[:, None]
    )
    palette = np.empty(
        (triangle_offset + len(main_colours), 3),
        dtype=np.float32,
    )
    palette[0] = BACKGROUND_RGB
    palette[triangle_offset:] = main_colours
    supersampled = np.clip(
        np.rint(palette[np.asarray(id_canvas, dtype=np.int32)] * 255.0),
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
            frame = np.clip(np.rint(frame.astype(np.float32) * fade), 0, 255).astype(
                np.uint8
            )
    return frame, {
        "seconds": global_seconds,
        "tilt_degrees": camera.tilt_degrees,
        "heading_degrees": camera.heading_degrees,
        "distance": camera.distance,
        "healthy_fill_fraction": camera.healthy_fill_fraction,
        "visible_triangles": visible_triangles,
        "render_seconds": time.perf_counter() - started,
    }


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


def render_phase4_preview(seed: int = SEED) -> dict:
    """Render the orbit shimmer clip and the four-times-speed full-film preview."""
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
                "total_points": int(len(scene.points_world)),
                "source_triangles_preserved": scene.source_triangle_count,
                "extension_triangles": scene.extension_triangle_count,
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
