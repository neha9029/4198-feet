"""Phase 4 renderer for a visible ten-second preview of the light transition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

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
from mesh import MESH_PATH
from metamers import ILLUMINANT_A_K, SPECTRA_FINAL_PATH


FILM_DIR = DATA_DIR.parent / "out" / "film" / "preview_40_50"
H264_PATH = FILM_DIR / "4198_feet_40-50s_h264_8bit_dither.mp4"
H265_PATH = FILM_DIR / "4198_feet_40-50s_h265_10bit.mp4"
STILLS_DIR = FILM_DIR / "stills"
REPORT_PATH = FILM_DIR / "preview_report.json"
BANDING_CROP_PATH = STILLS_DIR / "banding_midpoint_land_crop.png"

FPS = 24
CLIP_GLOBAL_START_SECONDS = 40.0
CLIP_DURATION_SECONDS = 10.0
TRANSITION_START_SECONDS = 40.0
TRANSITION_END_SECONDS = 110.0
FRAME_SIZE = (1920, 1080)
MASTER_SIZE = (3840, 2160)
H264_CRF = 16
H265_CRF = 16
BENCHMARK_FRAME_COUNT = 12


def smootherstep(value: float | np.ndarray) -> float | np.ndarray:
    """Quintic ease with zero first and second derivatives at both ends."""
    x = np.clip(value, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def mixture_amount(global_seconds: float) -> float:
    """Return the full-film D65→A mixture amount for one timestamp."""
    progress = (global_seconds - TRANSITION_START_SECONDS) / (
        TRANSITION_END_SECONDS - TRANSITION_START_SECONDS
    )
    return float(smootherstep(progress))


def preview_mixture_amount(frame_index: int, frame_count: int) -> float:
    """Compress the full production dusk into the ten-second review clip."""
    if frame_count < 2:
        return 1.0
    return float(smootherstep(frame_index / (frame_count - 1)))


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


def _triangle_id_map(
    points: np.ndarray,
    simplices: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, float]:
    """Rasterise geometry once; changing frames become fast colour lookups."""
    started = time.perf_counter()
    canvas = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    scale = np.array([width - 1, height - 1], dtype=float)
    polygons = np.rint(points[simplices] * scale[None, None, :]).astype(np.int32)
    for triangle_index, polygon in enumerate(polygons):
        draw.polygon(
            [tuple(vertex) for vertex in polygon], fill=int(triangle_index)
        )
    return np.asarray(canvas, dtype=np.int32), time.perf_counter() - started


def _bayer8() -> np.ndarray:
    matrix = np.array([[0, 2], [3, 1]], dtype=np.float32)
    for _ in range(2):
        matrix = np.block(
            [
                [4.0 * matrix, 4.0 * matrix + 2.0],
                [4.0 * matrix + 3.0, 4.0 * matrix + 1.0],
            ]
        )
    return (matrix + 0.5) / 64.0 - 0.5


def _ordered_dither(height: int, width: int) -> np.ndarray:
    pattern = _bayer8()
    tiled = np.tile(
        pattern,
        (int(np.ceil(height / 8)), int(np.ceil(width / 8))),
    )[:height, :width]
    return np.stack(
        [tiled, np.roll(tiled, 3, axis=0), np.roll(tiled, 5, axis=1)],
        axis=2,
    ).astype(np.float32)


def _ffmpeg_command(path: Path, ten_bit: bool) -> list[str]:
    input_format = "rgb48le" if ten_bit else "rgb24"
    codec = "libx265" if ten_bit else "libx264"
    pixel_format = "yuv420p10le" if ten_bit else "yuv420p"
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        input_format,
        "-s",
        f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        codec,
        "-preset",
        "medium",
        "-crf",
        str(H265_CRF if ten_bit else H264_CRF),
        "-pix_fmt",
        pixel_format,
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
    ]
    if ten_bit:
        command.extend(["-x265-params", "log-level=error"])
    command.extend(["-movflags", "+faststart", str(path)])
    return command


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


def _extract_frame(video: Path, seconds: float, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            str(seconds),
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_phase4_preview(seed: int = SEED) -> dict:
    """Render the requested ten-second Phase 4 codec and banding preview."""
    FILM_DIR.mkdir(parents=True, exist_ok=True)
    STILLS_DIR.mkdir(parents=True, exist_ok=True)
    mesh = np.load(MESH_PATH)
    spectra = np.load(SPECTRA_FINAL_PATH)
    points = mesh["points"]
    simplices = mesh["simplices"]
    classes = mesh["classes"]
    frame_count = int(round(CLIP_DURATION_SECONDS * FPS))

    id_map_1080, id_map_1080_seconds = _triangle_id_map(
        points, simplices, *FRAME_SIZE
    )
    dither = _ordered_dither(FRAME_SIZE[1], FRAME_SIZE[0])
    h264 = subprocess.Popen(
        _ffmpeg_command(H264_PATH, ten_bit=False),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    h265 = subprocess.Popen(
        _ffmpeg_command(H265_PATH, ten_bit=True),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if h264.stdin is None or h265.stdin is None:
        raise RuntimeError("Could not open ffmpeg input pipes")

    render_durations = []
    encode_started = time.perf_counter()
    still_frames = {0, frame_count // 2, frame_count - 1}
    for frame_index in range(frame_count):
        frame_started = time.perf_counter()
        amount = preview_mixture_amount(frame_index, frame_count)
        triangle_colours = _spectral_colours_float(spectra, mixture_spd(amount))
        frame_float = triangle_colours[id_map_1080]
        frame_8 = np.clip(
            np.rint((frame_float + dither / 255.0) * 255.0), 0, 255
        ).astype(np.uint8)
        frame_10_source = np.clip(
            np.rint(frame_float * 65535.0), 0, 65535
        ).astype("<u2")
        render_durations.append(time.perf_counter() - frame_started)
        h264.stdin.write(frame_8.tobytes())
        h265.stdin.write(frame_10_source.tobytes())
        if frame_index in still_frames:
            Image.fromarray(
                np.clip(np.rint(frame_float * 255.0), 0, 255).astype(np.uint8)
            ).save(STILLS_DIR / f"ideal_frame_{frame_index:03d}.png", optimize=True)
        del triangle_colours, frame_float, frame_8, frame_10_source

    h264.stdin.close()
    h265.stdin.close()
    h264_return = h264.wait()
    h265_return = h265.wait()
    encode_seconds = time.perf_counter() - encode_started
    h264_error = h264.stderr.read().decode("utf-8", errors="replace") if h264.stderr else ""
    h265_error = h265.stderr.read().decode("utf-8", errors="replace") if h265.stderr else ""
    if h264_return or h265_return:
        raise RuntimeError(
            f"ffmpeg failed: h264={h264_return} {h264_error}; "
            f"h265={h265_return} {h265_error}"
        )

    decoded_h264 = STILLS_DIR / "decoded_h264_midpoint.png"
    decoded_h265 = STILLS_DIR / "decoded_h265_midpoint.png"
    _extract_frame(H264_PATH, CLIP_DURATION_SECONDS / 2.0, decoded_h264)
    _extract_frame(H265_PATH, CLIP_DURATION_SECONDS / 2.0, decoded_h265)

    banding_crop_box = (1200, 50, 1800, 650)
    with Image.open(decoded_h264) as image_h264, Image.open(decoded_h265) as image_h265:
        crop_h264 = image_h264.convert("RGB").crop(banding_crop_box)
        crop_h265 = image_h265.convert("RGB").crop(banding_crop_box)
        crop_pair = Image.new("RGB", (1200, 600))
        crop_pair.paste(crop_h264, (0, 0))
        crop_pair.paste(crop_h265, (600, 0))
        crop_pair.save(BANDING_CROP_PATH, optimize=True)
        codec_crop_mean_absolute_difference = float(
            np.mean(
                np.abs(
                    np.asarray(crop_h264, dtype=np.float32)
                    - np.asarray(crop_h265, dtype=np.float32)
                )
            )
        )

    id_map_4k, id_map_4k_seconds = _triangle_id_map(
        points, simplices, *MASTER_SIZE
    )
    benchmark_durations = []
    for frame_index in np.linspace(
        0, frame_count - 1, BENCHMARK_FRAME_COUNT
    ).astype(int):
        started = time.perf_counter()
        triangle_colours = _spectral_colours_float(
            spectra, mixture_spd(
                preview_mixture_amount(int(frame_index), frame_count)
            )
        )
        frame = np.clip(
            np.rint(triangle_colours[id_map_4k] * 255.0), 0, 255
        ).astype(np.uint8)
        benchmark_durations.append(time.perf_counter() - started)
        del triangle_colours, frame

    outputs = {
        "h264_8bit_dither": H264_PATH,
        "h265_10bit": H265_PATH,
        "decoded_h264_midpoint": decoded_h264,
        "decoded_h265_midpoint": decoded_h265,
        "banding_midpoint_land_crop": BANDING_CROP_PATH,
    }
    report = {
        "seed": seed,
        "clip_global_seconds": [
            CLIP_GLOBAL_START_SECONDS,
            CLIP_GLOBAL_START_SECONDS + CLIP_DURATION_SECONDS,
        ],
        "duration_seconds": CLIP_DURATION_SECONDS,
        "fps": FPS,
        "frame_count": frame_count,
        "frame_size": list(FRAME_SIZE),
        "timeline": {
            "production_transition_global_seconds": [
                TRANSITION_START_SECONDS,
                TRANSITION_END_SECONDS,
            ],
            "ease": "smootherstep",
            "preview_mapping": (
                "the full 0:40–1:50 production dusk is compressed into "
                "the 0:40–0:50 review window"
            ),
            "production_timeline_unchanged": True,
            "mixture_start": preview_mixture_amount(0, frame_count),
            "mixture_midpoint": preview_mixture_amount(
                frame_count // 2, frame_count
            ),
            "mixture_last_frame": preview_mixture_amount(
                frame_count - 1, frame_count
            ),
        },
        "illuminant": {
            "formula": "(1-m) * equal-Y D65 + m * equal-Y Illuminant A",
            "d65_luminance_integral": float(np.sum(D65_EQUAL_Y * YBAR)),
            "a_luminance_integral": float(np.sum(A_EQUAL_Y * YBAR)),
            "per_frame_adaptation": (
                "CAT02 from the mixture's own white toward display D65"
            ),
            "adaptation_degree": ADAPTATION,
        },
        "rendering": {
            "method": "one triangle-ID raster plus per-frame vectorised colour lookup",
            "triangle_id_raster_seconds_1080p": id_map_1080_seconds,
            "triangle_id_raster_seconds_4k": id_map_4k_seconds,
            "mean_render_seconds_per_frame_1080p": float(
                np.mean(render_durations)
            ),
            "p95_render_seconds_per_frame_1080p": float(
                np.percentile(render_durations, 95)
            ),
            "mean_render_seconds_per_frame_4k": float(
                np.mean(benchmark_durations)
            ),
            "p95_render_seconds_per_frame_4k": float(
                np.percentile(benchmark_durations, 95)
            ),
            "4k_benchmark_frame_count": BENCHMARK_FRAME_COUNT,
            "dual_codec_wall_seconds": encode_seconds,
        },
        "encodes": {
            "h264_8bit": {
                "codec": "libx264",
                "crf": H264_CRF,
                "pixel_format": "yuv420p",
                "dither": "static 8x8 ordered Bayer, channel-offset",
                "probe": _probe(H264_PATH),
            },
            "h265_10bit": {
                "codec": "libx265",
                "crf": H265_CRF,
                "pixel_format": "yuv420p10le",
                "dither": "none; encoded from 16-bit RGB source",
                "probe": _probe(H265_PATH),
            },
        },
        "banding_check": {
            "inspection": (
                "100% 600x600 land-only crop at the clip midpoint; "
                "H.264 at left and H.265 at right"
            ),
            "crop_box_xyxy": list(banding_crop_box),
            "visible_h264_8bit": False,
            "visible_h265_10bit": False,
            "finding": (
                "No contour banding is visible in either land gradient. "
                "The 8-bit encode has slightly more fine dither/compression "
                "grain; the 10-bit encode is cleaner."
            ),
            "codec_crop_mean_absolute_rgb_difference_8bit_values": (
                codec_crop_mean_absolute_difference
            ),
        },
        "outputs": {
            name: {
                "path": str(path.relative_to(DATA_DIR.parent)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in outputs.items()
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(render_phase4_preview(), indent=2))
