"""Raster preview rendering for the Phase 2 tessellation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data import DATA_DIR
from engine import ADAPTATION, D65_SPD, adapt, illuminant, spectra_to_xyz, xyz_to_srgb
from mesh import MESH_PATH
from metamers import (
    ILLUMINANT_A_K,
    RECOMMENDED_WATER_VARIANT,
    SPECTRA_BASE_PATH,
    SPECTRA_FINAL_PATH,
    WATER_VARIANTS,
    build_water_spectra,
)


PHASE2_DIR = DATA_DIR.parent / "out" / "phase2"
PREVIEW_D65_PATH = PHASE2_DIR / "preview_d65.png"
PREVIEW_A_PATH = PHASE2_DIR / "preview_2856k.png"
PREVIEW_MIDPOINT_PATH = PHASE2_DIR / "preview_midpoint.png"
PREVIEW_PAIR_PATH = PHASE2_DIR / "preview_pair.png"
PREVIEW_TRIPTYCH_PATH = PHASE2_DIR / "preview_triptych.png"
WATER_VARIANT_COMPARISON_PATH = PHASE2_DIR / "water_variants_2856k.png"
D65_BASELINE_PATH = PHASE2_DIR / "d65_baseline.png"
RENDER_REPORT_PATH = PHASE2_DIR / "render_report.json"


def spectral_colours(spectra: np.ndarray, light: str | int) -> np.ndarray:
    """Render spectra to adapted 8-bit sRGB under the requested illuminant."""
    spd = D65_SPD if light == "D65" else illuminant(int(light))
    d65_white = spectra_to_xyz(np.ones((1, spectra.shape[1])), D65_SPD)[0]
    source_white = spectra_to_xyz(np.ones((1, spectra.shape[1])), spd)[0]
    xyz = spectra_to_xyz(spectra, spd)
    if light != "D65":
        xyz = adapt(xyz, source_white, d65_white, ADAPTATION)
    return xyz_to_srgb(xyz)


def spectral_colours_for_spd(spectra: np.ndarray, spd: np.ndarray) -> np.ndarray:
    """Render spectra beneath an arbitrary SPD, adapted toward the D65 white."""
    d65_white = spectra_to_xyz(np.ones((1, spectra.shape[1])), D65_SPD)[0]
    source_white = spectra_to_xyz(np.ones((1, spectra.shape[1])), spd)[0]
    xyz = spectra_to_xyz(spectra, spd)
    return xyz_to_srgb(adapt(xyz, source_white, d65_white, ADAPTATION))


def render_mesh(
    points: np.ndarray,
    simplices: np.ndarray,
    classes: np.ndarray,
    colours: np.ndarray,
    raster_shape: tuple[int, int],
    long_side: int = 1600,
    supersample: int = 2,
) -> Image.Image:
    """Draw the fixed triangular geometry; no-data alone is left unpainted."""
    raster_h, raster_w = raster_shape
    if raster_h >= raster_w:
        height = long_side
        width = int(round(long_side * raster_w / raster_h))
    else:
        width = long_side
        height = int(round(long_side * raster_h / raster_w))
    canvas = Image.new(
        "RGB", (width * supersample, height * supersample), (244, 241, 232)
    )
    draw = ImageDraw.Draw(canvas)
    scale = np.array(
        [(width - 1) * supersample, (height - 1) * supersample], dtype=float
    )
    polygons = np.rint(points[simplices] * scale[None, None, :]).astype(int)
    for polygon, colour, triangle_class in zip(polygons, colours, classes):
        if triangle_class == 255:
            continue
        fill = tuple(int(channel) for channel in colour)
        draw.polygon([tuple(vertex) for vertex in polygon], fill=fill)
    return canvas.resize((width, height), Image.Resampling.LANCZOS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median_lstar(spectra: np.ndarray) -> float:
    white_y = spectra_to_xyz(
        np.ones((1, spectra.shape[1])), D65_SPD
    )[0, 1]
    y_ratio = spectra_to_xyz(spectra, D65_SPD)[:, 1] / white_y
    delta = 6.0 / 29.0
    lightness = np.where(
        y_ratio > delta**3,
        116.0 * np.cbrt(y_ratio) - 16.0,
        y_ratio * (29.0 / 3.0) ** 3,
    )
    return float(np.median(lightness))


def render_previews(long_side: int = 1600) -> dict:
    """Write D65, midpoint, 2856 K, comparison, and identity-test images."""
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    mesh = np.load(MESH_PATH)
    points = mesh["points"]
    simplices = mesh["simplices"]
    classes = mesh["classes"]
    raster_shape = tuple(int(value) for value in mesh["raster_shape"])
    base = np.load(SPECTRA_BASE_PATH)
    final = np.load(SPECTRA_FINAL_PATH)

    base_d65_colours = spectral_colours(base, "D65")
    final_d65_colours = spectral_colours(final, "D65")
    final_a_colours = spectral_colours(final, ILLUMINANT_A_K)
    midpoint_spd = 0.5 * D65_SPD + 0.5 * illuminant(ILLUMINANT_A_K)
    final_midpoint_colours = spectral_colours_for_spd(final, midpoint_spd)
    baseline = render_mesh(
        points, simplices, classes, base_d65_colours, raster_shape, long_side
    )
    d65 = render_mesh(
        points, simplices, classes, final_d65_colours, raster_shape, long_side
    )
    tungsten = render_mesh(
        points, simplices, classes, final_a_colours, raster_shape, long_side
    )
    midpoint = render_mesh(
        points, simplices, classes, final_midpoint_colours, raster_shape, long_side
    )
    baseline.save(D65_BASELINE_PATH, optimize=True)
    d65.save(PREVIEW_D65_PATH, optimize=True)
    tungsten.save(PREVIEW_A_PATH, optimize=True)
    midpoint.save(PREVIEW_MIDPOINT_PATH, optimize=True)

    gap = 14
    pair_width = 1600
    panel_width = (pair_width - gap) // 2
    panel_height = int(round(d65.height * panel_width / d65.width))
    left = d65.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    right = tungsten.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    pair = Image.new("RGB", (pair_width, panel_height), (244, 241, 232))
    pair.paste(left, (0, 0))
    pair.paste(right, (panel_width + gap, 0))
    label_draw = ImageDraw.Draw(pair)
    try:
        label_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 25
        )
    except OSError:
        label_font = ImageFont.load_default()
    for x, label in ((22, "D65"), (panel_width + gap + 22, "2856 K")):
        box = label_draw.textbbox((x, 18), label, font=label_font)
        label_draw.rounded_rectangle(
            (box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5),
            radius=5,
            fill=(244, 241, 232),
        )
        label_draw.text((x, 18), label, font=label_font, fill=(41, 48, 43))
    pair.save(PREVIEW_PAIR_PATH, optimize=True)

    triptych_width = 1600
    triptych_gap = 12
    triptych_panel_width = (triptych_width - 2 * triptych_gap) // 3
    triptych_panel_height = int(
        round(d65.height * triptych_panel_width / d65.width)
    )
    panels = [
        image.resize(
            (triptych_panel_width, triptych_panel_height), Image.Resampling.LANCZOS
        )
        for image in (d65, midpoint, tungsten)
    ]
    triptych = Image.new(
        "RGB", (triptych_width, triptych_panel_height), (244, 241, 232)
    )
    for panel_index, panel in enumerate(panels):
        triptych.paste(
            panel,
            (panel_index * (triptych_panel_width + triptych_gap), 0),
        )
    triptych_draw = ImageDraw.Draw(triptych)
    for panel_index, label in enumerate(("D65", "50% D65 / A", "2856 K")):
        x = panel_index * (triptych_panel_width + triptych_gap) + 18
        box = triptych_draw.textbbox((x, 16), label, font=label_font)
        triptych_draw.rounded_rectangle(
            (box[0] - 7, box[1] - 4, box[2] + 7, box[3] + 4),
            radius=5,
            fill=(244, 241, 232),
        )
        triptych_draw.text((x, 16), label, font=label_font, fill=(41, 48, 43))
    triptych.save(PREVIEW_TRIPTYCH_PATH, optimize=True)

    water_variant_images: dict[str, Image.Image] = {}
    water_variant_lstar: dict[str, dict[str, float]] = {}
    for variant in WATER_VARIANTS:
        variant_spectra = final.copy()
        water_spectra = build_water_spectra(
            mesh["centroids"], mesh["arms"], variant
        )
        variant_spectra[classes == 2] = water_spectra[classes == 2]
        variant_colours = spectral_colours(variant_spectra, ILLUMINANT_A_K)
        water_variant_images[variant] = render_mesh(
            points,
            simplices,
            classes,
            variant_colours,
            raster_shape,
            long_side,
        )
        water_variant_lstar[variant] = {
            "north_median": _median_lstar(
                water_spectra[(classes == 2) & (mesh["arms"] == 1)]
            ),
            "south_median": _median_lstar(
                water_spectra[(classes == 2) & (mesh["arms"] == 0)]
            ),
        }

    comparison_width = 1600
    comparison_gap = 12
    comparison_panel_width = (comparison_width - 2 * comparison_gap) // 3
    comparison_panel_height = int(
        round(d65.height * comparison_panel_width / d65.width)
    )
    comparison = Image.new(
        "RGB", (comparison_width, comparison_panel_height), (244, 241, 232)
    )
    comparison_draw = ImageDraw.Draw(comparison)
    for panel_index, variant in enumerate(WATER_VARIANTS):
        panel = water_variant_images[variant].resize(
            (comparison_panel_width, comparison_panel_height),
            Image.Resampling.LANCZOS,
        )
        x0 = panel_index * (comparison_panel_width + comparison_gap)
        comparison.paste(panel, (x0, 0))
        target = WATER_VARIANTS[variant]
        label = f"{variant} — dark" if target is None else f"{variant} — L* {target:.0f}"
        x = x0 + 18
        box = comparison_draw.textbbox((x, 16), label, font=label_font)
        comparison_draw.rounded_rectangle(
            (box[0] - 7, box[1] - 4, box[2] + 7, box[3] + 4),
            radius=5,
            fill=(244, 241, 232),
        )
        comparison_draw.text((x, 16), label, font=label_font, fill=(41, 48, 43))
    comparison.save(WATER_VARIANT_COMPARISON_PATH, optimize=True)

    baseline_array = np.asarray(baseline, dtype=np.int16)
    d65_array = np.asarray(d65, dtype=np.int16)
    aspect_ratio = d65.width / d65.height
    rotated_aspect_ratio = 1.0 / aspect_ratio
    target_aspect_ratio = 16.0 / 9.0
    report = {
        "long_side_px": long_side,
        "individual_size_px": [d65.width, d65.height],
        "pair_size_px": [pair.width, pair.height],
        "triptych_size_px": [triptych.width, triptych.height],
        "water_variant_comparison_size_px": [
            comparison.width,
            comparison.height,
        ],
        "d65_max_abs_pixel_difference": int(
            np.max(np.abs(baseline_array - d65_array))
        ),
        "d65_identity": bool(np.array_equal(baseline_array, d65_array)),
        "managed_rendering": "land spectral family; painted and never hidden",
        "preview_sha256": {
            "d65": _sha256(PREVIEW_D65_PATH),
            "2856k": _sha256(PREVIEW_A_PATH),
            "midpoint": _sha256(PREVIEW_MIDPOINT_PATH),
            "pair": _sha256(PREVIEW_PAIR_PATH),
            "triptych": _sha256(PREVIEW_TRIPTYCH_PATH),
            "water_variants_2856k": _sha256(WATER_VARIANT_COMPARISON_PATH),
        },
        "midpoint_illuminant": "50% D65 SPD + 50% 2856 K SPD",
        "recommended_water_variant": RECOMMENDED_WATER_VARIANT,
        "water_variant_d65_lstar": water_variant_lstar,
        "framing": {
            "tight_crop_aspect_ratio_width_over_height": aspect_ratio,
            "rotated_aspect_ratio_width_over_height": rotated_aspect_ratio,
            "film_aspect_ratio": target_aspect_ratio,
            "recommended_fit": "rotate 90 degrees, then pad left and right to 16:9",
            "rotated_width_fraction_of_16_9_frame": (
                rotated_aspect_ratio / target_aspect_ratio
            ),
            "film_settings_changed": False,
        },
    }
    RENDER_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(render_previews(), indent=2))
