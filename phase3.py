"""Phase 3 production stills, vector panels, diptych, and film frames."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data import DATA_DIR, SEED
from mesh import MESH_PATH
from metamers import ILLUMINANT_A_K, SPECTRA_BASE_PATH, SPECTRA_FINAL_PATH
from render import render_mesh, spectral_colours


STILLS_DIR = DATA_DIR.parent / "out" / "stills"
D65_PNG_PATH = STILLS_DIR / "4198_feet_d65_4000.png"
A_PNG_PATH = STILLS_DIR / "4198_feet_2856k_4000.png"
D65_SVG_PATH = STILLS_DIR / "4198_feet_d65.svg"
A_SVG_PATH = STILLS_DIR / "4198_feet_2856k.svg"
DIPTYCH_PATH = STILLS_DIR / "4198_feet_diptych.png"
D65_FRAME_PATH = STILLS_DIR / "4198_feet_d65_3840x2160.png"
A_FRAME_PATH = STILLS_DIR / "4198_feet_2856k_3840x2160.png"
REPORT_PATH = STILLS_DIR / "phase3_report.json"

PANEL_LONG_SIDE = 4_000
FRAME_SIZE = (3_840, 2_160)
SUPERSAMPLE = 2
GUTTER_PX = 32
BACKGROUND = (244, 241, 232)
CAPTIONS = ("D65 · daylight", "2856 K · tungsten")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use a portable sans-serif stack, falling back to Pillow's bundled font."""
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _svg_panel(
    path: Path,
    points: np.ndarray,
    simplices: np.ndarray,
    classes: np.ndarray,
    colours: np.ndarray,
    width: int,
    height: int,
    title: str,
) -> None:
    """Write the exact triangle panel as a compact standalone SVG."""
    scale = np.array([width - 1, height - 1], dtype=float)
    polygons = points[simplices] * scale[None, None, :]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'shape-rendering="geometricPrecision">'
        ),
        f"  <title>{title}</title>",
        f'  <rect width="100%" height="100%" fill="rgb{BACKGROUND}"/>',
        '  <g stroke="none">',
    ]
    for polygon, colour, triangle_class in zip(polygons, colours, classes):
        if triangle_class == 255:
            continue
        coordinates = " ".join(
            f"{vertex[0]:.3f},{vertex[1]:.3f}" for vertex in polygon
        )
        hexadecimal = "#{:02x}{:02x}{:02x}".format(*(int(value) for value in colour))
        lines.append(f'    <polygon points="{coordinates}" fill="{hexadecimal}"/>')
    lines.extend(["  </g>", "</svg>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _caption(image: Image.Image, label: str, x_offset: int) -> None:
    draw = ImageDraw.Draw(image)
    font = _font(44)
    x, y = x_offset + 54, 46
    box = draw.textbbox((x, y), label, font=font)
    draw.rounded_rectangle(
        (box[0] - 18, box[1] - 13, box[2] + 18, box[3] + 13),
        radius=10,
        fill=BACKGROUND,
    )
    draw.text((x, y), label, font=font, fill=(41, 48, 43))


def run_phase3(seed: int = SEED) -> dict:
    """Render and verify all accepted Phase 3 deliverables."""
    STILLS_DIR.mkdir(parents=True, exist_ok=True)
    mesh = np.load(MESH_PATH)
    points = mesh["points"]
    simplices = mesh["simplices"]
    classes = mesh["classes"]
    raster_shape = tuple(int(value) for value in mesh["raster_shape"])
    base = np.load(SPECTRA_BASE_PATH)
    final = np.load(SPECTRA_FINAL_PATH)

    base_d65_colours = spectral_colours(base, "D65")
    d65_colours = spectral_colours(final, "D65")
    a_colours = spectral_colours(final, ILLUMINANT_A_K)
    d65_colour_max_difference = int(
        np.max(
            np.abs(
                base_d65_colours.astype(np.int16) - d65_colours.astype(np.int16)
            )
        )
    )

    d65 = render_mesh(
        points,
        simplices,
        classes,
        d65_colours,
        raster_shape,
        PANEL_LONG_SIDE,
        SUPERSAMPLE,
    )
    d65.save(D65_PNG_PATH, optimize=True)
    d65.resize(FRAME_SIZE, Image.Resampling.LANCZOS).save(
        D65_FRAME_PATH, optimize=True
    )
    d65_size = d65.size
    del d65

    tungsten = render_mesh(
        points,
        simplices,
        classes,
        a_colours,
        raster_shape,
        PANEL_LONG_SIDE,
        SUPERSAMPLE,
    )
    tungsten.save(A_PNG_PATH, optimize=True)
    tungsten.resize(FRAME_SIZE, Image.Resampling.LANCZOS).save(
        A_FRAME_PATH, optimize=True
    )
    tungsten_size = tungsten.size
    del tungsten

    _svg_panel(
        D65_SVG_PATH,
        points,
        simplices,
        classes,
        d65_colours,
        d65_size[0],
        d65_size[1],
        CAPTIONS[0],
    )
    _svg_panel(
        A_SVG_PATH,
        points,
        simplices,
        classes,
        a_colours,
        tungsten_size[0],
        tungsten_size[1],
        CAPTIONS[1],
    )

    with Image.open(D65_PNG_PATH) as left, Image.open(A_PNG_PATH) as right:
        diptych = Image.new(
            "RGB",
            (left.width + GUTTER_PX + right.width, max(left.height, right.height)),
            BACKGROUND,
        )
        diptych.paste(left, (0, 0))
        diptych.paste(right, (left.width + GUTTER_PX, 0))
    _caption(diptych, CAPTIONS[0], 0)
    _caption(diptych, CAPTIONS[1], d65_size[0] + GUTTER_PX)
    diptych.save(DIPTYCH_PATH, optimize=True)

    outputs = {
        "d65_png": D65_PNG_PATH,
        "2856k_png": A_PNG_PATH,
        "d65_svg": D65_SVG_PATH,
        "2856k_svg": A_SVG_PATH,
        "diptych_png": DIPTYCH_PATH,
        "d65_frame_3840x2160": D65_FRAME_PATH,
        "2856k_frame_3840x2160": A_FRAME_PATH,
    }
    report = {
        "seed": seed,
        "panel_size_px": list(d65_size),
        "panel_long_side_px": PANEL_LONG_SIDE,
        "supersample": SUPERSAMPLE,
        "north_up": True,
        "film_frame_size_px": list(FRAME_SIZE),
        "diptych_size_px": list(diptych.size),
        "diptych_gutter_px": GUTTER_PX,
        "captions": list(CAPTIONS),
        "svg_triangle_count_per_panel": int(np.count_nonzero(classes != 255)),
        "d65_identity_max_colour_channel_difference": d65_colour_max_difference,
        "d65_identity": d65_colour_max_difference == 0,
        "out_of_data_treatment": (
            "land spectral-family texture only; no metameric reveal"
        ),
        "outputs": {
            name: {
                "path": str(path.relative_to(DATA_DIR.parent)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_phase3(), indent=2))
