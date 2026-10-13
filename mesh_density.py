"""Build and compare the 15k, 25k, and 30k texture meshes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data import DATA_DIR, SEED
from mesh import (
    COMPARISON_POINT_COUNTS,
    MESH_PATH,
    MESH_REPORT_PATH,
    TARGET_POINTS,
    build_mesh,
)
from metamers import _adjacent_camouflage, _water_report, build_base_spectra
from render import render_mesh, spectral_colours


OUTPUT_DIR = DATA_DIR.parent / "out" / "phase2" / "mesh_density"
COMPARISON_PATH = OUTPUT_DIR / "mesh_density_comparison_1600.png"
REPORT_PATH = OUTPUT_DIR / "mesh_density_comparison_report.json"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_density_comparison(seed: int = SEED) -> dict:
    """Render the three requested point counts and retain 25k as production."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    images = []
    for target_points in COMPARISON_POINT_COUNTS:
        mesh_path = (
            MESH_PATH
            if target_points == TARGET_POINTS
            else DATA_DIR / f"gsl_mesh_{target_points}.npz"
        )
        report_path = (
            MESH_REPORT_PATH
            if target_points == TARGET_POINTS
            else OUTPUT_DIR / f"mesh_{target_points}_report.json"
        )
        mesh_report = build_mesh(
            seed,
            target_points,
            mesh_path=mesh_path,
            report_path=report_path,
        )
        mesh = np.load(mesh_path)
        bases = build_base_spectra(
            mesh["centroids"],
            mesh["classes"],
            mesh["arms"],
            mesh["occurrence"],
            seed,
        )
        camouflage = _adjacent_camouflage(mesh, bases)
        water = _water_report(mesh, bases)
        preview = render_mesh(
            mesh["points"],
            mesh["simplices"],
            mesh["classes"],
            spectral_colours(bases, "D65"),
            tuple(int(value) for value in mesh["raster_shape"]),
            long_side=1600,
            supersample=2,
        )
        preview_path = OUTPUT_DIR / f"mesh_{target_points}_d65_1600.png"
        preview.save(preview_path, optimize=True)
        images.append(preview)
        results.append(
            {
                "target_points": target_points,
                "actual_points": mesh_report["points"],
                "triangles": mesh_report["triangles"],
                "density_camouflage": mesh_report["density_camouflage"],
                "spectral_camouflage": camouflage,
                "water_facets": water,
                "preview": str(preview_path.relative_to(DATA_DIR.parent)),
            }
        )

    gap = 12
    width = 1600
    panel_width = (width - 2 * gap) // 3
    panel_height = int(round(images[0].height * panel_width / images[0].width))
    comparison = Image.new("RGB", (width, panel_height), (244, 241, 232))
    draw = ImageDraw.Draw(comparison)
    font = _font(25)
    for index, (image, result) in enumerate(zip(images, results)):
        panel = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        x0 = index * (panel_width + gap)
        comparison.paste(panel, (x0, 0))
        label = (
            f"{result['target_points'] // 1000}k points · "
            f"{result['triangles']:,} triangles"
        )
        x, y = x0 + 18, 16
        box = draw.textbbox((x, y), label, font=font)
        draw.rounded_rectangle(
            (box[0] - 7, box[1] - 5, box[2] + 7, box[3] + 5),
            radius=5,
            fill=(244, 241, 232),
        )
        draw.text((x, y), label, font=font, fill=(41, 48, 43))
    comparison.save(COMPARISON_PATH, optimize=True)

    report = {
        "seed": seed,
        "production_target_points": TARGET_POINTS,
        "comparison_size_px": list(comparison.size),
        "comparison_path": str(COMPARISON_PATH.relative_to(DATA_DIR.parent)),
        "variants": results,
        "all_density_tests_pass": all(
            item["density_camouflage"]["passes_target"] for item in results
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_density_comparison(), indent=2))
