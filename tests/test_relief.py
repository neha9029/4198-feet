"""Optional terrain-relief source, modulation, and acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PHASE2_REPORT = ROOT / "out" / "phase2" / "phase2_report.json"
RELIEF_REPORT = ROOT / "out" / "phase2" / "relief_report.json"


def test_relief_source_and_factor_range() -> None:
    result = json.loads(RELIEF_REPORT.read_text(encoding="utf-8"))
    assert result["enabled"] is True
    assert result["source"]["bucket"] == "s3://elevation-tiles-prod"
    assert result["source"]["tile_count"] == 144
    assert result["hillshade"]["sun_azimuth_degrees"] == 315.0
    assert result["hillshade"]["sun_altitude_degrees"] == 45.0
    assert result["hillshade"]["smoothing_metres"] == 300.0
    assert result["hillshade"]["modulation_amplitude"] == 0.08
    assert np.allclose(result["hillshade"]["factor_min_max"], [0.92, 1.08])
    assert result["triangle_factors"]["water_untouched"] is True


def test_relief_retains_camouflage_density_and_d65_identity() -> None:
    result = json.loads(PHASE2_REPORT.read_text(encoding="utf-8"))
    assert result["metamers"]["terrain_relief"]["enabled"] is True
    assert result["metamers"]["terrain_relief"]["water_untouched"] is True
    camouflage = result["metamers"]["d65_camouflage"]
    assert camouflage["ks_pvalue"] > 0.05
    assert camouflage["median_relative_difference"] <= 0.10
    assert camouflage["passes_target"] is True
    density = result["mesh"]["density_camouflage"]
    assert density["relative_difference"] <= 0.10
    assert density["passes_target"] is True
    assert result["render"]["d65_identity"] is True
    assert result["render"]["d65_max_abs_pixel_difference"] == 0


def test_relief_comparison_is_1600_pixels_wide() -> None:
    comparison = Image.open(ROOT / "out" / "stills" / "relief_comparison_1600.png")
    assert comparison.width == 1600
