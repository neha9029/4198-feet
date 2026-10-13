"""Phase 3 production-export tests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STILLS = ROOT / "out" / "stills"


def report() -> dict:
    return json.loads((STILLS / "phase3_report.json").read_text(encoding="utf-8"))


def test_phase3_png_dimensions_and_captions() -> None:
    result = report()
    assert result["panel_size_px"] == [4000, 2250]
    assert result["film_frame_size_px"] == [3840, 2160]
    assert result["captions"] == ["D65 · daylight", "2856 K · tungsten"]
    assert result["north_up"] is True
    for name in ("4198_feet_d65_4000.png", "4198_feet_2856k_4000.png"):
        assert Image.open(STILLS / name).size == (4000, 2250)
    for name in ("4198_feet_d65_3840x2160.png", "4198_feet_2856k_3840x2160.png"):
        assert Image.open(STILLS / name).size == (3840, 2160)


def test_phase3_d65_identity_and_vector_panels() -> None:
    result = report()
    assert result["d65_identity"] is True
    assert result["d65_identity_max_colour_channel_difference"] == 0
    assert result["out_of_data_treatment"].endswith("no metameric reveal")
    for name in ("4198_feet_d65.svg", "4198_feet_2856k.svg"):
        svg = (STILLS / name).read_text(encoding="utf-8")
        assert 'viewBox="0 0 4000 2250"' in svg
        assert svg.count("<polygon ") == result["svg_triangle_count_per_panel"]


def test_bear_river_bay_crops_are_native_frame_scale() -> None:
    result = report()["bear_river_bay_crop"]
    assert result["crop_size_px"] == [600, 600]
    assert result["scale"].startswith("100%")
    assert Image.open(STILLS / "bear_river_bay_d65_600.png").size == (600, 600)
    assert Image.open(STILLS / "bear_river_bay_2856k_600.png").size == (600, 600)
    assert Image.open(STILLS / "bear_river_bay_crops.png").size == (1212, 600)
