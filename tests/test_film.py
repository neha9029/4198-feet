"""Phase 4 timeline, codec, timing, and banding-preview tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from film import A_EQUAL_Y, D65_EQUAL_Y, YBAR, mixture_amount, smootherstep


ROOT = Path(__file__).resolve().parents[1]
FILM_DIR = ROOT / "out" / "film" / "preview_40_50"
REPORT = FILM_DIR / "preview_report.json"


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_smootherstep_timeline_and_equal_luminance_mix() -> None:
    assert smootherstep(-1.0) == 0.0
    assert smootherstep(0.0) == 0.0
    assert smootherstep(1.0) == 1.0
    assert smootherstep(2.0) == 1.0
    assert mixture_amount(40.0) == 0.0
    assert np.isclose(mixture_amount(50.0), 0.02326429535276956)
    assert np.isclose(np.sum(D65_EQUAL_Y * YBAR), 1.0)
    assert np.isclose(np.sum(A_EQUAL_Y * YBAR), 1.0)


def test_phase4_preview_encodes_match_spec() -> None:
    result = report()
    assert result["duration_seconds"] == 10.0
    assert result["fps"] == 24
    assert result["frame_count"] == 240
    assert result["frame_size"] == [1920, 1080]
    h264 = result["encodes"]["h264_8bit"]
    h265 = result["encodes"]["h265_10bit"]
    assert h264["crf"] == 16 and h264["probe"]["pix_fmt"] == "yuv420p"
    assert "ordered Bayer" in h264["dither"]
    assert h265["crf"] == 16 and h265["probe"]["pix_fmt"] == "yuv420p10le"
    for name in (
        "4198_feet_40-50s_h264_8bit_dither.mp4",
        "4198_feet_40-50s_h265_10bit.mp4",
    ):
        assert (FILM_DIR / name).stat().st_size > 0


def test_phase4_timings_and_banding_inspection() -> None:
    result = report()
    timing = result["rendering"]
    assert timing["mean_render_seconds_per_frame_1080p"] > 0.0
    assert timing["mean_render_seconds_per_frame_4k"] > 0.0
    banding = result["banding_check"]
    assert banding["visible_h264_8bit"] is False
    assert banding["visible_h265_10bit"] is False
    crop = Image.open(FILM_DIR / "stills" / "banding_midpoint_land_crop.png")
    assert crop.size == (1200, 600)
