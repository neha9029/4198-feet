"""Phase 4 light, camera, identity, and preview-render tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from film import (
    A_EQUAL_Y,
    CAMERA_HEADING_EASE_END_SECONDS,
    CAMERA_HEADING_END_DEGREES,
    CAMERA_HEALTHY_FILL_FRACTION,
    CAMERA_SUPERSAMPLE,
    CAMERA_TILT_HOLD_DEGREES,
    D65_EQUAL_Y,
    END_CARD_LINES,
    FILM_DURATION_SECONDS,
    TERRAIN_FINAL_MAP_DURATION_SECONDS,
    TERRAIN_FINAL_TOTAL_DURATION_SECONDS,
    TOTAL_DURATION_SECONDS,
    YBAR,
    camera_angles,
    end_card_opacity,
    mixture_amount,
    render_end_card,
    smootherstep,
    terrain_final_source_seconds,
)


ROOT = Path(__file__).resolve().parents[1]
FILM_DIR = ROOT / "out" / "film"
REPORT = FILM_DIR / "camera_preview_report.json"
GEOMETRY_REPORT = FILM_DIR / "camera_geometry_report.json"
FINAL_REPORT = FILM_DIR / "final_film_report.json"


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_light_timeline_and_equal_luminance_mix() -> None:
    assert smootherstep(-1.0) == 0.0
    assert smootherstep(0.0) == 0.0
    assert smootherstep(1.0) == 1.0
    assert smootherstep(2.0) == 1.0
    assert mixture_amount(0.0) == 0.0
    assert mixture_amount(40.0) == 0.0
    assert mixture_amount(110.0) == 1.0
    assert mixture_amount(150.0) == 1.0
    assert mixture_amount(200.0) == 0.0
    assert np.isclose(np.sum(D65_EQUAL_Y * YBAR), 1.0)
    assert np.isclose(np.sum(A_EQUAL_Y * YBAR), 1.0)


def test_camera_angle_endpoints_and_holds() -> None:
    assert camera_angles(0.0) == (0.0, 0.0)
    tilt_40, heading_40 = camera_angles(40.0)
    assert tilt_40 == CAMERA_TILT_HOLD_DEGREES
    assert 0.0 < heading_40 < CAMERA_HEADING_END_DEGREES
    assert camera_angles(150.0)[0] == CAMERA_TILT_HOLD_DEGREES
    tilt_200, heading_200 = camera_angles(200.0)
    assert tilt_200 == 0.0
    assert 89.9 < heading_200 < CAMERA_HEADING_END_DEGREES
    assert camera_angles(CAMERA_HEADING_EASE_END_SECONDS) == (
        0.0,
        CAMERA_HEADING_END_DEGREES,
    )


def test_per_frame_daylight_identity_is_exact() -> None:
    result = report()
    assert result["identity_max_pixel_difference"] == 0
    assert [row["seconds"] for row in result["identity_checks"]] == [
        0.0,
        20.0,
        39.0,
        200.0,
        205.0,
    ]
    assert all(row["mixture_amount"] == 0.0 for row in result["identity_checks"])
    assert all(row["max_pixel_difference"] == 0 for row in result["identity_checks"])


def test_camera_geometry_framing_and_end_states() -> None:
    camera = report()["camera"]
    assert camera["geometry"].startswith("flat plane")
    assert camera["supersample"] == CAMERA_SUPERSAMPLE == 2
    assert camera["extended_land"]["hard_edge_visible"] is False
    assert (
        abs(camera["opening"]["healthy_fill_fraction"] - CAMERA_HEALTHY_FILL_FRACTION)
        < 1e-6
    )
    assert camera["opening"]["tilt_degrees"] == 0.0
    assert camera["opening"]["heading_degrees"] == 0.0
    assert camera["final_daylight"]["tilt_degrees"] == 0.0
    assert camera["final_daylight"]["heading_degrees"] == 90.0
    assert Image.open(FILM_DIR / "camera_opening_d65_720p.png").size == (
        1280,
        720,
    )
    assert Image.open(FILM_DIR / "camera_final_daylight_720p.png").size == (
        1280,
        720,
    )


def test_shimmer_and_accelerated_preview_outputs() -> None:
    result = report()
    shimmer = result["shimmer_clip"]
    assert shimmer["global_seconds"] == [40.0, 50.0]
    assert shimmer["size"] == [1920, 1080]
    assert shimmer["fps"] == 24
    assert shimmer["supersample"] == 2
    assert shimmer["visible_flicker"] is False
    assert shimmer["visible_moire"] is False
    assert shimmer["probe"]["nb_frames"] == "240"

    fast = result["fast_preview"]
    assert fast["speed"] == 4.0
    assert fast["encoded_duration_seconds"] == 52.5
    assert fast["size"] == [1280, 720]
    assert fast["probe"]["nb_frames"] == "1260"
    for path in (
        FILM_DIR / "shimmer_40_50" / "4198_feet_40-50s_camera_h264.mp4",
        FILM_DIR / "4198_feet_full_4x_720p.mp4",
    ):
        assert path.stat().st_size > 0


def test_render_timings_are_recorded() -> None:
    result = report()
    assert result["shimmer_clip"]["timing"]["mean_seconds_per_frame"] > 0.0
    assert result["fast_preview"]["timing"]["mean_seconds_per_frame"] > 0.0
    assert result["master_4k_benchmark"]["mean_seconds_per_frame"] > 0.0


def test_geometry_gate_samples_every_two_seconds() -> None:
    geometry = json.loads(GEOMETRY_REPORT.read_text(encoding="utf-8"))
    assert geometry["passes"] is True
    assert geometry["sample_interval_seconds"] == 2.0
    assert geometry["sample_count"] == 106
    assert geometry["max_center_deviation_fraction"] <= 0.03
    assert geometry["max_outside_lake_triangle_frame_fraction"] <= 0.005
    assert geometry["minimum_plane_to_frustum_multiple"] >= 3.0
    assert geometry["fog_opaque_before_plane_edge"] is True
    assert geometry["final_heading_degrees"] == 90.0
    assert geometry["camouflage"]["passes"] is True
    assert geometry["density"]["passes"] is True


def test_end_card_timing_copy_and_fades() -> None:
    assert TOTAL_DURATION_SECONDS == 215.0
    assert END_CARD_LINES == (
        "4,198 Feet",
        "Great Salt Lake, 4,189.6 ft, 9 October 2026",
        "Neha Naik, 2026",
    )
    assert end_card_opacity(FILM_DURATION_SECONDS) == 0.0
    assert end_card_opacity(FILM_DURATION_SECONDS + 1.0) == 1.0
    assert end_card_opacity(TOTAL_DURATION_SECONDS - 1.0) == 1.0
    assert end_card_opacity(TOTAL_DURATION_SECONDS) == 0.0
    middle, metadata = render_end_card(
        FILM_DURATION_SECONDS + 2.5,
        (640, 360),
    )
    assert metadata["opacity"] == 1.0
    assert middle.shape == (360, 640, 3)
    assert middle.max() > 0


def test_terrain_final_cut_is_brisk_and_preserves_full_arc() -> None:
    assert TERRAIN_FINAL_MAP_DURATION_SECONDS == 85.0
    assert TERRAIN_FINAL_TOTAL_DURATION_SECONDS == 90.0
    assert terrain_final_source_seconds(0.0) == 0.0
    assert terrain_final_source_seconds(85.0) == FILM_DURATION_SECONDS
    assert np.isclose(
        terrain_final_source_seconds(42.5),
        FILM_DURATION_SECONDS / 2.0,
    )


def test_final_deliveries_are_exactly_3m35s() -> None:
    final = json.loads(FINAL_REPORT.read_text(encoding="utf-8"))
    assert final["frame_count"] == 5_160
    assert final["total_duration_seconds"] == 215.0
    assert final["duration_check"]["passes"] is True
    assert final["identity_max_pixel_difference"] == 0
    assert final["geometry_tests_pass"] is True
    assert final["final_heading_degrees"] == 90.0
    expected = {
        "h264_2160p": (3840, 2160, "h264", "yuv420p"),
        "prores_422_hq_10bit": (3840, 2160, "prores", "yuv422p10le"),
        "h264_1080p": (1920, 1080, "h264", "yuv420p"),
    }
    for label, (width, height, codec, pixel_format) in expected.items():
        output = final["outputs"][label]
        probe = output["probe"]
        assert Path(ROOT / output["path"]).stat().st_size == output["bytes"]
        assert probe["width"] == width
        assert probe["height"] == height
        assert probe["codec_name"] == codec
        assert probe["pix_fmt"] == pixel_format
        assert probe["r_frame_rate"] == "24/1"
        assert probe["duration"] == "215.000000"
        assert probe["nb_frames"] == "5160"
