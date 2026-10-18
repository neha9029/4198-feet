"""Regression and acceptance tests for opt-in Phase 4 terrain geometry."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import film
from data import SEED
from film import (
    build_scene,
    render_camera_frame,
    source_tile_edge_luminance_check,
)
from mesh import MESH_PATH
from terrain import build_vertex_terrain


FLAT_FRAME_HASHES = {
    0.0: "e0f745e3d77c0e9862c0e63208364fe9258d65c1d5ae1470b8b102ca2de42c3a",
    105.0: "39026bee23f455692b9897c363cda5647b923f045b3a2b61dc09c6ed2594f264",
    180.0: "057718ebbaf6f07e76984b9a65a2089073e2287de7fb1fedb660302cda618c89",
}


@pytest.fixture(scope="module")
def terrain_scene():
    return build_scene(SEED, terrain=True, exaggeration=2.0)


@pytest.fixture(scope="module")
def shaded_terrain_scene():
    return build_scene(
        SEED,
        terrain=True,
        exaggeration=6.0,
        terrain_shade=True,
        shade_strength=0.6,
    )


def test_terrain_off_frame_hashes_match_approved_renderer() -> None:
    scene = build_scene(SEED)
    for seconds, expected in FLAT_FRAME_HASHES.items():
        frame, _ = render_camera_frame(
            scene,
            seconds,
            (640, 360),
            apply_fade=False,
        )
        assert hashlib.sha256(frame.tobytes()).hexdigest() == expected


def test_terrain_d65_metamer_identity_is_exact(terrain_scene) -> None:
    for seconds in (0.0, 105.0, 180.0):
        final, _ = render_camera_frame(
            terrain_scene,
            seconds,
            (640, 360),
            apply_fade=False,
            mixture_override=0.0,
        )
        baseline, _ = render_camera_frame(
            terrain_scene,
            seconds,
            (640, 360),
            spectra=terrain_scene.identity_spectra,
            apply_fade=False,
            mixture_override=0.0,
        )
        assert np.max(np.abs(final.astype(np.int16) - baseline.astype(np.int16))) == 0


def test_shaded_terrain_d65_metamer_identity_is_exact(
    shaded_terrain_scene,
) -> None:
    for seconds in (0.0, 105.0, 180.0):
        final, _ = render_camera_frame(
            shaded_terrain_scene,
            seconds,
            (640, 360),
            apply_fade=False,
            mixture_override=0.0,
        )
        baseline, _ = render_camera_frame(
            shaded_terrain_scene,
            seconds,
            (640, 360),
            spectra=shaded_terrain_scene.identity_spectra,
            apply_fade=False,
            mixture_override=0.0,
        )
        assert np.max(np.abs(final.astype(np.int16) - baseline.astype(np.int16))) == 0


def test_shaded_water_facets_are_exactly_unshaded(
    shaded_terrain_scene,
) -> None:
    assert shaded_terrain_scene.terrain_shade_enabled is True
    assert shaded_terrain_scene.baked_hillshade_enabled is False
    for class_value in (1, 2, 3):
        shades = shaded_terrain_scene.facet_shades[
            shaded_terrain_scene.classes == class_value
        ]
        assert len(shades) > 0
        assert np.all(shades == 1.0)


def test_lakebed_vertices_and_outer_boundary_are_exactly_flat() -> None:
    mesh = np.load(MESH_PATH)
    geometry = build_vertex_terrain(mesh["points"], exaggeration=2.0)
    lakebed_vertices = np.unique(
        mesh["simplices"][np.isin(mesh["classes"], (1, 2, 3))]
    )
    assert np.all(geometry.flat_vertex_mask[lakebed_vertices])
    assert np.all(geometry.heights_world[lakebed_vertices] == 0.0)
    outside = geometry.radial_distance >= geometry.radial_zero_radius
    assert np.count_nonzero(outside) > 0
    assert np.all(geometry.radial_falloff[outside] == 0.0)
    assert np.all(geometry.heights_world[outside] == 0.0)


def test_source_tile_boundary_vertices_are_exactly_flat() -> None:
    mesh = np.load(MESH_PATH)
    geometry = build_vertex_terrain(mesh["points"], exaggeration=2.0)
    points = mesh["points"]
    boundary = np.any((points == 0.0) | (points == 1.0), axis=1)
    assert np.count_nonzero(boundary) > 0
    assert np.all(geometry.heights_world[boundary] == 0.0)


def test_source_tile_edge_luminance_is_continuous(
    shaded_terrain_scene,
) -> None:
    terrain_off = source_tile_edge_luminance_check(build_scene(SEED))
    terrain_shade = source_tile_edge_luminance_check(shaded_terrain_scene)
    assert terrain_off["relative_difference"] < 0.01
    assert terrain_shade["relative_difference"] < 0.01


def test_non_source_terrain_tiles_are_projected_flat(
    monkeypatch: pytest.MonkeyPatch,
    terrain_scene,
) -> None:
    original_project = film._project
    projected_tile_count = 0

    def checking_project(points_world, *args, **kwargs):
        nonlocal projected_tile_count
        if (
            points_world.shape == terrain_scene.points_world.shape
            and not np.allclose(
                points_world[:, :2],
                terrain_scene.points_world[:, :2],
            )
        ):
            projected_tile_count += 1
            assert np.all(points_world[:, 2] == 0.0)
        return original_project(points_world, *args, **kwargs)

    monkeypatch.setattr(film, "_project", checking_project)
    render_camera_frame(
        terrain_scene,
        105.0,
        (640, 360),
        apply_fade=False,
    )
    assert projected_tile_count > 0


def test_terrain_corners_are_covered(terrain_scene) -> None:
    for seconds in (0.0, 105.0, 180.0):
        _, metadata = render_camera_frame(
            terrain_scene,
            seconds,
            (640, 360),
            apply_fade=False,
        )
        assert metadata["corner_background_pixels"] == [0, 0, 0, 0]


def test_terrain_render_is_deterministic_for_seed() -> None:
    first_scene = build_scene(SEED, terrain=True, exaggeration=2.0)
    second_scene = build_scene(SEED, terrain=True, exaggeration=2.0)
    first, _ = render_camera_frame(
        first_scene,
        105.0,
        (640, 360),
        apply_fade=False,
    )
    second, _ = render_camera_frame(
        second_scene,
        105.0,
        (640, 360),
        apply_fade=False,
    )
    assert np.array_equal(first, second)
