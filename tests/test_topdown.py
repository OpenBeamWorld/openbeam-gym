# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Top-down image <-> table math."""

import math

import numpy as np
import pytest

from openbeam_gym import topdown
from openbeam_gym.topdown import Camera, Plane


def nadir(height=1.0, fov=50.0, w=640, h=480, at=(0.55, 0.0)):
    p = np.array([at[0], at[1], height])
    return Camera.from_nodes(p, p + [0, 0, -0.1], p + [0.1, 0, 0], fov, w, h)


def tilted(pitch_deg=30.0, height=1.0):
    """Looking forward-down (+x), pitch_deg below the horizon."""
    a = math.radians(pitch_deg)
    p = np.array([0.0, 0.0, height])
    fwd = np.array([math.cos(a), 0.0, -math.sin(a)])
    up = np.array([math.sin(a), 0.0, math.cos(a)])
    return Camera.from_nodes(p, p + fwd, p + 0.2 * up, 50.0, 640, 480)


def test_camera_pose_from_three_nodes():
    pose = topdown.camera_pose([1, 2, 3], [1, 2, 2.5], [1.3, 2, 3.4])     # up not orthogonal: fixed
    assert np.allclose(pose.position, [1, 2, 3])
    assert np.allclose(pose.forward, [0, 0, -1]) and np.allclose(pose.up, [1, 0, 0])
    assert np.allclose(pose.right, [0, -1, 0])
    r = pose.rotation
    assert np.allclose(r.T @ r, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)
    assert np.allclose(r[:, 2], pose.forward) and np.allclose(r[:, 1], -pose.up)
    # Scale of the node offsets does not matter.
    other = topdown.camera_pose([1, 2, 3], [1, 2, -7], [9, 2, 3])
    assert np.allclose(other.rotation, r)
    with pytest.raises(ValueError):
        topdown.camera_pose([0, 0, 0], [0, 0, 0], [1, 0, 0])               # no optical axis
    with pytest.raises(ValueError):
        topdown.camera_pose([0, 0, 0], [0, 0, -1], [0, 0, -2])             # up along the axis


def test_nadir_centre_scale_and_orientation():
    cam = nadir()
    table = Plane.horizontal(0.0)
    assert np.allclose(topdown.pixel_to_plane(cam, table, [320, 240]), [0.55, 0, 0])
    # Ground sample distance = height / focal.
    assert topdown.metres_per_pixel(cam, table, [320, 240]) == pytest.approx(1.0 / cam.focal)
    assert cam.focal == pytest.approx(240 / math.tan(math.radians(25)))
    # Image top = +x, image right = -y.
    top = topdown.pixel_to_plane(cam, table, [320, 0])
    right = topdown.pixel_to_plane(cam, table, [640, 240])
    assert top[0] == pytest.approx(0.55 + math.tan(math.radians(25))) and top[1] == pytest.approx(0)
    assert right[1] == pytest.approx(-math.tan(math.radians(25)) * 640 / 480)


@pytest.mark.parametrize("cam", [nadir(), nadir(0.8, 60, 1280, 720, (0.2, -0.1)), tilted(30), tilted(70, 1.2)])
def test_round_trips(cam):
    table = Plane.horizontal(0.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        px = rng.uniform([0, cam.height * 0.5 if cam.pose.forward[2] > -0.9 else 0], [cam.width, cam.height])
        x = topdown.pixel_to_plane(cam, table, px)
        assert x[2] == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(topdown.plane_to_pixel(cam, x), px, atol=1e-7)
    for _ in range(200):
        x = np.array([*rng.uniform([0.2, -0.3], [0.9, 0.3]), 0.0])
        px = topdown.plane_to_pixel(cam, x)
        assert np.allclose(topdown.pixel_to_plane(cam, table, px), x, atol=1e-9)


def test_round_trip_on_a_sloped_plane():
    cam = nadir()
    slope = Plane(np.array([0.5, 0, 0.1]), np.array([0.2, -0.1, 1.0]))
    for px in ([10, 10], [320, 240], [600, 400]):
        x = topdown.pixel_to_plane(cam, slope, px)
        assert (x - slope.point) @ slope.normal == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(topdown.plane_to_pixel(cam, x), px)


def test_drag_matches_jacobian_and_is_exact():
    table = Plane.horizontal(0.0)
    for cam in (nadir(), tilted(40)):
        px = np.array([300.0, 330.0])
        j = topdown.drag_jacobian(cam, table, px)
        eps = 1e-3
        fd = np.column_stack([(topdown.pixel_to_plane(cam, table, px + e) - topdown.pixel_to_plane(cam, table, px - e))
                              / (2 * eps) for e in (np.array([eps, 0]), np.array([0, eps]))])
        assert np.allclose(j, fd, rtol=1e-5, atol=1e-10)
        assert np.allclose(j[2], 0.0)                                   # moves stay on the table
        d = topdown.drag_to_plane_delta(cam, table, px, [25, -40])
        assert np.allclose(topdown.plane_to_pixel(cam, topdown.pixel_to_plane(cam, table, px) + d), px + [25, -40])
        assert np.allclose(topdown.drag_to_table_xy(cam, table, px, [25, -40]), d[:2])
        assert np.allclose(topdown.drag_to_table_xy(cam, table, px, [0, 0]), 0.0)


def test_nadir_drag_is_uniform_and_linear():
    cam, table = nadir(), Plane.horizontal(0.0)
    s = 1.0 / cam.focal
    for px in ([0, 0], [320, 240], [640, 480]):
        assert np.allclose(topdown.drag_to_table_xy(cam, table, px, [10, 0]), [0, -10 * s])
        assert np.allclose(topdown.drag_to_table_xy(cam, table, px, [0, 10]), [-10 * s, 0])


def test_tilted_far_pixels_cover_more_ground():
    cam, table = tilted(30), Plane.horizontal(0.0)
    near = topdown.metres_per_pixel(cam, table, [320, 470])
    far = topdown.metres_per_pixel(cam, table, [320, 250])
    assert far > near


def test_parallel_and_behind_rays_are_refused():
    table = Plane.horizontal(0.0)
    # Horizontal camera: the centre ray is parallel to the table.
    p = np.array([0.0, 0.0, 1.0])
    level = Camera.from_nodes(p, p + [1, 0, 0], p + [0, 0, 1], 50, 640, 480)
    with pytest.raises(ValueError, match="parallel"):
        topdown.pixel_to_plane(level, table, [320, 240])
    with pytest.raises(ValueError, match="parallel"):
        topdown.drag_jacobian(level, table, [320, 240])
    # Rays above the horizon meet the plane behind the camera.
    with pytest.raises(ValueError, match="behind"):
        topdown.pixel_to_plane(level, table, [320, 100])
    assert topdown.pixel_to_plane(level, table, [320, 400])[2] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        topdown.drag_to_plane_delta(tilted(10), table, [320, 470], [0, -400])
    # Camera under the table looking down: the table is behind it.
    with pytest.raises(ValueError, match="behind"):
        topdown.pixel_to_plane(nadir(height=-0.5), table, [320, 240])
    # Points behind the camera do not project.
    with pytest.raises(ValueError):
        topdown.plane_to_pixel(nadir(), [0.55, 0, 2.0])


def test_camera_from_spec_block():
    block = {"position_node": "cam", "target_node": "cam_fwd", "up_node": "cam_up", "fov_deg": 50.0,
             "fov_axis": "vertical", "aspect": 16 / 9}
    nodes = {"cam": [0, 0, 1], "cam_fwd": [0, 0, 0.9], "cam_up": [0.1, 0, 1]}
    cam = topdown.camera_from_spec(block, nodes, 1280)
    assert (cam.width, cam.height) == (1280, 720)
    assert np.allclose(cam.pose.forward, [0, 0, -1])
    with pytest.raises(ValueError):
        topdown.camera_from_spec({**block, "fov_axis": "horizontal"}, nodes, 1280)
    with pytest.raises(ValueError):
        Camera(cam.pose, 180.0, 10, 10)
