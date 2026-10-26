# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Top-down teleoperation math: an overhead camera image <-> a table plane.

The overhead camera (``assets/openbeam_camera_boom.zip``) is declared by three
nodes in its ``openbeam_robot.json`` ``camera`` block: the optical centre
(``position_node``), a point on the optical axis (``target_node``) and a point
toward the image's up direction (``up_node``), plus the vertical field of view
and the image aspect. ``camera_pose`` turns those node positions into a pose;
``Camera`` is a pinhole camera with that pose, and the functions below map an
image pixel to the table and back, and a drag in the image to a move on the
table: what a player (or a phone) uses to move a gripper in the overhead view.

Conventions:

* World: any right-handed frame the node positions are given in (the game's
  world, or a robot's base frame).
* Camera axes (OpenCV): x image right, y image down, z the optical axis.
* Pixels (u, v): u to the right, v down, origin at the image's top-left corner;
  pixel (i, j) covers [i, i+1) x [j, j+1). Square pixels; the principal point is
  the image centre; ``fov_deg`` is the vertical field of view.

Numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

# A ray is refused when the cosine between it and the table plane's normal is below this.
MIN_COS = 1e-6


def _unit(v, what: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError(f"{what} has no direction")
    return v / n


@dataclass(frozen=True)
class CameraPose:
    """Position, unit forward (optical axis) and unit up (image up, orthogonal to forward)."""

    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray

    @property
    def right(self) -> np.ndarray:
        return np.cross(self.forward, self.up)

    @property
    def rotation(self) -> np.ndarray:
        """World-from-camera rotation; columns are the OpenCV camera axes (right, down, forward)."""
        return np.column_stack([self.right, -self.up, self.forward])


def camera_pose(position, target, up) -> CameraPose:
    """The camera pose from three node positions: the optical centre, a point
    on the optical axis, and a point toward image up (``up - position`` is
    orthogonalised against the optical axis, Gram-Schmidt).

    This is what a viewer needs to place a render camera every frame: at
    ``position``, looking at ``position + forward``, with ``up``."""
    p = np.asarray(position, dtype=np.float64).reshape(3)
    forward = _unit(np.asarray(target, dtype=np.float64) - p, "the optical axis (target - position)")
    u = np.asarray(up, dtype=np.float64).reshape(3) - p
    u = u - forward * float(forward @ u)
    up_dir = _unit(u, "image up (up - position, orthogonal to the optical axis)")
    return CameraPose(p, forward, up_dir)


@dataclass(frozen=True)
class Camera:
    """A pinhole camera: pose, vertical field of view and image size in pixels."""

    pose: CameraPose
    fov_deg: float
    width: int
    height: int

    def __post_init__(self):
        if not 0.0 < self.fov_deg < 180.0:
            raise ValueError("fov_deg must be in (0, 180)")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("the image size must be positive")

    @classmethod
    def from_nodes(cls, position, target, up, fov_deg: float, width: int, height: int) -> "Camera":
        return cls(camera_pose(position, target, up), fov_deg, width, height)

    @property
    def focal(self) -> float:
        """Focal length in pixels (square pixels)."""
        return 0.5 * self.height / math.tan(math.radians(self.fov_deg) / 2)

    @property
    def principal(self) -> np.ndarray:
        return np.array([self.width / 2.0, self.height / 2.0])

    def ray(self, pixel) -> np.ndarray:
        """The (unnormalised, camera-z = 1) world direction through ``pixel``."""
        u, v = np.asarray(pixel, dtype=np.float64).reshape(2) - self.principal
        return self.pose.rotation @ np.array([u / self.focal, v / self.focal, 1.0])

    def project(self, point) -> np.ndarray:
        """The pixel of a world point. Refuses points at or behind the camera plane."""
        c = self.pose.rotation.T @ (np.asarray(point, dtype=np.float64).reshape(3) - self.pose.position)
        if c[2] <= 1e-12:
            raise ValueError("the point is not in front of the camera")
        return self.principal + self.focal * c[:2] / c[2]

    def in_image(self, pixel) -> bool:
        u, v = np.asarray(pixel, dtype=np.float64).reshape(2)
        return 0.0 <= u <= self.width and 0.0 <= v <= self.height


def camera_from_spec(block: Mapping, nodes: Mapping[str, object], width: int, height: int | None = None) -> Camera:
    """A Camera from an ``openbeam_robot.json`` ``camera`` block and node
    positions (name -> xyz). ``height`` defaults to ``width / aspect``."""
    if block.get("fov_axis", "vertical") != "vertical":
        raise ValueError("only a vertical field of view is supported")
    if height is None:
        height = int(round(width / float(block.get("aspect", 4.0 / 3.0))))
    return Camera.from_nodes(nodes[block["position_node"]], nodes[block["target_node"]], nodes[block["up_node"]],
                             float(block["fov_deg"]), width, height)


@dataclass(frozen=True)
class Plane:
    """A plane through ``point`` with normal ``normal`` (normalised on use)."""

    point: np.ndarray
    normal: np.ndarray

    @classmethod
    def horizontal(cls, z: float = 0.0) -> "Plane":
        return cls(np.array([0.0, 0.0, z]), np.array([0.0, 0.0, 1.0]))

    def unit_normal(self) -> np.ndarray:
        return _unit(self.normal, "the plane normal")


def _hit(camera: Camera, plane: Plane, pixel, min_cos: float):
    d = camera.ray(pixel)
    n = plane.unit_normal()
    dn = float(d @ n)
    if abs(dn) < min_cos * float(np.linalg.norm(d)):
        raise ValueError("the ray through this pixel is parallel to the plane")
    t = float(n @ (np.asarray(plane.point, dtype=np.float64) - camera.pose.position)) / dn
    if t <= 0.0:
        raise ValueError("the ray through this pixel meets the plane behind the camera (above the horizon)")
    return camera.pose.position + t * d, d, n, t


def pixel_to_plane(camera: Camera, plane: Plane, pixel, min_cos: float = MIN_COS) -> np.ndarray:
    """The world point on the plane seen at ``pixel``. Raises ValueError for a
    ray (nearly) parallel to the plane or one that meets it behind the camera."""
    return _hit(camera, plane, pixel, min_cos)[0]


def plane_to_pixel(camera: Camera, point) -> np.ndarray:
    """The pixel at which a world point (on the table or not) is seen."""
    return camera.project(point)


def drag_to_plane_delta(camera: Camera, plane: Plane, pixel, delta, min_cos: float = MIN_COS) -> np.ndarray:
    """The world displacement on the plane for a drag of ``delta`` pixels
    starting at ``pixel`` (exact: the difference of the two hits)."""
    start = np.asarray(pixel, dtype=np.float64).reshape(2)
    end = start + np.asarray(delta, dtype=np.float64).reshape(2)
    return pixel_to_plane(camera, plane, end, min_cos) - pixel_to_plane(camera, plane, start, min_cos)


def drag_to_table_xy(camera: Camera, plane: Plane, pixel, delta, min_cos: float = MIN_COS) -> np.ndarray:
    """``drag_to_plane_delta`` as a world (dx, dy): the gripper move for a drag
    on a horizontal table. For a tilted table use ``drag_to_plane_delta``."""
    return drag_to_plane_delta(camera, plane, pixel, delta, min_cos)[:2]


def drag_jacobian(camera: Camera, plane: Plane, pixel, min_cos: float = MIN_COS) -> np.ndarray:
    """d(world point on the plane) / d(pixel) at ``pixel``, a 3 x 2 matrix: the
    linear map from a small drag (or a pixel velocity) to a world move."""
    x, d, n, t = _hit(camera, plane, pixel, min_cos)
    dd = camera.pose.rotation[:, :2] / camera.focal              # d ray / d (u, v)
    return t * (np.eye(3) - np.outer(d, n) / float(d @ n)) @ dd


def metres_per_pixel(camera: Camera, plane: Plane, pixel, min_cos: float = MIN_COS) -> float:
    """The ground sample distance at ``pixel``: the geometric mean of how far
    one pixel right and one pixel down move on the plane."""
    j = drag_jacobian(camera, plane, pixel, min_cos)
    return math.sqrt(float(np.linalg.norm(j[:, 0]) * np.linalg.norm(j[:, 1])))
