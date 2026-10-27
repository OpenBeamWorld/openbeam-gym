# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Joint table of ``openbeam_humanoid_g1``, a G1-class humanoid (about 1.3 m,
about 35 kg; the publicly described 23-DoF base layout plus waist roll and
pitch: 25 joints). Shared by the builder (``build_humanoid.py``) and the pose
retargeter (``openbeam_gym/retarget.py``); stdlib only.

Frame: the pelvis (base) frame, x forward, y left, z up, metres. Every joint is
zero in the authored (spawn) pose: standing straight, legs vertical, arms
hanging at the sides with straight elbows, palms facing the thighs. A joint's
value is the right-handed rotation about its ``axis`` (the axis as it is at
zero, in the base frame). Left and right joints share axis directions, so a
mirrored pose has equal pitch values and negated roll and yaw values; the roll
and yaw limits mirror accordingly.

Sign conventions that follow from the axes:

* ``*_hip_pitch``, ``*_shoulder_pitch`` (+y): negative swings the limb forward.
* ``*_knee`` (+y), ``*_elbow`` (-y): positive flexes.
* ``*_ankle_pitch`` (+y): positive points the toes down.
* ``left_hip_roll``, ``left_shoulder_roll`` (+x): positive abducts (outward);
  the right side abducts with negative values.
* ``*_hip_yaw``, ``*_shoulder_yaw`` and ``waist_yaw`` (+z): positive turns
  the limb (or torso) counter-clockwise seen from above.
* ``waist_roll`` (+x): positive leans the torso to the right;
  ``waist_pitch`` (+y): positive leans it forward.
* ``*_ankle_roll`` (+x), ``*_wrist_roll`` (-z, along the hanging forearm):
  right-handed about those axes.

Limits are close to those published for G1-class robots, kept within
(-pi, pi) because a torsion bar's angle is unique only there; the elbow's zero
here is a straight arm. Rates (rad/s) are how fast each joint's actuator
follows its target: quick for wrists and elbows, slower for the heavy hips and
the waist.
"""

from __future__ import annotations

# name, axis (base frame, at zero), lower, upper (rad), rate (rad/s), home (rad)
_LEG = [
    ("hip_pitch", (0.0, 1.0, 0.0), (-2.53, 2.88), 4.0, -0.10),
    ("hip_roll", (1.0, 0.0, 0.0), (-0.52, 2.97), 3.0, 0.0),
    ("hip_yaw", (0.0, 0.0, 1.0), (-2.76, 2.76), 3.0, 0.0),
    ("knee", (0.0, 1.0, 0.0), (-0.09, 2.88), 5.0, 0.25),
    ("ankle_pitch", (0.0, 1.0, 0.0), (-0.87, 0.52), 4.0, -0.15),
    ("ankle_roll", (1.0, 0.0, 0.0), (-0.26, 0.26), 2.0, 0.0),
]
_WAIST = [
    ("waist_yaw", (0.0, 0.0, 1.0), (-2.62, 2.62), 2.0, 0.0),
    ("waist_roll", (1.0, 0.0, 0.0), (-0.52, 0.52), 1.5, 0.0),
    ("waist_pitch", (0.0, 1.0, 0.0), (-0.52, 0.52), 1.5, 0.0),
]
_ARM = [
    ("shoulder_pitch", (0.0, 1.0, 0.0), (-3.0, 2.67), 6.0, 0.0),
    ("shoulder_roll", (1.0, 0.0, 0.0), (-1.59, 2.25), 6.0, 0.15),
    ("shoulder_yaw", (0.0, 0.0, 1.0), (-2.62, 2.62), 6.0, 0.0),
    ("elbow", (0.0, -1.0, 0.0), (-0.09, 2.62), 8.0, 0.35),
    ("wrist_roll", (0.0, 0.0, -1.0), (-1.97, 1.97), 8.0, 0.0),
]
# Joints whose limits and home mirror (negate and swap) on the right side.
MIRRORED = {"hip_roll", "hip_yaw", "ankle_roll", "shoulder_roll", "shoulder_yaw", "wrist_roll"}


def _side(rows, side):
    out = []
    for base, axis, (lo, hi), rate, home in rows:
        if side == "right" and base in MIRRORED:
            lo, hi, home = -hi, -lo, -home
        out.append({"name": f"{side}_{base}", "base": base, "side": side, "axis": axis,
                    "lower": lo, "upper": hi, "rate": rate, "home": home})
    return out


def _center(rows):
    return [{"name": n, "base": n, "side": "center", "axis": a, "lower": lo, "upper": hi, "rate": r, "home": h}
            for n, a, (lo, hi), r, h in rows]


# The G1 ordering: left leg, right leg, waist, left arm, right arm.
JOINTS: list[dict] = (_side(_LEG, "left") + _side(_LEG, "right") + _center(_WAIST)
                      + _side(_ARM, "left") + _side(_ARM, "right"))
JOINT_NAMES: list[str] = [j["name"] for j in JOINTS]
LIMITS: dict[str, tuple[float, float]] = {j["name"]: (j["lower"], j["upper"]) for j in JOINTS}
RATES: dict[str, float] = {j["name"]: j["rate"] for j in JOINTS}
# The neutral standing pose ('home' key): knees slightly bent with the feet
# level (hip + knee + ankle pitch = 0), arms slightly out, elbows bent.
HOME: dict[str, float] = {j["name"]: j["home"] for j in JOINTS}
ZERO: dict[str, float] = {j["name"]: 0.0 for j in JOINTS}


def mirror_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def mirror_pose(pose: dict[str, float]) -> dict[str, float]:
    """The left/right mirror image of a pose (reflection through the x-z plane)."""
    out = {}
    for name, value in pose.items():
        base = name.split("_", 1)[1] if name.startswith(("left_", "right_")) else name
        flip = base in MIRRORED or name in ("waist_yaw", "waist_roll")
        out[mirror_name(name)] = -value if flip else value
    return out
