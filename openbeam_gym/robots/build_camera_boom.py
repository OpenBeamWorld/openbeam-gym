# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Builds assets/openbeam_camera_boom.zip: an overhead camera boom as a JBeam
vehicle mod. It stands beside a tabletop arm (an SO-101, say) and holds a
camera looking straight down at the arm's workspace, for top-down teleoperation.

It follows ``build_test_arm`` exactly: a vehicle package
(``vehicles/openbeam_camera_boom/``) that ``install_mod`` takes as it is, stiff
node-cluster links, hinge joints on two axis nodes, each driven by four JBeam
``torsionHydros`` whose rest angle follows the electric ``boom_cmd_<joint>``
(radians from the spawn pose), an ``openbeam_robot.json`` (``openbeam-robot/1``)
for the game's robot tools and a vehicle Lua that keeps it awake, publishes
the generic robot block and binds keys.

Geometry (base frame: x forward, toward the workspace; y left; z up; metres;
z = 0 is the surface the boom and the arm stand on, the table top):

* A fixed stand: the base plate and the mast nodes are pinned (``fixed``),
  so the mast is a clamp stand and adds no compliance.
* ``base_pan`` about the vertical mast axis, 0.85 m up.
* ``lift`` (shoulder pitch) 1.15 m up; a 0.45 m upper boom to ``elbow``; a
  0.45 m forearm to ``tilt`` (camera pitch). At spawn the elbow is raised
  (upper boom 52 deg up, bent 105 deg) so the tilt axis is 0.55 m out at the
  lift's height: home is mid-reach, away from the straight-arm singularity.
* The camera head hangs 0.15 m below the tilt axis. Its optical centre is the
  node ``cam``; ``cam_fwd`` lies 0.1 m along the optical axis (below it) and
  ``cam_up`` 0.1 m along the image's up direction. The three are braced to
  each other, to the tilt lever ring and to the tilt axis like any link
  cluster, and weigh 0.5 kg each.

The spawn pose (all joints 0) is the home pose: the camera at (0.55, 0, 1.00)
looking straight down (nadir), image up = +x (away from the mast). With a 50
deg vertical field of view at 4:3 it sees about 1.24 x 0.93 m of the table,
so the default 0.6 x 0.6 m workspace centred at (0.55, 0) fits with margin.
The camera looks straight down whenever lift + elbow + tilt = 0 (all three
pitch about -y); ``nadir_joints`` solves that pose in closed form, and reaches
any point within 0.25 m of the workspace centre from 0.8 to 1.2 m high.

Stability (a camera mount must not sway). The game holds every node to its
0.5 ms fixed-step budgets and scales down whatever asks for more (``sizing.py``
has the law). This builder sizes everything within them itself, as the
humanoid's does, so the game scales nothing at spawn:

* The mass sits on the joints' lever rings and axis nodes (``RING_MASS``),
  because a torsion bar's budget is its nodes' m r^2. Each lever ring is
  centred on its own axis, so its own mass adds no gravity torque to that
  joint. The joints that carry the boom have long levers (base pan and lift
  20 cm), the outer joints shorter ones (elbow 16 cm, tilt 12 cm).
* Beams take an equal slice of each node's beam budget (at most
  ``BEAM_CEILING`` per node); torsion bars at most ``TORSION_CEILING`` per
  node, split between spring and damper for the damping ratio ``ZETA``
  against the inertia each joint moves.
* ``analysis()`` and ``--report`` give each joint's stiffness, damping,
  natural frequency and gravity sag (at home, and worst case: the moved mass
  held out level at its radius of gyration).
* Joints move at 0.5 rad/s (the test arm: 1), so the picture glides rather
  than jerks.

The camera:

* The ``camera`` block of ``openbeam_robot.json`` is the source of truth: the
  position, target and up nodes, the vertical field of view and the image
  aspect, plus the home pose and default workspace. A viewer places the
  render camera from those three node positions every frame (position at
  ``cam``, looking at ``cam_fwd``, up toward ``cam_up``);
  ``openbeam_gym.topdown`` has that pose and the image <-> table math.
* ``end_effectors`` declares ``camera`` at the optical centre (frame nodes
  ``cam``, ``cam_fwd``, ``cam_up``; axes the OpenCV camera axes: x image
  right, y image down, z the optical axis), so robot_fk/robot_ik treat the
  boom like the test arm.
* No JBeam camera is declared at the tip: the game's importer keeps one
  ``camerasInternal`` eye, always oriented along the vehicle's refNodes frame
  (it cannot look down), drops its FOV, and makes it a heavy sprung node that
  would sag a thin boom. The one ``camerasInternal`` row is the player's seat,
  a ``driver`` eye at the base like the test arm's, sprung only to pinned base
  nodes (tested), so it cannot load the boom.

Visuals (``visuals()``, written by ``visuals.py``): one flat-coloured mesh per
link bound to its node group (grey stand, dark tubes, orange joint housings,
a black camera body whose lens ends at the optical centre ``cam``).

Keys (seated in the boom): A/D base pan, W/S lift, E/Q elbow, X/Z tilt, 3 home.
While the nadir lock is on (after spawn or home) W/S/E/Q also move the tilt
goal so the camera keeps its pitch; X/Z turn the lock off and 3 turns it on.

    python -m openbeam_gym.robots.build_camera_boom --out assets/openbeam_camera_boom.zip [--report]

Stdlib only; the zip is deterministic (fixed timestamps, sorted members).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from openbeam_gym.robots.build_test_arm import Arm, basis, to_jbeam, write_zip
from openbeam_gym.robots.visuals import Mesh, flexbodies, package_members
from openbeam_gym.robots.sizing import (GRAVITY, add, cross, dot, gravity_torque_bound, inertia_about, joint_need,
                                        mul, size_beams, size_torsion)
from openbeam_gym.robots.sizing import dynamics as joint_dynamics
from openbeam_gym.robots.sizing import node_loads as _node_loads

NAME = "openbeam_camera_boom"
RATE = 0.5                # rad/s, joint speed

# Mass, kg per node. A torsion bar's fixed-step budget is its nodes' m r^2,
# so the mass sits on each joint's lever rings and axis nodes, where the
# motors and bearings of a real boom are: RING_MASS per ring node (parent and
# child side alike, they load equally), AXIS_SHARE of it per axis node (an
# axis node carries all four bars but sits nearer the axis). A child ring is
# centred on its own axis, so its mass costs that joint inertia but no
# gravity torque; the outer joints' rings are lighter because the lift
# carries them. The moving mass (about 114 kg) is about the earlier uniform
# 3 kg-per-node boom's (111 kg), redistributed. Pinned stand nodes weigh
# STAND_NODE_MASS; the three camera nodes CAMERA_NODE_MASS each (a 1.5 kg camera).
RING_MASS = {"base_pan": 4.0, "lift": 5.0, "elbow": 3.0, "tilt": 2.0}
AXIS_SHARE = 0.8
STAND_NODE_MASS = 3.0
CAMERA_NODE_MASS = 0.5
# Fixed-step budgets (see sizing.py): beams at most BEAM_CEILING per node,
# torsion bars at most TORSION_CEILING (the game scales bars above 0.8 and
# measures about 0.1 % above this builder, hence the margin below 0.7).
BEAM_CEILING = 2.0
BEAM_SPRING_SHARE = 0.6
BEAM_SPRING_MAX = 2.0e6
TORSION_CEILING = 0.68
# Damping ratio of every joint, chosen deliberately low. Per unit, damping
# costs 4000x stiffness in the budget (2 h C against h^2 K), so at a damping
# ratio of 0.15 over 95 % of the lift's budget would be damper. In that regime
# the sway's decay rate C / 2I is set by the budget and hardly depends on
# zeta, while K grows as 1 / zeta^2: 0.05 keeps the decay rate within about
# 15 % of its maximum and makes the lift several times stiffer than 0.15
# would (worst-case sag about 1.4 mrad, under a millimetre at the camera).
ZETA = 0.05
SAG_TARGET = 0.01         # rad, the worst-case sag a joint's need is sized for (only splits shared nodes)
K_FLOOR = 50.0            # N m/rad

MAST_Z = 1.15             # lift axis height
UPPER = 0.45              # lift -> elbow
FORE = 0.45               # elbow -> tilt
HEAD = 0.15               # tilt axis -> optical centre (straight down at home)
HOME_X = 0.55             # the camera's reach at home: over the workspace centre
CAM_FOV_DEG = 50.0        # vertical field of view
CAM_ASPECT = 4.0 / 3.0    # image width / height the fov and coverage are planned for
WORKSPACE_CENTER = (HOME_X, 0.0, 0.0)
WORKSPACE_SIZE = (0.6, 0.6)


def planar_ik(tx: float, tz: float, elbow_up: bool = True) -> tuple[float, float]:
    """The upper boom's pitch above horizontal and the elbow's bend (forearm
    pitch minus upper pitch) that put the tilt axis at (tx, tz) from the lift
    axis, in the boom's vertical plane; positive pitch raises the far end."""
    c = (tx * tx + tz * tz - UPPER ** 2 - FORE ** 2) / (2 * UPPER * FORE)
    if not -1.0 <= c <= 1.0:
        raise ValueError(f"tilt axis ({tx:.3f}, {tz:.3f}) from the lift is out of the boom's reach")
    bend = -math.acos(c) if elbow_up else math.acos(c)
    return math.atan2(tz, tx) - math.atan2(FORE * math.sin(bend), UPPER + FORE * math.cos(bend)), bend


# The spawn (= home) pose: the elbow raised, the tilt axis HOME_X out at the lift's height.
LIFT_0, BEND_0 = planar_ik(HOME_X, 0.0)
ELBOW_AT = (UPPER * math.cos(LIFT_0), 0.0, MAST_Z + UPPER * math.sin(LIFT_0))
TILT_AT = (HOME_X, 0.0, MAST_Z)

# name, parent link, child link, axis point, axis direction, limits (deg), lever (m), axis half-length (m).
# Lift, elbow and tilt pitch about -y: positive raises the far end, as the test arm's shoulder.
JOINTS = [
    ("base_pan", "root", "turret", (0.0, 0.0, 0.85), (0.0, 0.0, 1.0), (-170, 170), 0.20, 0.15),
    ("lift", "turret", "upper", (0.0, 0.0, MAST_Z), (0.0, -1.0, 0.0), (-60, 60), 0.20, 0.15),
    ("elbow", "upper", "forearm", ELBOW_AT, (0.0, -1.0, 0.0), (-60, 95), 0.16, 0.12),
    ("tilt", "forearm", "head", TILT_AT, (0.0, -1.0, 0.0), (-120, 120), 0.12, 0.10),
]
LEVERS = {j[0]: j[6] for j in JOINTS}
LINKS = ["root", "turret", "upper", "forearm", "head"]
PITCH_JOINTS = ("lift", "elbow", "tilt")   # camera pitch = their sum; nadir when it is 0

CAM = (HOME_X, 0.0, MAST_Z - HEAD)
CAM_FWD = (CAM[0], 0.0, CAM[2] - 0.1)
CAM_UP = (CAM[0] + 0.1, 0.0, CAM[2])
DRIVER_EYE = (0.0, 0.0, 0.05)
MAST_TOP = {"m0", "m1", "m2", "m3"}
# The camera frame's axes in the basis of the frame nodes (cam, cam_fwd, cam_up):
# node x = optical axis, node y = image up, node z = x cross y = image right.
# Columns are the camera's x (image right), y (image down), z (optical axis); row-major.
EE_ROTATION = [0.0, 0.0, 1.0,
               0.0, -1.0, 0.0,
               1.0, 0.0, 0.0]


class Boom(Arm):
    """The test arm's node/beam/hydro containers plus per-node mass, sized
    beam and bar coefficients and the per-joint dynamics."""

    def __init__(self):
        super().__init__()
        self.mass: dict[str, float] = {}
        self.beam_coeffs: dict[tuple[str, str], tuple[float, float]] = {}
        self.joint_dynamics: dict[str, dict] = {}

    def node(self, name, p, link, mass, fixed=False):
        self.mass[name] = mass
        return super().node(name, p, link, fixed)


def build() -> Boom:
    boom = Boom()
    # The pinned stand: a 0.4 m base plate and a mast up to the pan bearing.
    for i, (x, y) in enumerate(((0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2))):
        boom.node(f"b{i}", (x, y, 0.0), "root", STAND_NODE_MASS, fixed=True)
    for i, (x, y) in enumerate(((0.08, 0.08), (-0.08, 0.08), (-0.08, -0.08), (0.08, -0.08))):
        boom.node(f"m{i}", (x, y, 0.55), "root", STAND_NODE_MASS, fixed=True)
    boom.node("bref", (0.0, 0.0, 0.0), "root", STAND_NODE_MASS, fixed=True)
    boom.node("bback", (-0.2, 0.0, 0.0), "root", STAND_NODE_MASS, fixed=True)
    boom.node("bleft", (0.0, 0.2, 0.0), "root", STAND_NODE_MASS, fixed=True)
    boom.node("bup", (0.0, 0.0, 0.1), "root", STAND_NODE_MASS, fixed=True)
    axes = {}
    for name, parent, child, p, axis, (lo, hi), lever, half in JOINTS:
        e1, e2 = basis(axis)
        ring, hub = RING_MASS[name], round(AXIS_SHARE * RING_MASS[name], 6)
        pos = boom.node(f"{name}_ap", add(p, mul(axis, half)), parent, hub)
        neg = boom.node(f"{name}_an", add(p, mul(axis, -half)), parent, hub)
        base, moving = [], []
        for k in range(4):
            ang = k * math.pi / 2
            radial = add(mul(e1, math.cos(ang)), mul(e2, math.sin(ang)))
            base.append(boom.node(f"{name}_p{k}", add(p, mul(axis, -0.03), mul(radial, lever)), parent, ring))
            moving.append(boom.node(f"{name}_c{k}", add(p, mul(axis, 0.03), mul(radial, lever)), child, ring))
        axes[child] = (pos, neg)
        lo_r, hi_r = math.radians(lo), math.radians(hi)
        for k in range(4):
            boom.hydros.append({"nodes": [base[k], neg, pos, moving[k]], "joint": name, "lo": lo_r, "hi": hi_r})
        boom.spec_joints.append({"name": name, "axis": [neg, pos], "parent_ref": base[0], "child_ref": moving[0],
                                 "sign": 1.0, "lower": round(lo_r, 6), "upper": round(hi_r, 6), "reference": 0.0})
    # The camera head: optical centre, a point along the optical axis, a point toward image up.
    boom.node("cam", CAM, "head", CAMERA_NODE_MASS)
    boom.node("cam_fwd", CAM_FWD, "head", CAMERA_NODE_MASS)
    boom.node("cam_up", CAM_UP, "head", CAMERA_NODE_MASS)
    by_link = links(boom)
    for members in by_link.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pinned = (a in boom.fixed) + (b in boom.fixed)
                # Root bearing nodes brace to each other and to the mast top only,
                # which keeps their beam count (and so each beam's slice) moderate.
                if pinned == 0 or (pinned == 1 and ({a, b} & MAST_TOP)):
                    boom.beams.add((a, b))
    for child, (pos, neg) in axes.items():
        for n in by_link[child]:
            boom.beams.add((n, pos))
            boom.beams.add((n, neg))
    for a, b in (("b0", "b2"), ("b1", "b3"), ("b0", "bup"), ("b2", "bup"), ("bref", "bup"),
                 ("b0", "m0"), ("b1", "m1"), ("b2", "m2"), ("b3", "m3"), ("m0", "m2"), ("m1", "m3"),
                 ("bref", "bback"), ("bref", "bleft"), ("bback", "bleft"), ("bback", "b1"), ("bleft", "b0")):
        boom.beams.add((a, b))
    size(boom)
    return boom


def size(boom: Boom) -> None:
    """Beams and torsion bars within the fixed-step budgets (``sizing.py``),
    then the per-joint dynamics."""
    boom.beam_coeffs = size_beams({tuple(sorted(b)) for b in boom.beams}, boom.mass, BEAM_CEILING,
                                  BEAM_SPRING_SHARE, BEAM_SPRING_MAX)
    asks = []
    for i, (name, _parent, child, p, axis, *_rest) in enumerate(JOINTS):
        moved = [n for n in boom.nodes if boom.link[n] in LINKS[LINKS.index(child):]]
        inertia = inertia_about(boom.nodes, boom.mass, moved, p, axis)
        m_moved = sum(boom.mass[n] for n in moved)
        torque_max = gravity_torque_bound(m_moved, inertia)
        bars = [h["nodes"] for h in boom.hydros if h["joint"] == name]
        asks.append({"name": name, "bars": bars, "inertia": inertia, "moved_mass": m_moved,
                     "torque_max": torque_max, "torque_home": home_gravity_torque(boom, i),
                     "need": joint_need(inertia, torque_max, SAG_TARGET, ZETA, K_FLOOR)})
    sized = size_torsion(boom.nodes, boom.mass, asks, TORSION_CEILING, ZETA)
    by_bar = {}
    for ask, s in zip(asks, sized):
        for ns, kc in zip(ask["bars"], s["bars"]):
            by_bar[tuple(ns)] = kc
        boom.joint_dynamics[ask["name"]] = {
            "lever": LEVERS[ask["name"]], "inertia": ask["inertia"], "moved_mass": ask["moved_mass"],
            "spring": s["spring"], "damp": s["damp"],
            **joint_dynamics(s["spring"], s["damp"], ask["inertia"], ask["torque_max"]),
            "gravity_torque_max": ask["torque_max"], "gravity_torque_home": ask["torque_home"],
            "gravity_sag_home": abs(ask["torque_home"]) / s["spring"]}
    for h in boom.hydros:
        h["spring"], h["damp"] = by_bar[tuple(h["nodes"])]


def home_gravity_torque(boom: Boom, i: int) -> float:
    """Gravity torque (N m, about the joint's +axis) on joint i at the spawn pose."""
    name, _parent, child, p, axis, *_ = JOINTS[i]
    carried = set(LINKS[LINKS.index(child):])
    torque = 0.0
    for n, link in boom.link.items():
        if link in carried:
            torque += dot(cross(add(boom.nodes[n], mul(p, -1.0)), (0.0, 0.0, -GRAVITY * boom.mass[n])), axis)
    return torque


def node_loads(boom: Boom) -> dict[str, dict[str, float]]:
    """Per node: the beams' and torsion bars' share of the fixed-step criterion."""
    return _node_loads(boom.nodes, boom.mass, boom.beam_coeffs,
                       ((h["nodes"], h["spring"], h["damp"]) for h in boom.hydros))


def links(arm: Arm) -> dict[str, list[str]]:
    by_link: dict[str, list[str]] = {}
    for n, l in arm.link.items():
        by_link.setdefault(l, []).append(n)
    return by_link


# --- rigid kinematics of the spawn geometry (zero = spawn pose) ---------------------------------------

def _rotate(v, axis, angle):
    """Rodrigues: v rotated right-handedly about the unit axis."""
    c, s = math.cos(angle), math.sin(angle)
    d = sum(a * b for a, b in zip(axis, v))
    k = cross(axis, v)
    return tuple(v[i] * c + k[i] * s + axis[i] * d * (1 - c) for i in range(3))


def fk(q: dict[str, float], points=None) -> dict[str, tuple]:
    """Rigid forward kinematics: where the head nodes (or ``points``, head-link
    positions at spawn) are for joint angles ``q`` (radians, missing = 0)."""
    points = points or {"cam": CAM, "cam_fwd": CAM_FWD, "cam_up": CAM_UP}
    out = {}
    for name, p in points.items():
        x = p
        for jname, _, _, origin, axis, *_ in reversed(JOINTS):
            x = add(_rotate(add(x, mul(origin, -1.0)), axis, q.get(jname, 0.0)), origin)
        out[name] = x
    return out


def nadir_joints(x: float, y: float, height: float, elbow_up: bool = True) -> dict[str, float]:
    """Joint angles that put the optical centre at base-frame (x, y, height),
    looking straight down (image up pointing away from the mast). Raises
    ValueError when the point is out of reach or a joint out of its limits."""
    rho = math.hypot(x, y)
    pan = math.atan2(y, x) if rho > 1e-9 else 0.0
    # The tilt axis sits HEAD above the camera; solve the planar lift/elbow pair for it.
    try:
        lift_abs, bend = planar_ik(rho, height + HEAD - MAST_Z, elbow_up)
    except ValueError:
        raise ValueError(f"camera point ({x:.3f}, {y:.3f}, {height:.3f}) is out of the boom's reach") from None
    lift, elbow = lift_abs - LIFT_0, bend - BEND_0
    q = {"base_pan": pan, "lift": lift, "elbow": elbow, "tilt": -(lift + elbow)}
    for name, *_, (lo, hi), _lever, _half in JOINTS:
        if not math.radians(lo) <= q[name] <= math.radians(hi):
            raise ValueError(f"{name} = {math.degrees(q[name]):.1f} deg is outside [{lo}, {hi}]")
    return q


# --- sizing report -------------------------------------------------------------------------------------

def analysis(boom: Boom) -> dict:
    loads = node_loads(boom)
    worst_beam = max(loads, key=lambda n: (loads[n]["beams"], n))
    worst_torsion = max(loads, key=lambda n: (loads[n]["torsion"], n))
    return {
        "nodes": len(boom.nodes), "beams": len(boom.beams), "torsion_hydros": len(boom.hydros),
        "moving_mass_kg": round(sum(m for n, m in boom.mass.items() if boom.link[n] != "root"), 3),
        "worst_beam_load": [worst_beam, round(loads[worst_beam]["beams"], 4)],
        "worst_torsion_load": [worst_torsion, round(loads[worst_torsion]["torsion"], 4)],
        "joints_dynamics": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in boom.joint_dynamics.items()},
    }


def report_table(boom: Boom) -> str:
    rows = [f"{'joint':10s} {'lever m':>7s} {'I kgm2':>8s} {'K Nm/rad':>9s} {'C Nms':>7s} {'w rad/s':>8s} "
            f"{'zeta':>5s} {'sag home':>9s} {'sag max':>8s}"]
    for name, d in boom.joint_dynamics.items():
        rows.append(f"{name:10s} {d['lever']:7.2f} {d['inertia']:8.4f} {d['spring']:9.1f} {d['damp']:7.3f} "
                    f"{d['omega']:8.1f} {d['zeta']:5.2f} {d['gravity_sag_home']:9.5f} {d['gravity_sag_max']:8.5f}")
    return "\n".join(rows)


def driver_eye_anchors(arm: Arm, k: int = 8) -> list[str]:
    """The k structural nodes nearest the driver eye (exact distance, then node
    order), which the importer springs the eye node to."""
    ranked = sorted((sum((a - b) ** 2 for a, b in zip(arm.nodes[n], DRIVER_EYE)), i, n)
                    for i, n in enumerate(arm.nodes))
    return [n for d, _, n in ranked if d > 1e-12][:k]


# --- package ------------------------------------------------------------------------------------------

def visuals(boom: Boom | None = None) -> list:
    """One flat-coloured mesh per link, bound to the link's node group
    (``visuals.py``): the grey stand, dark boom tubes, an orange housing per
    joint (on its parent link) and a black camera with its lens pointing
    along the optical axis on the head."""
    meshes = {link: Mesh(f"{NAME}_{link}", f"boom_{link}") for link in LINKS}
    x, y, z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    root, turret, upper, fore, head = (meshes[k] for k in LINKS)
    lift, tilt = (0.0, 0.0, MAST_Z), TILT_AT
    root.box((0.0, 0.0, 0.015), (x, y, z), (0.2, 0.2, 0.015), "stand")
    root.prism((0.0, 0.0, 0.03), (0.0, 0.0, 0.79), 0.05, "stand")
    root.housing((0.0, 0.0, 0.80), z, 0.04, 0.10)
    turret.prism((0.0, 0.0, 0.85), (0.0, 0.0, 0.91), 0.12, "link")
    turret.prism((0.0, 0.0, 0.91), (0.0, 0.0, MAST_Z - 0.06), 0.05, "link")
    turret.housing(lift, y, 0.16, 0.075)
    upper.bar(lift, ELBOW_AT, 0.06, 0.06, "link", hint=y)
    upper.housing(ELBOW_AT, y, 0.12, 0.06)
    fore.bar(ELBOW_AT, tilt, 0.05, 0.05, "link", hint=y)
    fore.housing(tilt, y, 0.10, 0.05)
    # The head: a hanger from the tilt axis down to the camera body, the body
    # just above the optical centre and the lens down to it.
    head.bar(tilt, (CAM[0], 0.0, CAM[2] + 0.08), 0.03, 0.04, "link", hint=y)
    head.box((CAM[0], 0.0, CAM[2] + 0.045), (x, y, z), (0.06, 0.05, 0.035), "camera")
    head.prism((CAM[0], 0.0, CAM[2] + 0.01), CAM, 0.025, "lens")
    return [meshes[k] for k in LINKS]


def jbeam_text(arm: Boom) -> str:
    nodes = [["id", "posX", "posY", "posZ"], {"collision": False}, {"selfCollision": False},
             {"frictionCoef": 0.5}]
    weight = None
    for link in LINKS:
        nodes.append({"group": f"boom_{link}"})
        for n, l in arm.link.items():
            if l != link:
                continue
            if arm.mass[n] != weight:
                weight = arm.mass[n]
                nodes.append({"nodeWeight": weight})
            x, y, z = to_jbeam(arm.nodes[n])
            row = [n, x, y, z]
            if n in arm.fixed:
                row.append({"fixed": True})
            nodes.append(row)
    nodes.append({"group": ""})
    beams = [["id1:", "id2:"], {"beamDeform": 1.0e9, "beamStrength": 1.0e12}]
    current = None
    for (a, b), (k, c) in sorted(arm.beam_coeffs.items(), key=lambda t: (t[1], t[0])):
        if (k, c) != current:
            beams.append({"beamSpring": k, "beamDamp": c})
            current = (k, c)
        beams.append([a, b])
    hydros = [["id1:", "id2:", "id3:", "id4:"]]
    for h in arm.hydros:
        j = h["joint"]
        hydros.append(h["nodes"] + [{
            "spring": h["spring"], "damp": h["damp"], "inputSource": f"boom_cmd_{j}",
            "inLimit": round(h["lo"], 6), "outLimit": round(h["hi"], 6),
            "inputInLimit": round(h["lo"], 6), "inputCenter": 0, "inputOutLimit": round(h["hi"], 6),
            "inRate": RATE, "outRate": RATE, "autoCenterRate": RATE}])
    ex, ey, ez = to_jbeam(DRIVER_EYE)
    part = {
        "information": {"authors": "OpenBeam, LLC (openbeam-gym)", "name": "OpenBeam camera boom", "value": 0},
        "slotType": "main",
        "refNodes": [["ref:", "back:", "left:", "up:"], ["bref", "bback", "bleft", "bup"]],
        # The player's seat at the pinned base (see the module docstring); the
        # overhead camera is the node triple named in openbeam_robot.json.
        "camerasInternal": [["type", "x", "y", "z", "fov"], ["driver", ex, ey, ez, 60]],
        "nodes": nodes,
        "beams": beams,
        "torsionHydros": hydros,
        "flexbodies": flexbodies(visuals(arm)),
    }
    return json.dumps({NAME: part}, indent=1) + "\n"


def spec(arm: Arm) -> dict:
    return {
        "schema": "openbeam-robot/1",
        "name": NAME,
        "source": "openbeam-gym openbeam_gym/robots/build_camera_boom.py",
        "joints": arm.spec_joints,
        # The camera's optical centre, in OpenCV camera axes (x image right,
        # y image down, z optical axis), expressed in the frame of the nodes
        # (origin cam; x toward cam_fwd; y toward cam_up), as so101.draft.json.
        "end_effectors": [{"name": "camera", "nodes": ["cam", "cam_fwd", "cam_up"],
                           "offset": [0.0, 0.0, 0.0], "rotation": EE_ROTATION}],
        "targets": {"electrics": "boom_cmd_{joint}"},
        "camera": {
            "position_node": "cam",       # optical centre
            "target_node": "cam_fwd",     # a point on the optical axis
            "up_node": "cam_up",          # image up = up_node - position_node, orthogonalised
            "fov_deg": CAM_FOV_DEG,
            "fov_axis": "vertical",
            "aspect": round(CAM_ASPECT, 6),
            "end_effector": "camera",
            "home": {"joints": {j[0]: 0.0 for j in JOINTS}, "nadir": True},
            "nadir_when": "lift + elbow + tilt = 0",
            # Base frame (x forward, y left, z up), z = 0 the surface the boom stands on.
            "workspace": {"frame": "base: x forward, y left, z up; z = 0 the surface the boom stands on",
                          "center": list(WORKSPACE_CENTER), "size": list(WORKSPACE_SIZE),
                          "normal": [0.0, 0.0, 1.0]},
        },
    }


LUA = """-- Copyright 2026 OpenBeam, LLC
-- SPDX-License-Identifier: Apache-2.0
-- openbeam_camera_boom: publishes the joint state (generic robot block and
-- boom_<joint>_pos|vel|target) at the 60 Hz GFX cadence and keeps the boom
-- awake (a vehicle with Lua never auto-sleeps). The torsion hydros follow
-- boom_cmd_<joint> themselves, every physics step. Keys nudge the goals; with
-- the nadir lock on, lift/elbow keys also move the tilt goal so the camera
-- keeps its pitch (straight down after home).
-- Generated by openbeam_gym/robots/build_camera_boom.py.
local M = {}
local JOINTS = __JOINTS__
local TILT = "tilt"
local ids, last = {}, {}

local function id(name)
  local nodes = v and v.data and v.data.nodes
  if not nodes then return nil end
  for k, n in pairs(nodes) do
    if type(n) == "table" and n.name == name then return n.cid or k end
  end
  return nil
end

local function sub(a, b) return {a.x - b.x, a.y - b.y, a.z - b.z} end
local function dot(a, b) return a[1] * b[1] + a[2] * b[2] + a[3] * b[3] end
local function crossv(a, b) return {a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]} end

local function angle(j)
  local n = ids[j.name]
  if not n then return nil end
  local an, ap = obj:getNodePosition(n.an), obj:getNodePosition(n.ap)
  local axis = sub(ap, an)
  local l = math.sqrt(dot(axis, axis))
  if l < 1e-9 then return nil end
  axis = {axis[1] / l, axis[2] / l, axis[3] / l}
  local function lever(node)
    local r = sub(obj:getNodePosition(node), an)
    local d = dot(r, axis)
    return {r[1] - d * axis[1], r[2] - d * axis[2], r[3] - d * axis[3]}
  end
  local a, b = lever(n.parent), lever(n.child)
  return math.atan2(dot(crossv(a, b), axis), dot(a, b)) - n.offset
end

local function clamp(j, x) return math.max(j.lower, math.min(j.upper, x)) end

local function init()
  ids, last = {}, {}
  for i, j in ipairs(JOINTS) do
    local n = {an = id(j.axis[1]), ap = id(j.axis[2]), parent = id(j.parent_ref), child = id(j.child_ref), offset = 0}
    if n.an and n.ap and n.parent and n.child then
      ids[j.name] = n
      n.offset = angle(j) or 0
    end
    electrics.values["boom_cmd_" .. j.name] = 0
  end
  electrics.values.robot_dof = #JOINTS
  electrics.values.boom_nadir_lock = 1
end

local function teleop(dt)
  if (electrics.values.boom_key_home or 0) > 0 then
    electrics.values.boom_key_home = 0
    for _, j in ipairs(JOINTS) do electrics.values["boom_cmd_" .. j.name] = 0 end
    electrics.values.boom_nadir_lock = 1
    return
  end
  local pitch, tilt = 0, nil
  for _, j in ipairs(JOINTS) do
    if j.name == TILT then tilt = j end
    local d = (electrics.values["boom_key_" .. j.name .. "_p"] or 0) - (electrics.values["boom_key_" .. j.name .. "_n"] or 0)
    if d ~= 0 then
      local k = "boom_cmd_" .. j.name
      local old = electrics.values[k] or 0
      local new = clamp(j, old + d * __RATE__ * dt)
      electrics.values[k] = new
      if j.pitch then pitch = pitch + (new - old) end
      if j.name == TILT then electrics.values.boom_nadir_lock = 0 end
    end
  end
  if pitch ~= 0 and tilt and (electrics.values.boom_nadir_lock or 0) ~= 0 then
    local k = "boom_cmd_" .. TILT
    electrics.values[k] = clamp(tilt, (electrics.values[k] or 0) - pitch)
  end
end

local function updateGFX(dt)
  teleop(dt)
  for i, j in ipairs(JOINTS) do
    local q = angle(j)
    if q then
      local qd = (last[i] and dt > 0) and (q - last[i]) / dt or 0
      last[i] = q
      local cmd = electrics.values["boom_cmd_" .. j.name] or 0
      electrics.values["boom_" .. j.name .. "_pos"] = q
      electrics.values["boom_" .. j.name .. "_vel"] = qd
      electrics.values["boom_" .. j.name .. "_target"] = cmd
      electrics.values["robot_q" .. i] = q
      electrics.values["robot_qd" .. i] = qd
      electrics.values["robot_cmd" .. i] = cmd
    end
  end
end

M.onInit = init
M.onReset = init
M.updateGFX = updateGFX
return M
"""


def lua_text(arm: Arm) -> str:
    rows = ", ".join(
        '{name = "%s", axis = {"%s", "%s"}, parent_ref = "%s", child_ref = "%s", lower = %r, upper = %r, pitch = %s}'
        % (j["name"], j["axis"][0], j["axis"][1], j["parent_ref"], j["child_ref"], j["lower"], j["upper"],
           "true" if j["name"] in ("lift", "elbow") else "false")
        for j in arm.spec_joints)
    return LUA.replace("__JOINTS__", "{" + rows + "}").replace("__RATE__", repr(RATE))


KEYS = [("base_pan", "d", "a"), ("lift", "w", "s"), ("elbow", "e", "q"), ("tilt", "x", "z")]


def input_files() -> dict[str, bytes]:
    actions, bindings = {}, []
    for joint, plus, minus in KEYS:
        for sign, key in (("p", plus), ("n", minus)):
            name = f"boom_{joint}_{sign}"
            e = f"electrics.values.boom_key_{joint}_{sign}"
            actions[name] = {"cat": "vehicle_specific", "order": len(actions), "ctx": "vlua",
                             "onDown": f"{e} = 1", "onUp": f"{e} = 0",
                             "title": f"Camera boom {joint} {'+' if sign == 'p' else '-'}",
                             "desc": f"Moves the {joint} goal while held"}
            bindings.append({"control": key, "action": name})
    actions["boom_home"] = {"cat": "vehicle_specific", "order": len(actions), "ctx": "vlua",
                            "onDown": "electrics.values.boom_key_home = 1", "title": "Camera boom home pose",
                            "desc": "Points the camera straight down over the default workspace"}
    bindings.append({"control": "3", "action": "boom_home"})
    return {
        f"vehicles/{NAME}/input_actions_{NAME}.json": (json.dumps(actions, indent=2) + "\n").encode(),
        f"vehicles/{NAME}/inputmaps/keyboard.json": (json.dumps({"bindings": bindings}, indent=2) + "\n").encode(),
    }


def info_json() -> str:
    return json.dumps({"Name": "OpenBeam camera boom", "Brand": "OpenBeam", "Type": "Prop",
                       "Author": "OpenBeam, LLC (openbeam-gym)",
                       "Description": "An overhead camera boom for top-down teleoperation (base pan, lift, elbow, "
                                      "camera tilt), driven by torsion hydros on the electrics boom_cmd_<joint>."},
                      indent=2) + "\n"


def members(arm: Boom | None = None) -> dict[str, bytes]:
    arm = arm or build()
    prefix = f"vehicles/{NAME}"
    return {
        f"{prefix}/{NAME}.jbeam": jbeam_text(arm).encode(),
        f"{prefix}/openbeam_robot.json": (json.dumps(spec(arm), indent=1) + "\n").encode(),
        f"{prefix}/lua/{NAME}.lua": lua_text(arm).encode(),
        f"{prefix}/info.json": info_json().encode(),
        **input_files(),
        **package_members(prefix, NAME, visuals(arm)),
        f"{prefix}/LICENSE.txt": b"Copyright 2026 OpenBeam, LLC. Licensed under the Apache License, Version 2.0.\n",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("assets") / f"{NAME}.zip")
    ap.add_argument("--report", action="store_true", help="print the per-joint stiffness/damping/sag table")
    args = ap.parse_args()
    boom = build()
    digest = write_zip(args.out, members(boom))
    a = analysis(boom)
    print(json.dumps({"zip": str(args.out), "sha256": digest, **{k: v for k, v in a.items() if k != "joints_dynamics"}}))
    if args.report:
        print(report_table(boom))


if __name__ == "__main__":
    main()
