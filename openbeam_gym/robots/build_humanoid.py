# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Builds assets/openbeam_humanoid_g1.zip: a G1-class humanoid as a JBeam
vehicle mod (``vehicles/openbeam_humanoid_g1/``) that the game's
``install_mod`` takes as it is.

G1-class means the publicly described proportions and joint layout of that
robot class, not its design files: about 1.3 m tall, about 35 kg, six joints
per leg (hip pitch, roll, yaw, knee, ankle pitch, roll), a waist (yaw, plus
roll and pitch here), five per arm (shoulder pitch, roll, yaw, elbow, wrist
roll): 25 joints. The geometry is this file's own: node clusters, and simple
generated visual meshes per link (``visuals.py``).
The joint table (names, axes, limits, rates, home pose) is in
``humanoid_g1.py``.

Construction, as the test arm (``build_test_arm.py``):

* Every link is a cluster of nodes braced by every pairwise beam. Each joint is
  a hinge: two axis nodes in the parent link, to which every node of the child
  link is tied.
* Each joint has two lever rings, as the test arm: two parent-link nodes and
  two child-link nodes on opposite sides of the axis, at the same radius and
  direction, 3 cm apart along it. Two JBeam ``torsionHydros`` (one per
  aligned parent/child pair) drive the joint; their rest angle follows the
  electric ``g1_cmd_<joint>`` (radians from the spawn pose) at the joint's
  rate. The ring nodes carry most of each link's mass, like the motors of a
  real robot, because a bar's fixed-step budget is its lever nodes' m r^2.
* **v1 is fixed-base** (``STAND = True``): the pelvis is braced to a heavy
  pedestal behind it (``STAND_MASS`` kg, part of the vehicle, resting on the
  ground on four collision nodes), so the body follows poses without balance
  control. ``PELVIS_BALLAST`` kg of the pedestal's mass sits on the pelvis
  nodes (they are rigidly mounted, so it only stiffens the hip and waist
  drives); the body itself is about 35 kg. The soles hang ``FOOT_CLEARANCE``
  above the ground; the foot nodes collide, so a toe pointed far enough down
  rests on the ground lightly. ``build(stand=False)`` leaves the pedestal and
  the ballast out: a free-standing body that needs a balance controller.

Fixed-step sizing (0.5 ms; the shared law and solvers are in ``sizing.py``).
The game holds every node to the explicit
criterion ``omega^2 h^2 + 2 c h / m <= 4``: it lowers beams to a 2.6 share and
scales the torsion bars on a node to a 0.8 share (per bar
``(h^2 k + 2 h c) |g_i| sum_j |g_j| / m_i``, g the gradient of the bar's
angle). This builder sizes everything below those caps itself, so the game
scales nothing at spawn: beams use at most ``BEAM_CEILING`` per node (spring
and damper split ``BEAM_SPRING_SHARE``), torsion bars at most
``TORSION_CEILING``. The torsion budget is shared out by progressive filling,
in proportion to what each joint needs to hold its worst-case gravity torque
within ``SAG_TARGET`` at damping ratio ``ZETA``; each joint's share is then
split between spring and damper for ``ZETA`` against the inertia of
everything it moves. Explicit damping is the expensive part (2 h c against
h^2 k), which is why the rings are wide. ``analysis()`` reports stiffness,
damping, natural frequency and worst-case gravity sag per joint
(``--report``).

* ``openbeam_robot.json`` (``openbeam-robot/1``) declares the joints (axis
  nodes, reference levers, limits), the pelvis base frame, end effectors for
  both hands and both feet, ``targets.electrics`` ``g1_cmd_{joint}``, the
  ``home`` pose and the ``retarget`` block (``openbeam_gym/retarget.py``: how
  each joint maps from MediaPipe BlazePose 33-landmark world coordinates).
* The vehicle Lua keeps the robot awake, publishes the generic robot block
  (``robot_dof``, ``robot_q<i>``, ``robot_qd<i>``, ``robot_cmd<i>``) and
  ``g1_<joint>_pos|vel|target`` at 60 Hz, and has a bulk setter for streamed
  targets: one command per frame,
  ``extensions.get("openbeam_humanoid_g1").set({q1, ..., q25})`` (joint order
  of the spec) or ``.set({left_knee = 0.4, ...})``; values are clamped to the
  limits. Keys: 3 home pose, 4 zero pose, A/D waist yaw (the bindings also
  make it a seat for the player, which stepped physics needs).

    python -m openbeam_gym.robots.build_humanoid --out assets/openbeam_humanoid_g1.zip [--report]

Stdlib only; the zip is deterministic (fixed timestamps, sorted members).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import humanoid_g1
from .build_test_arm import write_zip
from .sizing import (add, cross, dot, gravity_torque_bound, inertia_about, joint_need, mul, norm, size_beams,
                     size_torsion, sub, unit)
from .sizing import dihedral_gradient  # noqa: F401  (re-exported: the bars' angle gradient)
from .visuals import Mesh, flexbodies, package_members
from .sizing import dynamics as joint_dynamics
from .sizing import node_loads as _node_loads

NAME = "openbeam_humanoid_g1"
STAND = True               # v1: fixed base on a pedestal
STAND_MASS = 150.0         # kg, the pedestal, including the pelvis ballast
PELVIS_BALLAST = 30.0      # kg of the pedestal carried on the (mounted) pelvis nodes
FOOT_CLEARANCE = 0.05      # m, soles above the ground when on the pedestal
BEAM_CEILING = 2.0         # per-node beam share of the criterion (the game lowers beams above 2.6)
BEAM_SPRING_SHARE = 0.6    # of a beam's share: spring (the rest is its damper)
BEAM_SPRING_MAX = 2.0e6    # N/m
TORSION_CEILING = 0.7      # per-node torsion share (the game scales bars above 0.8)
ZETA = 0.1                 # joint damping ratio the torsion budget is split for
SAG_TARGET = 0.05          # rad, worst-case gravity sag each joint's need is sized for
K_FLOOR = 20.0             # N m/rad, the least stiffness any joint is asked to need
RING_DELTA = 0.015         # m, half the axial gap between a joint's parent and child rings
CMD = "g1_cmd_{joint}"

# kg per link (35.1 kg in all). Within a link, ring nodes weigh RING_WEIGHT,
# axis nodes AXIS_WEIGHT and other nodes 1.
LINK_MASS = {
    "pelvis": 3.5, "hip_pitch_link": 1.8, "hip_roll_link": 1.8, "thigh": 1.9, "shin": 1.4,
    "ankle_link": 0.3, "foot": 0.5, "waist_yaw_link": 1.5, "waist_roll_link": 1.5, "torso": 6.6,
    "shoulder_pitch_link": 0.8, "shoulder_roll_link": 0.8, "upper_arm": 0.8, "forearm": 0.6, "hand": 0.3,
}
RING_WEIGHT = 3.0
AXIS_WEIGHT = 1.5


def to_jbeam(p):
    """Base frame (x forward, y left, z up) -> JBeam (x left, y back, z up)."""
    return (round(p[1], 5), round(-p[0], 5), round(p[2], 5))


class Robot:
    def __init__(self, stand: bool):
        self.stand = stand
        self.nodes: dict[str, tuple] = {}
        self.link: dict[str, str] = {}          # node -> link instance (e.g. left_thigh)
        self.kind: dict[str, str] = {}          # link instance -> LINK_MASS key
        self.weight: dict[str, float] = {}      # node -> share of its link's mass
        self.colliding: set[str] = set()
        self.joints: list[dict] = []
        self.beams: dict[tuple[str, str], tuple[float, float]] = {}
        self.mass: dict[str, float] = {}
        self.ballast = 0.0
        self.hydros: list[dict] = []
        self.spec_joints: list[dict] = []
        self.joint_dynamics: dict[str, dict] = {}

    def node(self, name, p, link, kind=None, weight=1.0):
        assert name not in self.nodes, name
        z = 0.0 if link == "stand" else FOOT_CLEARANCE
        self.nodes[name] = (p[0], p[1], p[2] + z)
        self.link[name] = link
        self.weight[name] = weight
        self.kind.setdefault(link, kind or link)
        return name

    def joint(self, name, parent, child, center, hw, ring_dir, ring_r, child_kind):
        row = humanoid_g1.JOINTS[humanoid_g1.JOINT_NAMES.index(name)]
        axis = row["axis"]
        e = unit(ring_dir)
        if name.startswith("right_") and abs(e[1]) > 0.5:
            e = mul(e, -1.0)        # ring 0 mirrors ring 0 of the left side
        assert abs(dot(e, axis)) < 1e-9, name
        self.kind.setdefault(child, child_kind)
        neg = self.node(f"{name}_an", add(center, mul(axis, -hw)), parent, weight=AXIS_WEIGHT)
        pos = self.node(f"{name}_ap", add(center, mul(axis, hw)), parent, weight=AXIS_WEIGHT)
        # The rings' axial gap mirrors with the body: on the right side a
        # left-right (y) axis keeps its direction, so the gap is flipped.
        gap = mul(axis, RING_DELTA * (-1.0 if name.startswith("right_") and abs(axis[1]) > 0.5 else 1.0))
        bars = []
        for k, sgn in ((0, 1.0), (1, -1.0)):
            radial = mul(e, sgn * ring_r)
            p = self.node(f"{name}_p{k}", add(center, radial, mul(gap, -1.0)), parent, weight=RING_WEIGHT)
            c = self.node(f"{name}_c{k}", add(center, radial, gap), child, weight=RING_WEIGHT)
            bars.append((p, c))
        self.joints.append({"name": name, "parent": parent, "child": child, "axis": axis,
                            "center": (center[0], center[1], center[2] + FOOT_CLEARANCE),
                            "neg": neg, "pos": pos, "bars": bars, "ring_r": ring_r, "row": row})


X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def geometry(stand: bool = STAND) -> Robot:
    """Nodes, links and joints in the authored (zero) pose; soles at z = 0
    before FOOT_CLEARANCE, head top at 1.32 m."""
    r = Robot(stand)
    for name, p in (("pv_c", (0.0, 0.0, 0.76)), ("pv_f", (0.09, 0.0, 0.76)), ("pv_b", (-0.09, 0.0, 0.76)),
                    ("pv_l", (0.0, 0.10, 0.76)), ("pv_r", (0.0, -0.10, 0.76))):
        r.node(name, p, "pelvis")
    for side, s in (("left", 1.0), ("right", -1.0)):
        q = side[0] + "_"
        hpl, hrl, thigh, shin, ank, foot = (f"{side}_{k}" for k in
                                             ("hip_pitch_link", "hip_roll_link", "thigh", "shin", "ankle_link", "foot"))
        r.joint(f"{side}_hip_pitch", "pelvis", hpl, (0.0, s * 0.085, 0.72), 0.05, X, 0.16, "hip_pitch_link")
        r.joint(f"{side}_hip_roll", hpl, hrl, (0.0, s * 0.10, 0.66), 0.08, Z, 0.14, "hip_roll_link")
        r.joint(f"{side}_hip_yaw", hrl, thigh, (0.0, s * 0.10, 0.54), 0.07, X, 0.10, "thigh")
        r.joint(f"{side}_knee", thigh, shin, (0.0, s * 0.10, 0.38), 0.05, X, 0.10, "shin")
        r.joint(f"{side}_ankle_pitch", shin, ank, (0.0, s * 0.10, 0.08), 0.05, X, 0.08, "ankle_link")
        r.joint(f"{side}_ankle_roll", ank, foot, (0.0, s * 0.10, 0.045), 0.06, Y, 0.05, "foot")
        # Sole corners: inner (toward the body) and outer, toe and heel.
        for tag, x, y in (("ti", 0.15, 0.06), ("to", 0.15, 0.14), ("hi", -0.05, 0.06), ("ho", -0.05, 0.14)):
            r.colliding.add(r.node(q + "ft_" + tag, (x, s * y, 0.0), foot))
    r.joint("waist_yaw", "pelvis", "waist_yaw_link", (0.0, 0.0, 0.85), 0.05, Y, 0.14, "waist_yaw_link")
    r.joint("waist_roll", "waist_yaw_link", "waist_roll_link", (0.0, 0.0, 0.92), 0.07, Y, 0.14, "waist_roll_link")
    r.joint("waist_pitch", "waist_roll_link", "torso", (0.0, 0.0, 0.97), 0.07, X, 0.14, "torso")
    for name, p in (("to_fl", (0.07, 0.09, 1.06)), ("to_fr", (0.07, -0.09, 1.06)), ("to_bl", (-0.06, 0.09, 1.06)),
                    ("to_br", (-0.06, -0.09, 1.06)), ("neck", (0.0, 0.0, 1.16)), ("head_f", (0.06, 0.0, 1.24)),
                    ("head_b", (-0.05, 0.0, 1.24)), ("head_t", (0.0, 0.0, 1.32))):
        r.node(name, p, "torso")
    for side, s in (("left", 1.0), ("right", -1.0)):
        q = side[0] + "_"
        spl, srl, ua, fa, hand = (f"{side}_{k}" for k in
                                  ("shoulder_pitch_link", "shoulder_roll_link", "upper_arm", "forearm", "hand"))
        r.joint(f"{side}_shoulder_pitch", "torso", spl, (0.0, s * 0.13, 1.10), 0.04, X, 0.12, "shoulder_pitch_link")
        r.joint(f"{side}_shoulder_roll", spl, srl, (0.0, s * 0.16, 1.07), 0.06, Z, 0.10, "shoulder_roll_link")
        r.joint(f"{side}_shoulder_yaw", srl, ua, (0.0, s * 0.18, 0.98), 0.05, X, 0.06, "upper_arm")
        r.joint(f"{side}_elbow", ua, fa, (0.0, s * 0.18, 0.86), 0.04, X, 0.07, "forearm")
        r.joint(f"{side}_wrist_roll", fa, hand, (0.0, s * 0.18, 0.72), 0.04, X, 0.05, "hand")
        r.node(q + "h_tip", (0.0, s * 0.18, 0.58), hand)
        r.node(q + "h_in", (0.01, s * 0.155, 0.64), hand)
    if stand:
        # The pedestal: a ground frame open at the front (around the feet) and a
        # braced mast behind the pelvis. World coordinates, ground at z = 0.
        top = 0.76 + FOOT_CLEARANCE
        for name, p in (("st_gbl", (-0.55, 0.40, 0.0)), ("st_gbr", (-0.55, -0.40, 0.0)),
                        ("st_gfl", (0.30, 0.40, 0.0)), ("st_gfr", (0.30, -0.40, 0.0))):
            r.colliding.add(r.node(name, p, "stand"))
        for name, p in (("st_ml", (-0.30, 0.12, 0.0)), ("st_mr", (-0.30, -0.12, 0.0)), ("st_mb", (-0.45, 0.0, 0.0)),
                        ("st_tl", (-0.17, 0.10, top)), ("st_tr", (-0.17, -0.10, top)), ("st_tb", (-0.30, 0.0, top))):
            r.node(name, p, "stand")
    assert [j["name"] for j in r.joints] == humanoid_g1.JOINT_NAMES
    return r


def frame_of(o, x, y):
    """Node frame (o, x, y) as the spec reads it: columns ex, ey, ez."""
    ex = unit(sub(x, o))
    v = sub(y, o)
    ey = unit(sub(v, mul(ex, dot(ex, v))))
    return ex, ey, cross(ex, ey)


def frame_spec(r: Robot, nodes, point):
    """{nodes, offset, rotation}: a frame at `point` with the base frame's axes
    in the authored pose, expressed in the basis of three nodes."""
    o, x, y = (r.nodes[n] for n in nodes)
    ex, ey, ez = frame_of(o, x, y)
    rel = sub(point, o)
    offset = [round(dot(e, rel), 6) + 0.0 for e in (ex, ey, ez)]
    # rotation: the frame's axes (base x, y, z) in the node basis, row major.
    rotation = [round(e[i], 6) + 0.0 for e in (ex, ey, ez) for i in range(3)]
    return {"nodes": list(nodes), "offset": offset, "rotation": rotation}


# -- assembly ------------------------------------------------------------------

def _subtree(r: Robot, link: str) -> set[str]:
    out, todo = set(), [link]
    while todo:
        l = todo.pop()
        out.add(l)
        todo += [j["child"] for j in r.joints if j["parent"] == l]
    return out


def build(stand: bool = STAND) -> Robot:
    r = geometry(stand)
    members: dict[str, list[str]] = {}
    for n, l in r.link.items():
        members.setdefault(l, []).append(n)
    for l, ns in members.items():
        if l == "stand":
            total = STAND_MASS - PELVIS_BALLAST
        else:
            total = LINK_MASS[r.kind[l]] + (PELVIS_BALLAST if (l == "pelvis" and stand) else 0.0)
        weights = sum(r.weight[n] for n in ns)
        for n in ns:
            r.mass[n] = round(total * r.weight[n] / weights, 6)
    r.ballast = PELVIS_BALLAST if stand else 0.0
    # Beam topology: every pair within a link; every child-link node to its
    # joint's axis nodes; the pelvis body to the pedestal.
    pairs: set[tuple[str, str]] = set()
    for ns in members.values():
        for i, a in enumerate(ns):
            for b in ns[i + 1:]:
                pairs.add(tuple(sorted((a, b))))
    for j in r.joints:
        for n in members[j["child"]]:
            pairs.add(tuple(sorted((n, j["neg"]))))
            pairs.add(tuple(sorted((n, j["pos"]))))
    if stand:
        for p in ("pv_c", "pv_f", "pv_b", "pv_l", "pv_r"):
            for s in ("st_tl", "st_tr", "st_tb", "st_ml", "st_mr", "st_mb"):
                pairs.add(tuple(sorted((p, s))))
    r.beams.update(size_beams(pairs, r.mass, BEAM_CEILING, BEAM_SPRING_SHARE, BEAM_SPRING_MAX))
    # Per joint: the inertia it moves, its worst-case gravity torque, and the
    # torsion coefficient (h^2 K + 2 h C) that would hold that torque within
    # SAG_TARGET at damping ratio ZETA: its need.
    asks = []
    for j in r.joints:
        sub_links = _subtree(r, j["child"])
        sub_nodes = [n for n in r.nodes if r.link[n] in sub_links]
        inertia = inertia_about(r.nodes, r.mass, sub_nodes, j["center"], j["axis"])
        m_sub = sum(r.mass[n] for n in sub_nodes)
        # Worst-case gravity torque, bounded by the moved mass at its radius of
        # gyration (the limb held out level). Yaw joints of an upright body
        # carry none, but are sized the same way, for leaning poses.
        torque = gravity_torque_bound(m_sub, inertia)
        j.update(inertia=inertia, moved_mass=m_sub, gravity_torque=torque)
        asks.append({"bars": [[pl, j["neg"], j["pos"], cl] for pl, cl in j["bars"]], "inertia": inertia,
                     "need": joint_need(inertia, torque, SAG_TARGET, ZETA, K_FLOOR)})
    sized = size_torsion(r.nodes, r.mass, asks, TORSION_CEILING, ZETA)
    for j, ask, s in zip(r.joints, asks, sized):
        lo, hi = j["row"]["lower"], j["row"]["upper"]
        for ns, (k, c) in zip(ask["bars"], s["bars"]):
            r.hydros.append({"nodes": ns, "joint": j["name"], "spring": k, "damp": c, "lo": lo, "hi": hi,
                             "rate": j["row"]["rate"]})
        r.joint_dynamics[j["name"]] = {
            "bars": len(ask["bars"]), "ring_r": j["ring_r"], "inertia": j["inertia"], "moved_mass": j["moved_mass"],
            "spring": s["spring"], "damp": s["damp"],
            **joint_dynamics(s["spring"], s["damp"], j["inertia"], j["gravity_torque"]),
            "gravity_torque_max": j["gravity_torque"]}
        pl, cl = j["bars"][0]
        r.spec_joints.append({"name": j["name"], "axis": [j["neg"], j["pos"]], "parent_ref": pl, "child_ref": cl,
                              "sign": 1.0, "lower": lo, "upper": hi, "reference": 0.0})
    return r


def node_loads(r: Robot) -> dict[str, dict[str, float]]:
    """Per node: the beams' and the torsion bars' share of the fixed-step
    criterion omega^2 h^2 + 2 c h / m <= 4, as asked (before any game scaling)."""
    return _node_loads(r.nodes, r.mass, r.beams, ((h["nodes"], h["spring"], h["damp"]) for h in r.hydros))


def robot_mass(r: Robot) -> float:
    """The body's mass: every node but the pedestal's, less the pelvis ballast."""
    return sum(m for n, m in r.mass.items() if r.link[n] != "stand") - r.ballast


# -- visuals ------------------------------------------------------------------------

def visuals(r: Robot | None = None) -> list:
    """One flat-coloured mesh per link, bound to the link's node group
    (``visuals.py``): light shell limbs, torso and head, dark small links and
    feet, an orange housing per joint (on its parent link), a dark visor and
    a grey pedestal."""
    r = r or build()
    order = (["stand"] if r.stand else []) + ["pelvis"] + [j["child"] for j in r.joints]
    meshes = {link: Mesh(f"{NAME}_{link}", f"g1_{link}") for link in order}
    c = {j["name"]: j["center"] for j in r.joints}
    lift = FOOT_CLEARANCE

    def at(x, y, z):
        return (x, y, z + lift)

    xa, ya, za = X, Y, Z
    meshes["pelvis"].box(at(0.0, 0.0, 0.77), (xa, ya, za), (0.07, 0.11, 0.05), "shell")
    for side, s in (("left", 1.0), ("right", -1.0)):
        m = {k: meshes[f"{side}_{k}"] for k in ("hip_pitch_link", "hip_roll_link", "thigh", "shin", "ankle_link",
                                                "foot", "shoulder_pitch_link", "shoulder_roll_link", "upper_arm",
                                                "forearm", "hand")}
        q = side + "_"
        m["hip_pitch_link"].bar(c[q + "hip_pitch"], c[q + "hip_roll"], 0.06, 0.06, "link", hint=xa)
        m["hip_roll_link"].bar(c[q + "hip_roll"], c[q + "hip_yaw"], 0.07, 0.07, "link", hint=xa)
        m["thigh"].prism(c[q + "hip_yaw"], c[q + "knee"], 0.055, "shell", cap=0.02)
        m["shin"].prism(c[q + "knee"], c[q + "ankle_pitch"], 0.045, "shell", cap=0.015)
        m["ankle_link"].bar(c[q + "ankle_pitch"], c[q + "ankle_roll"], 0.04, 0.04, "link", hint=xa)
        m["foot"].box(at(0.05, s * 0.10, 0.02), (xa, ya, za), (0.10, 0.04, 0.02), "link")
        m["shoulder_pitch_link"].bar(c[q + "shoulder_pitch"], c[q + "shoulder_roll"], 0.05, 0.05, "link", hint=xa)
        m["shoulder_roll_link"].bar(c[q + "shoulder_roll"], c[q + "shoulder_yaw"], 0.05, 0.05, "link", hint=xa)
        m["upper_arm"].prism(c[q + "shoulder_yaw"], c[q + "elbow"], 0.035, "shell", cap=0.01)
        m["forearm"].prism(c[q + "elbow"], c[q + "wrist_roll"], 0.03, "shell", cap=0.01)
        m["hand"].bar(c[q + "wrist_roll"], at(0.0, s * 0.18, 0.59), 0.05, 0.025, "link", hint=xa)
    meshes["waist_yaw_link"].bar(c["waist_yaw"], c["waist_roll"], 0.10, 0.08, "link", hint=xa)
    meshes["waist_roll_link"].bar(c["waist_roll"], c["waist_pitch"], 0.10, 0.08, "link", hint=xa)
    torso = meshes["torso"]
    torso.box(at(0.005, 0.0, 1.06), (xa, ya, za), (0.075, 0.115, 0.07), "shell")
    torso.prism(at(0.0, 0.0, 1.13), at(0.0, 0.0, 1.18), 0.03, "link")
    torso.prism(at(0.0, 0.0, 1.20), at(0.0, 0.0, 1.27), 0.07, "shell", cap=0.03)
    torso.box(at(0.062, 0.0, 1.24), (xa, ya, za), (0.012, 0.05, 0.018), "lens")
    for j in r.joints:
        hw = norm(sub(r.nodes[j["pos"]], r.nodes[j["neg"]])) / 2
        radius = min(0.05, max(0.025, 0.35 * j["ring_r"]))
        meshes[j["parent"]].housing(j["center"], j["axis"], min(0.08, 1.6 * hw), radius)
    if r.stand:
        st, top = meshes["stand"], 0.76 + FOOT_CLEARANCE
        g = {n: r.nodes[n] for n in r.nodes if r.link[n] == "stand"}
        for a, b in (("st_gbl", "st_gbr"), ("st_gbl", "st_gfl"), ("st_gbr", "st_gfr")):
            st.bar(add(g[a], (0.0, 0.0, 0.02)), add(g[b], (0.0, 0.0, 0.02)), 0.05, 0.04, "stand", hint=za)
        st.prism((-0.30, 0.0, 0.0), (-0.30, 0.0, top), 0.04, "stand")
        st.bar((-0.30, 0.0, top), (-0.10, 0.0, top), 0.06, 0.05, "stand", hint=za)
    return [meshes[k] for k in order]


# -- package members --------------------------------------------------------------

def jbeam_text(r: Robot) -> str:
    nodes = [["id", "posX", "posY", "posZ"], {"collision": False}, {"selfCollision": False},
             {"frictionCoef": 0.5}]
    order = ["stand", "pelvis"] + [j["child"] for j in r.joints]
    weight = None
    for link in order:
        names = [n for n in r.nodes if r.link[n] == link]
        if not names:
            continue
        nodes.append({"group": f"g1_{link}"})
        for n in names:
            if r.mass[n] != weight:
                weight = r.mass[n]
                nodes.append({"nodeWeight": weight})
            x, y, z = to_jbeam(r.nodes[n])
            row = [n, x, y, z]
            if n in r.colliding:
                row.append({"collision": True, "frictionCoef": 1.0 if link == "stand" else 0.8})
            nodes.append(row)
    nodes.append({"group": ""})
    beams = [["id1:", "id2:"], {"beamDeform": 1.0e9, "beamStrength": 1.0e12}]
    current = None
    for (a, b), (k, c) in sorted(r.beams.items(), key=lambda t: (t[1], t[0])):
        if (k, c) != current:
            beams.append({"beamSpring": k, "beamDamp": c})
            current = (k, c)
        beams.append([a, b])
    hydros = [["id1:", "id2:", "id3:", "id4:"]]
    for h in r.hydros:
        lo, hi, rate = round(h["lo"], 6), round(h["hi"], 6), h["rate"]
        hydros.append(h["nodes"] + [{
            "spring": h["spring"], "damp": h["damp"], "inputSource": CMD.format(joint=h["joint"]),
            "inLimit": lo, "outLimit": hi, "inputInLimit": lo, "inputCenter": 0, "inputOutLimit": hi,
            "inRate": rate, "outRate": rate, "autoCenterRate": rate}])
    part = {
        "information": {"authors": "OpenBeam, LLC (openbeam-gym)", "name": "OpenBeam humanoid G1-class",
                        "value": 0},
        "slotType": "main",
        "refNodes": [["ref:", "back:", "left:", "up:"], ["pv_c", "pv_b", "pv_l", "waist_yaw_ap"]],
        "camerasInternal": [["type", "x", "y", "z", "fov"], ["driver", 0.0, -0.06, 1.24 + FOOT_CLEARANCE, 70]],
        "nodes": nodes,
        "beams": beams,
        "torsionHydros": hydros,
        "flexbodies": flexbodies(visuals(r)),
    }
    return json.dumps({NAME: part}, indent=1) + "\n"


def end_effectors(r: Robot) -> list[dict]:
    out = []
    for side, s in (("left", 1.0), ("right", -1.0)):
        q = side[0] + "_"
        palm = (0.0, s * 0.18, 0.615 + FOOT_CLEARANCE)
        ring = f"{side}_wrist_roll_c"
        out.append({"name": f"{side}_hand", **frame_spec(r, (ring + "1", ring + "0", q + "h_in"), palm)})
    for side, s in (("left", 1.0), ("right", -1.0)):
        q = side[0] + "_"
        inner, outer = ("hi", "ho") if s > 0 else ("ho", "hi")   # node y increasing
        toe = "ti" if s > 0 else "to"
        sole = (0.0, s * 0.10, FOOT_CLEARANCE)
        out.append({"name": f"{side}_foot", **frame_spec(r, (q + "ft_" + inner, q + "ft_" + toe, q + "ft_" + outer),
                                                         sole)})
    return out


def spec(r: Robot) -> dict:
    from .. import retarget

    return {
        "schema": "openbeam-robot/1",
        "name": NAME,
        "source": "openbeam-gym openbeam_gym/robots/build_humanoid.py",
        "description": "G1-class humanoid (about 1.3 m, 35 kg, 25 joints), "
                       + ("fixed base: pelvis on a pedestal" if r.stand else "free base"),
        "base_frame": frame_spec(r, ("pv_c", "pv_f", "pv_l"), r.nodes["pv_c"]),
        "joints": r.spec_joints,
        "end_effectors": end_effectors(r),
        "targets": {"electrics": CMD},
        "bulk_targets": {"lua": f'extensions.get("{NAME}").set({{...}})',
                         "order": humanoid_g1.JOINT_NAMES,
                         "note": "one vehicle-Lua command sets every joint: an array in this order or a "
                                 "name -> radians table; values are clamped to the limits"},
        "home": {k: round(v, 6) for k, v in humanoid_g1.HOME.items()},
        "retarget": retarget.retarget_spec(),
    }


LUA = """-- Copyright 2026 OpenBeam, LLC
-- SPDX-License-Identifier: Apache-2.0
-- openbeam_humanoid_g1: a G1-class humanoid on a pedestal. The torsion hydros
-- follow g1_cmd_<joint> (radians from the spawn pose) every physics step;
-- this file sets those electrics in bulk, binds a few keys and publishes the
-- joint state (generic robot block and g1_<joint>_pos|vel|target) at the
-- 60 Hz GFX cadence. A vehicle with Lua never auto-sleeps, which keeps the
-- robot awake. Generated by openbeam_gym/robots/build_humanoid.py.
--
-- Streaming (one command per frame, e.g. 30 Hz):
--   extensions.get("__NAME__").set({q1, ..., q__DOF__})   -- the spec's joint order
--   extensions.get("__NAME__").set({left_knee = 0.4, waist_yaw = -0.2})
local M = {}
local JOINTS = __JOINTS__
local ids, last = {}, {}

local function id(name)
  local map = v and v.nodeNameToCid
  if map and map[name] then return map[name] end
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
  local q = math.atan2(dot(crossv(a, b), axis), dot(a, b)) - n.offset
  if q > math.pi then q = q - 2 * math.pi elseif q < -math.pi then q = q + 2 * math.pi end
  return q
end

local function clamp(j, x)
  if x ~= x then return nil end
  if x < j.lower then return j.lower elseif x > j.upper then return j.upper end
  return x
end

-- Bulk setter: an array in JOINTS order, or a name -> radians table.
function M.set(t)
  if type(t) ~= "table" then return false end
  local ev = electrics.values
  local array = t[1] ~= nil
  for i, j in ipairs(JOINTS) do
    local x = tonumber(array and t[i] or t[j.name])
    if x then
      x = clamp(j, x)
      if x then ev[j.cmd] = x end
    end
  end
  return true
end
M.setJoints = M.set

function M.home()
  for _, j in ipairs(JOINTS) do electrics.values[j.cmd] = j.home end
end

function M.zero()
  for _, j in ipairs(JOINTS) do electrics.values[j.cmd] = 0 end
end

local function init()
  ids, last = {}, {}
  for _, j in ipairs(JOINTS) do
    local n = {an = id(j.axis[1]), ap = id(j.axis[2]), parent = id(j.parent_ref), child = id(j.child_ref), offset = 0}
    if n.an and n.ap and n.parent and n.child then
      ids[j.name] = n
      n.offset = angle(j) or 0
    end
    electrics.values[j.cmd] = 0
  end
  electrics.values.robot_dof = #JOINTS
end

local WAIST_YAW
local function teleop(dt)
  local ev = electrics.values
  if (ev.g1_key_home or 0) > 0 then ev.g1_key_home = 0; M.home(); return end
  if (ev.g1_key_zero or 0) > 0 then ev.g1_key_zero = 0; M.zero(); return end
  local d = (ev.g1_key_waist_p or 0) - (ev.g1_key_waist_n or 0)
  if d ~= 0 and WAIST_YAW then
    ev[WAIST_YAW.cmd] = clamp(WAIST_YAW, (ev[WAIST_YAW.cmd] or 0) + d * WAIST_YAW.rate * dt) or 0
  end
end

local function updateGFX(dt)
  teleop(dt)
  local ev = electrics.values
  for i, j in ipairs(JOINTS) do
    local q = angle(j)
    if q then
      local qd = (last[i] and dt > 0) and (q - last[i]) / dt or 0
      last[i] = q
      local cmd = ev[j.cmd] or 0
      ev[j.pos] = q
      ev[j.vel] = qd
      ev[j.target] = cmd
      ev["robot_q" .. i] = q
      ev["robot_qd" .. i] = qd
      ev["robot_cmd" .. i] = cmd
    end
  end
end

for _, j in ipairs(JOINTS) do
  j.cmd = "g1_cmd_" .. j.name
  j.pos, j.vel, j.target = "g1_" .. j.name .. "_pos", "g1_" .. j.name .. "_vel", "g1_" .. j.name .. "_target"
  if j.name == "waist_yaw" then WAIST_YAW = j end
end

M.onInit = init
M.onReset = init
M.updateGFX = updateGFX
return M
"""


def lua_text(r: Robot) -> str:
    rows = []
    for j in r.spec_joints:
        row = humanoid_g1.JOINTS[humanoid_g1.JOINT_NAMES.index(j["name"])]
        rows.append('  {name = "%s", axis = {"%s", "%s"}, parent_ref = "%s", child_ref = "%s", '
                    'lower = %r, upper = %r, home = %r, rate = %r}'
                    % (j["name"], j["axis"][0], j["axis"][1], j["parent_ref"], j["child_ref"], j["lower"],
                       j["upper"], row["home"], row["rate"]))
    return (LUA.replace("__JOINTS__", "{\n" + ",\n".join(rows) + "\n}").replace("__NAME__", NAME)
            .replace("__DOF__", str(len(rows))))


def input_files() -> dict[str, bytes]:
    actions, bindings = {}, []

    def action(name, key, flag, title, desc, hold):
        e = f"electrics.values.{flag}"
        row = {"cat": "vehicle_specific", "order": len(actions), "ctx": "vlua", "onDown": f"{e} = 1",
               "title": title, "desc": desc}
        if hold:
            row["onUp"] = f"{e} = 0"
        actions[name] = row
        bindings.append({"control": key, "action": name})

    action("g1_waist_yaw_p", "a", "g1_key_waist_p", "Humanoid waist yaw left", "Turns the waist left while held", True)
    action("g1_waist_yaw_n", "d", "g1_key_waist_n", "Humanoid waist yaw right", "Turns the waist right while held",
           True)
    action("g1_home", "3", "g1_key_home", "Humanoid home pose", "Moves every joint to the neutral standing pose",
           False)
    action("g1_zero", "4", "g1_key_zero", "Humanoid zero pose", "Moves every joint to the spawn pose", False)
    return {
        f"vehicles/{NAME}/input_actions_{NAME}.json": (json.dumps(actions, indent=2) + "\n").encode(),
        f"vehicles/{NAME}/inputmaps/keyboard.json": (json.dumps({"bindings": bindings}, indent=2) + "\n").encode(),
    }


def info_json() -> str:
    return json.dumps({"Name": "OpenBeam humanoid G1-class", "Brand": "OpenBeam", "Type": "Prop",
                       "Author": "OpenBeam, LLC (openbeam-gym)",
                       "Description": "A G1-class humanoid (about 1.3 m, 35 kg, 25 joints) on a pedestal for "
                                      "openbeam-gym, driven by torsion hydros on the electrics g1_cmd_<joint>."},
                      indent=2) + "\n"


def members(r: Robot | None = None) -> dict[str, bytes]:
    r = r or build()
    prefix = f"vehicles/{NAME}"
    return {
        f"{prefix}/{NAME}.jbeam": jbeam_text(r).encode(),
        f"{prefix}/openbeam_robot.json": (json.dumps(spec(r), indent=1) + "\n").encode(),
        f"{prefix}/lua/{NAME}.lua": lua_text(r).encode(),
        f"{prefix}/info.json": info_json().encode(),
        **input_files(),
        **package_members(prefix, NAME, visuals(r)),
        f"{prefix}/LICENSE.txt": b"Copyright 2026 OpenBeam, LLC. Licensed under the Apache License, Version 2.0.\n",
    }


def analysis(r: Robot) -> dict:
    loads = node_loads(r)
    worst_beam = max(loads, key=lambda n: (loads[n]["beams"], n))
    worst_torsion = max(loads, key=lambda n: (loads[n]["torsion"], n))
    return {
        "nodes": len(r.nodes), "robot_nodes": sum(1 for n in r.nodes if r.link[n] != "stand"),
        "beams": len(r.beams), "torsion_hydros": len(r.hydros), "joints": len(r.spec_joints),
        "robot_mass_kg": round(robot_mass(r), 3),
        "stand_mass_kg": STAND_MASS if r.stand else 0.0,
        "worst_beam_load": [worst_beam, round(loads[worst_beam]["beams"], 4)],
        "worst_torsion_load": [worst_torsion, round(loads[worst_torsion]["torsion"], 4)],
        "joints_dynamics": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                            for k, v in r.joint_dynamics.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("assets") / f"{NAME}.zip")
    ap.add_argument("--report", action="store_true", help="print the per-joint stiffness/damping table")
    args = ap.parse_args()
    r = build()
    digest = write_zip(args.out, members(r))
    a = analysis(r)
    summary = {"zip": str(args.out), "sha256": digest, **{k: v for k, v in a.items() if k != "joints_dynamics"}}
    print(json.dumps(summary))
    if args.report:
        print(f"{'joint':24s} {'bars':>4s} {'I kgm2':>8s} {'K Nm/rad':>9s} {'C Nms':>7s} {'w rad/s':>8s} "
              f"{'zeta':>5s} {'sag rad':>8s}")
        for name, d in r.joint_dynamics.items():
            print(f"{name:24s} {d['bars']:4d} {d['inertia']:8.4f} {d['spring']:9.1f} {d['damp']:7.3f} "
                  f"{d['omega']:8.1f} {d['zeta']:5.2f} {d['gravity_sag_max']:8.4f}")


if __name__ == "__main__":
    main()
