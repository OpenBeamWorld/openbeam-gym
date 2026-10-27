# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Builds assets/openbeam_test_arm.zip: a small four-joint arm as a JBeam
vehicle mod, the fast test robot for openbeam-gym.

The arm is a vehicle package (``vehicles/openbeam_test_arm/``) that the game's
``install_mod`` takes as it is; no legacy format is involved.

* Each link is a stiff cluster of nodes braced by every pairwise beam.
* Each joint is a hinge: two nodes on the joint axis, in the parent link, to
  which every node of the child link is tied.
* Each joint is driven by four JBeam ``torsionHydros`` around that axis, one
  per lever pair. They are torsion bars whose rest angle follows the electric
  ``arm_cmd_<joint>`` (radians from the spawn pose) at 1 rad/s, the speed the
  previous rotator arm moved at.
* ``openbeam_robot.json`` (schema ``openbeam-robot/1``) names the joints, their
  axis nodes, reference levers and limits, so the game's robot tools derive
  the kinematics. It writes targets to ``arm_cmd_<joint>``.
* Keys (seated in the arm) move the goals: A/D base pan, W/S shoulder, Q/E
  elbow, Z/X wrist roll, 3 home. The key bindings also make the arm a seat
  for the player, which stepped physics needs.
* Visuals (``visuals()``, written by ``visuals.py``): one flat-coloured mesh
  per link in ``openbeam_test_arm.dae`` with ``openbeam_test_arm.materials.json``,
  bound to the link's node group by a JBeam ``flexbodies`` row. They add no
  mass or beams: the physics is the same as before they were added.
* The vehicle Lua keeps the arm awake and publishes the generic robot block
  (``robot_dof``, ``robot_q<i>``, ``robot_qd<i>``, ``robot_cmd<i>``) and
  ``arm_<joint>_pos|vel|target`` at 60 Hz, following the SO-101 contract's
  naming.

The geometry is the earlier rotator test arm's: base pan about the vertical
axis 0.2 m above the base, the shoulder 0.45 m up, the elbow 0.5 m out and
the wrist roll 0.3 m beyond it, with the tool point 0.2 m past the wrist (only the
levers of the joint rings are longer, 16 cm, see LEVER). The
masses (3 kg per node) and beams (400 kN/m, 200 N s/m) are also unchanged. So
the Reach task and its PPO results stay comparable.

    python -m openbeam_gym.robots.build_test_arm --out assets/openbeam_test_arm.zip

Stdlib only; the zip is deterministic (fixed timestamps, sorted members).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

from .visuals import Mesh, flexbodies, package_members

NAME = "openbeam_test_arm"
NODE_MASS = 3.0          # kg, every node (the previous arm's minimass)
BEAM_SPRING = 400000.0   # N/m
BEAM_DAMP = 200.0        # N s/m
RATE = 1.0               # rad/s, joint speed
# Asked torsion stiffness per hydro; the game sizes each bar to its
# fixed-step stability budget at spawn, so this is an upper bound.
TORSION_SPRING = 200000.0  # N m/rad
TORSION_DAMP = 400.0       # N m s/rad
# Lever radius of each joint's ring nodes. The game caps a bar's stiffness at
# about m r^2 / h^2, so the levers are twice the rotator arm's 8 cm: with 8 cm
# the arm sagged and swung enough that reaching a goal's joint values often
# missed its point (an oracle reached 3 of 8 goals, the rotator arm 6 of 8;
# 16 cm: 6 of 8).
LEVER = 0.16

# name, parent link, child link, axis point, axis direction, limits (deg).
# Base frame: x forward (the reach), y left, z up, metres.
JOINTS = [
    ("base_pan", "root", "turret", (0.0, 0.0, 0.2), (0.0, 0.0, 1.0), (-170, 170)),
    ("shoulder", "turret", "upper_arm", (0.0, 0.0, 0.45), (0.0, -1.0, 0.0), (-90, 90)),
    ("elbow", "upper_arm", "forearm", (0.5, 0.0, 0.45), (0.0, -1.0, 0.0), (-135, 135)),
    # +-175 rather than +-180: a torsion bar's angle is unique within (-pi, pi).
    ("wrist_roll", "forearm", "hand", (0.8, 0.0, 0.45), (1.0, 0.0, 0.0), (-175, 175)),
]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def add(*vs):
    return tuple(sum(c) for c in zip(*vs))


def mul(v, s):
    return tuple(c * s for c in v)


def basis(axis):
    a = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 0.0, 1.0)
    e1 = cross(axis, a)
    n = math.sqrt(sum(c * c for c in e1))
    e1 = tuple(c / n for c in e1)
    return e1, cross(axis, e1)


def to_jbeam(p):
    """Base frame (x forward, y left, z up) -> JBeam (x left, y back, z up)."""
    return (round(p[1], 5), round(-p[0], 5), round(p[2], 5))


class Arm:
    def __init__(self):
        self.nodes: dict[str, tuple] = {}
        self.link: dict[str, str] = {}
        self.fixed: set[str] = set()
        self.beams: set[tuple[str, str]] = set()
        self.hydros: list[dict] = []
        self.spec_joints: list[dict] = []

    def node(self, name, p, link, fixed=False):
        self.nodes[name] = p
        self.link[name] = link
        if fixed:
            self.fixed.add(name)
        return name


def build() -> Arm:
    arm = Arm()
    for i, (x, y) in enumerate(((0.25, 0.25), (-0.25, 0.25), (-0.25, -0.25), (0.25, -0.25))):
        arm.node(f"b{i}", (x, y, 0.0), "root", fixed=True)
    # refNodes: ref at the base origin; back (+y JBeam = -x base), left (+x JBeam = +y base), up.
    arm.node("bref", (0.0, 0.0, 0.0), "root", fixed=True)
    arm.node("bback", (-0.25, 0.0, 0.0), "root", fixed=True)
    arm.node("bleft", (0.0, 0.25, 0.0), "root", fixed=True)
    arm.node("bup", (0.0, 0.0, 0.1), "root", fixed=True)
    axes = {}
    for name, parent, child, p, axis, (lo, hi) in JOINTS:
        e1, e2 = basis(axis)
        # Axis nodes 15 cm either side of the joint: the bearing's lever against bending.
        pos = arm.node(f"{name}_ap", add(p, mul(axis, 0.15)), parent)
        neg = arm.node(f"{name}_an", add(p, mul(axis, -0.15)), parent)
        base, moving = [], []
        for k in range(4):
            ang = k * math.pi / 2
            radial = add(mul(e1, math.cos(ang)), mul(e2, math.sin(ang)))
            base.append(arm.node(f"{name}_p{k}", add(p, mul(axis, -0.03), mul(radial, LEVER)), parent))
            moving.append(arm.node(f"{name}_c{k}", add(p, mul(axis, 0.03), mul(radial, LEVER)), child))
        axes[child] = (pos, neg)
        lo_r, hi_r = math.radians(lo), math.radians(hi)
        for k in range(4):
            # Positive = right-handed about neg -> pos (the +axis), the bar's own sign.
            arm.hydros.append({"nodes": [base[k], neg, pos, moving[k]], "joint": name, "lo": lo_r, "hi": hi_r})
        arm.spec_joints.append({"name": name, "axis": [neg, pos], "parent_ref": base[0], "child_ref": moving[0],
                                "sign": 1.0, "lower": round(lo_r, 6), "upper": round(hi_r, 6), "reference": 0.0})
    # Link bodies between the joints.
    arm.node("ua0", (0.25, -0.05, 0.45), "upper_arm")
    arm.node("ua1", (0.25, 0.05, 0.45), "upper_arm")
    arm.node("fa0", (0.65, 0.0, 0.48), "forearm")
    arm.node("fa1", (0.65, 0.0, 0.42), "forearm")
    arm.node("hd0", (0.92, -0.03, 0.47), "hand")
    arm.node("tcp", (1.0, 0.0, 0.45), "hand")
    # Beams: every pair within a link; every child-link node to its joint's axis nodes.
    by_link: dict[str, list[str]] = {}
    for n, l in arm.link.items():
        by_link.setdefault(l, []).append(n)
    for members in by_link.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if not (a in arm.fixed and b in arm.fixed):
                    arm.beams.add((a, b))
    for child, (pos, neg) in axes.items():
        for n in by_link[child]:
            arm.beams.add((n, pos))
            arm.beams.add((n, neg))
    # The pinned base still needs its own frame braced for the importer's checks.
    for a, b in (("b0", "b2"), ("b1", "b3"), ("b0", "bup"), ("b2", "bup"), ("bref", "bup")):
        arm.beams.add((a, b))
    return arm


LINKS = ("root", "turret", "upper_arm", "forearm", "hand")


def visuals(arm: Arm | None = None) -> list:
    """One flat-coloured mesh per link, bound to the link's node group
    (``visuals.py``); each joint's orange housing rides on its parent link."""
    meshes = {link: Mesh(f"{NAME}_{link}", f"arm_{link}") for link in LINKS}
    z, y, x = (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)
    root, turret, upper, fore, hand = (meshes[k] for k in LINKS)
    root.box((0.0, 0.0, 0.02), (x, y, z), (0.25, 0.25, 0.02), "stand")
    root.prism((0.0, 0.0, 0.04), (0.0, 0.0, 0.17), 0.06, "stand")
    root.housing((0.0, 0.0, 0.18), z, 0.04, 0.11)
    turret.prism((0.0, 0.0, 0.20), (0.0, 0.0, 0.25), 0.12, "link")
    turret.bar((0.0, 0.0, 0.25), (0.0, 0.0, 0.42), 0.09, 0.09, "link", hint=x)
    turret.housing((0.0, 0.0, 0.45), y, 0.16, 0.07)
    upper.bar((0.0, 0.0, 0.45), (0.5, 0.0, 0.45), 0.08, 0.07, "link", hint=z)
    upper.housing((0.5, 0.0, 0.45), y, 0.13, 0.06)
    fore.bar((0.5, 0.0, 0.45), (0.77, 0.0, 0.45), 0.06, 0.06, "link", hint=z)
    fore.housing((0.765, 0.0, 0.45), x, 0.05, 0.05)
    hand.prism((0.79, 0.0, 0.45), (0.86, 0.0, 0.45), 0.045, "link")
    hand.bar((0.86, 0.0, 0.45), (0.98, 0.0, 0.45), 0.05, 0.03, "link", hint=z)
    hand.box((0.99, 0.0, 0.45), (x, y, z), (0.01, 0.015, 0.015), "housing")
    return [meshes[k] for k in LINKS]


def jbeam_text(arm: Arm) -> str:
    nodes = [["id", "posX", "posY", "posZ"], {"nodeWeight": NODE_MASS}, {"collision": False},
             {"selfCollision": False}, {"frictionCoef": 0.5}]
    for link in LINKS:
        nodes.append({"group": f"arm_{link}"})
        for n, l in arm.link.items():
            if l != link:
                continue
            x, y, z = to_jbeam(arm.nodes[n])
            row = [n, x, y, z]
            if n in arm.fixed:
                row.append({"fixed": True})
            nodes.append(row)
    nodes.append({"group": ""})
    beams = [["id1:", "id2:"], {"beamSpring": BEAM_SPRING, "beamDamp": BEAM_DAMP},
             {"beamDeform": 1.0e9, "beamStrength": 1.0e12}]
    beams += [[a, b] for a, b in sorted(arm.beams)]
    hydros = [["id1:", "id2:", "id3:", "id4:"]]
    for h in arm.hydros:
        j = h["joint"]
        hydros.append(h["nodes"] + [{
            "spring": TORSION_SPRING, "damp": TORSION_DAMP, "inputSource": f"arm_cmd_{j}",
            "inLimit": round(h["lo"], 6), "outLimit": round(h["hi"], 6),
            "inputInLimit": round(h["lo"], 6), "inputCenter": 0, "inputOutLimit": round(h["hi"], 6),
            "inRate": RATE, "outRate": RATE, "autoCenterRate": RATE}])
    part = {
        "information": {"authors": "OpenBeam, LLC (openbeam-gym)", "name": "OpenBeam test arm", "value": 0},
        "slotType": "main",
        "refNodes": [["ref:", "back:", "left:", "up:"], ["bref", "bback", "bleft", "bup"]],
        "camerasInternal": [["type", "x", "y", "z", "fov"], ["driver", 0.0, 0.0, 0.05, 60]],
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
        "source": "openbeam-gym openbeam_gym/robots/build_test_arm.py",
        "joints": arm.spec_joints,
        "targets": {"electrics": "arm_cmd_{joint}"},
    }


LUA = """-- Copyright 2026 OpenBeam, LLC
-- SPDX-License-Identifier: Apache-2.0
-- openbeam_test_arm: publishes the joint state (generic robot block and
-- arm_<joint>_pos|vel|target) at the 60 Hz GFX cadence and keeps the arm
-- awake (a vehicle with Lua never auto-sleeps). The torsion hydros follow
-- arm_cmd_<joint> themselves, every physics step; this file only reads.
-- Generated by openbeam_gym/robots/build_test_arm.py.
local M = {}
local JOINTS = __JOINTS__
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

local function init()
  ids, last = {}, {}
  for i, j in ipairs(JOINTS) do
    local n = {an = id(j.axis[1]), ap = id(j.axis[2]), parent = id(j.parent_ref), child = id(j.child_ref), offset = 0}
    if n.an and n.ap and n.parent and n.child then
      ids[j.name] = n
      n.offset = angle(j) or 0
    end
    electrics.values["arm_cmd_" .. j.name] = 0
  end
  electrics.values.robot_dof = #JOINTS
end

local function teleop(dt)
  if (electrics.values.arm_key_home or 0) > 0 then
    electrics.values.arm_key_home = 0
    for _, j in ipairs(JOINTS) do electrics.values["arm_cmd_" .. j.name] = 0 end
    return
  end
  for _, j in ipairs(JOINTS) do
    local d = (electrics.values["arm_key_" .. j.name .. "_p"] or 0) - (electrics.values["arm_key_" .. j.name .. "_n"] or 0)
    if d ~= 0 then
      local k = "arm_cmd_" .. j.name
      electrics.values[k] = math.max(j.lower, math.min(j.upper, (electrics.values[k] or 0) + d * __RATE__ * dt))
    end
  end
end

local function updateGFX(dt)
  teleop(dt)
  for i, j in ipairs(JOINTS) do
    local q = angle(j)
    if q then
      local qd = (last[i] and dt > 0) and (q - last[i]) / dt or 0
      last[i] = q
      local cmd = electrics.values["arm_cmd_" .. j.name] or 0
      electrics.values["arm_" .. j.name .. "_pos"] = q
      electrics.values["arm_" .. j.name .. "_vel"] = qd
      electrics.values["arm_" .. j.name .. "_target"] = cmd
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
        '{name = "%s", axis = {"%s", "%s"}, parent_ref = "%s", child_ref = "%s", lower = %r, upper = %r}'
        % (j["name"], j["axis"][0], j["axis"][1], j["parent_ref"], j["child_ref"], j["lower"], j["upper"])
        for j in arm.spec_joints)
    return LUA.replace("__JOINTS__", "{" + rows + "}").replace("__RATE__", repr(RATE))


KEYS = [("base_pan", "d", "a"), ("shoulder", "w", "s"), ("elbow", "e", "q"), ("wrist_roll", "x", "z")]


def input_files() -> dict[str, bytes]:
    actions, bindings = {}, []
    for joint, plus, minus in KEYS:
        for sign, key in (("p", plus), ("n", minus)):
            name = f"arm_{joint}_{sign}"
            e = f"electrics.values.arm_key_{joint}_{sign}"
            actions[name] = {"cat": "vehicle_specific", "order": len(actions), "ctx": "vlua",
                             "onDown": f"{e} = 1", "onUp": f"{e} = 0",
                             "title": f"Test arm {joint} {'+' if sign == 'p' else '-'}",
                             "desc": f"Moves the {joint} goal while held"}
            bindings.append({"control": key, "action": name})
    actions["arm_home"] = {"cat": "vehicle_specific", "order": len(actions), "ctx": "vlua",
                           "onDown": "electrics.values.arm_key_home = 1", "title": "Test arm home pose",
                           "desc": "Moves every goal to the spawn pose"}
    bindings.append({"control": "3", "action": "arm_home"})
    return {
        f"vehicles/{NAME}/input_actions_{NAME}.json": (json.dumps(actions, indent=2) + "\n").encode(),
        f"vehicles/{NAME}/inputmaps/keyboard.json": (json.dumps({"bindings": bindings}, indent=2) + "\n").encode(),
    }


def info_json() -> str:
    return json.dumps({"Name": "OpenBeam test arm", "Brand": "OpenBeam", "Type": "Prop",
                       "Author": "OpenBeam, LLC (openbeam-gym)",
                       "Description": "A four-joint test arm for openbeam-gym (base pan, shoulder, elbow, wrist roll), "
                                      "driven by torsion hydros on the electrics arm_cmd_<joint>."}, indent=2) + "\n"


def members(arm: Arm | None = None) -> dict[str, bytes]:
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


def write_zip(path: Path, files: dict[str, bytes]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budget(arm: Arm | None = None) -> dict:
    """The arm against the game's fixed-step budgets (``sizing.py``), as the
    game reads its JBeam. The arm predates that sizing and asks for more than
    the budgets: the game scales its torsion bars and lowers its busiest
    beams at spawn. It is left as it is so its PPO results stay comparable."""
    from .sizing import budget_report

    return budget_report(json.loads(jbeam_text(arm or build()))[NAME])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("assets") / f"{NAME}.zip")
    ap.add_argument("--report", action="store_true", help="print the fixed-step budget report")
    args = ap.parse_args()
    arm = build()
    digest = write_zip(args.out, members(arm))
    print(json.dumps({"zip": str(args.out), "sha256": digest, "nodes": len(arm.nodes), "beams": len(arm.beams),
                      "torsion_hydros": len(arm.hydros)}))
    if args.report:
        print(json.dumps({k: v for k, v in budget(arm).items() if k != "loads"}))


if __name__ == "__main__":
    main()
