# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The published G1-class humanoid is the builder's output, a JBeam vehicle
mod with a consistent openbeam-robot/1 spec, mirror-symmetric, and sized
within the game's fixed-step budgets."""

import hashlib
import json
import math
import zipfile
from pathlib import Path

import pytest

from openbeam_gym.env import HUMANOID_G1_FILE, HUMANOID_G1_SHA256, HUMANOID_G1_URL, humanoid_g1_scene
from openbeam_gym.robots import build_humanoid as bh
from openbeam_gym.robots import humanoid_g1

ASSET = Path(__file__).resolve().parents[1] / "assets" / "openbeam_humanoid_g1.zip"
PREFIX = "vehicles/openbeam_humanoid_g1/"


@pytest.fixture(scope="module")
def robot():
    return bh.build()


@pytest.fixture(scope="module")
def package():
    with zipfile.ZipFile(ASSET) as z:
        return {n: z.read(n) for n in z.namelist()}


def test_the_published_zip_is_the_builders_output(tmp_path):
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    assert bh.write_zip(a, bh.members()) == bh.write_zip(b, bh.members()) == HUMANOID_G1_SHA256
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == HUMANOID_G1_SHA256
    assert HUMANOID_G1_URL.endswith("/main/assets/openbeam_humanoid_g1.zip")


def test_it_is_a_jbeam_vehicle_package_with_a_robot_spec(package):
    names = set(package)
    assert not any(n.endswith((".truck", ".load", ".fixed")) for n in names)
    for member in ("openbeam_humanoid_g1.jbeam", "openbeam_robot.json", "lua/openbeam_humanoid_g1.lua", "info.json",
                   "input_actions_openbeam_humanoid_g1.json", "inputmaps/keyboard.json"):
        assert PREFIX + member in names
    part = json.loads(package[PREFIX + "openbeam_humanoid_g1.jbeam"])["openbeam_humanoid_g1"]
    spec = json.loads(package[PREFIX + "openbeam_robot.json"])
    node_ids = {row[0] for row in part["nodes"][1:] if isinstance(row, list)}
    assert spec["schema"] == "openbeam-robot/1"
    assert [j["name"] for j in spec["joints"]] == humanoid_g1.JOINT_NAMES
    assert len(spec["joints"]) == 25
    hydros = [row for row in part["torsionHydros"][1:]]
    sources = {row[4]["inputSource"] for row in hydros}
    assert sources == {f"g1_cmd_{j['name']}" for j in spec["joints"]}
    assert spec["targets"] == {"electrics": "g1_cmd_{joint}"}
    for row in hydros:
        assert set(row[:4]) <= node_ids
        o = row[4]
        assert o["inLimit"] < 0 < o["outLimit"] and o["inputInLimit"] < o["inputCenter"] < o["inputOutLimit"]
        assert o["spring"] > 0 and o["damp"] > 0 and o["inRate"] == o["outRate"] > 0
    for j in spec["joints"]:
        assert set(j["axis"]) | {j["parent_ref"], j["child_ref"]} <= node_ids
        assert -math.pi < j["lower"] < 0 < j["upper"] < math.pi
    assert {e["name"] for e in spec["end_effectors"]} == {"left_hand", "right_hand", "left_foot", "right_foot"}
    for frame in [spec["base_frame"]] + spec["end_effectors"]:
        assert set(frame["nodes"]) <= node_ids and len(frame["offset"]) == 3 and len(frame["rotation"]) == 9
    assert set(spec["home"]) == set(humanoid_g1.JOINT_NAMES)
    assert set(spec["retarget"]["joints"]) == set(humanoid_g1.JOINT_NAMES)
    assert spec["bulk_targets"]["order"] == humanoid_g1.JOINT_NAMES
    refs = part["refNodes"][1]
    assert set(refs) <= node_ids
    scene = humanoid_g1_scene()
    assert scene.robot_file == HUMANOID_G1_FILE and scene.install_sha256 == HUMANOID_G1_SHA256
    assert scene.home == humanoid_g1.HOME


def test_joint_table_limits_and_home():
    assert len(humanoid_g1.JOINTS) == 25
    for j in humanoid_g1.JOINTS:
        assert -math.pi < j["lower"] < 0 < j["upper"] < math.pi, j["name"]
        assert j["lower"] <= j["home"] <= j["upper"]
        assert 0 < j["rate"] <= 10
        assert abs(math.sqrt(sum(c * c for c in j["axis"])) - 1) < 1e-12
    home = humanoid_g1.HOME
    for side in ("left", "right"):     # feet level in the home pose
        assert abs(home[f"{side}_hip_pitch"] + home[f"{side}_knee"] + home[f"{side}_ankle_pitch"]) < 1e-12
    assert humanoid_g1.RATES["left_elbow"] > humanoid_g1.RATES["left_hip_pitch"]


def test_left_and_right_mirror(robot):
    lim, rates = humanoid_g1.LIMITS, humanoid_g1.RATES
    for name in humanoid_g1.JOINT_NAMES:
        if not name.startswith("left_"):
            continue
        other = humanoid_g1.mirror_name(name)
        base = name[5:]
        if base in humanoid_g1.MIRRORED:
            assert lim[other] == (-lim[name][1], -lim[name][0])
        else:
            assert lim[other] == lim[name]
        assert rates[other] == rates[name]
        for key in ("inertia", "spring", "damp", "gravity_torque_max"):
            assert robot.joint_dynamics[other][key] == pytest.approx(robot.joint_dynamics[name][key], rel=1e-6)
    assert humanoid_g1.mirror_pose(humanoid_g1.mirror_pose(humanoid_g1.HOME)) == humanoid_g1.HOME
    assert humanoid_g1.mirror_pose(humanoid_g1.HOME) == humanoid_g1.HOME

    axes = {j["name"]: j["axis"] for j in humanoid_g1.JOINTS}

    def mirror_node(n):
        for a, b in (("l_", "r_"), ("r_", "l_"), ("left_", "right_"), ("right_", "left_")):
            if n.startswith(a):
                m = b + n[len(a):]
                joint, suffix = m.rsplit("_", 1)
                # A left-right axis keeps its direction in the mirror: its ends swap.
                if suffix in ("an", "ap") and abs(axes[joint][1]) > 0.5:
                    m = joint + ("_ap" if suffix == "an" else "_an")
                return m
        return {"pv_l": "pv_r", "pv_r": "pv_l"}.get(n, n)

    by_place = {(round(p[0], 9), round(p[1], 9) + 0.0, round(p[2], 9)): n for n, p in robot.nodes.items()}
    for n, p in robot.nodes.items():
        m = mirror_node(n)
        if m == n and n.startswith("waist_pitch_"):
            continue    # the one centred left-right axis: its rings sit 1.5 cm to either side
        if m == n:      # a centre node: some node of the same mass sits at its mirror image
            m = by_place[(round(p[0], 9), round(-p[1], 9) + 0.0, round(p[2], 9))]
        assert (p[0], -p[1], p[2]) == pytest.approx(robot.nodes[m], abs=1e-9), n
        assert robot.mass[n] == robot.mass[m]


def test_size_mass_and_counts(robot):
    a = bh.analysis(robot)
    assert a["joints"] == 25 and a["torsion_hydros"] == 50
    assert a["nodes"] < 300 and a["beams"] < 2000        # real time; far below the game's 4000/40000 check
    assert a["robot_mass_kg"] == pytest.approx(35.0, abs=1.0)
    body = [p for n, p in robot.nodes.items() if robot.link[n] != "stand"]
    soles = min(p[2] for p in body)
    assert soles == pytest.approx(bh.FOOT_CLEARANCE)
    assert max(p[2] for p in body) - soles == pytest.approx(1.32, abs=0.05)
    stand = [n for n in robot.nodes if robot.link[n] == "stand"]
    assert min(robot.nodes[n][2] for n in stand) == 0.0
    assert {n for n in robot.colliding if robot.link[n] == "stand"} == {"st_gbl", "st_gbr", "st_gfl", "st_gfr"}
    free = bh.build(stand=False)
    assert not any(l == "stand" for l in free.link.values()) and free.ballast == 0
    assert bh.robot_mass(free) == pytest.approx(a["robot_mass_kg"])


@pytest.mark.parametrize("stand", [True, False])
def test_fixed_step_budgets(stand):
    r = bh.build(stand=stand)
    loads = bh.node_loads(r)
    assert max(v["beams"] for v in loads.values()) <= bh.BEAM_CEILING + 1e-9 < 2.6
    assert max(v["torsion"] for v in loads.values()) <= bh.TORSION_CEILING + 1e-9 < 0.8
    assert all(v["beams"] + v["torsion"] < 4.0 for v in loads.values())
    for name, d in r.joint_dynamics.items():
        assert d["zeta"] == pytest.approx(bh.ZETA, rel=0.05), name
        assert d["gravity_sag_max"] < (0.1 if stand else 0.2), name
        assert d["omega"] > 15, name


def test_beams_and_hinges_are_well_formed(robot):
    seen = set()
    for (a, b), (k, c) in robot.beams.items():
        assert a != b and (a, b) not in seen and a in robot.nodes and b in robot.nodes
        seen.add((a, b))
        assert k > 0 and c > 0
        d = [x - y for x, y in zip(robot.nodes[a], robot.nodes[b])]
        assert math.sqrt(sum(v * v for v in d)) > 0.005
    adjacency = {n: set() for n in robot.nodes}
    for a, b in robot.beams:
        adjacency[a].add(b)
        adjacency[b].add(a)
    for j in robot.joints:
        # The game's derivation: the child side, reached without crossing the
        # axis nodes, must not reach the parent side (no closed loop).
        blocked = {j["neg"], j["pos"]}
        start = j["bars"][0][1]
        seen, todo = {start}, [start]
        while todo:
            for m in adjacency[todo.pop()] - blocked - seen:
                seen.add(m)
                todo.append(m)
        assert j["bars"][0][0] not in seen and "pv_c" not in seen, j["name"]
        # Every torsion bar's consecutive nodes are beamed (or the game leaves it inert).
        for p, c in j["bars"]:
            for a, b in ((p, j["neg"]), (j["neg"], j["pos"]), (j["pos"], c)):
                assert tuple(sorted((a, b))) in robot.beams, (j["name"], a, b)


def test_torsion_gradient_matches_the_lever_geometry(robot):
    for j in robot.joints:
        p, c = j["bars"][0]
        g = bh.dihedral_gradient(*(robot.nodes[n] for n in (p, j["neg"], j["pos"], c)))
        total = [sum(v[i] for v in g) for i in range(3)]
        assert max(abs(t) for t in total) < 1e-9
        assert math.sqrt(sum(v * v for v in g[3])) == pytest.approx(1 / j["ring_r"], rel=1e-6)


def test_frames_reproduce_their_points(robot):
    spec = bh.spec(robot)

    def pose(frame):
        o, x, y = (robot.nodes[n] for n in frame["nodes"])
        cols = bh.frame_of(o, x, y)
        point = [o[i] + sum(cols[k][i] * frame["offset"][k] for k in range(3)) for i in range(3)]
        rot = frame["rotation"]
        world = [[sum(cols[k][i] * rot[3 * k + c] for k in range(3)) for c in range(3)] for i in range(3)]
        return point, world

    point, world = pose(spec["base_frame"])
    assert point == pytest.approx(robot.nodes["pv_c"], abs=1e-6)
    assert world == [pytest.approx(row, abs=1e-5) for row in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
    effectors = {e["name"]: e for e in spec["end_effectors"]}
    for side, s in (("left", 1), ("right", -1)):
        p, w = pose(effectors[f"{side}_hand"])
        assert p == pytest.approx((0.0, s * 0.18, 0.615 + bh.FOOT_CLEARANCE), abs=1e-5)
        assert w == [pytest.approx(row, abs=1e-5) for row in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
        p, w = pose(effectors[f"{side}_foot"])
        assert p == pytest.approx((0.0, s * 0.10, bh.FOOT_CLEARANCE), abs=1e-5)


def test_lua_has_the_bulk_setter_and_every_joint(package):
    lua = package[PREFIX + "lua/openbeam_humanoid_g1.lua"].decode()
    assert "function M.set(t)" in lua and "M.setJoints = M.set" in lua and "function M.home()" in lua
    assert 'extensions.get("openbeam_humanoid_g1")' in lua
    for name in humanoid_g1.JOINT_NAMES:
        assert f'name = "{name}"' in lua
    assert lua.count("{name = ") == 25
    assert "robot_dof" in lua and "robot_q" in lua and "M.updateGFX = updateGFX" in lua
    keys = json.loads(package[PREFIX + "inputmaps/keyboard.json"])["bindings"]
    assert {k["control"] for k in keys} == {"a", "d", "3", "4"}
