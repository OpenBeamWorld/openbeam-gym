# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The published camera boom is the builder's output, a JBeam vehicle mod
whose camera block, end effector and kinematics agree."""

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from openbeam_gym import topdown
from openbeam_gym.env import CAMERA_BOOM_FILE, CAMERA_BOOM_SHA256, CAMERA_BOOM_URL, TEST_ARM_URL
from openbeam_gym.robots import build_camera_boom as B
from openbeam_gym.robots import build_test_arm as T

ASSET = Path(__file__).resolve().parents[1] / "assets" / "openbeam_camera_boom.zip"
PREFIX = "vehicles/openbeam_camera_boom"


def load():
    with zipfile.ZipFile(ASSET) as z:
        names = set(z.namelist())
        jbeam = json.loads(z.read(f"{PREFIX}/openbeam_camera_boom.jbeam"))
        spec = json.loads(z.read(f"{PREFIX}/openbeam_robot.json"))
        lua = z.read(f"{PREFIX}/lua/openbeam_camera_boom.lua").decode()
        keys = json.loads(z.read(f"{PREFIX}/inputmaps/keyboard.json"))
    return names, jbeam["openbeam_camera_boom"], spec, lua, keys


def node_table(part):
    rows = {}
    for row in part["nodes"][1:]:
        if isinstance(row, list):
            rows[row[0]] = row
    return rows


def base_frame(part):
    """Node positions back in the base frame (x forward, y left, z up)."""
    return {n: (-r[2], r[1], r[3]) for n, r in node_table(part).items()}


def test_the_published_zip_is_the_builders_output(tmp_path):
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    digest = B.write_zip(a, B.members())
    assert B.write_zip(b, B.members()) == digest
    assert a.read_bytes() == b.read_bytes()
    assert digest == CAMERA_BOOM_SHA256
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == CAMERA_BOOM_SHA256
    assert CAMERA_BOOM_URL == TEST_ARM_URL.replace("openbeam_test_arm.zip", "openbeam_camera_boom.zip")
    assert CAMERA_BOOM_FILE == "openbeam_camera_boom.jbeam"


def test_it_is_a_jbeam_vehicle_package_with_a_robot_spec():
    names, part, spec, lua, keys = load()
    assert not any(n.endswith((".truck", ".load", ".fixed")) for n in names)
    assert {f"{PREFIX}/info.json", f"{PREFIX}/input_actions_openbeam_camera_boom.json",
            f"{PREFIX}/LICENSE.txt"} <= names
    nodes = node_table(part)
    assert spec["schema"] == "openbeam-robot/1"
    assert spec["targets"] == {"electrics": "boom_cmd_{joint}"}
    assert [j["name"] for j in spec["joints"]] == ["base_pan", "lift", "elbow", "tilt"]
    sources = {row[4]["inputSource"] for row in part["torsionHydros"][1:]}
    assert sources == {f"boom_cmd_{j['name']}" for j in spec["joints"]}
    assert len(part["torsionHydros"]) - 1 == 4 * len(spec["joints"])
    for j in spec["joints"]:
        assert set(j["axis"]) | {j["parent_ref"], j["child_ref"]} <= set(nodes)
        assert -math.pi < j["lower"] < 0 < j["upper"] < math.pi
    for row in part["torsionHydros"][1:]:
        assert set(row[:4]) <= set(nodes)
        opts = row[4]
        joint = next(j for j in spec["joints"] if opts["inputSource"] == f"boom_cmd_{j['name']}")
        assert (opts["inLimit"], opts["outLimit"]) == (joint["lower"], joint["upper"])
        assert opts["inputInLimit"] == joint["lower"] and opts["inputOutLimit"] == joint["upper"]


def test_nodes_and_beams_are_sane():
    _, part, _, _, _ = load()
    nodes = node_table(part)
    pos = base_frame(part)
    beams = [tuple(r) for r in part["beams"][1:] if isinstance(r, list)]
    assert len(beams) == len(set(beams))
    seen = set()
    for a, b in beams:
        assert a in nodes and b in nodes and a != b
        assert math.dist(pos[a], pos[b]) > 0.02          # no zero-length beams
        seen |= {a, b}
    assert seen == set(nodes)                             # nothing floats
    # Distinct node positions.
    pts = sorted(pos.values())
    assert all(math.dist(p, q) > 0.02 for p, q in zip(pts, pts[1:]))
    # Every node sits above the surface; only the stand is pinned.
    fixed = {n for n, r in nodes.items() if len(r) > 4 and r[4].get("fixed")}
    assert all(p[2] >= 0.0 for p in pos.values())
    assert fixed == {f"b{i}" for i in range(4)} | {f"m{i}" for i in range(4)} | {"bref", "bback", "bleft", "bup"}
    for r in part["refNodes"][1]:
        assert r in fixed


def test_camera_block_and_end_effector_agree():
    _, part, spec, _, _ = load()
    pos = base_frame(part)
    cam = spec["camera"]
    for key in ("position_node", "target_node", "up_node"):
        assert cam[key] in pos
    assert cam["fov_axis"] == "vertical" and 20 < cam["fov_deg"] < 90 and cam["aspect"] == pytest.approx(4 / 3)
    ee = {e["name"]: e for e in spec["end_effectors"]}[cam["end_effector"]]
    assert ee["nodes"] == [cam["position_node"], cam["target_node"], cam["up_node"]]
    assert ee["offset"] == [0.0, 0.0, 0.0]
    # The EE rotation (columns = camera axes in the node frame) is a proper rotation ...
    r = np.array(ee["rotation"]).reshape(3, 3)
    assert np.allclose(r.T @ r, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)
    # ... and with the node frame (x to target, y to up) it is exactly topdown's camera rotation.
    p, t, u = (np.array(pos[cam[k]]) for k in ("position_node", "target_node", "up_node"))
    ex = (t - p) / np.linalg.norm(t - p)
    ey = (u - p) - ex * (ex @ (u - p))
    ey /= np.linalg.norm(ey)
    node_basis = np.column_stack([ex, ey, np.cross(ex, ey)])
    pose = topdown.camera_pose(p, t, u)
    assert np.allclose(node_basis @ r, pose.rotation)
    # Home: straight down from 1.0 m over the workspace centre, image up away from the mast.
    assert np.allclose(pose.position, [0.55, 0.0, 1.0])
    assert np.allclose(pose.forward, [0, 0, -1]) and np.allclose(pose.up, [1, 0, 0])
    assert cam["home"]["nadir"] and set(cam["home"]["joints"]) == {j["name"] for j in spec["joints"]}
    # The camera nodes are a braced cluster tied to the tilt axis.
    beams = {frozenset(r) for r in part["beams"][1:] if isinstance(r, list)}
    tilt = next(j for j in spec["joints"] if j["name"] == "tilt")
    for n in ee["nodes"]:
        for other in ee["nodes"] + tilt["axis"]:
            if other != n:
                assert frozenset((n, other)) in beams


def test_the_default_workspace_fits_the_home_image():
    _, part, spec, _, _ = load()
    cam = topdown.camera_from_spec(spec["camera"], base_frame(part), 640)
    assert (cam.width, cam.height) == (640, 480)
    ws = spec["camera"]["workspace"]
    plane = topdown.Plane(np.array(ws["center"]), np.array(ws["normal"]))
    cx, cy, _ = ws["center"]
    sx, sy = ws["size"]
    for dx in (-sx / 2, sx / 2):
        for dy in (-sy / 2, sy / 2):
            px = topdown.plane_to_pixel(cam, [cx + dx, cy + dy, 0.0])
            assert 0.1 * cam.height < px[1] < 0.9 * cam.height and 0.1 * cam.width < px[0] < 0.9 * cam.width
    centre = topdown.plane_to_pixel(cam, ws["center"])
    assert np.allclose(centre, [320, 240])
    # A drag up the image moves away from the mast (+x); right moves to the boom's right (-y).
    assert np.allclose(np.sign(topdown.drag_to_table_xy(cam, plane, centre, [0, -10])), [1, 0])
    assert np.allclose(np.sign(topdown.drag_to_table_xy(cam, plane, centre, [10, 0])), [0, -1])


def test_joint_limits_and_the_lua_agree():
    _, _, spec, lua, keys = load()
    for j in spec["joints"]:
        assert f'name = "{j["name"]}"' in lua
        assert f"lower = {j['lower']!r}, upper = {j['upper']!r}" in lua
    for token in ("boom_cmd_", "robot_dof", "robot_q", "robot_qd", "robot_cmd", '"_pos"', '"_vel"', '"_target"',
                  "boom_nadir_lock", "boom_key_home"):
        assert token in lua
    assert "pitch = true" in lua and lua.count("pitch = true") == 2     # lift and elbow
    controls = {b["control"]: b["action"] for b in keys["bindings"]}
    assert controls["3"] == "boom_home" and len(controls) == 9


def test_rigid_kinematics_and_nadir_solutions():
    home = B.fk({})
    assert np.allclose(home["cam"], B.CAM) and np.allclose(home["cam_fwd"], B.CAM_FWD)
    lim = {j[0]: tuple(math.radians(a) for a in j[5]) for j in B.JOINTS}
    for h in (0.8, 1.0, 1.2):
        for x in (0.30, 0.55, 0.80):
            for y in (-0.25, 0.0, 0.25):
                q = B.nadir_joints(x, y, h)
                for name, v in q.items():
                    assert lim[name][0] <= v <= lim[name][1]
                assert q["lift"] + q["elbow"] + q["tilt"] == pytest.approx(0.0, abs=1e-12)
                p = B.fk(q)
                pose = topdown.camera_pose(p["cam"], p["cam_fwd"], p["cam_up"])
                assert np.allclose(pose.position, [x, y, h], atol=1e-9)
                assert np.allclose(pose.forward, [0, 0, -1], atol=1e-9)
                # Image up points away from the mast.
                assert pose.up @ np.array([x, y, 0]) > 0
    assert B.nadir_joints(*B.WORKSPACE_CENTER[:2], 1.0) == pytest.approx({j[0]: 0.0 for j in B.JOINTS}, abs=1e-12)
    with pytest.raises(ValueError):
        B.nadir_joints(1.5, 0.0, 1.0)


def test_the_driver_seat_only_touches_the_pinned_stand():
    boom = B.build()
    assert set(B.driver_eye_anchors(boom)) <= boom.fixed


def _beam_network_margins(arm, spring, damp, mass=3.0, h=5e-4):
    free = [n for n in arm.nodes if n not in arm.fixed]
    idx = {n: i for i, n in enumerate(free)}
    k = np.zeros((3 * len(free),) * 2)
    for a, b in arm.beams:
        d = np.subtract(arm.nodes[b], arm.nodes[a])
        d = d / np.linalg.norm(d)
        p = np.outer(d, d)
        for x in (a, b):
            if x in idx:
                i = 3 * idx[x]
                k[i:i + 3, i:i + 3] += p
        if a in idx and b in idx:
            i, j = 3 * idx[a], 3 * idx[b]
            k[i:i + 3, j:j + 3] -= p
            k[j:j + 3, i:i + 3] -= p
    lam = float(np.linalg.eigvalsh(k).max())
    return h * math.sqrt(spring * lam / mass), h * damp * lam / mass


def test_stability_margins_against_the_test_arm():
    boom, arm = B.build(), T.build()
    boom_w, boom_c = _beam_network_margins(boom, B.BEAM_SPRING, B.BEAM_DAMP)
    arm_w, arm_c = _beam_network_margins(arm, T.BEAM_SPRING, T.BEAM_DAMP)
    assert B.BEAM_SPRING > T.BEAM_SPRING
    assert boom_w < 0.9 and boom_c < 0.7                 # explicit step limit is 2
    assert boom_w < 1.2 * arm_w
    # Gravity sag per unit of the m r^2 / h^2 torsion cap: below the test arm's worst joint.
    sb = B.stability(boom, B.JOINTS, B.LEVERS)["joints"]
    sa = B.stability(arm, T.JOINTS, {j[0]: T.LEVER for j in T.JOINTS})["joints"]
    worst_arm = max(v["sag_per_cap"] for v in sa.values())
    assert all(v["sag_per_cap"] < worst_arm for v in sb.values())
    assert sb["lift"]["sag_per_cap"] < 0.5 * sa["shoulder"]["sag_per_cap"]
    assert B.LEVERS["lift"] > T.LEVER and B.RATE < T.RATE
