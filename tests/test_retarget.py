# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""BlazePose world landmarks -> G1-class joint angles, on synthetic poses.

`landmarks_for(pose)` poses a human skeleton with the robot's own joint chain
(the conventions retarget.py documents) and returns the 33 landmarks in
MediaPipe world axes, so a retargeter that follows its conventions recovers
the pose exactly."""

import math
import random

import pytest

from openbeam_gym import retarget as rt
from openbeam_gym.robots import humanoid_g1

X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
NEG_Y, NEG_Z = (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)
THIGH, SHIN, UPPER, FORE, TORSO = 0.42, 0.40, 0.28, 0.25, 0.50


def chain(*rotations):
    return rt._compose((X, Y, Z), *rotations)


def at(origin, frame, v):
    return rt._add(origin, rt._from(frame, v))


def landmarks_for(pose: dict, visibility: float = 0.99, hidden=()):
    q = {**humanoid_g1.ZERO, **pose}
    pts = {"left_hip": (0.0, 0.1, 0.0), "right_hip": (0.0, -0.1, 0.0)}
    for side in ("left", "right"):
        j = lambda n: q[f"{side}_{n}"]  # noqa: E731
        hip = pts[f"{side}_hip"]
        thigh = chain(rt._rot(Y, j("hip_pitch")), rt._rot(X, j("hip_roll")))
        knee = at(hip, thigh, (0, 0, -THIGH))
        shin = rt._compose(thigh, rt._rot(Z, j("hip_yaw")), rt._rot(Y, j("knee")))
        ankle = at(knee, shin, (0, 0, -SHIN))
        foot = rt._compose(shin, rt._rot(Y, j("ankle_pitch")), rt._rot(X, j("ankle_roll")))
        pts.update({f"{side}_knee": knee, f"{side}_ankle": ankle, f"{side}_heel": at(ankle, foot, (-0.05, 0, -0.06)),
                    f"{side}_foot_index": at(ankle, foot, (0.15, 0, -0.06))})
    torso = chain(rt._rot(Z, q["waist_yaw"]), rt._rot(X, q["waist_roll"]), rt._rot(Y, q["waist_pitch"]))
    for side, s in (("left", 1.0), ("right", -1.0)):
        j = lambda n: q[f"{side}_{n}"]  # noqa: E731
        sh = at((0, 0, 0), torso, (0, s * 0.18, TORSO))
        arm = rt._compose(torso, rt._rot(Y, j("shoulder_pitch")), rt._rot(X, j("shoulder_roll")))
        elbow = at(sh, arm, (0, 0, -UPPER))
        fore = rt._compose(arm, rt._rot(Z, j("shoulder_yaw")), rt._rot(NEG_Y, j("elbow")))
        wrist = at(elbow, fore, (0, 0, -FORE))
        hand = rt._compose(fore, rt._rot(NEG_Z, j("wrist_roll")))
        pts.update({f"{side}_shoulder": sh, f"{side}_elbow": elbow, f"{side}_wrist": wrist,
                    f"{side}_index": at(wrist, hand, (0.03, 0, -0.08)), f"{side}_pinky": at(wrist, hand, (-0.03, 0, -0.07)),
                    f"{side}_thumb": at(wrist, hand, (0.04, -s * 0.01, -0.03))})
    head = at((0, 0, 0), torso, (0.05, 0, TORSO + 0.15))
    out = []
    for name in rt.NAMES:
        p = pts.get(name, head)
        if name in hidden:
            out.append(None)
        else:
            x, y, z = rt.robot_to_mediapipe(p)
            out.append({"x": x, "y": y, "z": z, "visibility": visibility})
    return out


def exact(**kw):
    """A retargeter without smoothing or rate limits, starting at zero."""
    return rt.Retargeter(smoothing=None, rate_scale=1e9, initial=dict(humanoid_g1.ZERO), **kw)


def expected(pose):
    out = {**humanoid_g1.ZERO, **pose}
    for side in ("left", "right"):
        out[f"{side}_ankle_roll"] = rt.JOINT_MAP[f"{side}_ankle_roll"]["gain"] * out[f"{side}_ankle_roll"]
    return {n: min(max(v, humanoid_g1.LIMITS[n][0]), humanoid_g1.LIMITS[n][1]) for n, v in out.items()}


def assert_pose(got, want, tol=1e-6, names=None):
    for n in names or want:
        assert got[n] == pytest.approx(want[n], abs=tol), n


def test_axes_round_trip_and_landmark_formats():
    p = (0.1, -0.2, 0.3)
    assert rt.mediapipe_to_robot(rt.robot_to_mediapipe(p)) == pytest.approx(p)
    # MediaPipe: the subject's left is image-right (+x), up is -y, toward the camera is -z.
    assert rt.mediapipe_to_robot((1, 0, 0)) == (0, 1, 0)
    assert rt.mediapipe_to_robot((0, -1, 0)) == (0, 0, 1)
    assert rt.mediapipe_to_robot((0, 0, -1)) == (1, 0, 0)
    lms = landmarks_for({})
    as_tuples = [(d["x"], d["y"], d["z"], d["visibility"]) for d in lms]

    class Lm:
        def __init__(self, d):
            self.x, self.y, self.z, self.visibility = d["x"], d["y"], d["z"], d["visibility"]

    for variant in (as_tuples, [Lm(d) for d in lms], [t[:3] for t in as_tuples]):
        assert rt.landmark_points(variant).keys() == rt.landmark_points(lms).keys()
    assert len(rt.LANDMARKS) == 33 and rt.LANDMARKS["left_hip"] == 23 and rt.LANDMARKS["right_foot_index"] == 32


def test_arms_down_is_the_zero_pose():
    r = exact()
    assert_pose(r.update(landmarks_for({})), humanoid_g1.ZERO)


def test_t_pose():
    r = exact()
    pose = {"left_shoulder_roll": math.pi / 2, "right_shoulder_roll": -math.pi / 2}
    got = r.update(landmarks_for(pose))
    assert_pose(got, expected(pose))     # shoulder pitch is held (0) at the gimbal, not a jump


def test_elbow_bent_90_with_yaw_and_wrist():
    pose = {"left_elbow": math.pi / 2, "left_shoulder_yaw": 0.3, "left_wrist_roll": 0.5,
            "right_elbow": math.pi / 2, "right_shoulder_yaw": -0.4, "right_wrist_roll": -1.0,
            "left_shoulder_pitch": -0.6}
    assert_pose(exact().update(landmarks_for(pose)), expected(pose))


def test_squat_and_lean():
    pose = {}
    for side in ("left", "right"):
        pose.update({f"{side}_hip_pitch": -1.2, f"{side}_knee": 2.0, f"{side}_ankle_pitch": -0.8})
    pose.update({"left_hip_yaw": 0.2, "right_hip_yaw": -0.2, "left_hip_roll": 0.15, "right_hip_roll": -0.15,
                 "waist_pitch": 0.3, "waist_yaw": -0.25, "waist_roll": 0.1})
    assert_pose(exact().update(landmarks_for(pose)), expected(pose))


def test_random_poses_round_trip():
    rng = random.Random(7)
    for _ in range(50):
        pose = {}
        for side, s in (("left", 1), ("right", -1)):
            pose[f"{side}_hip_pitch"] = rng.uniform(-1.5, 0.4)
            pose[f"{side}_hip_roll"] = s * rng.uniform(-0.3, 0.8)
            pose[f"{side}_hip_yaw"] = rng.uniform(-0.8, 0.8)
            pose[f"{side}_knee"] = rng.uniform(0.6, 2.2)
            pose[f"{side}_ankle_pitch"] = rng.uniform(-0.8, 0.5)
            pose[f"{side}_ankle_roll"] = rng.uniform(-0.25, 0.25)
            pose[f"{side}_shoulder_pitch"] = rng.uniform(-2.0, 0.8)
            pose[f"{side}_shoulder_roll"] = s * rng.uniform(-0.3, 1.2)
            pose[f"{side}_shoulder_yaw"] = rng.uniform(-1.2, 1.2)
            pose[f"{side}_elbow"] = rng.uniform(0.6, 2.4)
            pose[f"{side}_wrist_roll"] = rng.uniform(-1.5, 1.5)
        pose.update({"waist_yaw": rng.uniform(-0.8, 0.8), "waist_roll": rng.uniform(-0.4, 0.4),
                     "waist_pitch": rng.uniform(-0.4, 0.4)})
        assert_pose(exact().update(landmarks_for(pose)), expected(pose))


def test_left_right_mirror_symmetry():
    pose = {"left_shoulder_roll": 0.7, "left_shoulder_pitch": -1.0, "left_shoulder_yaw": 0.5, "left_elbow": 1.2,
            "left_wrist_roll": 0.8, "left_hip_pitch": -0.7, "left_hip_roll": 0.3, "left_hip_yaw": 0.4,
            "left_knee": 1.0, "left_ankle_pitch": 0.3, "waist_yaw": 0.3, "waist_roll": 0.2, "waist_pitch": 0.1}
    a = exact().update(landmarks_for(pose))
    b = exact().update(landmarks_for(humanoid_g1.mirror_pose(pose)))
    assert_pose(b, humanoid_g1.mirror_pose(a))
    # mirror=True on the original landmarks gives the mirrored pose.
    assert_pose(exact(mirror=True).update(landmarks_for(pose)), humanoid_g1.mirror_pose(a))


def test_limits_clamp():
    pose = {"left_hip_roll": -0.9, "right_hip_roll": 0.9, "left_ankle_pitch": 0.5,
            "left_elbow": 0.0, "right_shoulder_roll": 0.0}
    got = exact().update(landmarks_for(pose))
    assert got["left_hip_roll"] == humanoid_g1.LIMITS["left_hip_roll"][0] == -0.52
    assert got["right_hip_roll"] == humanoid_g1.LIMITS["right_hip_roll"][1] == 0.52
    overhead = {"left_shoulder_pitch": -math.pi + 0.05, "right_shoulder_pitch": -math.pi + 0.05}
    got = exact().update(landmarks_for(overhead))
    for side in ("left", "right"):
        assert got[f"{side}_shoulder_pitch"] == humanoid_g1.LIMITS[f"{side}_shoulder_pitch"][0]
    for n, v in got.items():
        lo, hi = humanoid_g1.LIMITS[n]
        assert lo <= v <= hi


def test_missing_or_low_visibility_landmarks_keep_previous_values():
    first = {"left_elbow": 1.0, "left_shoulder_yaw": 0.4, "left_wrist_roll": 0.3, "right_knee": 1.0,
             "right_hip_pitch": -0.5}
    second = {"left_elbow": 2.0, "left_shoulder_yaw": -0.4, "left_wrist_roll": -0.3, "right_knee": 0.3,
              "right_hip_pitch": -0.2, "waist_yaw": 0.2}
    r = exact()
    a = r.update(landmarks_for(first))
    b = r.update(landmarks_for(second, hidden={"left_wrist"}))
    for n in ("left_elbow", "left_shoulder_yaw", "left_wrist_roll"):
        assert b[n] == a[n], n
    assert b["right_knee"] == pytest.approx(0.3) and b["waist_yaw"] == pytest.approx(0.2)
    # Low visibility: the whole frame is ignored.
    c = r.update(landmarks_for(first, visibility=0.2))
    assert c == b
    # NaN coordinates count as missing.
    lms = landmarks_for(first)
    lms[rt.LANDMARKS["right_ankle"]] = {"x": math.nan, "y": 0.0, "z": 0.0, "visibility": 1.0}
    d = r.update(lms)
    assert d["right_knee"] == b["right_knee"] and d["left_elbow"] == pytest.approx(1.0)
    # Nothing at all: everything holds.
    assert r.update([None] * 33) == d
    assert r.update([]) == d


def test_rate_limit_and_smoothing():
    target = {"left_elbow": 2.0, "left_shoulder_pitch": -1.5}
    lms = landmarks_for(target)
    raw = rt.Retargeter(smoothing=None, initial=dict(humanoid_g1.ZERO))
    got = raw.update(lms, dt=1 / 30)
    assert got["left_elbow"] == pytest.approx(humanoid_g1.RATES["left_elbow"] / 30)
    assert got["left_shoulder_pitch"] == pytest.approx(-humanoid_g1.RATES["left_shoulder_pitch"] / 30)
    for _ in range(60):
        got = raw.update(lms, dt=1 / 30)
    assert_pose(got, {**humanoid_g1.ZERO, **target}, names=list(target))
    for smoothing in ("one_euro", "exponential"):
        r = rt.Retargeter(smoothing=smoothing, initial=dict(humanoid_g1.ZERO))
        r.update(landmarks_for({}), t=0.0)
        values = [r.update(lms, t=k / 30)["left_elbow"] for k in range(1, 120)]
        assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))      # no overshoot from a step
        assert all(b - a <= humanoid_g1.RATES["left_elbow"] / 30 + 1e-9 for a, b in zip(values, values[1:]))
        assert values[-1] == pytest.approx(2.0, abs=1e-3)
    # Jitter is attenuated.
    rng = random.Random(3)
    r = rt.Retargeter(initial=dict(humanoid_g1.ZERO))
    out = []
    for k in range(200):
        noisy = landmarks_for({"left_elbow": 1.0 + rng.uniform(-0.05, 0.05)})
        out.append(r.update(noisy, t=k / 30)["left_elbow"])
    tail = out[100:]
    assert max(tail) - min(tail) < 0.05


def test_lua_command_and_spec():
    r = exact()
    r.update(landmarks_for({"left_elbow": 1.0}))
    cmd = r.lua_command()
    assert cmd.startswith('extensions.get("openbeam_humanoid_g1").set({') and cmd.endswith("})")
    values = [float(v) for v in cmd[cmd.index("{") + 1:-2].split(",")]
    assert len(values) == 25 and values[humanoid_g1.JOINT_NAMES.index("left_elbow")] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rt.lua_set_command([0.0] * 3)
    spec = rt.retarget_spec()
    assert spec["landmarks"]["left_shoulder"] == 11 and set(spec["joints"]) == set(humanoid_g1.JOINT_NAMES)
    for name, row in spec["joints"].items():
        assert set(row["landmarks"]) <= set(rt.LANDMARKS)
        assert row["limits"] == list(humanoid_g1.LIMITS[name])
