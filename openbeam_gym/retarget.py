# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Human pose -> G1-class humanoid joint angles (``openbeam_humanoid_g1``).

Input: the 33 MediaPipe BlazePose *world* landmarks of one frame (metres,
origin between the hips; MediaPipe's axes: x to the image right, y down, z
away from the camera), each with an optional visibility. Output: the 25 joint
angles of ``openbeam_gym/robots/humanoid_g1.py`` in radians, inside the
limits, smoothed (one-euro or exponential) and rate limited per joint. Pure
Python, stdlib only, so a phone app or a server can run it as it is.

Conventions (also published as the ``retarget`` block of the robot's
``openbeam_robot.json``, see ``retarget_spec()``):

* Landmarks are first turned into robot axes: ``(x, y, z)_robot =
  (-z, x, -y)_mediapipe`` (forward toward the camera, the subject's left, up).
  Only "up" depends on the camera: it is assumed upright (pass ``up`` for a
  tilted one). ``mirror=True`` swaps left and right (a selfie-style mirror).
* **Pelvis frame** P: y from the right hip to the left hip, z the camera's up
  made orthogonal to y, x = y cross z. The robot's pelvis is fixed, so the
  joints are angles of the body *relative* to the subject's pelvis.
* **Torso frame** T: y from the right to the left shoulder, z from the hip
  centre to the shoulder centre made orthogonal to y, x = y cross z. The waist
  is ``P^T T = Rz(waist_yaw) Rx(waist_roll) Ry(waist_pitch)``.
* **Legs**, in P: the thigh direction (hip -> knee) gives hip roll and pitch
  (``Ry(pitch) Rx(roll)`` applied to straight down); the shin direction in that
  frame gives the knee (the angle between thigh and shin) and, when the knee is
  bent, the hip yaw (the plane of the bend); with a straight knee hip yaw comes
  from the foot's heading (heel -> foot index). The foot in the shin frame
  gives ankle pitch (toes down positive) and, from the ankle's height above
  the heel-toe line, ankle roll (scaled by ``ankle_roll_gain``: BlazePose sees
  little of it).
* **Arms**, in T: the upper arm (shoulder -> elbow) gives shoulder roll and
  pitch, the forearm (elbow -> wrist) in that frame gives the elbow and, when
  the elbow is bent, shoulder yaw; the hand's index-minus-pinky direction
  about the forearm gives wrist roll (zero: palm facing the thigh, thumb
  forward).
* Gimbal and ambiguity: shoulder pitch is undefined with the arm straight out
  sideways (roll near 90 degrees), shoulder yaw with a straight elbow, hip yaw
  with a straight knee and no visible foot. Near those poses the value blends
  into (or holds) its previous value instead of jumping.
* Each joint then gets ``value = sign * gain * angle + offset`` (all 1, 1, 0
  by default, kept as parameters for calibration), is unwrapped next to its
  previous value, clamped to the limits, filtered, rate limited to
  ``rate_scale`` times the robot's actuator rate, and clamped again.
* A joint whose landmarks are missing (None, NaN, or visibility below
  ``visibility``) keeps its previous value.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .robots import humanoid_g1

NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
LANDMARKS = {n: i for i, n in enumerate(NAMES)}

JOINT_NAMES = humanoid_g1.JOINT_NAMES
LIMITS = humanoid_g1.LIMITS

_TORSO = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
_PELVIS = ["left_hip", "right_hip"]


def _joint_map() -> dict[str, dict]:
    """Per joint: the landmarks it needs and what it measures."""
    m: dict[str, dict] = {}
    for side in ("left", "right"):
        hip, knee, ankle = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
        heel, toe = f"{side}_heel", f"{side}_foot_index"
        sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
        idx, pk = f"{side}_index", f"{side}_pinky"
        m[f"{side}_hip_pitch"] = {"landmarks": _PELVIS + [knee],
                                  "quantity": "thigh (hip->knee) in the pelvis frame: atan2(-d.x, -d.z)"}
        m[f"{side}_hip_roll"] = {"landmarks": _PELVIS + [knee],
                                 "quantity": "thigh in the pelvis frame: asin(d.y)"}
        m[f"{side}_hip_yaw"] = {"landmarks": _PELVIS + [knee, ankle, heel, toe],
                                "quantity": "bent knee: plane of thigh and shin, atan2(-e.y, -e.x) of the shin in the "
                                            "hip frame; straight knee: foot heading (heel->foot index), atan2(f.y, f.x)"}
        m[f"{side}_knee"] = {"landmarks": _PELVIS + [knee, ankle],
                             "quantity": "angle between thigh and shin, acos(-e.z)"}
        m[f"{side}_ankle_pitch"] = {"landmarks": _PELVIS + [knee, ankle, heel, toe],
                                    "quantity": "foot (heel->foot index) in the shin frame: atan2(-f.z, f.x)"}
        m[f"{side}_ankle_roll"] = {"landmarks": _PELVIS + [knee, ankle, heel, toe],
                                   "quantity": "ankle above the heel-toe line, about the foot axis: atan2(-u.y, u.z)",
                                   "gain": 0.5}
        m[f"{side}_shoulder_pitch"] = {"landmarks": _TORSO + [el],
                                       "quantity": "upper arm (shoulder->elbow) in the torso frame: atan2(-d.x, -d.z); "
                                                   "holds near roll = +-90 deg"}
        m[f"{side}_shoulder_roll"] = {"landmarks": _TORSO + [el], "quantity": "upper arm in the torso frame: asin(d.y)"}
        m[f"{side}_shoulder_yaw"] = {"landmarks": _TORSO + [el, wr],
                                     "quantity": "forearm (elbow->wrist) in the shoulder frame: atan2(e.y, e.x); holds "
                                                 "with a straight elbow"}
        m[f"{side}_elbow"] = {"landmarks": _TORSO + [el, wr], "quantity": "angle between upper arm and forearm, "
                                                                          "acos(-e.z)"}
        m[f"{side}_wrist_roll"] = {"landmarks": _TORSO + [el, wr, idx, pk],
                                   "quantity": "index-minus-pinky in the forearm frame: atan2(-v.y, v.x)"}
    m["waist_yaw"] = {"landmarks": list(_TORSO), "quantity": "P^T T = Rz(yaw) Rx(roll) Ry(pitch): atan2(-R01, R11)"}
    m["waist_roll"] = {"landmarks": list(_TORSO), "quantity": "asin(R21)"}
    m["waist_pitch"] = {"landmarks": list(_TORSO), "quantity": "atan2(-R20, R22)"}
    for v in m.values():
        v.setdefault("sign", 1.0)
        v.setdefault("offset", 0.0)
        v.setdefault("gain", 1.0)
    return {n: m[n] for n in JOINT_NAMES}


JOINT_MAP = _joint_map()


# -- small vector / matrix helpers ----------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a):
    n = math.sqrt(_dot(a, a))
    if n < 1e-9:
        return None
    return (a[0] / n, a[1] / n, a[2] / n)


def _frame(y_dir, z_hint):
    """Columns (x, y, z): y along y_dir, z the part of z_hint orthogonal to it."""
    y = _unit(y_dir)
    if y is None:
        return None
    z = _unit(_sub(z_hint, _mul(y, _dot(z_hint, y))))
    if z is None:
        return None
    return (_cross(y, z), y, z)


def _to(frame, v):
    """Coordinates of v in a frame given as columns."""
    return (_dot(frame[0], v), _dot(frame[1], v), _dot(frame[2], v))


def _from(frame, v):
    return _add(_add(_mul(frame[0], v[0]), _mul(frame[1], v[1])), _mul(frame[2], v[2]))


def _rot(axis, angle):
    """Rotation about a unit axis as a function on vectors (Rodrigues)."""
    c, s = math.cos(angle), math.sin(angle)

    def apply(v):
        return _add(_add(_mul(v, c), _mul(_cross(axis, v), s)), _mul(axis, _dot(axis, v) * (1 - c)))
    return apply


def _compose(frame, *rotations):
    """frame * R1 * R2 ...: the columns rotated, as body-fixed rotations."""
    cols = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for r in reversed(rotations):
        cols = [r(c) for c in cols]
    return tuple(_from(frame, c) for c in cols)


X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
NEG_Y, NEG_Z = (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _blend_angle(a, b, w):
    """w * a + (1 - w) * b on the circle (b may be None: then a)."""
    if b is None:
        return a
    return math.atan2(w * math.sin(a) + (1 - w) * math.sin(b), w * math.cos(a) + (1 - w) * math.cos(b))


def _ramp(x, start, width):
    return _clamp((x - start) / width, 0.0, 1.0)


# -- landmarks ----------------------------------------------------------------------

def mediapipe_to_robot(p) -> tuple:
    """MediaPipe world axes (x right, y down, z away) -> robot axes (x toward
    the camera, y the subject's left, z up)."""
    return (-p[2], p[0], -p[1])


def robot_to_mediapipe(p) -> tuple:
    return (p[1], -p[2], -p[0])


def _read(lm):
    """(x, y, z, visibility) from a tuple/list, dict or landmark object."""
    if lm is None:
        return None
    if isinstance(lm, dict):
        vals = (lm.get("x"), lm.get("y"), lm.get("z"), lm.get("visibility", 1.0))
    elif hasattr(lm, "x"):
        vals = (lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0))
    else:
        seq = list(lm)
        if len(seq) < 3:
            return None
        vals = (seq[0], seq[1], seq[2], seq[3] if len(seq) > 3 else 1.0)
    try:
        x, y, z, v = (float(c) if c is not None else math.nan for c in vals)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(c) for c in (x, y, z)):
        return None
    return x, y, z, (v if math.isfinite(v) else 0.0)


def landmark_points(landmarks: Sequence, visibility: float = 0.5, mirror: bool = False) -> dict[str, tuple]:
    """Usable landmarks by name, in robot axes; missing or less visible than
    `visibility` are left out."""
    pts = {}
    lms = list(landmarks) if landmarks is not None else []
    for name, i in LANDMARKS.items():
        row = _read(lms[i]) if i < len(lms) else None
        if row is None or row[3] < visibility:
            continue
        p = mediapipe_to_robot(row[:3])
        if mirror:
            p = (p[0], -p[1], p[2])
            name = humanoid_g1.mirror_name(name)
        pts[name] = p
    return pts


# -- geometry -> raw angles -------------------------------------------------------------

def raw_angles(pts: dict[str, tuple], previous: dict[str, float] | None = None,
               up=(0.0, 0.0, 1.0)) -> dict[str, float]:
    """Joint angles (radians, before sign/gain/offset, limits and filtering)
    measurable from `pts` (robot axes). `previous` resolves the ambiguous
    angles (see the module notes); joints that cannot be measured are absent."""
    prev = previous or {}
    out: dict[str, float] = {}
    if "left_hip" not in pts or "right_hip" not in pts:
        return out
    pelvis = _frame(_sub(pts["left_hip"], pts["right_hip"]), up)
    if pelvis is None:
        return out
    hip_mid = _mul(_add(pts["left_hip"], pts["right_hip"]), 0.5)
    torso = None
    if "left_shoulder" in pts and "right_shoulder" in pts:
        sh_mid = _mul(_add(pts["left_shoulder"], pts["right_shoulder"]), 0.5)
        tf = _frame(_sub(pts["left_shoulder"], pts["right_shoulder"]), _sub(sh_mid, hip_mid))
        if tf is not None:
            # Waist: R = P^T T = Rz(yaw) Rx(roll) Ry(pitch).
            cols = [_to(pelvis, c) for c in tf]          # columns of R
            r = [[cols[c][rr] for c in range(3)] for rr in range(3)]
            roll = math.asin(_clamp(r[2][1], -1.0, 1.0))
            out["waist_roll"] = roll
            out["waist_pitch"] = math.atan2(-r[2][0], r[2][2])
            out["waist_yaw"] = math.atan2(-r[0][1], r[1][1])
            torso = tf
    for side in ("left", "right"):
        _leg(side, pts, pelvis, prev, out)
        if torso is not None:
            _arm(side, pts, torso, prev, out)
    return out


def _limb_root(d, prev_pitch, hold_below):
    """Roll and pitch of a limb direction d (frame coordinates): d =
    Ry(pitch) Rx(roll) (0, 0, -1). Pitch is held near roll = +-90 deg."""
    roll = math.asin(_clamp(d[1], -1.0, 1.0))
    cr = math.cos(roll)
    pitch = math.atan2(-d[0], -d[2]) if cr > 1e-6 else (prev_pitch or 0.0)
    w = _ramp(cr, hold_below, 0.2)
    return roll, _blend_angle(pitch, prev_pitch, w) if prev_pitch is not None else pitch


def _leg(side, pts, pelvis, prev, out):
    hip, knee, ankle = (pts.get(f"{side}_{k}") for k in ("hip", "knee", "ankle"))
    if hip is None or knee is None:
        return
    d = _unit(_to(pelvis, _sub(knee, hip)))
    if d is None:
        return
    roll, pitch = _limb_root(d, prev.get(f"{side}_hip_pitch"), 0.1)
    out[f"{side}_hip_roll"], out[f"{side}_hip_pitch"] = roll, pitch
    hip_frame = _compose(pelvis, _rot(Y, pitch), _rot(X, roll))
    heel, toe = pts.get(f"{side}_heel"), pts.get(f"{side}_foot_index")
    foot = _unit(_sub(toe, heel)) if heel is not None and toe is not None else None
    if ankle is None:
        return
    e = _unit(_to(hip_frame, _sub(ankle, knee)))
    if e is None:
        return
    k = math.acos(_clamp(-e[2], -1.0, 1.0))
    out[f"{side}_knee"] = k
    yaw_knee = math.atan2(-e[1], -e[0]) if math.sin(k) > 1e-6 else None
    yaw_foot = None
    if foot is not None:
        f = _to(hip_frame, foot)
        if f[0] * f[0] + f[1] * f[1] > 0.04:
            yaw_foot = math.atan2(f[1], f[0])
    w = _ramp(k, 0.2, 0.3)                 # bent knee: trust the bend plane
    prev_yaw = prev.get(f"{side}_hip_yaw")
    if yaw_knee is not None and yaw_foot is not None:
        yaw = _blend_angle(yaw_knee, yaw_foot, w)
    elif yaw_knee is not None and w > 0:
        yaw = _blend_angle(yaw_knee, prev_yaw, w)
    elif yaw_foot is not None and k < 1.2:
        yaw = yaw_foot
    else:
        yaw = None
    if yaw is not None:
        out[f"{side}_hip_yaw"] = yaw
    else:
        yaw = prev_yaw or 0.0
    if foot is None:
        return
    shin = _compose(hip_frame, _rot(Z, yaw), _rot(Y, k))
    f = _to(shin, foot)
    ap = math.atan2(-f[2], f[0])
    out[f"{side}_ankle_pitch"] = ap
    up = _sub(ankle, heel)
    u = _sub(up, _mul(foot, _dot(up, foot)))
    u = _unit(_to(_compose(shin, _rot(Y, ap)), u))
    if u is not None:
        out[f"{side}_ankle_roll"] = math.atan2(-u[1], u[2])


def _arm(side, pts, torso, prev, out):
    sh, el, wr = (pts.get(f"{side}_{k}") for k in ("shoulder", "elbow", "wrist"))
    if sh is None or el is None:
        return
    d = _unit(_to(torso, _sub(el, sh)))
    if d is None:
        return
    roll, pitch = _limb_root(d, prev.get(f"{side}_shoulder_pitch"), 0.1)
    out[f"{side}_shoulder_roll"], out[f"{side}_shoulder_pitch"] = roll, pitch
    if wr is None:
        return
    arm = _compose(torso, _rot(Y, pitch), _rot(X, roll))
    e = _unit(_to(arm, _sub(wr, el)))
    if e is None:
        return
    theta = math.acos(_clamp(-e[2], -1.0, 1.0))
    out[f"{side}_elbow"] = theta
    prev_yaw = prev.get(f"{side}_shoulder_yaw")
    w = _ramp(theta, 0.25, 0.3)            # straight elbow: yaw is unobservable
    if w > 0 and math.sin(theta) > 1e-6:
        yaw = _blend_angle(math.atan2(e[1], e[0]), prev_yaw, w)
        out[f"{side}_shoulder_yaw"] = yaw
    else:
        yaw = prev_yaw or 0.0
    idx, pk = pts.get(f"{side}_index"), pts.get(f"{side}_pinky")
    if idx is None or pk is None:
        return
    fore = _compose(arm, _rot(Z, yaw), _rot(NEG_Y, theta))
    v = _to(fore, _sub(idx, pk))
    if v[0] * v[0] + v[1] * v[1] > 1e-8:
        out[f"{side}_wrist_roll"] = math.atan2(-v[1], v[0])


# -- filters ------------------------------------------------------------------------

class OneEuro:
    """The one-euro filter (Casiez, Roussel and Vogel, 2012) for one scalar."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.3, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, dt: float) -> float:
        if self.x is None:
            self.x = x
            return x
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


class Exponential:
    """Exponential smoothing with time constant `tau` seconds."""

    def __init__(self, tau: float = 0.08):
        self.tau = tau
        self.x = None

    def __call__(self, x: float, dt: float) -> float:
        if self.x is None or self.tau <= 0:
            self.x = x
            return x
        a = 1.0 - math.exp(-dt / self.tau)
        self.x += a * (x - self.x)
        return self.x


class Retargeter:
    """Streams BlazePose world landmarks to joint targets.

        rt = Retargeter()
        targets = rt.update(result.pose_world_landmarks[0], t=timestamp_s)
        command = rt.lua_command()      # one vehicle-Lua call for every joint

    `smoothing`: "one_euro" (default), "exponential" or None. `rate_scale`
    scales each joint's rate limit (the robot's actuator rate, RATES).
    `initial`: the pose before the first frame (default HOME)."""

    def __init__(self, *, smoothing: str | None = "one_euro", min_cutoff: float = 1.5, beta: float = 0.3,
                 d_cutoff: float = 1.0, tau: float = 0.08, rate_scale: float = 1.0, visibility: float = 0.5,
                 mirror: bool = False, up=(0.0, -1.0, 0.0), dt: float = 1.0 / 30.0,
                 joint_map: dict[str, dict] | None = None, initial: dict[str, float] | None = None):
        self.joint_map = {n: dict(v) for n, v in (joint_map or JOINT_MAP).items()}
        self.visibility = visibility
        self.mirror = mirror
        self.up = mediapipe_to_robot(up)
        self.default_dt = dt
        self.rate = {n: humanoid_g1.RATES[n] * rate_scale for n in JOINT_NAMES}
        if smoothing == "one_euro":
            self.filters = {n: OneEuro(min_cutoff, beta, d_cutoff) for n in JOINT_NAMES}
        elif smoothing == "exponential":
            self.filters = {n: Exponential(tau) for n in JOINT_NAMES}
        elif smoothing is None:
            self.filters = None
        else:
            raise ValueError(f"unknown smoothing {smoothing!r}")
        self.values = dict(initial or humanoid_g1.HOME)
        self.raw: dict[str, float] = {}          # last measured angle per joint (geometry)
        self.measured: set[str] = set()          # joints updated by the last frame
        self.t = None

    def update(self, landmarks: Sequence, t: float | None = None, dt: float | None = None) -> dict[str, float]:
        if dt is None:
            dt = (t - self.t) if (t is not None and self.t is not None) else self.default_dt
        if t is not None:
            self.t = t
        dt = max(1e-4, float(dt))
        pts = landmark_points(landmarks, self.visibility, self.mirror)
        raw = raw_angles(pts, self.raw, self.up)
        self.measured = set()
        for name in JOINT_NAMES:
            m = self.joint_map[name]
            if name not in raw or not all(lm in pts for lm in self._required(name)):
                continue
            self.raw[name] = raw[name]
            lo, hi = LIMITS[name]
            target = m["sign"] * m["gain"] * raw[name] + m["offset"]
            prev = self.values[name]
            target += 2 * math.pi * round((prev - target) / (2 * math.pi))
            target = _clamp(target, lo, hi)
            if self.filters is not None:
                target = self.filters[name](target, dt)
            step = self.rate[name] * dt
            target = _clamp(target, prev - step, prev + step)
            self.values[name] = _clamp(target, lo, hi)
            self.measured.add(name)
        return dict(self.values)

    def _required(self, name):
        # Hip yaw and the ankle need the foot or a bent knee; raw_angles
        # already leaves them out when they cannot be measured.
        need = self.joint_map[name]["landmarks"]
        if name.endswith("hip_yaw"):
            side = name.split("_", 1)[0]
            return [lm for lm in need if lm not in (f"{side}_heel", f"{side}_foot_index", f"{side}_ankle")]
        return need

    def array(self) -> list[float]:
        return [self.values[n] for n in JOINT_NAMES]

    def lua_command(self, values: dict[str, float] | None = None) -> str:
        return lua_set_command(values or self.values)


def lua_set_command(values: dict[str, float] | Iterable[float]) -> str:
    """The single vehicle-Lua command that sets every joint target at once:
    ``extensions.get("openbeam_humanoid_g1").set({...})`` in the spec's order."""
    if isinstance(values, dict):
        seq = [values.get(n, humanoid_g1.HOME[n]) for n in JOINT_NAMES]
    else:
        seq = list(values)
    if len(seq) != len(JOINT_NAMES):
        raise ValueError(f"expected {len(JOINT_NAMES)} joint values")
    body = ",".join(f"{float(v):.5f}" for v in seq)
    return f'extensions.get("openbeam_humanoid_g1").set({{{body}}})'


def retarget_spec() -> dict:
    """The ``retarget`` block of openbeam_robot.json: the conventions above as data."""
    return {
        "source": "mediapipe-blazepose-33-world-landmarks",
        "implementation": "openbeam-gym openbeam_gym/retarget.py",
        "units": {"input": "metres, origin between the hips", "output": "radians"},
        "axes": {"input": "MediaPipe world: x image right, y down, z away from the camera",
                 "robot_from_input": "(x, y, z)_robot = (-z, x, -y)_mediapipe",
                 "robot": "x forward (toward the camera), y the subject's left, z up",
                 "up": "the camera's up, (0, -1, 0) in MediaPipe axes; overridable",
                 "mirror": "optional: swap left and right (selfie-style mirror)"},
        "landmarks": LANDMARKS,
        "frames": {
            "pelvis": "y = unit(left_hip - right_hip); z = up made orthogonal to y; x = y cross z",
            "torso": "y = unit(left_shoulder - right_shoulder); z = (shoulder centre - hip centre) made "
                     "orthogonal to y; x = y cross z",
            "waist": "pelvis^T torso = Rz(waist_yaw) Rx(waist_roll) Ry(waist_pitch)",
            "leg": "pelvis Ry(hip_pitch) Rx(hip_roll) Rz(hip_yaw) Ry(knee) Ry(ankle_pitch) Rx(ankle_roll); zero = "
                   "straight down, foot forward",
            "arm": "torso Ry(shoulder_pitch) Rx(shoulder_roll) Rz(shoulder_yaw) R(-y, elbow) R(-z, wrist_roll); "
                   "zero = hanging, palm facing the thigh, thumb forward",
        },
        "joints": {n: {"landmarks": m["landmarks"], "quantity": m["quantity"], "sign": m["sign"],
                       "gain": m["gain"], "offset": m["offset"], "limits": list(LIMITS[n]),
                       "rate_limit": humanoid_g1.RATES[n]} for n, m in JOINT_MAP.items()},
        "value": "clamp(sign * gain * angle + offset, limits), unwrapped next to the previous value, then filtered "
                 "and rate limited (rad/s), then clamped",
        "ambiguities": {"shoulder_pitch": "held near shoulder_roll = +-90 deg (blend over cos(roll) 0.1..0.3)",
                        "hip_pitch": "held near hip_roll = +-90 deg",
                        "shoulder_yaw": "held with a straight elbow (blend over elbow 0.25..0.55 rad)",
                        "hip_yaw": "bend plane for a bent knee (0.2..0.5 rad blend), else foot heading"},
        "missing": {"visibility_below": 0.5, "policy": "keep the previous value"},
        "smoothing": {"default": "one_euro", "one_euro": {"min_cutoff_hz": 1.5, "beta": 0.3, "d_cutoff_hz": 1.0},
                      "exponential": {"tau_s": 0.08}},
        "command": {"order": JOINT_NAMES,
                    "lua": 'extensions.get("openbeam_humanoid_g1").set({q1, ..., q25})'},
    }
