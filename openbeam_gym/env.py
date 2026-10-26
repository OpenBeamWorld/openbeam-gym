"""Gymnasium environment over an OpenBeam game's robot tools.

One env = one game session. Each `step` is ONE MCP call: `sim_step` applies
the joint targets and advances exactly `physics_steps_per_action` fixed
0.5 ms physics steps, then answers the robot_state. The session must opt in
to stepped physics (`sim_mode stepped`), which freezes the simulation between
calls for anyone watching that session; `reset` does it.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .mcp import GameClient, ToolError
from .tasks import Reach, Task

PHYSICS_DT = 0.0005

# The default robot: the four-joint test arm, a JBeam vehicle mod published in
# this repository (assets/openbeam_test_arm.zip, built by
# openbeam_gym/robots/build_test_arm.py). install_mod takes it as it is.
TEST_ARM_FILE = "openbeam_test_arm.jbeam"
TEST_ARM_URL = "https://raw.githubusercontent.com/OpenBeam-OSS/openbeam-gym/main/assets/openbeam_test_arm.zip"
TEST_ARM_SHA256 = "1464f6b06318edb3c3abfc4385fa1690fbffe582c143729d67cc314d12e892a9"

# A G1-class humanoid (25 joints, fixed base on a pedestal), a JBeam vehicle mod
# published in this repository (assets/openbeam_humanoid_g1.zip, built by
# openbeam_gym/robots/build_humanoid.py). See humanoid_g1_scene().
HUMANOID_G1_FILE = "openbeam_humanoid_g1.jbeam"
HUMANOID_G1_URL = "https://raw.githubusercontent.com/OpenBeamWorld/openbeam-gym/main/assets/openbeam_humanoid_g1.zip"
HUMANOID_G1_SHA256 = "c70fbb353c0cfb6c6a80356e3d301f59bc1cb93cd58a3017276a1823c51edb74"


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass
class SceneConfig:
    """What reset() makes sure exists in the session."""

    terrain: str | None = "simple2.terrn2"     # load_level {terrain}; None to keep whatever is loaded
    level_id: str | None = None                # or a BeamNG level id (load_level {level_id})
    robot_file: str | None = TEST_ARM_FILE     # spawn_vehicle {file} when no robot is present
    robot_position: list | None = None         # world position for that spawn
    install_path: str | None = None            # install_mod {path} before spawning (local games)
    install_url: str | None = TEST_ARM_URL     # or install_mod {url, sha256} (cloud games); None: already installed
    install_sha256: str | None = TEST_ARM_SHA256
    spec: dict | None = None                   # robot_attach {spec} (an openbeam-robot/1 table)
    home: dict = field(default_factory=dict)   # joint name -> radians for reset; default mid-range
    settle_steps: int = 600                    # physics steps after moving to home (0.3 s)
    end_effector: str | None = None


def humanoid_g1_scene(end_effector: str = "left_hand", **overrides) -> SceneConfig:
    """SceneConfig for the G1-class humanoid: installs it by URL, spawns it and
    uses its neutral standing pose as the reset home (the default mid-range
    home would splay the hips). End effectors: left_hand, right_hand,
    left_foot, right_foot."""
    from .robots.humanoid_g1 import HOME

    args = dict(robot_file=HUMANOID_G1_FILE, install_url=HUMANOID_G1_URL, install_sha256=HUMANOID_G1_SHA256,
                home=dict(HOME), end_effector=end_effector)
    args.update(overrides)
    return SceneConfig(**args)


class OpenBeamRobotEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, game: GameClient | Callable[[], GameClient], task: Task | None = None,
                 scene: SceneConfig | None = None, control_hz: float = 20.0, action_mode: str = "delta",
                 max_delta: float = 0.15, joints: list[str] | None = None, image: dict | None = None,
                 render_mode: str | None = None, robot=None):
        super().__init__()
        self._factory = game if callable(game) and not isinstance(game, GameClient) else None
        self.game: GameClient = game if isinstance(game, GameClient) else None  # type: ignore[assignment]
        self.task = task or Reach()
        self.scene = scene or SceneConfig()
        self.physics_steps_per_action = max(1, int(round(1.0 / (control_hz * PHYSICS_DT))))
        self.action_mode = action_mode
        self.max_delta = max_delta
        self.image = image                    # e.g. {"max_width": 128, "format": "jpeg"}
        self.render_mode = render_mode
        self.robot_arg = {} if robot is None else {"robot": robot}
        self.state: dict = {}
        self.timings: list[float] = []        # seconds per step() round trip
        self._steps = 0
        self._ready = False
        self._requested_joints = joints
        self._connect_and_describe()

    # -- setup ------------------------------------------------------------------
    def _connect_and_describe(self) -> None:
        if self.game is None:
            self.game = self._factory()
        self._ensure_scene()
        info = self.game.call("robot_info", self.robot_arg).data
        self.robot_id = info["robot"]["instance_id"]
        self.robot_arg = {"robot": self.robot_id}
        self.end_effector = self.scene.end_effector or info["default_end_effector"]
        effector = next(e for e in info["end_effectors"] if e["name"] == self.end_effector)
        self.chain_joints = list(effector["chain"])
        all_joints = [j["name"] for j in info["joints"]]
        wants_gripper = self.task.name == "pick"
        self.joints = self._requested_joints or [
            j for j in all_joints if j in self.chain_joints or (wants_gripper and j == "gripper")]
        self.limits = {}
        for j in info["joints"]:
            lim = j.get("limits")
            self.limits[j["name"]] = (lim["lower"], lim["upper"]) if lim else (-np.pi, np.pi)
        self.home = {n: self.scene.home.get(n, 0.5 * sum(self.limits[n])) for n in self.joints}
        n = len(self.joints)
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        state_dim = 2 * n + 3 + self.task.extra_dim
        state_space = spaces.Box(-np.inf, np.inf, (state_dim,), np.float32)
        if self.image:
            w = int(self.image.get("max_width", 128))
            h = int(self.image.get("height", round(w * 9 / 16)))
            self.observation_space = spaces.Dict({
                "state": state_space, "image": spaces.Box(0, 255, (h, w, 3), np.uint8)})
        else:
            self.observation_space = state_space
        self._targets = dict(self.home)

    def _ensure_scene(self) -> None:
        status = self.game.call("status").data
        terrain_loaded = bool(status.get("terrain_loaded"))
        if self.scene.terrain and not terrain_loaded:
            self.game.call("load_level", {"terrain": self.scene.terrain, "wait_seconds": 120}, timeout=180)
        elif self.scene.level_id and not terrain_loaded:
            self.game.call("load_level", {"level_id": self.scene.level_id, "wait_seconds": 300}, timeout=360)
        try:
            self.game.call("robot_state", self.robot_arg)
            have_robot = True
        except ToolError:
            have_robot = False
        if not have_robot and self.scene.robot_file:
            if self.scene.install_path:
                self.game.call("install_mod", {"path": self.scene.install_path}, timeout=300)
            elif self.scene.install_url:
                self.game.call("install_mod", {"url": self.scene.install_url, "sha256": self.scene.install_sha256},
                               timeout=300)
            # Entered: stepped physics advances the player's actor.
            args = {"file": self.scene.robot_file, "enter": True}
            if self.scene.robot_position:
                args["position"] = self.scene.robot_position
            self.game.call("spawn_vehicle", args, timeout=120)
            deadline = time.time() + 60
            while True:
                try:
                    self.game.call("robot_state", self.robot_arg)
                    break
                except ToolError:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.5)
        if self.scene.spec:
            self.game.call("robot_attach", {**self.robot_arg, "spec": self.scene.spec})

    # -- helpers ----------------------------------------------------------------
    def joint_position(self, name: str) -> float:
        for j in self.state.get("joints", []):
            if j["name"] == name:
                return float(j["position"])
        raise KeyError(name)

    def world_to_base(self, p) -> np.ndarray:
        base = self.state["base_pose"]
        r = quat_xyzw_to_matrix(base["orientation"])
        return r.T @ (np.asarray(p, dtype=np.float64) - np.asarray(base["position"]))

    def _observe(self) -> np.ndarray | dict:
        by_name = {j["name"]: j for j in self.state["joints"]}
        q = np.array([by_name[n]["position"] for n in self.joints], dtype=np.float32)
        qd = np.array([by_name[n]["velocity"] for n in self.joints], dtype=np.float32)
        ee = np.asarray(self.state["end_effector"]["position"], dtype=np.float32)
        vec = np.concatenate([q, 0.1 * qd, ee, self.task.observation_extra(self)]).astype(np.float32)
        if not self.image:
            return vec
        return {"state": vec, "image": self._camera()}

    def _camera(self) -> np.ndarray:
        answer = self.game.call("camera_frame", {"max_width": int(self.image.get("max_width", 128)),
                                                 "format": self.image.get("format", "jpeg")})
        shape = self.observation_space["image"].shape
        if not answer.images:
            return np.zeros(shape, np.uint8)
        from PIL import Image  # optional dependency: pip install openbeam-gym[image]
        img = Image.open(io.BytesIO(answer.images[0][1])).convert("RGB").resize((shape[1], shape[0]))
        return np.asarray(img, dtype=np.uint8)

    def _sim_step(self, positions: dict, steps: int) -> dict:
        try:
            answer = self.game.call("sim_step", {**self.robot_arg, "steps": steps, "positions": positions})
        except ToolError as e:
            if "stepped mode" not in str(e):
                raise
            # Stepped mode was released (idle timeout, level change): take it again.
            self.game.call("sim_mode", {"mode": "stepped"})
            answer = self.game.call("sim_step", {**self.robot_arg, "steps": steps, "positions": positions})
        self.state = answer.data["state"]
        return answer.data

    # -- gym API ----------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.game.call("sim_mode", {"mode": "stepped", "idle_release_s": 120})
        self._targets = dict(self.home)
        self._sim_step(self._targets, self.scene.settle_steps)
        info = self.task.reset(self, self.np_random)
        self._sim_step(self._targets, 1)
        self._steps = 0
        return self._observe(), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for i, name in enumerate(self.joints):
            lo, hi = self.limits[name]
            if self.action_mode == "delta":
                value = self._targets[name] + float(action[i]) * self.max_delta
            else:
                value = lo + (float(action[i]) + 1.0) * 0.5 * (hi - lo)
            self._targets[name] = float(np.clip(value, lo, hi))
        started = time.perf_counter()
        self._sim_step(self._targets, self.physics_steps_per_action)
        self.timings.append(time.perf_counter() - started)
        self._steps += 1
        reward, terminated, info = self.task.reward(self, action)
        truncated = self._steps >= self.task.max_steps and not terminated
        info["sim_time"] = self.state.get("sim_time")
        return self._observe(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "rgb_array":
            if not self.image:
                self.image = {"max_width": 320}
            return self._camera()
        return None

    def close(self):
        if self.game is not None:
            try:
                self.game.call("sim_mode", {"mode": "realtime"}, raise_on_error=False)
            except Exception:
                pass
            self.game.close()
