# openbeam-gym

[Gymnasium](https://gymnasium.farama.org/) environments for robot learning on
[OpenBeam](https://openbeam.world), a soft-body physics simulation, driven
through the game's own MCP tools. One environment step is one MCP call: the
game applies the joint targets, advances an exact number of 0.5 ms physics
steps in a stepped (paused) mode and answers the robot's state. The same
environment runs against a game on your machine or a training session in the
OpenBeam cloud.

## Install

```sh
pip install "openbeam-gym[train,image]"      # or, from a checkout: pip install -e ".[train,image,test]"
```

## Quickstart

```python
from openbeam_gym import OpenBeamRobotEnv, Reach, SceneConfig, LocalGame, CloudAccount, CloudGame

# A game on this machine, started with ROR_AGENT_PORT=0: its endpoint file names host, port and token.
game = LocalGame.from_endpoint_file("path/to/agent-endpoint.json")

# Or the OpenBeam cloud. The first call opens OpenBeam's approval page in your browser
# (OAuth); tokens are cached owner-only in ~/.config/openbeam-gym and refreshed.
account = CloudAccount()                              # https://cloud.openbeam.world/mcp
session = account.start_session()                     # a training session: no player tab needed
game = CloudGame(account, session["session_id"]); game.wait_ready()

# SceneConfig() installs and spawns the test arm (assets/openbeam_test_arm.zip) by URL.
env = OpenBeamRobotEnv(game, Reach(), SceneConfig())
obs, info = env.reset(seed=0)
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

The default robot is the test arm in `assets/openbeam_test_arm.zip`, a JBeam
vehicle mod that the game's `install_mod` takes as it is: no classic `.truck`
format is needed. Any vehicle mod with an `openbeam_robot.json`
(`openbeam-robot/1`) works the same way: set `SceneConfig(robot_file=...,
install_url=..., install_sha256=...)`.

For top-down teleoperation, `assets/openbeam_camera_boom.zip` is an overhead
camera boom that stands beside a tabletop arm and looks straight down at its
workspace (`openbeam_gym.env.CAMERA_BOOM_URL` / `CAMERA_BOOM_SHA256`, spawn
`openbeam_camera_boom.jbeam`). Its `openbeam_robot.json` names the camera's
nodes and field of view; `openbeam_gym.topdown` turns them into a camera pose
and maps image pixels and drags to points and moves on the table:

```python
from openbeam_gym import topdown
cam = topdown.camera_from_spec(spec["camera"], node_positions, width=640)   # height from the aspect
table = topdown.Plane.horizontal(z=0.0)
point = topdown.pixel_to_plane(cam, table, (320, 240))       # where the image centre looks
dx, dy = topdown.drag_to_table_xy(cam, table, (320, 240), (12, -30))   # a drag -> a gripper move
```

`assets/openbeam_humanoid_g1.zip` is a G1-class humanoid (25 joints, fixed
base on a pedestal) for pose tracking; `humanoid_g1_scene()` sets it up and
`openbeam_gym.retarget.Retargeter` turns MediaPipe BlazePose world landmarks
into its joint targets (smoothed, rate limited, within the limits):

```python
from openbeam_gym.env import humanoid_g1_scene
from openbeam_gym.retarget import Retargeter

rt = Retargeter()
targets = rt.update(pose_world_landmarks, t=timestamp)   # {joint: radians}
command = rt.lua_command()   # one vehicle-Lua call that sets all 25 joints
```

See `assets/README.md` for both robots.

No game at hand? `openbeam_gym.fake.FakeGame` is an in-process stand-in with
the same tool surface (a rigid kinematic arm), for tests and dry runs:

```python
from openbeam_gym.fake import FakeGame
env = OpenBeamRobotEnv(FakeGame(), Reach(), SceneConfig())
```

## Sessions

- **Local**: `openbeam_gym.launch.LocalGameProcess(executable, home)` starts a game
  with an isolated user directory, `ROR_AGENT_PORT=0`, a random token and an
  endpoint file. Connections are persistent (the game keeps `POST /mcp` alive).
- **Cloud**: `CloudAccount.start_session()` creates a **training** session. It
  may use stepped physics (refused in play and showcase sessions), runs without
  a player tab, and ends after 10 minutes without an MCP request, after 2 hours
  without a viewer, or at the account's normal session limit. It uses your play
  time and session limits like the Play button. Open its `player_url` only if
  you want to watch. Headless runs can set `OPENBEAM_MCP_ACCESS_TOKEN`.

## The environment

- **Robot**: whatever the game's `robot_info` tool reports. That can be an actor with
  rotators, an actor with an `openbeam-robot/1` joint table (`SceneConfig.spec`,
  or shipped in its package as `openbeam_robot.json`), or the SO-101 arm.
  `openbeam_gym/robots/so101.draft.json` is a draft SO-101 table.
- **reset(seed)** loads the scene if no terrain is loaded and spawns the robot if
  none is present. It enters stepped mode, drives the robot home and settles,
  then the task draws its goal from the seeded generator.
- **step(action)** takes actions in [-1, 1] per joint, as target deltas
  (`max_delta` rad) or as absolute targets across the limits. `control_hz`
  (default 20) sets the physics steps per action (100 at 20 Hz).
- **Observation**: joint positions, scaled velocities, the end-effector position
  (base frame) and the task's extras. With `image={"max_width": 128}` the
  observation becomes a Dict that adds the camera (an in-memory JPEG frame).
- **Tasks**: `Reach` (goal points the soft body can actually reach; reward is
  -distance, success within `tolerance`) and `Pick` (grasp an object actor and
  lift it).
- **Vector envs**: `vector.local_vec_env(n, ...)` and
  `vector.start_cloud_sessions` with `vector.cloud_vec_env`. Each sub-environment
  gets its own session, stepped in parallel processes.

## Training example

```sh
python examples/train_ppo_reach.py --fake 4 --timesteps 60000          # no game
python examples/train_ppo_reach.py --endpoint G0/agent-endpoint.json --endpoint G1/agent-endpoint.json \
    --control-hz 10 --tolerance 0.05 --episode-steps 50 --timesteps 50000
```

Each run writes `curve.csv`, `model.zip` and `summary.json`. `examples/results/`
holds a reference run on a live four-joint rotator arm: 2 local sessions, 50k
steps, about 32 environment steps/s; success rose from 9% to 44% and mean
return from -31.7 to -12.1.

## Throughput

```sh
python -m openbeam_gym.bench --endpoint G0/agent-endpoint.json
python -m openbeam_gym.bench --cloud --session SESSION_ID
```

The game serves MCP between frames, so one call costs one game frame and
steps/s per session is about the game's frame rate. Run more sessions for more
throughput.

## Tests

```sh
pip install -e ".[test]" && pytest
```

## License

Apache-2.0. Copyright 2026 OpenBeam, LLC. See `LICENSE` and `NOTICE`.
