# assets

`openbeam_test_arm.zip`: the fast test robot, a four-joint arm (`base_pan`,
`shoulder`, `elbow`, `wrist_roll`, with a `tcp` tool node) packaged as a JBeam
vehicle mod (`vehicles/openbeam_test_arm/`). The game's `install_mod` takes it
as it is; no classic `.truck` format is involved. `SceneConfig()` installs it
with `install_mod {"url": <raw URL of this file>, "sha256": <its SHA-256>}`
(`openbeam_gym.env.TEST_ARM_URL` / `TEST_ARM_SHA256`) and spawns
`openbeam_test_arm.jbeam`.

- Links are stiff node clusters; each joint is a hinge on two axis nodes,
  driven by four JBeam `torsionHydros` whose rest angle follows the electric
  `arm_cmd_<joint>` (radians from the spawn pose, 1 rad/s).
- `openbeam_robot.json` (`openbeam-robot/1`) declares the joints, limits and
  target electrics for the game's robot tools; the vehicle Lua publishes
  `robot_dof`, `robot_q<i>`, `robot_qd<i>`, `robot_cmd<i>` and
  `arm_<joint>_pos|vel|target`, and binds keys (A/D, W/S, Q/E, Z/X, 3 home)
  that also make it a seat for the player, which stepped physics needs.
- It is a soft body: expect a few centimetres of sag under gravity, which
  `robot_fk` reports as `simulated` error.
- It predates the fixed-step sizing of the other robots and asks for more
  than the game's per-node budgets (worst torsion share 14.1 against 0.8,
  beams 5.0 against 2.6), so the game scales its torsion bars (to 5.7 % at
  worst) and lowers its busiest beams at spawn. It is kept as it is so the
  Reach results stay comparable; `--report` prints these numbers.

Rebuild it (deterministic, stdlib only):

```sh
python -m openbeam_gym.robots.build_test_arm --out assets/openbeam_test_arm.zip
```

and update `TEST_ARM_SHA256` in `openbeam_gym/env.py` when it changes.

## openbeam_camera_boom.zip

An overhead camera boom (`vehicles/openbeam_camera_boom/`) that stands beside
a tabletop arm and holds a camera looking straight down at its workspace, for
top-down teleoperation. Same pattern as the test arm: a pinned stand, four
torsion-hydro hinges driven by `boom_cmd_<joint>` (`base_pan`, `lift`,
`elbow`, `tilt`, 0.5 rad/s), an `openbeam_robot.json` and a vehicle Lua that
publishes the robot block and `boom_<joint>_pos|vel|target`.

- Home (the spawn pose, key 3): the camera 1.0 m above the surface the boom
  stands on, over (0.55, 0) in the boom's base frame, looking straight down,
  image up pointing away from the mast. A 50 deg vertical field of view at 4:3
  covers the default 0.6 x 0.6 m workspace with margin. The camera stays nadir
  while `lift + elbow + tilt = 0`; with the nadir lock on (after home), W/S and
  E/Q also move the tilt goal. Keys: A/D pan, W/S lift, E/Q elbow, X/Z tilt.
- The `camera` block of `openbeam_robot.json` is the source of truth: the
  nodes `cam` (optical centre), `cam_fwd` (on the optical axis) and `cam_up`
  (image up), `fov_deg` (vertical) and `aspect`. A viewer places its render
  camera from those node positions each frame. `end_effectors` declares
  `camera` at the optical centre (OpenCV axes), so `robot_fk`/`robot_ik` work.
  No JBeam camera is placed at the tip: the importer's single
  `camerasInternal` eye always looks along the vehicle frame and is a heavy
  sprung node; the one `driver` row is the player's seat on the pinned base.
- `openbeam_gym.topdown` maps image pixels to the table and back, and image
  drags to table moves.
- Sizing: beams and torsion bars are sized within the game's 0.5 ms
  fixed-step budgets (`openbeam_gym/robots/sizing.py`, shared with the
  humanoid), so the game scales nothing at spawn (worst torsion share 0.68 of
  the 0.8 limit). The mass sits on the joints' lever rings and axis nodes
  (about 114 kg moving), the joints are damped at zeta = 0.05 (damping costs
  4000x stiffness in that budget), and `--report` prints each joint's
  stiffness, damping and gravity sag (worst case under 2.3 mrad).

```sh
python -m openbeam_gym.robots.build_camera_boom --out assets/openbeam_camera_boom.zip --report
```

and update `CAMERA_BOOM_SHA256` in `openbeam_gym/env.py` when it changes.

## openbeam_humanoid_g1.zip

A G1-class humanoid (about 1.3 m, 35 kg; 25 joints: six per leg, waist yaw,
roll and pitch, five per arm), packaged as a JBeam vehicle mod
(`vehicles/openbeam_humanoid_g1/`) and built from public proportions and
joint layout with this repository's own geometry. `humanoid_g1_scene()` in
`openbeam_gym.env` installs it with `install_mod {"url": HUMANOID_G1_URL,
"sha256": HUMANOID_G1_SHA256}`, spawns `openbeam_humanoid_g1.jbeam` and uses
the neutral standing pose as the reset home.

- **Fixed base (v1)**: the pelvis is braced to a 150 kg pedestal (part of
  the vehicle) that stands on the ground behind and around the feet. The soles
  hang 5 cm above the ground, and the foot nodes collide, so a toe pointed
  far enough down touches the ground lightly. Building with
  `build(stand=False)` leaves the pedestal out (free-standing, needs balance
  control).
- Joints are hinges between node clusters, each driven by two `torsionHydros`
  whose rest angle follows the electric `g1_cmd_<joint>` (radians from the
  spawn pose) at a per-joint rate (2-8 rad/s: hips and waist slowest, elbows
  and wrists fastest). The joint table is `openbeam_gym/robots/humanoid_g1.py`.
- `openbeam_robot.json` declares the joints, limits, the pelvis base frame,
  end effectors `left_hand`, `right_hand`, `left_foot`, `right_foot`, the
  `home` pose and a `retarget` block (MediaPipe BlazePose landmarks ->
  joints, see `openbeam_gym/retarget.py`).
- To stream targets (e.g. 30 Hz), send one vehicle-Lua command per frame:
  `extensions.get("openbeam_humanoid_g1").set({q1, ..., q25})` in the spec's
  joint order (or a name -> radians table); `retarget.lua_set_command()`
  builds it. `robot_set_joints` works too, through the electrics.
- The vehicle Lua publishes `robot_dof`, `robot_q<i>`, `robot_qd<i>`,
  `robot_cmd<i>` and `g1_<joint>_pos|vel|target`, and binds 3 (home pose),
  4 (zero pose) and A/D (waist yaw).
- Size and sizing: 185 nodes (175 body, 10 pedestal), 1022 beams, 50 torsion
  hydros. Beams and bars are sized within the game's 0.5 ms fixed-step
  budgets, so the game scales nothing at spawn; `--report` prints each joint's
  stiffness, damping and worst-case gravity sag (under 0.08 rad).

Rebuild it (deterministic, stdlib only) and update `HUMANOID_G1_SHA256`:

```sh
python -m openbeam_gym.robots.build_humanoid --out assets/openbeam_humanoid_g1.zip --report
```
