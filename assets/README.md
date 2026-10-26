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

```sh
python -m openbeam_gym.robots.build_camera_boom --out assets/openbeam_camera_boom.zip
```

and update `CAMERA_BOOM_SHA256` in `openbeam_gym/env.py` when it changes.
