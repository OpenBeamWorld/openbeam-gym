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
