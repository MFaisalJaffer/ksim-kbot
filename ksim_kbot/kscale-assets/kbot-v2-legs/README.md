# kbot-v2-legs

Self-contained MuJoCo + URDF asset package for the **physical robot under test
in this repo**: torso + both legs, no arms, mass scaled to match the measured
real robot weight (28.8 lb / 13.06 kg).

Derived from upstream `kscale-assets/kbot-v2/` by [`../../strip_arms.py`](../../strip_arms.py),
which:

1. Removes the two arm subtrees (KC_C_104R/L_PitchHardstopDriven and descendants).
2. Removes the 10 arm actuators (shoulder pitch/roll/yaw, elbow, wrist on both
   sides).
3. Removes contact `<exclude>` pairs and unused `<asset>` mesh declarations
   that referenced arm bodies.
4. **Scales every body mass + inertia by 0.4611x** so the total robot mass
   matches 13.06 kg (the CAD masses are heavier than the physical build —
   electronics/payload/structural assumptions in CAD aren't all present).
5. Copies only the meshes actually referenced by the legs-only model (24 of
   the 44 upstream meshes).

## Files

| file | what |
|---|---|
| `robot.mjcf` | Floating-base (freejoint), for sim where the robot stands on its feet. |
| `robot.suspended.mjcf` | Torso welded to world, for bench-rig replays. |
| `robot.urdf` | URDF version (for `leg_test.py --urdf`, ROS, etc.). |
| `meshes/` | 24 leg+torso STL files (~5 MB). |
| `metadata.json` | Per-joint kp/kd/torque-limit metadata from upstream kbot-v2. |

## Verify

```python
import mujoco
m = mujoco.MjModel.from_xml_path('robot.suspended.mjcf')
assert m.nu == 10                       # 5 leg joints x 2
assert abs(sum(m.body_mass) - 13.06) < 0.01  # matches real robot
```

## Regenerate

If the upstream `kscale-assets/kbot-v2/` ever updates, regenerate with:

```bash
python3 gim_test/strip_arms.py \
  --out-dir gim_test/sim_assets/kbot-v2-legs \
  --target-mass-kg 13.06 \
  --include-urdf
```

## Use

For training, point your ksim task at this directory:

```python
robot_urdf_path: str = xax.field(
    value="/Users/faisaljaffer/Documents/GitHub/odrive-mit-control/gim_test/sim_assets/kbot-v2-legs/",
)
```

The `metadata.json` still contains all 20 joints (legs + arms). When ksim's
`MITPositionActuators` builds from it, it'll log a warning for the 10 arm
joints whose actuator names don't exist in the mjcf, and skip them. The 10
leg joints will pick up the correct kp/kd/torque_limit values.

If you want a strict metadata that matches the actuators, strip the 10 arm
entries from `metadata.json` manually.

## Why not use the upstream kbot-v2 directly?

- Upstream kbot-v2 includes both arms (don't exist on this build).
- Upstream CAD masses (~28.3 kg legs-only, ~35.6 kg full) are 2.2x heavier
  than the physical robot. PD gains tuned for the heavier CAD silently rely
  on inertia to damp oscillations; at real mass, the same gains oscillate.
- This package gives the trained policy a sim that matches what it will
  actually run on.
