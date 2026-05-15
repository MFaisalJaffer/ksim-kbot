# K-Bot Walking Policy — Status & Reference

Living document tracking the current state of `walking_joystick_rnn.py`: what's
trained, what's rewarded, what commands the policy accepts, and how to deploy it.

---

## Overview

A single RNN policy that walks the K-Bot 2 humanoid while accepting runtime
commands for **base velocity** *and* **arm pose**. Same policy handles:

- Standing still
- Walking forward / backward / sideways
- Turning
- Holding any commanded arm pose while walking (carry-task robustness)

Network is a GRU-based actor + critic (`KbotRNNActor` / `KbotRNNCritic`) trained
with PPO. Inputs include proprioception, IMU, projected gravity, gait phase,
last action, and all four commands listed below.

---

## Commands (runtime control surface)

All commands are `ksim.Command` subclasses produced once per episode (some
re-sample mid-episode, some don't). At deployment, replace the random sampler
with an external interface (joystick, ROS topic, API) writing into the same
slots — the policy treats them identically.

| Command | Dims | Switch | Notes |
|---|---|---|---|
| `linear_velocity_command` | 2 | every ~3s | `(vx, vy)` in m/s. Range trained: `vx ∈ [-0.3, 0.7]`, `vy ∈ [-0.2, 0.2]`. 30% zero probability for stand-still practice. |
| `angular_velocity_command` | 1 | every ~3s | `wz` in rad/s. Scale 0.1, 90% zero probability (turning is rare). |
| `gait_frequency_command` | 1 | per-episode | Step rate in Hz. Trained `[1.25, 1.5]`. Feeds into the foot-phase clock that defines the swing/stance schedule. |
| `arm_constraint_command` | 11 | per-episode | `[is_constrained, 10× target_joint_angle]`. `is_constrained=1` sampled with prob `0.3 × curriculum_level` (curriculum-gated — disabled at level 0 so walking is learned first, ramps in to full 30% by level 1). Arm targets uniformly sampled within MJCF joint ranges (conservative bounds). |

### Arm joint order (indices 1–10 of `arm_constraint_command`)
```
right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist
left arm:  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist
```

### Velocity sign convention
- `vx > 0` → forward
- `vy > 0` → strafe left
- `wz > 0` → yaw left

---

## Rewards (positive signal — encourage)

| Reward | Scale | What it rewards |
|---|---|---|
| `LinearVelocityTrackingReward` | 3.0 | Tracks the commanded `vx, vy` velocity. Exp-shaped, peaks when actual body velocity matches command. |
| `AngularVelocityTrackingReward` | 1.0 | Tracks the commanded `wz`. |
| `FeetPhaseReward` | 2.1 | Rewards each foot following the gait-clock height trajectory (cubic-Bezier stance→swing curve). Carrot for proper gait timing. |
| `StandStillReward` | 8 | Reward for *not moving* when commanded velocity is near zero. Has an `orientation_sensitivity=0.05` gate that fades the reward when the robot leans, so balance recovery isn't punished. Reduced 50→15→8 over training — previous values dominated and forced policy to a "stand still forever" local optimum. |
| `FootAirTimeReward` | 1.0 | **Bootstrap reward** for getting an entire foot off the ground (heel + center + toe all above 3cm) when commanded to walk. Ungated by gait clock — any lift earns reward. Designed to push the policy out of "both feet planted" attractor before the more selective gait rewards can pay out. |
| `SingleFootContactReward` | 0.5 | Rewards walking with only one foot on the ground at a time (canonical biped gait). |
| `WalkingPostureReward` | 2.0 | When commanded to walk: rewards (a) at least `min_knee_bend=0.4 rad` (~23°) of knee flex AND (b) all three foot points (heel + center + toe) above `min_clearance=0.08m` during swing. Both must be satisfied — gates together so the robot can't earn knee reward without lifting feet. |
| ~~`ArmConstraintReward`~~ | ~~3.0~~ | **Removed**. Arms are now driven by an external controller (the actor's arm output is overridden when `is_constrained=1`), so no reward is needed to incentivize matching — they match automatically. |

---

## Penalties (negative signal — discourage)

| Penalty | Scale | What it discourages |
|---|---|---|
| `JointDeviationPenalty` | -0.1 | Soft anchor toward `JOINT_TARGETS` (mostly zeros). Knees and hip pitch have weight `0.01` (effectively free); arms have weight `1.0–1.2`. |
| `HipDeviationPenalty` | – | Penalizes hip yaw/roll drifting from neutral (extra stability for stance leg). |
| `TerminationPenalty` | -1.0 | Flat penalty for episode termination (falling, going underground). |
| `OrientationPenalty` | -5.0 | Penalizes upright deviation (x/y components of rotated up-vector). Large scale because the robot was abusing the underground exploit when this was weaker. |
| `AngularVelocityXYPenalty` | – | Discourages rolling / pitching the torso. |
| `FeetSlipPenalty` | -0.25 | True foot-velocity slip detection: differences consecutive foot positions to compute world-frame xy velocity, then penalizes that × `is_in_contact`. Zero during normal stepping (planted foot has zero velocity), fires only when a foot actually slides along the ground. |
| `JointPositionLimitPenalty` | – | Quadratic penalty for joints near their hardware limits. |
| `ContactForcePenalty` | – | Penalizes excessive contact forces (hard floor strikes). |
| `CtrlPenalty` | -0.005 | L2 penalty on the action vector (torque smoothness). |
| `ActionAccelerationPenalty` | -0.005 | Penalty on action second-derivatives (jerk). |
| `JointVelocityPenalty` | -0.005 | Penalty on joint velocities (smoother motion). |
| `KneeRangeOfMotion` | – | Soft envelope around the usable knee range. |
| `NoContactPenalty` | -0.1 | Penalizes flying — at least one foot should be in contact most of the time. |
| `MarchInPlacePenalty` | -0.5 | Penalizes stepping while commanded velocity is zero. |
| `FeetPhasePenalty` | -1.0 | Complement of `FeetPhaseReward`: explicit stick for feet *not* matching the gait phase when commanded to walk. |
| `FootSwingClearancePenalty` | -2.0 | Per-foot, per-timestep penalty for the foot height being below `0.08m` while the gait clock says it should be swinging. Uses `min(heel_z, center_z, toe_z)` so tilting the toe up while the heel drags still triggers the penalty. |

---

## Terminations (episode-ending conditions)

| Termination | Threshold | Why |
|---|---|---|
| `NotUprightTermination` | 1.2 rad (~69°) | Robot has tipped too far. |
| `MinimumHeightTermination` | 0.5 m | Robot has fallen / sunk below half its standing height. Closes the "lie on back, feet up" exploit. |

---

## Sim-to-real features

### TV-curve actuator (`TVCurveMITActuators` in `common.py`)
Replaces the previous constant-torque-clip actuator. Each joint's max torque is
clipped per-step against a velocity-dependent T-V curve from the actuator
datasheet:

| Motor | Used for | Peak τ | No-load ω |
|---|---|---|---|
| GIM_8108_8 (`"04"`) | shoulder pitch/roll, elbow, hip pitch/roll, knee | 22 Nm | 21.5 rad/s |
| GIM_6010_8 (`"03"`,`"02"`) | shoulder yaw, hip yaw, ankle | 11 Nm | 29.8 rad/s |
| (constant clip) `"00"` | wrists | 5 Nm | – |

**Randomization**: each step, per-joint scale ∈ [0.85, 1.0] (only weaker, never
stronger) so the policy doesn't depend on exact peak torque. Real motors weaken
when hot.

### Domain randomization
Floor friction, static friction, body masses, armature (CoM offsets) all
randomized per episode. IMU and joint velocity observations have noise injected
during training.

### Random pushes
`XYPushEvent` and `TorquePushEvent` every 2–4 s, magnitude scaled by curriculum
level (off at level 0, full at level 1).

---

## Curriculum

`EpisodeLengthCurriculum`:
- 20 levels (0.05 increments)
- Bump up after sustained 120-s episodes
- Drop down after 10-s episodes
- Wide hysteresis prevents oscillation

Push strength and other dynamics scale with curriculum level.

---

## How to deploy

The policy expects four commands per step. In simulation/viewer:

```python
commands = {
    "linear_velocity_command":   jnp.array([vx, vy]),
    "angular_velocity_command":  jnp.array([wz]),
    "gait_frequency_command":    jnp.array([gait_hz]),
    "arm_constraint_command":    jnp.array([is_constrained, *target_arm_10]),
}
```

### Example: walk forward at 0.5 m/s, arms free
```python
arm_cmd = jnp.array([0.0,  0,0,0,0,0,  0,0,0,0,0])  # is_constrained = 0
```

### Example: walk forward holding a box (both arms forward, elbows bent)
```python
arm_cmd = jnp.array([
    1.0,                           # is_constrained = 1
    0.5, 0.3, 0.0, 1.8, 0.0,       # right arm: forward + roll + elbow bent
    0.5,-0.3, 0.0,-1.8, 0.0,       # left arm: symmetric
])
```

### Example: stand still with right hand raised (waving pose)
```python
vel_cmd = jnp.zeros(2)
arm_cmd = jnp.array([
    1.0,
    1.5, 0.5, 0.0, 1.8, 0.0,       # right arm raised
    0.0, 0.0, 0.0, -1.4, 0.0,      # left arm default
])
```

No retraining required — the policy was trained on random samples from a
continuous distribution covering the full joint ranges, so any pose within
the sample bounds should work.

---

## Run history (recent)

- `run_60..62`: stuck JIT compilation, killed
- `run_63`: fresh launch with TV curves + arm constraint, but had old FeetSlipPenalty
- `run_64`: current — fresh launch with all fixes applied (proper foot-velocity slip penalty, TV curves, arm constraint)

---

## Files

| File | What's in it |
|---|---|
| `walking/walking_joystick.py` | Base task: commands, observations, actuators, terminations, curriculum |
| `walking/walking_joystick_rnn.py` | RNN actor/critic and the override of `get_rewards()` for this experiment |
| `rewards.py` | All custom reward and penalty classes |
| `common.py` | Custom actuators, observations, commands, events |
| `kscale-assets/kbot-v2-feet/robot.mjcf` | Robot MJCF (symlinked to `~/.kscale/robots/kbot/robot/`, mirrored at `github.com/MFaisalJaffer/kbot-assets`) |
