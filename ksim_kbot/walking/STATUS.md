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
- Holding any commanded arm pose while walking (carry-task robustness — arms are
  driven by an external controller, not the walking policy)

Network is a GRU-based actor + critic (`KbotRNNActor` / `KbotRNNCritic`) trained
with PPO. Inputs include proprioception, IMU, projected gravity, gait phase,
last action, and all four commands listed below.

---

## Initial pose (JOINT_TARGETS)

**Unstable bootstrap pose** — intentionally cannot stand statically. Used to
drive walking discovery in early training (matches run_50's setup).

| Joint group | Values |
|---|---|
| Arms | All zeros except elbow R=+1.4 / L=−1.4 |
| Right leg | hip_pitch=**−0.23**, hip_roll=0, hip_yaw=0, **knee=−0.873** (~50° bend), **ankle=+0.195** |
| Left leg | hip_pitch=**+0.23** (mirrored), hip_roll=0, hip_yaw=0, **knee=+0.873**, **ankle=−0.195** |

**Geometry** (verified via forward-kinematics): heel z ≈ +0.04 m, toe z ≈ +0.15 m
— foot is **toe-up / heel-strike pose**. When the robot is held at this target
under gravity, only the heel touches and the body tips backward. The pose is
unmaintainable statically, so `StandStillReward` keeps pulling toward an
unstable target → policy is forced to discover stepping to balance.

**Phase-2 plan** (when walking is solid — episode length consistently > 30s and
foot_air_time consistently > 0.5): swap JOINT_TARGETS to all-zeros (straight
legs, flat feet) so the policy can finally learn stable static standing.

---

## Commands (runtime control surface)

All commands are `ksim.Command` subclasses produced once per episode (some
re-sample mid-episode, some don't). At deployment, replace the random sampler
with an external interface (joystick, ROS topic, API) writing into the same
slots — the policy treats them identically.

| Command | Dims | Switch | Notes |
|---|---|---|---|
| `linear_velocity_command` | 2 | every ~3s | `(vx, vy)` in m/s. Range trained: `vx ∈ [-0.3, 0.7]`, `vy ∈ [-0.2, 0.2]`. **10% zero probability** (down from 30% — more walking practice). |
| `angular_velocity_command` | 1 | every ~3s | `wz` in rad/s. Scale 0.1, 90% zero probability (turning is rare). |
| `gait_frequency_command` | 1 | per-episode | Step rate in Hz. Trained `[1.25, 1.5]`. Feeds into the foot-phase clock that defines the swing/stance schedule. |
| `arm_constraint_command` | 11 | per-episode | `[is_constrained, 10× target_joint_angle]`. `is_constrained=1` sampled with prob `0.3 × curriculum_level` (curriculum-gated — disabled at level 0). Arm targets uniformly sampled within MJCF joint ranges. |

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

## External arm controller (actor-side action override)

**Key architectural choice**: when `is_constrained = 1`, the policy is **not**
responsible for moving the arms to the target. The actor's `forward()` overrides
the arm portion of the output distribution:

- **Arm position delta** → `target_arm_pose - JOINT_TARGETS_arm` (deterministic)
- **Arm velocity delta** → 0
- **Arm std** → 0.05 (essentially deterministic)
- **Leg actions** → policy-controlled as normal

The PD actuator then drives arm joints toward the commanded target. The policy
gets ~zero gradient on arm outputs when constrained, so it learns to **ignore**
its arms in that mode and balance with legs/torso only.

**Sim-to-real parallel**: mirrors deployment where a separate manipulation
module owns the arms while the walking policy is only informed via the
`is_constrained` flag.

Because the override happens in the actor (not the actuator), it's a
deterministic function of the command — no PPO importance-sampling issues.

---

## Rewards (positive signal — encourage)

| Reward | Scale | What it rewards |
|---|---|---|
| `LinearVelocityTrackingReward` | 3.0 | Tracks the commanded `vx, vy` velocity. Exp-shaped. |
| `AngularVelocityTrackingReward` | 1.0 | Tracks the commanded `wz`. |
| `FeetPhaseReward` | 2.1 | Rewards each foot following the gait-clock height trajectory (cubic-Bezier stance→swing). Carrot for proper gait timing. |
| `StandStillReward` | **8** | Reward for not moving when commanded velocity is near zero. Pulls toward JOINT_TARGETS. Orientation gate (`orientation_sensitivity=0.05`) fades reward when leaning so recovery isn't punished. **Reduced 50→15→8** — previous values dominated and forced "stand still forever" local optimum. |
| `FootAirTimeReward` | 1.0 | Bootstrap reward for any whole-foot lift (heel + center + toe all > 3cm) when commanded to walk. Ungated by gait clock — pushes policy out of "both feet planted" attractor. |
| `SingleFootContactReward` | 0.5 | Rewards walking with only one foot on the ground (biped gait). |
| `WalkingPostureReward` | 2.0 | When walking: rewards (a) knee bend ≥ 0.4 rad AND (b) feet lifted (all three points heel+center+toe ≥ `min_clearance=0.04m`). Both gated together — bent knees alone don't pay. |

---

## Penalties (negative signal — discourage)

| Penalty | Scale | What it discourages |
|---|---|---|
| `JointDeviationPenalty` | -0.1 | Soft anchor toward `JOINT_TARGETS`. Knees and hip pitch have weight `0.01` (effectively free); arms have weight `1.0–1.2`. |
| `HipDeviationPenalty` | – | Penalizes hip yaw/roll drift (extra stance-leg stability). |
| `TerminationPenalty` | -1.0 | Flat penalty for episode termination. |
| `OrientationPenalty` | -5.0 | Penalizes upright deviation (x/y of rotated up-vector). |
| `AngularVelocityXYPenalty` | – | Discourages torso roll/pitch. |
| `FeetSlipPenalty` | -0.25 | True foot-velocity slip detection: differences foot positions for actual world-frame xy velocity, penalizes only while in contact. Zero during normal walking (planted foot has zero velocity), fires only when foot actually slides. |
| `JointPositionLimitPenalty` | – | Quadratic penalty near hardware joint limits. |
| `ContactForcePenalty` | – | Penalizes excessive contact forces (hard floor strikes). |
| `CtrlPenalty` | -0.005 | L2 penalty on action vector (torque smoothness). |
| `ActionAccelerationPenalty` | -0.005 | Penalty on action second-derivatives (jerk). |
| `JointVelocityPenalty` | -0.005 | Penalty on joint velocities (smoother motion). |
| `KneeRangeOfMotion` | – | Soft envelope around the usable knee range. |
| `NoContactPenalty` | -0.1 | Penalizes flying — at least one foot in contact most of the time. |
| `MarchInPlacePenalty` | -0.5 | Penalizes stepping while commanded velocity is zero. |
| `FeetPhasePenalty` | -1.0 | Stick complement of FeetPhaseReward — penalizes feet not matching gait phase while walking. |
| `FootSwingClearancePenalty` | -2.0 | Per-foot, per-step penalty for foot height below `min_clearance=0.04m` while gait clock says swinging. Uses `min(heel_z, center_z, toe_z)` so toe-up-heel-down exploit still triggers it. |

### Diagnostic loggers (scale = 0, do not affect training)

| Reward | What it logs |
|---|---|
| `TVCurveSaturationReward` | Mean `|τ| / max_τ_motoring(|qvel|)` across motoring joints — fraction of available motoring torque the policy is using. >0.85 sustained = T-V curve is the binding constraint. |
| `TVCurvePeakSaturationReward` | Worst-joint T-V saturation per step. |
| `AppliedTorqueMeanReward` | Mean `|applied_torque|` across joints — sanity check on observation. |
| `AppliedTorqueMaxReward` | Peak `|applied_torque|` per step. |
| `JointVelMeanReward` | Mean `|joint_velocity|` — operating speed range. |

(Note: the TV-saturation loggers read `applied_torque_observation` which reads
`data.ctrl` directly — `data.actuator_force` isn't reliably populated for motor
actuators in MJX, so we use the explicit ctrl we wrote.)

---

## Terminations (episode-ending conditions)

| Termination | Threshold | Why |
|---|---|---|
| `NotUprightTermination` | 1.2 rad (~69°) | Robot has tipped too far. |
| `MinimumHeightTermination` | 0.5 m | Robot has fallen / sunk below half standing height. Closes "lie on back, feet up" exploit. |

---

## Sim-to-real features

### TV-curve actuator (`TVCurveMITActuators` in `common.py`)

Replaces constant-torque-clip with a velocity-dependent torque limit per joint
from the actuator datasheet:

| Motor | Used for | Peak τ | No-load ω |
|---|---|---|---|
| GIM_8108_8 (`"04"`) | shoulder pitch/roll, elbow, hip pitch/roll, knee | 22 Nm | 21.5 rad/s |
| GIM_6010_8 (`"03"`,`"02"`) | shoulder yaw, hip yaw, ankle | 11 Nm | 29.8 rad/s |
| Wrist (`"00"`) | wrists | 5 Nm constant | – |

**Direction-aware clipping** (physically correct):
- When **motoring** (`sign(ctrl) == sign(qvel)`): limited by T-V curve (back-EMF)
- When **braking** (opposite signs): limited only by constant `ctrl_clip` (back-EMF aids the motor as a generator — full torque available)

**Randomization**: each step, per-joint motoring scale ∈ [0.85, 1.0] (only
weaker, never stronger). Models real motors weakening when hot.

### Domain randomization
Floor friction (0.1–2×), static friction (0.5–2×), body masses (0.85–1.15×),
armature, joint damping, joint zero positions (±0.05 rad). IMU lag 0–100ms.
Action latency 0–5ms.

### Per-step noise
- Position action noise σ=0.05 (Gaussian)
- Velocity action noise σ=0.05
- IMU acc σ=0.4 m/s², gyro σ=0.4 rad/s
- Joint velocity obs σ=1.8 (large — policy must rely more on positions)
- Joint position obs σ=0.05
- Projected gravity σ=0.05

### Random pushes
`XYPushEvent` and `TorquePushEvent` every 2–4 s, magnitude scales with
curriculum level (off at level 0, full at level 1).

---

## Curriculum

`EpisodeLengthCurriculum`:
- 20 levels (0.05 increments)
- **`increase_threshold = 60s`** (raised from 30s) — must sustain 60-s episodes before bumping up. Gives policy time to fully master each level before adding difficulty.
- `decrease_threshold = 10s` — drop only if episodes truly collapse
- Wide hysteresis prevents thrashing

Three things scale with curriculum level:
1. Push magnitude (off at level 0, full at level 1)
2. **Arm constraint probability** (0% at level 0 → 30% at level 1) — added so walking is learned before arm-task complexity phases in
3. (Push frequency stays constant; only magnitude scales)

---

## Training configuration

| Config | Value | Note |
|---|---|---|
| `num_envs` | 3072 | Parallel rollout count |
| `rollout_length_seconds` | **2.0** | Reduced from 5s for early-training speed (~2.5× more PPO updates/min). Bump back to 5s once episodes survive longer. |
| `ctrl_dt` | 0.02 | 50 Hz control rate |
| `iterations` (PPO epochs) | 6 | Per training step |
| `batch_size` | 256 | |
| `learning_rate` | 1e-4 | |
| `entropy_coef` | 0.008 | |
| `kl_coef` | 0.001 | |
| `clip_param` | 0.2 | PPO clip |

---

## How to deploy

The policy expects four commands per step:

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

### Example: stand still with right hand raised (waving)
```python
vel_cmd = jnp.zeros(2)
arm_cmd = jnp.array([
    1.0,
    1.5, 0.5, 0.0, 1.8, 0.0,       # right arm raised
    0.0, 0.0, 0.0, -1.4, 0.0,      # left arm default
])
```

`view_mac.py` provides keyboard joystick + 6 preset arm poses for interactive
testing (T = toggle constraint, 1-6 = cycle preset, WASD/QE = velocity).

---

## Run history (recent)

| Run | Notes | Result |
|---|---|---|
| `run_65` | First curriculum-gated arm constraint | Stuck — StandStill=50 dominated; episode 8s, curriculum stuck at 0 |
| `run_66` | Rebalance: StandStill 50→15, FAR `min_clearance` 8→4cm, curriculum 120s→30s | Episode 6.6→19.7s |
| `run_67` | + FootAirTimeReward, TV direction-aware fix, StandStill 15→8, zero_prob 0.3→0.1 | Curriculum first move 0→0.15 |
| `run_68` | Fresh start with full stack | Failed: `distrax.Normal.mean()` not available |
| `run_69` | Fix: use `.loc`/`.scale` | TV saturation reads 0 (bug found later) |
| `run_70` | Fix: read `data.ctrl` via `AppliedTorqueObservation` | Episode 1.1→29.2s, TV metric still 0 (root cause: data.actuator_force unreliable in MJX) |
| `run_71` | First unstable bootstrap pose (deep crouch, hip=0) | (killed for run_72) |
| `run_72` | **Current** — full run_50 pose match (hip_pitch=±0.23 mirrored, knee=±0.873, ankle=±0.195) | In progress |

---

## Files

| File | What's in it |
|---|---|
| `walking/walking_joystick.py` | Base task: JOINT_TARGETS, commands, observations, actuators, terminations, curriculum |
| `walking/walking_joystick_rnn.py` | RNN actor (with arm override in forward()) + critic + custom `get_rewards()`. Network sizing (NUM_INPUTS, CMD_SIZE). |
| `rewards.py` | All custom reward and penalty classes |
| `common.py` | Custom actuators (`TVCurveMITActuators`), observations (`AppliedTorqueObservation`, `FeetEndpointsObservation`), commands (`ArmConstraintCommand`), events |
| `kscale-assets/kbot-v2-feet/robot.mjcf` | Robot MJCF (symlinked to `~/.kscale/robots/kbot/robot/`, mirrored at `github.com/MFaisalJaffer/kbot-assets`) — includes the added heel/toe sites for multi-point clearance checking |
| `walking/SIM_TO_REAL.md` | Reference table of sim-to-real gaps and mitigations |
| `walking/STATUS.md` | This document |
