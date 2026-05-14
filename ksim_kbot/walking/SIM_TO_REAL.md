# Sim-to-Real Transfer — Reference Table

Each row identifies a sim-to-real gap (a way that the simulator's idealized
physics differs from the real K-Bot 2 hardware) and what we're doing in training
to harden the policy against that gap.

The goal: a policy that doesn't depend on any *one* assumption holding exactly,
so when the assumptions break in slightly different ways on hardware, it still
walks.

---

## Actuator dynamics

| Gap | Why it matters | Mitigation |
|---|---|---|
| **Constant torque limit (sim) vs velocity-dependent limit (real)** | Real BLDC motors lose torque as speed rises (back-EMF). Sim with constant 22 Nm clip lets the policy demand peak torque at any joint speed — actions the real motor cannot execute. | `TVCurveMITActuators` clips torque per joint per step against the actual T-V curve from the actuator datasheet (GIM_8108_8, GIM_6010_8). At high joint speeds, max torque is much lower. |
| **Motor heating / degradation** | Real motors weaken when hot — the published T-V curve is for cold motors. Hot motors deliver ~10–20% less peak torque. | TV curve randomization: each step the per-joint τ-curve is scaled by a uniform factor in `[0.85, 1.0]`. Policy can't bet on exact peak torque values. |
| **Actuator backlash / compliance** | Real harmonic drives have a small dead-band before torque transmits. | `pos_action_noise=0.05, vel_action_noise=0.05` (Gaussian) injects noise into the position/velocity targets the PD controller is given. Forces robustness to a fuzzy command-to-output mapping. |
| **Per-joint zero-position calibration drift** | Encoders may be calibrated slightly differently from the URDF. | `JointZeroPositionRandomizer(scale_lower=-0.05, scale_upper=0.05)` shifts each joint's zero by ±0.05 rad per episode. |
| **Joint damping uncertainty** | Friction in joints is hard to measure and varies between robots. | `JointDampingRandomizer()` perturbs damping coefficients each episode. |
| **Armature / rotor inertia uncertainty** | Reflected rotor inertia is not perfectly captured in the URDF. | `ArmatureRandomizer()` randomizes armature values each episode. |

---

## Contact & friction

| Gap | Why it matters | Mitigation |
|---|---|---|
| **Unknown floor friction** | Carpet vs concrete vs slick gym floor — friction varies 10×. Policy that depends on high friction (e.g. pushes off hard) will slip on smooth floors; policy that depends on slip will face-plant on grippy surfaces. | `FloorFrictionRandomizer(scale_lower=0.1, scale_upper=2.0)` — friction varies 20× across episodes. |
| **Static friction (foot-floor)** | First instant of contact has different dynamics from sliding contact. | `StaticFrictionRandomizer(scale_lower=0.5, scale_upper=2.0)`. |
| **Foot dragging / shuffling** | Sim contact lets feet slide easily; real foot-floor has more variation. Shuffling gait may slip on hardware. | `FeetSlipPenalty(-0.25)` using true foot velocity (not COM): penalizes any horizontal foot motion while in contact. `FootSwingClearancePenalty(-2.0)` forces feet to clear 8cm during swing. |
| **Tilt exploits (heel down, toe up)** | Sim doesn't realistically penalize a foot that's mostly off ground; policy can tilt and look like it's stepping. | Multi-point clearance: `min(heel_z, center_z, toe_z) ≥ 0.08m` per foot. The MJCF now includes 4 extra sites at the actual heel and toe bottom corners; clearance is checked at all three points. |

---

## Mass, inertia & geometry

| Gap | Why it matters | Mitigation |
|---|---|---|
| **Mass inaccuracies** | CAD masses are within 5–15% of real (battery weight, wiring, screws not in CAD). | `AllBodiesMassMultiplicationRandomizer(scale_lower=0.85, scale_upper=1.15)` — every body's mass scaled ±15% per episode. |
| **Center-of-mass offsets** | CoM in real differs from URDF due to actual component placement and cabling. | Randomized armature (rotor inertia offsets) + mass randomization together create CoM uncertainty the policy must handle. |
| **Foot geometry** | Foot collision geom is two capsules; real foot has a different contact patch. | Multi-point clearance check (heel + center + toe) doesn't assume a specific contact patch — uses three discrete points. |

---

## Sensors

| Gap | Why it matters | Mitigation |
|---|---|---|
| **IMU acceleration noise** | Real IMU acc has ~0.2-0.4 m/s² noise + bias. | `imu_acc_noise=0.4` Gaussian noise on `sensor_observation_imu_acc`. |
| **IMU gyro noise** | Real IMU gyro has ~0.2-0.4 rad/s noise. | `imu_gyro_noise=0.4` Gaussian noise. |
| **Projected gravity noise** | Estimated from filtered IMU + quaternion; noisy in practice. | `local_gvec_noise=0.05` on projected gravity observation. |
| **Joint velocity noise** | Joint velocities are computed by differencing encoder ticks — high frequency noise. | `vel_obs_noise=1.8` (large) on joint velocity observation; the policy must rely more on positions than instantaneous velocities. |
| **Joint position noise / quantization** | Encoder quantization is small but real. | `pos_action_noise=0.05` injected at the actuator level + `JointPositionObservation(noise=0.05)`. |
| **IMU lag** | IMU readings arrive 5–15 ms late on hardware. | `ProjectedGravityObservation` has `lag_range=(0.0, 0.1)` — randomized lag up to 100ms during training. |

---

## Latency & control loop

| Gap | Why it matters | Mitigation |
|---|---|---|
| **Action latency** | Time from sensor read → action computation → motor command is 5–20 ms; sim assumes 0. Policy that depends on instant response oscillates on hardware. | `action_latency_range=(0.0, 0.005)` — 0-5 ms randomized action latency per step. |
| **Dropped actions / control loop hiccups** | Real control loop occasionally misses a step (network glitch, processing spike). | (Available but currently off) `drop_action_prob` parameter — set in config to randomly hold last action with some probability. |
| **PD gain mismatch** | Sim PD gains may differ from real motor controller gains. | Actuator metadata (kps, kds) is loaded from kscale metadata, which is calibrated against real hardware. Plus position/velocity action noise creates robustness. |

---

## External disturbances

| Gap | Why it matters | Mitigation |
|---|---|---|
| **Pushes / unexpected forces** | Real robot will get bumped, walked into, knocked off-balance. | `XYPushEvent(interval_range=(2.0, 4.0), force_range=(0.0, 1.8))` — random horizontal push every 2–4s, magnitude scales with curriculum level. |
| **Torque disturbances** | Tether tugs, momentary contact forces. | `TorquePushEvent(interval_range=(2.0, 4.0))` — random torque pushes alongside force pushes. |
| **Initial state variation** | Real robot won't always start in the canonical default pose. | `ResetDefaultJointPosition` reset with noise; `KbotStandingTask` resets with randomized initial state. |

---

## Joint limits & safety

| Gap | Why it matters | Mitigation |
|---|---|---|
| **Hitting hardware joint limits** | Real motors stall hard against end-stops; can damage gearbox. Policy must learn to stay away from limits. | `JointPositionLimitPenalty` — quadratic penalty as joints approach the MJCF range limits. |
| **Excessive contact forces** | Hard ground strikes break feet and ankles on hardware. | `ContactForcePenalty` — penalizes high-magnitude contact forces between feet and floor. |
| **Excessive torques / motor saturation** | Constant saturation overheats motors. | `CtrlPenalty(-0.005)` — L2 penalty on action vector. Combined with the TV-curve clip, large action requests just don't translate to large torques anyway. |
| **Jerky motions** | High-frequency actuator commands stress hardware and look bad. | `ActionAccelerationPenalty(-0.005)` — penalty on action second-derivatives. `JointVelocityPenalty(-0.005)` discourages high joint velocities. |

---

## Behavioral robustness (not pure-dynamics gaps)

| Gap | Why it matters | Mitigation |
|---|---|---|
| **Relying on arm swings for balance** | If the robot is asked to carry an object, it can't swing arms — must balance with legs/torso only. | `ArmConstraintCommand` + `ArmConstraintReward(3.0)`. 30% of episodes (curriculum-gated) lock the arms to a random target pose; policy must walk without arm motion. |
| **Knee-locked / stiff walking** | Stiff-legged walking is brittle, can't absorb impacts. | `WalkingPostureReward(2.0)` requires `≥ 0.4 rad` of knee flex when walking + foot clearance. Combined `JointDeviationPenalty(joint_weights=0.01)` on knees gives policy freedom to bend them. |
| **Single-foot contact discipline** | Bipeds need clean stance/swing alternation. | `SingleFootContactReward(0.5)` rewards exactly one foot on ground during walking; `NoContactPenalty(-0.1)` discourages flying. |
| **Underground / "lie on back" exploits** | Sim's freejoint base lets the robot drop the torso through the floor in some configurations. Real robot would fall over. | `MinimumHeightTermination(0.5m)` ends the episode if base drops below 0.5m. `NotUprightTermination(1.2 rad)` ends episode at ~69° tilt. |
| **Marching in place** | Policy might oscillate feet without going anywhere when commanded zero velocity. | `MarchInPlacePenalty(-0.5)` + `StandStillReward(50)` with orientation gate. Stand-still reward is huge when commanded stop — but fades when leaning so recovery isn't suppressed. |

---

## Domain randomization summary

Per-episode randomization axes (all sampled fresh each episode):

- Floor friction (0.1× – 2.0×)
- Static friction (0.5× – 2.0×)
- Body masses (0.85× – 1.15×)
- Joint damping (default randomizer)
- Joint zero positions (±0.05 rad)
- Armature (default randomizer)
- IMU lag (0 – 100 ms)
- Action latency (0 – 5 ms)
- TV curve scale (0.85× – 1.0×, **per step** not per episode)

Per-step noise:
- Position action noise (Gaussian, σ=0.05)
- Velocity action noise (Gaussian, σ=0.05)
- IMU acc noise (Gaussian, σ=0.4 m/s²)
- IMU gyro noise (Gaussian, σ=0.4 rad/s)
- Joint velocity obs noise (Gaussian, σ=1.8)
- Projected gravity noise (Gaussian, σ=0.05)
- Joint position obs noise (Gaussian, σ=0.05)
- TV-curve torque-cap scale (per-joint, 0.85–1.0)

Per-2-to-4-seconds:
- Random horizontal force push (0 – 1.8 N, scaled by curriculum)
- Random torque push (scaled by curriculum)

---

## Curriculum

Episode-length curriculum has 20 levels (`EpisodeLengthCurriculum`). Three
things scale with curriculum level:

1. Push magnitude (off at level 0, full at level 1)
2. Arm constraint probability (0% at level 0, 30% at level 1) — added this session
3. Push frequency is constant; only magnitude scales

The curriculum ensures the policy bootstraps in a "soft" environment and is
progressively exposed to the full sim-to-real challenges as it gets stable.

---

## Known gaps we're NOT yet addressing

| Gap | Risk | Possible fix |
|---|---|---|
| **Terrain variation** | Real floors aren't perfectly flat. Slopes, bumps, gaps. | `terrain_type` is currently `"smooth"`. Switch to randomized terrain (slopes, height fields) once flat-ground walking is robust. |
| **Foot wear / asymmetric foot pads** | Left/right foot pads wear differently, asymmetric contact. | Could add per-foot friction randomization. |
| **Network latency for commands** | Joystick commands arrive over wireless with jitter. | Could add command lag / quantization. |
| **Battery voltage drop** | As battery drains, available current/torque drops. Currently captured in TV-curve scale randomization but only as ±15%. | Could correlate TV-curve scale across joints (whole-robot weakening) rather than independent per-joint. |
| **Vision-in-the-loop** | This policy uses only proprioception. Visual cues like floor markings could be exploited if added. | Out of scope for current policy. |
| **Thermal shutdown / drive faults** | Real motors can cut out mid-step. | Could implement random single-joint disable events. |

These are good candidates for the next round of robustness work after the
current training run stabilizes.
