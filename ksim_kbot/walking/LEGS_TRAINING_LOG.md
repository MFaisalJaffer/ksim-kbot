# Legs-only Training Log

Tracking attempts to train the kbot-v2-legs walking policy (`walking_legs_rnn.py`).

## Setup

- **Robot**: kbot-v2-legs — torso + legs only, 10 actuated DOFs, 13.06 kg
- **Framework**: ksim + JAX/MJX (GPU sim), PPO with RNN actor/critic
- **Goals**: (a) steady standing at near-zero velocity command; (b) good walking gait at non-zero velocity command
- **Asset path**: `ksim_kbot/kscale-assets/kbot-v2-legs/` (symlink to `odrive-mit-control/gim_test/sim_assets/kbot-v2-legs/`)

## Bugs discovered and fixed

| # | Bug | Where | Fixed in run | Symptom |
|---|---|---|---|---|
| 1 | `WalkingPostureReward` knee indices defaulted to (13, 18) — arms+legs values; out-of-bounds in legs-only model | `rewards.py` | run_7 | Reward signal was meaningless garbage. ~3900 steps trained without knee-bend gradient. Policy learned straight-leg shuffling. |
| 2 | `KneeDeviationPenalty(joint_targets=+0.3 for both knees)` — right knee bends NEGATIVE, so pulled it the wrong way | `walking_legs_rnn.py` | run_9 | Right knee actively pulled toward straight (or worse) for ~400 steps. |
| 3 | `KneeContactTermination` (custom) crashed JIT due to non-hashable `jax.Array` for geom indices | `walking_legs_rnn.py` | run_4 | `RuntimeError: unhashable type: 'jaxlib.xla_extension.ArrayImpl'`. Fix: store geom indices as Python `tuple[int, ...]`. |
| 4 | `SingleFootContactReward.stand_still_threshold` hardcoded to `1e-3` instead of system-wide `0.1` | `rewards.py` | run_13 | Gray-zone marching-in-place: any velocity command between 0.001 and 0.1 m/s gave +0.5 single-foot reward while other rewards considered the robot "standing." Policy marched in place during stand commands. |
| 5 | cuSolver crashes after `kill -9` on JAX processes | (operational) | run_12 | Zombie CUDA contexts (~284 MiB each) accumulate on GPU. Next training run hits stale memory → "cuSolver internal error" on first PPO step. Fix: after killing, run `nvidia-smi --query-compute-apps` and kill stragglers. |
| 6 | Pushes curriculum-scaled from 0 — never fired during fresh init | events config | run_17 | Pushes only activate after curriculum_level > 0, which only advances when episodes survive 60s+. So early training never gets pushes, exactly when they'd be most useful for forcing stepping discovery. Fix: custom `FixedXYPushEvent` that bypasses curriculum scaling. |

## Run-by-run log

### run_3 → run_7 — initial training with broken knee-index bug

- Loaded from arms+legs checkpoint, adapted to legs-only obs space (34 inputs)
- ~3900 steps over ~3 days
- **Result**: policy learned to balance and walk with **straight knees** (shuffling)
- **Root cause**: bug #1 — WalkingPostureReward was reading garbage qpos values, so no real knee-bend gradient was ever applied

### run_8 — first attempt at fixes (sign-bug introduced)

- Loaded from run_7 ckpt.3875
- Fixed WalkingPostureReward indices (3, 8)
- Added `KneeDeviationPenalty` with `joint_targets=(+0.3, +0.3)` — but right knee bends NEGATIVE, so this was bug #2
- Added `BentKneeReward(scale=3.0)`, `MotionTrackingReward(scale=4.0, sigma=0.6)`
- ~400 steps trained before next change
- **Issue**: motion_tracking_reward was 0.005 raw (~0.5% of max) — sigma=0.6 too tight for current policy state to provide meaningful gradient

### run_9 — sign fix + walking-gated bent-knee pull

- Replaced `KneeDeviationPenalty` with sign-aware `WalkingBentKneePenalty` (right target=-0.3, left target=+0.3)
- Added `FootLiftReward(scale=2.0)`
- **Killed before JIT completed** to test fresh start

### run_10/11 — cuSolver crashes

- Fresh-start attempts, both crashed within ~10s of ckpt.0 save
- **Root cause**: bug #5 — leftover zombie GPU processes from prior `kill -9`s

### run_12 — fresh, GPU cleaned, with all fixes from run_8/9

- ~1175 steps over ~12 hours
- **Result**: policy learned to **march in place during stand-still commands**
- **Root cause**: bug #4 — `SingleFootContactReward` was firing during gray-zone (cmd ∈ [0.001, 0.1]) commands where other rewards treated cmd as zero. Net: +0.38/step for marching even when "stand" was commanded.

### run_13/14 — fresh, all 4 fixes applied (rewards rebalanced)

- Dropped `WalkingBentKneePenalty` (conflicted with MotionTracking per-phase knee target)
- `WalkingPostureReward` scale 6.0 → 2.0 (was crowding out vel tracking)
- `JointDeviationPenalty` hip_roll/yaw weight 1.0 → 0.3 (stiff hip was blocking lateral shift)
- `FootProximityPenalty` min_distance 0.10 → 0.06 (10cm was forcing wide stance)
- ~1825 steps trained
- **Result**: **"defensive crouch"** — robot squats low, braces against pushes, eventually collapses (100% of terminations from `minimum_height_termination`)
- Episode length 24.8s, velocity tracking 0.008 raw (basically zero)
- foot_lift_reward DECREASED over training (0.003 → 0.002)
- **Diagnosis**: curriculum auto-advanced pushes to 35% while policy was still in defensive crouch local min — pushes punished the exploration needed to discover walking

### run_15 — disabled pushes + boosted velocity scale 3→6 + foot lift 2→4

- ~450 steps
- Episode length jumped to **42.4s** (vs run_14's 24.8s in 4× the training time)
- Velocity tracking nearly doubled (0.008 → 0.015)
- **But**: `orientation_penalty` *worsened* (more lean) while vel tracking improved
- single-foot contact STILL dead-flat at random-init baseline 0.0017
- foot_lift_reward decreased AGAIN
- **Diagnosis**: **"lean forward, fake velocity"** — policy got velocity reward by leaning forward without lifting feet at all

### run_16 — load run_15 + `SteppingGatedVelocityReward`

- Replaced LinearVelocityTrackingReward with custom variant that multiplies by `XOR(left_contact, right_contact)`
- **Killed the lean exploit** (orientation_penalty improved)
- **But**: policy didn't replace lean with stepping — single-foot contact still dead-flat
- ~187 new steps
- **Diagnosis**: deep "stand safely" local minimum. No reward channel strong enough to break out. Policy never *experiences* stepping in rollouts → can't learn it.

### run_17 — full restructure (CURRENT)

- **Massive simplification**: walking rewards cut from 13 to 6
- `MotionTrackingReward` boosted to scale 10 (primary walking driver)
- Standing rewards bumped: `StandStillReward` 4→8, `StandStillFootLiftPenalty` -3→-5
- Termination penalty -1 → -3
- **Custom `FixedXYPushEvent`** at 0.2-0.4 m/s every 2-4s, NOT curriculum-scaled, fires from step 0 — structural fix for the chicken-and-egg trap
- Dropped: WalkingPostureReward, SingleFootContactReward, FootAirTimeReward, MarchInPlacePenalty, NoContactPenalty, FeetPhasePenalty, FootSwingClearancePenalty, KneeRangeOfMotion, SteppingGatedVelocityReward
- See "Current reward set" below for the active list
- **Status**: running; awaiting results

## Patterns and learnings

### Local minima are sticky and exploit-driven
The policy will find the laziest "valid" behavior. Each run we identified an exploit; the policy promptly found another:
| Run | Exploit |
|---|---|
| 3-7 | Walk with straight knees (no pressure to bend) |
| 12 | March in place during stand commands (gray-zone reward bug) |
| 14 | Defensive crouch (collapse to lower COM, hard to topple) |
| 15 | Lean forward, fake velocity (no stepping required) |
| 16 | Stand safely (after lean was blocked) |

### Reward redundancy doesn't help — it dilutes
Having 4+ rewards for "knee bend" or "foot lift" did NOT make the signal stronger. Each individual channel was too weak. One strong reward >> many weak ones. Best practice: one **primary** reward per goal, optional secondary smooth signals for exploration credit.

### The chicken-and-egg trap
Every walking reward in the original set required the policy to already do *part of* walking. Without an external mechanism (pushes, reference state initialization, etc.) to generate stepping data, the policy can't learn what it never experiences in rollouts.

### Curriculum-scaled events are dangerous early
The episode-length curriculum auto-advances when episodes survive long. But "long enough" can be reached by NOT walking (just standing still and balancing). Bumping pushes before walking is learned ⇒ defensive crouch trap. Lesson: pushes should either be fixed-magnitude or gated on a different metric (e.g., velocity tracking quality).

### Sign conventions: hardware enforces, but rewards must agree
Joint limits in the MJCF enforce single-direction knee bending (right ∈ [-2.7, 0], left ∈ [0, +2.7]). This means **sign-agnostic** rewards using `|knee|` are SAFE (the joint physically can't bend the wrong way). But **signed-target** penalties MUST have correct per-leg signs — `KneeDeviationPenalty(targets=+0.3 for both)` actively pulled the right knee toward straight or past zero.

### Threshold consistency across rewards
Multiple rewards check "is the robot in walking mode?" via `cmd_norm > threshold`. **All must use the same threshold value** (we use `stand_still_threshold = 0.1`). A single reward with a different hardcoded threshold (1e-3) caused subtle gray-zone bugs that produced learned marching-in-place behavior.

### Operational: GPU zombies after kill -9
After force-killing a JAX training process, the CUDA context can persist as a "zombie" using ~284 MiB. Stacking multiple zombies fragments the GPU and the next launch hits cuSolver internal errors. Always:
```bash
ps aux | grep walking_legs_rnn | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 2
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# kill any remaining PIDs shown
```

## Current reward set (as of run_17)

**Standing-only** (`cmd_norm < 0.1`):
- `StandStillReward` (+8.0)
- `StandStillFootLiftPenalty` (-5.0)

**Walking-only** (`cmd_norm > 0.1`):
- `MotionTrackingReward` (+10.0, σ=1.5) — primary driver, DeepMimic-style imitation
- `LinearVelocityTrackingReward` (+5.0)
- `AngularVelocityTrackingReward` (+2.0)
- `BentKneeReward` (+2.0) — secondary smooth signal
- `FootLiftReward` (+2.0) — secondary smooth signal
- `FeetPhaseReward` (+2.0) — phase consistency

**Always-on** (safety/regularization):
- `OrientationPenalty` (-5.0)
- `AngularVelocityXYPenalty` (-0.15)
- `TerminationPenalty` (-3.0)
- `JointDeviationPenalty` (-0.1, weights 0.01 except hip_roll/yaw=0.3)
- `HipDeviationPenalty` (-0.10)
- `JointPositionLimitPenalty` (-1.0)
- `FeetSlipPenalty` (-0.25)
- `FootProximityPenalty` (-2.0, min_distance=0.06)
- `ContactForcePenalty` (-0.01)
- `CtrlPenalty`, `ActionAccelerationPenalty`, `JointVelocityPenalty` (-0.005 each)

**Events:**
- `FixedXYPushEvent` (0.2-0.4 m/s, every 2-4s) — not curriculum-scaled

**Curriculum:** `EpisodeLengthCurriculum(num_levels=20, increase_threshold=60s, decrease_threshold=10s)` — currently inert since the only events don't use curriculum_level

## Open questions / TODOs

- **JOINT_TARGETS for standing**: currently all zeros (straight legs). Real humanoid standing usually has 5-10° knee bend for stability. Worth experimenting if standing oscillates.
- **MotionTracking σ annealing**: σ=1.5 is forgiving (good for fresh start). Once the policy is close to the reference, drop to 1.0 or 0.6 to tighten and sharpen the gradient. Manual intervention or scheduled.
- **Re-enable curriculum-scaled pushes**: once walking is established, switch from `FixedXYPushEvent` back to `XYPushEvent` so robustness improves with policy capability.
- **Reference gait quality**: the analytical sinusoidal reference (±0.26 rad hip, ±0.6 rad knee peak) is a starting approximation. Real walking has asymmetric swing-vs-stance hip velocity, ground reaction force shaping, etc. If imitation tracking saturates but gait looks unnatural, consider replacing reference with mocap or trajectory optimization output.
