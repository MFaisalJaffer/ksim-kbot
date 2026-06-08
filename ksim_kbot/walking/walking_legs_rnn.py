# mypy: disable-error-code="override"
"""Legs-only RNN walking task for K-Bot v2 (torso + legs, no arms).

Derived from walking_joystick_rnn.py with all arm-related code removed:
- Actor and critic no longer receive arm_constraint_cmd_11.
- Actor.forward() has no arm-override block — all 10 outputs are leg actions.
- StandStillReward targets the legs-only JOINT_TARGETS (10 elements).
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx
from mujoco_scenes.mjcf import load_mjmodel
try:
    from xax.nn.export import export
except ModuleNotFoundError:
    export = None  # type: ignore[assignment]

from ksim_kbot import common
from ksim_kbot import rewards as kbot_rewards
from ksim_kbot.walking.walking_legs import (
    NUM_CRITIC_INPUTS,
    NUM_INPUTS,
    NUM_OUTPUTS,
    NUM_JOINTS,
    JOINT_TARGETS,
    KbotLegsWalkingTask,
    KbotLegsWalkingTaskConfig,
)

logger = logging.getLogger(__name__)

# Same obs space except without prev action.
RNN_NUM_INPUTS = NUM_INPUTS - NUM_OUTPUTS

RNN_NUM_CRITIC_INPUTS = NUM_CRITIC_INPUTS - NUM_OUTPUTS


@attrs.define(frozen=True, kw_only=True)
class KneeContactTermination(ksim.Termination):
    """Terminates when knee or thigh geoms contact the floor.

    Stores geom indices as a plain Python tuple (hashable) rather than a
    jax.Array so that JAX JIT can hash the enclosing RLLoopConstants without
    raising 'unhashable type: ArrayImpl'.  The JAX array is created lazily
    inside __call__ where it is used as a traced value, not a static key.
    """

    geom_idxs: tuple  # tuple of plain Python ints
    contact_eps: float = attrs.field(default=-0.001)

    def __call__(self, state: ksim.PhysicsData, curriculum_level: Array) -> Array:
        if state.ncon == 0:
            return jnp.array(0)
        illegal = jnp.array(self.geom_idxs, dtype=jnp.int32)
        hit1 = jnp.isin(state.contact.geom1, illegal)
        hit2 = jnp.isin(state.contact.geom2, illegal)
        any_hit = jnp.logical_or(hit1, hit2)
        penetrating = jnp.where(any_hit, state.contact.dist < self.contact_eps, False).any()
        return jnp.where(penetrating, -1, 0)

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        geom_names: list[str],
        contact_eps: float = -1e-3,
    ) -> "KneeContactTermination":
        from ksim.utils.mujoco import get_geom_data_idx_by_name
        geom_map = get_geom_data_idx_by_name(physics_model)
        missing = [n for n in geom_names if n not in geom_map]
        if missing:
            raise ValueError(f"Geoms not found in model: {missing}. Available: {sorted(geom_map)}")
        idxs = tuple(int(geom_map[n]) for n in geom_names)
        return cls(geom_idxs=idxs, contact_eps=contact_eps)


@attrs.define(frozen=True, kw_only=True)
class MotionTrackingReward(ksim.Reward):
    """Imitation reward for tracking an analytical walking reference gait.

    Reference is computed analytically from the gait phase (synced to the
    existing gait_frequency_command clock).  Per-joint weighted squared error
    fed through an exponential — exp(-err/sigma²) — gives a smooth 0..1 reward.
    Gated by is_walking so it doesn't fight StandStillReward during standing.

    Joint order matches JOINT_TARGETS:
        right: hip_pitch, hip_roll, hip_yaw, knee, ankle  (idx 0..4)
        left:  hip_pitch, hip_roll, hip_yaw, knee, ankle  (idx 5..9)

    Convention (per WalkingPostureReward comment): right knee bends negative,
    left knee bends positive.

    Reference cycle (phase ∈ [0, 1)):
        phase 0.0  — right heel-strike, left toe-off
        phase 0.5  — left  heel-strike, right toe-off

        hip_pitch — sinusoidal swing, ±0.26 rad (~15°).  Right peaks +0.26 at
                    heel-strike (leg forward), -0.26 at toe-off (leg back).
        knee      — bend during swing only, peaks at mid-swing.
                    Right knee:  -0.1 baseline, -0.6 peak (phase 0.75).
                    Left  knee:  +0.1 baseline, +0.6 peak (phase 0.25).
    """

    ctrl_dt: float = attrs.field(default=0.02)
    sigma: float = attrs.field(default=0.6)  # tolerance — bigger = more forgiving
    hip_amp: float = attrs.field(default=0.26)
    knee_baseline: float = attrs.field(default=0.1)
    knee_swing_amp: float = attrs.field(default=0.5)
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    gait_freq_cmd_name: str = attrs.field(default="gait_frequency_command")
    stand_still_threshold: float = attrs.field(default=0.1)
    # Per-joint weights — heavier for the gait-shape joints (hip_pitch, knee).
    joint_weights: tuple = attrs.field(default=(
        1.0, 0.2, 0.2, 1.5, 0.3,  # right: hip_pitch, hip_roll, hip_yaw, knee, ankle
        1.0, 0.2, 0.2, 1.5, 0.3,  # left
    ))

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # Phase clock — matches TimestepPhaseObservation / FeetPhaseReward.
        gait_freq = trajectory.command[self.gait_freq_cmd_name][..., 0]
        steps = jnp.int32(trajectory.timestep / self.ctrl_dt)
        phase = (steps.astype(jnp.float32) * gait_freq * self.ctrl_dt) % 1.0

        omega = 2.0 * jnp.pi * phase  # phase in radians

        # Sinusoidal hip swing, anti-phase legs.
        r_hip_pitch = self.hip_amp * jnp.cos(omega)
        l_hip_pitch = -self.hip_amp * jnp.cos(omega)

        # Knee bend during swing only.
        # Right swing during phase 0.5..1.0 → -sin(omega) > 0 there.
        # Left  swing during phase 0.0..0.5 →  sin(omega) > 0 there.
        r_knee = -self.knee_baseline - self.knee_swing_amp * jnp.maximum(0.0, -jnp.sin(omega))
        l_knee = self.knee_baseline + self.knee_swing_amp * jnp.maximum(0.0, jnp.sin(omega))

        zeros = jnp.zeros_like(phase)
        q_ref = jnp.stack(
            [r_hip_pitch, zeros, zeros, r_knee, zeros,
             l_hip_pitch, zeros, zeros, l_knee, zeros],
            axis=-1,
        )

        # Actual joint positions are qpos[7:17] for our 10-DOF legs-only model.
        q_actual = trajectory.qpos[..., 7:17]
        weights = jnp.array(self.joint_weights)
        err = jnp.sum(((q_actual - q_ref) * weights) ** 2, axis=-1)

        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        is_walking = cmd_norm > self.stand_still_threshold

        return jnp.exp(-err / (self.sigma ** 2)) * is_walking


@attrs.define(frozen=True, kw_only=True)
class KneeMotionWithoutProgressPenalty(ksim.Reward):
    """Penalize gait-like knee motion when not actually moving forward.

    Diagnosis from run_17/18 viewer: the policy converged to a "knee gait in
    place" exploit — knees oscillate like a walking pattern (one bent, the
    other extending) but feet stay essentially planted and body doesn't
    translate.  The existing MarchInPlacePenalty doesn't catch this because
    it checks foot HEIGHT (and the feet aren't lifting).  This penalty
    checks knee BEND instead, which is the actual signature of "trying to
    walk without going anywhere."

    Penalty = clamp(max(|r_knee|, |l_knee|) / threshold, 0, 1)
              × (1 - velocity_match)
              × cmd_active

    At "real walking" (knees bent + body translating): velocity_match ≈ 1,
    penalty ≈ 0.  At "knee gait in place" (knees bent, body still):
    velocity_match ≈ 0, penalty fires near max.  When standing still
    (cmd_active=0): penalty = 0.
    """

    right_knee_idx: int = attrs.field(default=3)
    left_knee_idx: int = attrs.field(default=8)
    knee_motion_threshold: float = attrs.field(default=0.1)
    velocity_match_sensitivity: float = attrs.field(default=0.1)
    linvel_obs_name: str = attrs.field(default="base_linear_velocity_observation")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    stand_still_threshold: float = attrs.field(default=0.1)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # Knee motion signal: how much at least one knee is bent.
        r_knee = trajectory.qpos[..., 7 + self.right_knee_idx]
        l_knee = trajectory.qpos[..., 7 + self.left_knee_idx]
        knee_motion = jnp.maximum(jnp.abs(r_knee), jnp.abs(l_knee))
        knee_motion_norm = jnp.minimum(knee_motion / self.knee_motion_threshold, 1.0)

        # Velocity match (tight sigma so mismatch ramps up sharply).
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        actual = trajectory.obs[self.linvel_obs_name][..., :2]
        vel_err_sq = jnp.sum(jnp.square(vel_cmd - actual), axis=-1)
        velocity_match = jnp.exp(-vel_err_sq / self.velocity_match_sensitivity)

        cmd_active = jnp.linalg.norm(vel_cmd, axis=-1) > self.stand_still_threshold
        return knee_motion_norm * (1.0 - velocity_match) * cmd_active


@attrs.define(frozen=True, kw_only=True)
class FixedXYPushEvent(ksim.Event):
    """XY velocity push at fixed magnitude — does NOT scale with curriculum_level.

    Existing common.XYPushEvent multiplies force_range by curriculum_level which
    starts at 0 for fresh training, so pushes never fire until episodes survive
    60s+ — exactly when we'd LIKE to be done with pushes.  This version provides
    constant small perturbations from the very first step to force the policy to
    discover stepping (single-foot stance) as a recovery mechanism.
    """

    interval_range: tuple[float, float] = attrs.field()
    force_range: tuple[float, float] = attrs.field()

    def __call__(
        self,
        model: ksim.PhysicsModel,
        data: ksim.PhysicsData,
        event_state: Array,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PhysicsData, Array]:
        dt = jnp.float32(model.opt.timestep)
        time_remaining = event_state - dt
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_push(data, rng),
            lambda: (data, time_remaining),
        )
        return updated_data, time_remaining

    def _apply_push(
        self, data: ksim.PhysicsData, rng: PRNGKeyArray
    ) -> tuple[ksim.PhysicsData, Array]:
        from ksim.utils.mujoco import slice_update, update_data_field
        rng_theta, rng_mag, rng_interval = jax.random.split(rng, 3)
        push_theta = jax.random.uniform(rng_theta, maxval=2 * jnp.pi)
        push_magnitude = jax.random.uniform(
            rng_mag, minval=self.force_range[0], maxval=self.force_range[1]
        )
        push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)]) * push_magnitude
        new_qvel_xy = data.qvel[:2] + push
        new_qvel = slice_update(data, "qvel", slice(0, 2), new_qvel_xy)
        updated_data = update_data_field(data, "qvel", new_qvel)
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(rng_interval, (), minval=minval, maxval=maxval)
        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)


@attrs.define(frozen=True, kw_only=True)
class SteppingGatedVelocityReward(ksim.Reward):
    """Velocity tracking gated by single-foot contact.

    Standard LinearVelocityTrackingReward computes exp(-||v_actual - v_cmd||/err),
    but rewards forward velocity REGARDLESS of how it's produced.  Run_14/15
    diagnosis: policy learned to lean forward and "fake" forward velocity
    without ever lifting a foot (single_foot_contact reward stayed dead-flat
    at the random-init baseline of 0.0017).

    This gate multiplies the velocity tracking reward by a "stepping" signal:
        gate = 1 if exactly one foot in contact (XOR of left/right contact)
        gate = 0 if both feet in contact (planted) or both feet off (flying)

    Effect: leaning-forward exploit gets zero velocity reward.  Only by
    actually stepping (alternating single-foot stance) does the policy earn
    the velocity reward.  Brief double-support during heel-strike transitions
    will lose some reward, but during clean walking the gate is ~1.0 for
    ~70-80% of the cycle.
    """

    error_scale: float = attrs.field(default=0.5)
    linvel_obs_name: str = attrs.field(default="base_linear_velocity_observation")
    command_name: str = attrs.field(default="linear_velocity_command")
    feet_contact_obs_name: str = attrs.field(default="feet_contact_observation")
    contact_threshold: float = attrs.field(default=0.1)
    stand_still_threshold: float = attrs.field(default=0.1)
    norm: xax.NormType = attrs.field(default="l2")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.linvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.linvel_obs_name} not found.")

        # Standard velocity tracking error.
        command = trajectory.command[self.command_name]
        actual = trajectory.obs[self.linvel_obs_name][..., :2]
        lin_vel_error = xax.get_norm(command - actual, self.norm).sum(axis=-1)
        track_reward = jnp.exp(-lin_vel_error / self.error_scale)

        # Stepping gate: 1 when exactly one foot in contact, 0 otherwise.
        feet_contact = trajectory.obs[self.feet_contact_obs_name]
        left = feet_contact[..., 0] > self.contact_threshold
        right = feet_contact[..., 1] > self.contact_threshold
        is_stepping = jnp.logical_xor(left, right).astype(jnp.float32)

        # Standard walking gate.
        command_norm = jnp.linalg.norm(command, axis=-1)
        is_walking = command_norm > self.stand_still_threshold

        return track_reward * is_stepping * is_walking


@attrs.define(frozen=True, kw_only=True)
class WalkingBentKneePenalty(ksim.Reward):
    """Walking-gated pull toward bent knees with correct per-leg sign convention.

    Robot convention: right knee bends NEGATIVE, left knee bends POSITIVE.
    So the "bent" target is asymmetric: right_knee → -bent_target,
    left_knee → +bent_target.  (My earlier KneeDeviationPenalty
    instantiation used +0.3 for both, pulling the right knee the WRONG
    way — that bug ran for run_7/run_8 until this fix.)

    Penalty = ((r_knee - (-bent_target))² + (l_knee - bent_target)²) * is_walking
    Gate by is_walking so it doesn't fight StandStillReward during stand cmds.
    """

    right_knee_idx: int = attrs.field(default=3)
    left_knee_idx: int = attrs.field(default=8)
    bent_target: float = attrs.field(default=0.3)
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    stand_still_threshold: float = attrs.field(default=0.1)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        r_knee = trajectory.qpos[..., 7 + self.right_knee_idx]
        l_knee = trajectory.qpos[..., 7 + self.left_knee_idx]
        # Targets respect sign convention.
        r_err = r_knee - (-self.bent_target)
        l_err = l_knee - self.bent_target
        penalty = r_err * r_err + l_err * l_err

        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        is_walking = cmd_norm > self.stand_still_threshold
        return penalty * is_walking


@attrs.define(frozen=True, kw_only=True)
class FootLiftReward(ksim.Reward):
    """Standalone additive reward for foot height during swing phase.

    Sibling of BentKneeReward — same idea but for feet, decoupled from
    knees so each signal carries its own gradient.  Uses the SAME phase
    clock as WalkingPostureReward / FootSwingClearancePenalty so only
    rewards lifting when the gait clock expects swing.

    Per-foot sigmoid: pays half reward at `half_lift`, saturates near
    `max_foot_height`.  Min-z across center/heel/toe to prevent tilt
    exploit.  Averaged across the two feet.
    """

    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    feet_endpoints_obs_name: str = attrs.field(default="feet_endpoints_observation")
    gait_freq_cmd_name: str = attrs.field(default="gait_frequency_command")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    stand_still_threshold: float = attrs.field(default=0.1)
    ctrl_dt: float = attrs.field(default=0.02)
    half_lift: float = attrs.field(default=0.03)        # 3 cm — half reward
    sensitivity: float = attrs.field(default=0.02)      # sigmoid slope
    max_foot_height: float = attrs.field(default=0.12)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # Gait clock — match WalkingPostureReward / FootSwingClearancePenalty.
        gait_freq_n = trajectory.command[self.gait_freq_cmd_name]
        phase_dt = 2 * jnp.pi * gait_freq_n * self.ctrl_dt
        steps = jnp.int32(trajectory.timestep / self.ctrl_dt)
        steps = jnp.repeat(steps[:, None], 2, axis=1)
        start_phase = jnp.broadcast_to(jnp.array([0.0, jnp.pi]), (steps.shape[0], 2))
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi
        x = jnp.clip((phase + jnp.pi) / (2 * jnp.pi), 0, 1)
        swing_h = jnp.array(self.max_foot_height)
        stance = xax.cubic_bezier_interpolation(jnp.array(0.0), swing_h, 2 * x)
        swing = xax.cubic_bezier_interpolation(swing_h, jnp.array(0.0), 2 * x - 1)
        ideal_z = jnp.where(x <= 0.5, stance, swing)
        in_swing = ideal_z > 0.0  # (T, 2)

        # Conservative foot z: min(center, heel, toe) per foot.
        foot_pos = trajectory.obs[self.feet_pos_obs_name]
        center_z = jnp.stack([foot_pos[..., 2], foot_pos[..., 5]], axis=-1)
        ep = trajectory.obs[self.feet_endpoints_obs_name]
        left_min_z = jnp.minimum(ep[..., 2], ep[..., 5])
        right_min_z = jnp.minimum(ep[..., 8], ep[..., 11])
        foot_z = jnp.minimum(center_z, jnp.stack([left_min_z, right_min_z], axis=-1))

        # Sigmoid lift reward per foot, only count during swing.
        lift_reward = jax.nn.sigmoid((foot_z - self.half_lift) / self.sensitivity)
        per_foot = jnp.where(in_swing, lift_reward, 0.0)
        mean_lift = jnp.mean(per_foot, axis=-1)

        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        is_walking = cmd_norm > self.stand_still_threshold
        return mean_lift * is_walking


@attrs.define(frozen=True, kw_only=True)
class BentKneeReward(ksim.Reward):
    """Standalone additive reward for bent knees when commanded to walk.

    Sigmoid per knee, summed (not multiplied) so each knee contributes its own
    gradient independently. Decoupled from foot-lift so the policy gets a
    direct signal to bend knees even before swing-phase clearance develops.

    Reward = (sigmoid_r + sigmoid_l) / 2 × is_walking
    """

    right_knee_idx: int = attrs.field(default=3)  # qpos[7+3]=qpos[10] = right knee
    left_knee_idx: int = attrs.field(default=8)   # qpos[7+8]=qpos[15] = left knee
    knee_half_bend: float = attrs.field(default=0.1)
    knee_sensitivity: float = attrs.field(default=0.05)
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    stand_still_threshold: float = attrs.field(default=0.1)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        is_walking = cmd_norm > self.stand_still_threshold
        r_knee = trajectory.qpos[..., 7 + self.right_knee_idx]
        l_knee = trajectory.qpos[..., 7 + self.left_knee_idx]
        r_growth = jax.nn.sigmoid((jnp.abs(r_knee) - self.knee_half_bend) / self.knee_sensitivity)
        l_growth = jax.nn.sigmoid((jnp.abs(l_knee) - self.knee_half_bend) / self.knee_sensitivity)
        return ((r_growth + l_growth) / 2.0) * is_walking


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array
    actor_carry: Array
    critic_carry: Array


class KbotRNNActor(eqx.Module):
    """RNN-based actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=NUM_OUTPUTS * 2,
            key=key,
        )

        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(
        self,
        timestep_phase_4: Array,
        joint_pos_n: Array,
        joint_vel_n: Array,
        projected_gravity_3: Array,
        # imu_acc_3: Array,
        imu_gyro_3: Array,
        lin_vel_cmd_2: Array,
        ang_vel_cmd: Array,
        gait_freq_cmd: Array,
        last_action_n: Array,
        carry: Array,
    ) -> tuple[distrax.Normal, Array]:
        # Legs-only: no arm_constraint_cmd, no arm-override post-processing.
        obs_n = jnp.concatenate(
            [
                timestep_phase_4,  # 4
                joint_pos_n,  # NUM_JOINTS (10)
                joint_vel_n,  # NUM_JOINTS (10)
                projected_gravity_3,  # 3
                # imu_acc_3,  # 3
                imu_gyro_3,  # 3
                lin_vel_cmd_2,  # 2
                ang_vel_cmd,  # 1
                gait_freq_cmd,  # 1
                # last_action_n,  # NUM_JOINTS
            ],
            axis=-1,
        )
        dist_n, new_carry = self.call_flat_obs(obs_n, carry)
        return dist_n, new_carry

    def call_flat_obs(self, obs_n: Array, carry: Array) -> tuple[distrax.Normal, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Converts the output to a distribution.
        mean_n = out_n[..., :NUM_OUTPUTS]
        std_n = out_n[..., NUM_OUTPUTS:]

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)
        dist_n = distrax.Normal(mean_n, std_n)
        return dist_n, jnp.stack(out_carries, axis=0)


class KbotRNNCritic(eqx.Module):
    """RNN-based critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

    def forward(
        self,
        timestep_phase_4: Array,
        joint_pos_n: Array,
        joint_vel_n: Array,
        projected_gravity_3: Array,
        lin_vel_cmd_2: Array,
        ang_vel_cmd: Array,
        gait_freq_cmd: Array,
        last_action_n: Array,
        # critic observations
        feet_contact_2: Array,
        feet_position_6: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        base_position_3: Array,
        base_orientation_4: Array,
        base_linear_velocity_3: Array,
        base_angular_velocity_3: Array,
        actuator_force_n: Array,
        true_height_1: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        # Legs-only critic: no arm_constraint_cmd in the input list.
        obs_n = jnp.concatenate(
            [
                timestep_phase_4,  # 4
                joint_pos_n,  # NUM_JOINTS (10)
                joint_vel_n,  # NUM_JOINTS (10)
                projected_gravity_3,  # 3
                lin_vel_cmd_2,  # 2
                ang_vel_cmd,  # 1
                gait_freq_cmd,  # 1
                # last_action_n,  # NUM_JOINTS
                feet_contact_2,  # 2
                feet_position_6,  # 6
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                base_position_3,  # 3
                base_orientation_4,  # 4
                base_linear_velocity_3,  # 3
                base_angular_velocity_3,  # 3
                actuator_force_n,  # NUM_JOINTS (10)
                true_height_1,  # 1
            ],
            axis=-1,
        )
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class KbotRNNModel(eqx.Module):
    actor: KbotRNNActor
    critic: KbotRNNCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        self.actor = KbotRNNActor(
            key,
            num_inputs=RNN_NUM_INPUTS,
            min_std=0.01,
            max_std=1.0,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.critic = KbotRNNCritic(
            key,
            num_inputs=RNN_NUM_CRITIC_INPUTS,
            num_outputs=1,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class KbotLegsWalkingRNNTaskConfig(KbotLegsWalkingTaskConfig):
    hidden_size: int = xax.field(value=256)
    depth: int = xax.field(value=5)


Config = TypeVar("Config", bound=KbotLegsWalkingRNNTaskConfig)


class KbotLegsWalkingRNNTask(KbotLegsWalkingTask[Config], Generic[Config]):
    config: Config

    def get_model(self, key: PRNGKeyArray) -> KbotRNNModel:
        return KbotRNNModel(
            key,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        import asyncio
        # Use the legs-only asset's metadata (resolved via robot_urdf_path).
        return asyncio.run(ksim.get_mujoco_model_metadata(self.config.robot_urdf_path, cache=False))

    def get_mujoco_model(self) -> mujoco.MjModel:
        # Resolve the legs-only MJCF relative to the configured robot_urdf_path.
        mjcf_path = (Path(self.config.robot_urdf_path) / "robot.mjcf").resolve().as_posix()
        logger.info("Loading MJCF model from %s", mjcf_path)

        mj_model = load_mjmodel(mjcf_path, scene=self.config.terrain_type)

        # NOTE: test the difference
        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        return mj_model

    def run_actor(
        self,
        model: KbotRNNActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Normal, Array]:
        timestep_phase_4 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        # imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        projected_gravity_3 = observations["projected_gravity_observation"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd = commands["angular_velocity_command"]
        gait_freq_cmd = commands["gait_frequency_command"]
        last_action_n = observations["last_action_observation"]

        return model.forward(
            timestep_phase_4=timestep_phase_4,
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            # imu_acc_3=imu_acc_3,
            imu_gyro_3=imu_gyro_3,
            projected_gravity_3=projected_gravity_3,
            lin_vel_cmd_2=lin_vel_cmd_2,
            ang_vel_cmd=ang_vel_cmd,
            gait_freq_cmd=gait_freq_cmd,
            last_action_n=last_action_n,
            carry=carry,
        )

    def run_critic(
        self,
        model: KbotRNNCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        timestep_phase_4 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        projected_gravity_3 = observations["projected_gravity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd = commands["angular_velocity_command"]
        gait_freq_cmd = commands["gait_frequency_command"]
        last_action_n = observations["last_action_observation"]
        # critic observations
        feet_contact_2 = observations["feet_contact_observation"]
        feet_position_6 = observations["feet_position_observation"]
        base_position_3 = observations["base_position_observation"]
        base_orientation_4 = observations["base_orientation_observation"]
        base_linear_velocity_3 = observations["base_linear_velocity_observation"]
        base_angular_velocity_3 = observations["base_angular_velocity_observation"]
        actuator_force_n = observations["actuator_force_observation"]
        true_height_1 = observations["true_height_observation"]
        return model.forward(
            timestep_phase_4=timestep_phase_4,
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            # imu_acc_3=imu_acc_3,
            # imu_gyro_3=imu_gyro_3,
            projected_gravity_3=projected_gravity_3,
            lin_vel_cmd_2=lin_vel_cmd_2,
            ang_vel_cmd=ang_vel_cmd,
            gait_freq_cmd=gait_freq_cmd,
            last_action_n=last_action_n,
            # critic observations
            feet_contact_2=feet_contact_2,
            feet_position_6=feet_position_6,
            imu_acc_3=imu_acc_3,
            imu_gyro_3=imu_gyro_3,
            base_position_3=base_position_3,
            base_orientation_4=base_orientation_4,
            base_linear_velocity_3=base_linear_velocity_3,
            base_angular_velocity_3=base_angular_velocity_3,
            actuator_force_n=actuator_force_n,
            true_height_1=true_height_1,
            carry=carry,
        )

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        # ─── PORTED FROM kbot_walking_joystick_rnntask/run_36 (May 9) ───
        # Goal: test whether the joystick task's working reward structure
        # also drives walking on the legs-only (no-arms) robot. This is an
        # A/B against the DeepMimic-style stack (runs 17–20) which had
        # MotionTracking@10 + symmetry + bent-knee + foot-lift + KneeMotionPenalty.
        # That stack produced symmetric joints but persistent foot-lift
        # asymmetry and weak forward translation. run_36's stack relies on
        # StandStillReward@50 as the dominant anchor + FeetPhaseReward@2.1
        # (gait phase clock) + SingleFootContact + MarchInPlacePenalty.
        #
        # Legs-only adaptations (vs run_36):
        #   - JointDeviationPenalty: 10 weights instead of 20 (no arms)
        #   - HipDeviationPenalty: _04 suffix on roll, _03 on yaw (legs MJCF)
        # Everything else is verbatim from run_36 (scales, signs, gating).
        #
        # Note: events (FixedXYPushEvent) and curriculum unchanged from our
        # legs task — this is a rewards-only port.
        return [
            # JointDeviationPenalty (run_36 verbatim, arms stripped):
            # Knees + hip_pitch + ankles free (weight 0.01); hip_roll/yaw
            # constrained (1.0). Anchors stable upper-leg-body pose.
            kbot_rewards.JointDeviationPenalty(
                scale=-0.1,
                joint_targets=JOINT_TARGETS,
                joint_weights=(
                    # right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle
                    0.01, 1.0, 1.0, 0.01, 0.01,
                    # left leg
                    0.01, 1.0, 1.0, 0.01, 0.01,
                ),
            ),
            # HipDeviationPenalty: legs MJCF uses _04 suffix on hip_roll, _03 on hip_yaw.
            kbot_rewards.HipDeviationPenalty.create(
                physics_model=physics_model,
                hip_names=(
                    "dof_right_hip_roll_04",
                    "dof_right_hip_yaw_03",
                    "dof_left_hip_roll_04",
                    "dof_left_hip_yaw_03",
                ),
                joint_targets=JOINT_TARGETS,
                scale=-0.10,
            ),
            kbot_rewards.TerminationPenalty(scale=-1.0),
            kbot_rewards.OrientationPenalty(scale=-2.0),
            kbot_rewards.LinearVelocityTrackingReward(
                scale=1.0,
                linvel_obs_name="base_linear_velocity_observation",
                stand_still_threshold=self.config.stand_still_threshold,
            ),
            kbot_rewards.AngularVelocityTrackingReward(
                scale=0.5,
                angvel_obs_name="base_angular_velocity_observation",
                stand_still_threshold=self.config.stand_still_threshold,
            ),
            kbot_rewards.AngularVelocityXYPenalty(
                scale=-0.15,
                angvel_obs_name="base_angular_velocity_observation",
                stand_still_threshold=self.config.stand_still_threshold,
            ),
            # FeetPhaseReward @ 2.1, translation_gated so marching-in-place
            # doesn't farm phase reward without forward progress.
            # use_endpoints=True: foot z = min(center, heel, toe) per foot.
            # Defeats the foot-tilt exploit observed in run_22 where the policy
            # rolled the inner edge down (tracked) while the outer edge lifted.
            kbot_rewards.FeetPhaseReward(
                foot_default_height=0.04,
                max_foot_height=0.12,
                scale=2.1,
                stand_still_threshold=self.config.stand_still_threshold,
                translation_gated=True,
                translation_gate_sensitivity=0.25,
                linvel_obs_name="base_linear_velocity_observation",
                use_endpoints=True,
            ),
            # FootSwingClearancePenalty: when the gait clock says foot should
            # be airborne (ideal_z > min_clearance), penalize proportionally
            # to how much the foot is actually below clearance. Uses the same
            # min(center, heel, toe) measurement as FeetPhaseReward — heel or
            # toe still on the floor = foot dragging, regardless of tilt.
            kbot_rewards.FootSwingClearancePenalty(
                scale=-1.0,
                min_clearance=0.06,
                max_foot_height=0.12,
                ctrl_dt=self.config.ctrl_dt,
                stand_still_threshold=self.config.stand_still_threshold,
            ),
            # REVERTED to run_23 reward stack for new experiment (run_27+):
            # - BentKneeReward (added run_26): removed
            # - PairwiseSymmetryReward x2 (added run_24): removed
            # - ContactForcePenalty (tightened run_25): reverted to original
            #   scale -0.01, default threshold 350 N (see below)
            # New addition: JointAccelerationPenalty for motion smoothness
            # (see below — added after JointVelocityPenalty).
            kbot_rewards.FeetSlipPenalty(scale=-0.25, ctrl_dt=self.config.ctrl_dt),
            # StandStillReward @ 50 — the dominant attractor for cmd_norm < threshold.
            # This is the run_36 design's heaviest hand: idle stand pose
            # gets +50, so the policy strongly defaults to standing when
            # not commanded to walk. Pairs with FeetPhase + SingleFoot
            # which only fire under non-zero commands.
            kbot_rewards.StandStillReward(
                scale=50.0,
                linear_velocity_cmd_name="linear_velocity_command",
                angular_velocity_cmd_name="angular_velocity_command",
                joint_targets=JOINT_TARGETS,
                stand_still_threshold=self.config.stand_still_threshold,
            ),
            kbot_rewards.JointPositionLimitPenalty.create(
                physics_model=physics_model,
                soft_limit_factor=0.95,
                scale=-1.0,
            ),
            # ContactForcePenalty — reverted to run_23 defaults for the
            # smoothness-focused experiment (run_27+). The aggressive
            # tightening (-0.10, 120 N) from run_25 is being replaced by
            # a different hypothesis: smoother joint motion → softer landings
            # as a side-effect (less velocity = less momentum to absorb).
            kbot_rewards.ContactForcePenalty(
                scale=-0.01,
                sensor_names=(
                    "sensor_observation_left_foot_force",
                    "sensor_observation_right_foot_force",
                ),
            ),
            ksim.CtrlPenalty(scale=-0.005),
            ksim.ActionAccelerationPenalty(scale=-0.005),
            ksim.JointVelocityPenalty(scale=-0.005),
            # JointAccelerationPenalty: penalize how fast joint velocities
            # change (= rate of change of joint angle changes). Encourages
            # smooth motion without slowing the gait down (unlike a stronger
            # velocity penalty which would force the policy to move slower).
            # Scale -0.01: moderate first try. If gait still looks jerky,
            # bump scale; if motion becomes too sluggish, dial back.
            ksim.JointAccelerationPenalty(scale=-0.01),
            kbot_rewards.KneeRangeOfMotion.create(
                physics_model=physics_model,
                knee_names=("dof_left_knee_04", "dof_right_knee_04"),
            ),
            # SingleFootContactReward @ 0.5, grace_period 0.2s.
            # Pays out when exactly one foot is in contact while walking.
            # Bug-fix: stand_still_threshold passed through from config so
            # the gray-zone marching-in-place exploit (caught in legs run_13)
            # doesn't reappear.
            kbot_rewards.SingleFootContactReward(
                scale=0.5,
                ctrl_dt=self.config.ctrl_dt,
                grace_period=0.2,
                stand_still_threshold=self.config.stand_still_threshold,
            ),
            kbot_rewards.NoContactPenalty(scale=-0.1),
            # MarchInPlacePenalty @ -2: in legs run_18 this registered 0.0000
            # because it required feet > 4cm to fire. Kept here for parity
            # with run_36; if the policy converges to ground-level shuffling
            # again, swap in KneeMotionWithoutProgressPenalty instead.
            kbot_rewards.MarchInPlacePenalty(
                scale=-2.0,
                foot_default_height=0.04,
                velocity_match_sensitivity=0.25,
                linvel_obs_name="base_linear_velocity_observation",
            ),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        if self.config.domain_randomize:
            vel_obs_noise = 1.8
            imu_acc_noise = 0.4
            imu_gyro_noise = 0.4
            local_gvec_noise = 0.05
            base_position_noise = 0.0
            base_orientation_noise = 0.0
            base_linear_velocity_noise = 0.0
            base_angular_velocity_noise = 0.0
        else:
            vel_obs_noise = 0.0
            imu_acc_noise = 0.0
            imu_gyro_noise = 0.0
            local_gvec_noise = 0.0
            base_position_noise = 0.0
            base_orientation_noise = 0.0
            base_linear_velocity_noise = 0.0
            base_angular_velocity_noise = 0.0

        # NOTE: JOINT_TARGETS is already imported at module scope from walking_legs
        # (10 elements). Do NOT re-import from walking_joystick — that's the
        # 20-element arms+legs version and would create a shape mismatch.
        from ksim_kbot.common import (
            TimestepPhaseObservation,
            JointPositionObservation,
            LocalProjectedGravityObservation,
            LastActionObservation,
            FeetContactObservation,
            FeetPositionObservation,
            FeetEndpointsObservation,
            TrueHeightObservation,
            AppliedTorqueObservation,
        )

        return [
            TimestepPhaseObservation(),
            JointPositionObservation(default_targets=JOINT_TARGETS, noise=0.05),
            ksim.JointVelocityObservation(noise=vel_obs_noise),
            ksim.ActuatorForceObservation(),
            # Custom obs reads data.ctrl (where our MIT actuator writes torque).
            # data.actuator_force isn't reliably populated for motor actuators in MJX,
            # but data.ctrl is what we explicitly set, so this is always correct.
            AppliedTorqueObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc", noise=imu_acc_noise),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro", noise=imu_gyro_noise),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="base_link_quat",
                lag_range=(0.0, 0.1),
                noise=local_gvec_noise,
            ),
            LocalProjectedGravityObservation.create(
                physics_model=physics_model, sensor_name="base_link_quat", noise=local_gvec_noise
            ),
            LastActionObservation(noise=0.0),
            # Additional critic observations
            ksim.BasePositionObservation(noise=base_position_noise),
            ksim.BaseOrientationObservation(noise=base_orientation_noise),
            ksim.BaseLinearVelocityObservation(noise=base_linear_velocity_noise),
            ksim.BaseAngularVelocityObservation(noise=base_angular_velocity_noise),
            ksim.CenterOfMassVelocityObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_force", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_force", noise=0.0),
            FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names="KB_D_501L_L_LEG_FOOT_collision",
                foot_right_geom_names="KB_D_501R_R_LEG_FOOT_collision",
                floor_geom_names="floor",
            ),
            FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_site_name="left_foot",
                foot_right_site_name="right_foot",
                floor_threshold=0.00,
            ),
            # Heel + toe corner positions for multi-point clearance checking in rewards.
            # NOT used as policy input — only for reward computation.
            FeetEndpointsObservation.create(physics_model=physics_model),
            TrueHeightObservation(),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.NotUprightTermination(max_radians=1.2),  # ~69°
            # Terminate if base drops below 0.65m.
            # Standing base height ~1.01m; knee-walking brings it to ~0.65-0.70m.
            # Raised from 0.5m to catch the knee-walking exploit.
            ksim.MinimumHeightTermination(min_height=0.65),
            # Terminate if any knee or thigh geom contacts the ground.
            # Explicitly prevents the "walking on knees" exploit where the shin/femur
            # bodies drop to the floor. Knee bodies sit at z=0.34m when standing.
            # Uses KneeContactTermination (not ksim.IllegalContactTermination) to
            # avoid the JAX JIT unhashable-ArrayImpl error.
            KneeContactTermination.create(
                physics_model=physics_model,
                geom_names=[
                    "KC_D_401R_R_Shin_Drive_collision",   # right knee/shin
                    "KC_D_401L_L_Shin_Drive_collision",   # left knee/shin
                    "KC_D_301R_R_Femur_Lower_Drive_collision",  # right thigh
                    "KC_D_301L_L_Femur_Lower_Drive_collision",  # left thigh
                ],
            ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        # Curriculum-gated pushes (matches run_36 design).
        # Both events multiply their magnitude by `curriculum_level` internally,
        # so at level 0 (fresh policy) pushes are ZERO. Once the policy can
        # sustain `increase_threshold` episodes, the curriculum advances and
        # pushes grow proportionally.
        # force_range and ang_vel_range here are the *full-strength* values at
        # curriculum_level=1.0; intermediate levels scale linearly.
        return [
            common.XYPushEvent(
                interval_range=(2.0, 4.0),
                force_range=(0.0, 1.8),
            ),
            common.TorquePushEvent(
                interval_range=(2.0, 4.0),
                ang_vel_range=(0.0, 1.8),
            ),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        # Auto-pacing curriculum with hysteresis to prevent thrashing.
        # - num_levels=20 → 0.05 increments
        # - increase_threshold=120s → must sustain 2-min episodes before bumping up.
        #   Combined with curriculum-scaled XY/Torque pushes (above), this means
        #   pushes stay OFF until the policy can stand/walk for 120s on its own.
        #   Then they grow gradually as the policy keeps proving stability.
        # - decrease_threshold=10s → if pushes break the policy and episodes
        #   collapse below 10s, curriculum backs off (level decreases, pushes
        #   shrink) until the policy recovers.
        return ksim.EpisodeLengthCurriculum(
            num_levels=20,
            increase_threshold=120.0,
            decrease_threshold=10.0,
        )

    def get_ppo_variables(
        self,
        model: KbotRNNModel,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        def scan_fn(
            actor_critic_carry: tuple[Array, Array], transition: ksim.Trajectory
        ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
            actor_carry, critic_carry = actor_critic_carry
            actor_dist, next_actor_carry = self.run_actor(
                model=model.actor,
                observations=transition.obs,
                commands=transition.command,
                carry=actor_carry,
            )
            log_probs = actor_dist.log_prob(transition.action)
            assert isinstance(log_probs, Array)
            value, next_critic_carry = self.run_critic(
                model=model.critic,
                observations=transition.obs,
                commands=transition.command,
                carry=critic_carry,
            )

            transition_ppo_variables = ksim.PPOVariables(
                log_probs=log_probs,
                values=value.squeeze(-1),
            )

            initial_carry = self.get_initial_model_carry(rng)
            next_carry = jax.tree.map(
                lambda x, y: jnp.where(transition.done, x, y), initial_carry, (next_actor_carry, next_critic_carry)
            )

            return next_carry, transition_ppo_variables

        next_model_carry, ppo_variables = jax.lax.scan(scan_fn, model_carry, trajectory)

        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: KbotRNNModel,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool = False,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
            aux_outputs=None,
        )

    def make_export_model(self, model: KbotRNNModel, stochastic: bool = False, batched: bool = False) -> Callable:
        """Makes a callable inference function that directly takes a flattened input vector and returns an action.

        Returns:
            A tuple containing the inference function and the size of the input vector.
        """

        def deterministic_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
            dist, carry = model.actor.call_flat_obs(obs, carry)
            return dist.mode(), carry

        def stochastic_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
            dist, carry = model.actor.call_flat_obs(obs, carry)
            return dist.sample(seed=jax.random.PRNGKey(0)), carry

        if stochastic:
            model_fn = stochastic_model_fn
        else:
            model_fn = deterministic_model_fn

        if batched:

            def batched_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
                return jax.vmap(model_fn)(obs, carry)

            return batched_model_fn

        return model_fn

    def on_after_checkpoint_save(self, ckpt_path: Path, state: xax.State) -> xax.State:
        if not self.config.export_for_inference:
            return state

        model: KbotRNNModel = self.load_ckpt(ckpt_path, part="model")[0]

        model_fn = self.make_export_model(model, stochastic=False, batched=True)
        input_shapes = [
            (RNN_NUM_INPUTS,),
            (
                self.config.depth,
                self.config.hidden_size,
            ),
        ]

        tf_path = (
            ckpt_path.parent / "tf_model"
            if self.config.only_save_most_recent
            else ckpt_path.parent / f"tf_model_{state.num_steps}"
        )

        if export is not None:
            export(model_fn, input_shapes, tf_path)

        return state


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m ksim_kbot.walking.walking_legs_rnn
    # To visualize the environment, use the following command:
    #   python -m ksim_kbot.walking.walking_legs_rnn run_model_viewer=True
    KbotLegsWalkingRNNTask.launch(
        KbotLegsWalkingRNNTaskConfig(
            num_envs=3072,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            # Simulation parameters.
            iterations=6,
            ls_iterations=6,
            dt=0.002,
            ctrl_dt=0.02,
            action_latency_range=(0.0, 0.005),
            # Bumped back to 5s now that walking is emerging. Longer rollouts
            # give better long-horizon credit assignment for full walking cycles
            # and balance recovery. (Was 2s during early bootstrap.)
            rollout_length_seconds=5.0,
            # PPO parameters
            action_scale=1.0,
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.005,
            learning_rate=1e-4,
            clip_param=0.3,
            max_grad_norm=0.3,
            valid_every_n_steps=25,
            save_every_n_steps=25,
            export_for_inference=False,
            only_save_most_recent=True,
            # Task parameters
            domain_randomize=True,
            gait_freq_lower=1.25,
            gait_freq_upper=1.5,
            reward_clip_min=0.0,
            reward_clip_max=1000.0,
        ),
    )
