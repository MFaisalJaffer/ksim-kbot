"""Common rewards for K-Bot 2.

If some logic will become more general, we can move it to ksim or xax.
"""

from typing import Literal, Self

import attrs
import jax
import jax.numpy as jnp
import ksim
import xax
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array, PRNGKeyArray, PyTree
from ksim.utils.mujoco import get_qpos_data_idxs_by_name


@attrs.define(frozen=True, kw_only=True)
class JointDeviationPenalty(ksim.Reward):
    """Penalty for joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    joint_targets: tuple[float, ...] = attrs.field()
    joint_weights: tuple[float, ...] = attrs.field(default=None)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        diff = trajectory.qpos[..., 7:] - jnp.array(self.joint_targets)
        cost = jnp.square(diff) * jnp.array(self.joint_weights)
        reward_value = jnp.sum(cost, axis=-1)
        return reward_value

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        *,
        joint_targets: tuple[float, ...],
        joint_weights: tuple[float, ...] | None = None,
    ) -> Self:
        if joint_weights is None:
            joint_weights = tuple([1.0] * len(joint_targets))

        return cls(
            scale=scale,
            joint_targets=joint_targets,
            joint_weights=joint_weights,
        )


@attrs.define(frozen=True, kw_only=True)
class FeetSlipPenalty(ksim.Reward):
    """Penalty for feet sliding along the ground while in contact.

    Measures the true horizontal velocity of each foot (computed by differencing
    consecutive foot positions along the trajectory) and penalizes that velocity
    only when the foot is in contact with the floor.

      penalty = sum_over_feet( ||foot_xy_velocity|| * is_in_contact )

    A foot that's planted while the body moves over it has velocity ≈ 0 in
    world frame — so this is zero during normal walking. A foot that's
    *sliding* along the ground generates a non-zero horizontal velocity while
    in contact, which is exactly what we want to penalize.
    """

    scale: float = -1.0
    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    feet_contact_obs_name: str = attrs.field(default="feet_contact_observation")
    ctrl_dt: float = attrs.field(default=0.02)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.feet_contact_obs_name not in trajectory.obs:
            raise ValueError(
                f"Observation {self.feet_contact_obs_name} not found; add it as an observation in your task."
            )
        contact = trajectory.obs[self.feet_contact_obs_name]              # (T, 2)
        foot_pos = trajectory.obs[self.feet_pos_obs_name]                 # (T, 6) = [Lxyz, Rxyz]
        # Stack xy positions per foot: (T, 2_feet, 2_xy)
        foot_xy = jnp.stack([foot_pos[..., 0:2], foot_pos[..., 3:5]], axis=-2)
        # Per-timestep horizontal velocity. First timestep gets zero by prepending itself.
        foot_xy_vel = jnp.diff(foot_xy, axis=0, prepend=foot_xy[:1]) / self.ctrl_dt
        # Speed per foot per timestep: (T, 2_feet)
        foot_speed = jnp.linalg.norm(foot_xy_vel, axis=-1)
        # Penalize only while in contact.
        penalty = jnp.sum(foot_speed * contact, axis=-1)
        return penalty


@attrs.define(frozen=True, kw_only=True)
class SensorOrientationPenalty(ksim.Reward):
    """Penalty for the orientation of the robot."""

    norm: xax.NormType = attrs.field(default="l2")
    obs_name: str = attrs.field(default="sensor_observation_upvector_origin")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        reward_value = xax.get_norm(trajectory.obs[self.obs_name][..., :2], self.norm).sum(axis=-1)
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class OrientationPenalty(ksim.Reward):
    """Penalizes deviation from upright orientation using the upvector approach.

    Rotates a unit up vector [0,0,1] by the current quaternion
    and penalizes any x,y components, which should be zero if perfectly upright.
    """

    scale: float = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        quat = trajectory.qpos[..., 3:7]
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = Rotation(quat).apply(up)
        orientation_penalty = jnp.sum(jnp.square(rot_up[..., :2]), axis=-1)
        return orientation_penalty


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)
    linvel_obs_name: str = attrs.field(default="sensor_observation_local_linvel_origin")
    command_name: str = attrs.field(default="linear_velocity_command")
    norm: xax.NormType = attrs.field(default="l2")
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.linvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.linvel_obs_name} not found; add it as an observation in your task.")

        command = trajectory.command[self.command_name]
        lin_vel_error = xax.get_norm(command - trajectory.obs[self.linvel_obs_name][..., :2], self.norm).sum(axis=-1)
        reward_value = jnp.exp(-lin_vel_error / self.error_scale)

        command_norm = jnp.linalg.norm(command, axis=-1)
        reward_value *= command_norm > self.stand_still_threshold

        return reward_value


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the angular velocity."""

    error_scale: float = attrs.field(default=0.25)
    angvel_obs_name: str = attrs.field(default="sensor_observation_gyro_origin")
    command_name: str = attrs.field(default="angular_velocity_command")
    norm: xax.NormType = attrs.field(default="l2")
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.angvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.angvel_obs_name} not found; add it as an observation in your task.")

        command = trajectory.command[self.command_name]
        ang_vel_error = jnp.square(command.flatten() - trajectory.obs[self.angvel_obs_name][..., 2])
        reward_value = jnp.exp(-ang_vel_error / self.error_scale)

        command_norm = jnp.linalg.norm(command, axis=-1)
        reward_value *= command_norm > self.stand_still_threshold

        return reward_value


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityXYPenalty(ksim.Reward):
    """Penalty for the angular velocity."""

    norm: xax.NormType = attrs.field(default="l2")
    angvel_obs_name: str = attrs.field(default="sensor_observation_global_angvel_origin")
    command_name: str = attrs.field(default="angular_velocity_command")
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.angvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.angvel_obs_name} not found; add it as an observation in your task.")
        ang_vel = trajectory.obs[self.angvel_obs_name][..., :2]
        command = trajectory.command[self.command_name]
        command_norm = jnp.linalg.norm(command, axis=-1)
        reward_value = xax.get_norm(ang_vel, self.norm).sum(axis=-1)
        reward_value *= command_norm > self.stand_still_threshold
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class HipDeviationPenalty(ksim.Reward):
    """Penalty for hip joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    hip_indices: tuple[int, ...] = attrs.field()
    joint_targets: tuple[float, ...] = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        diff = (
            trajectory.qpos[..., jnp.array(self.hip_indices) + 7]
            - jnp.array(self.joint_targets)[jnp.array(self.hip_indices)]
        )
        reward_value = xax.get_norm(diff, self.norm).sum(axis=-1)
        return reward_value

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        hip_names: tuple[str, ...],
        joint_targets: tuple[float, ...],
        scale: float = -1.0,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        mappings = get_qpos_data_idxs_by_name(physics_model)
        hip_indices = tuple([int(mappings[name][0]) - 7 for name in hip_names])
        return cls(
            hip_indices=hip_indices,
            joint_targets=joint_targets,
            scale=scale,
        )


@attrs.define(frozen=True, kw_only=True)
class KneeDeviationPenalty(ksim.Reward):
    """Penalty for knee joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    knee_indices: tuple[int, ...] = attrs.field()
    joint_targets: tuple[float, ...] = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        diff = (
            trajectory.qpos[..., jnp.array(self.knee_indices) + 7]
            - jnp.array(self.joint_targets)[jnp.array(self.knee_indices)]
        )
        reward_value = xax.get_norm(diff, self.norm).sum(axis=-1)
        return reward_value

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        knee_names: tuple[str, ...],
        joint_targets: tuple[float, ...],
        scale: float = -1.0,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        mappings = get_qpos_data_idxs_by_name(physics_model)
        knee_indices = tuple([int(mappings[name][0]) - 7 for name in knee_names])
        return cls(
            knee_indices=knee_indices,
            joint_targets=joint_targets,
            scale=scale,
        )


@attrs.define(frozen=True, kw_only=True)
class KneeRangeOfMotion(ksim.Reward):
    """Diagnostic metric: logs knee joint range of motion per trajectory. Not a real reward (scale=0)."""

    knee_indices: tuple[int, ...] = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        knee_pos = trajectory.qpos[..., jnp.array(self.knee_indices) + 7]
        # Return per-timestep max absolute angle across both knees
        return jnp.abs(knee_pos).max(axis=-1)

    @classmethod
    def create(cls, physics_model: ksim.PhysicsModel, knee_names: tuple[str, ...]) -> "KneeRangeOfMotion":
        mappings = get_qpos_data_idxs_by_name(physics_model)
        knee_indices = tuple([int(mappings[name][0]) - 7 for name in knee_names])
        return cls(scale=0.001, knee_indices=knee_indices)


@attrs.define(frozen=True, kw_only=True)
class SingleFootContactReward(ksim.StatefulReward):
    """Reward having one and only one foot in contact with the ground while walking.

    Allows a small grace period where both feet may be in contact, for less jumpy gaits.
    Adapted from kscalelabs/ksim/examples/kbot/train.py to use the
    `feet_contact_observation` (2-vec) and our split linear/angular velocity commands.
    """

    ctrl_dt: float = 0.02
    grace_period: float = 0.2  # seconds
    contact_threshold: float = 0.1
    feet_contact_obs_name: str = attrs.field(default="feet_contact_observation")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        return jnp.array([0.0])

    def get_reward_stateful(self, traj: ksim.Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        feet_contact = traj.obs[self.feet_contact_obs_name]  # (T, 2): [left, right]
        left_contact = feet_contact[..., 0] > self.contact_threshold
        right_contact = feet_contact[..., 1] > self.contact_threshold
        single = jnp.logical_xor(left_contact, right_contact)

        lin_cmd = traj.command[self.linear_velocity_cmd_name]
        ang_cmd = traj.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([lin_cmd, ang_cmd], axis=-1), axis=-1)
        is_zero_cmd = cmd_norm < 1e-3

        def _body(time_since_single_contact: Array, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
            is_single, is_zero = inputs
            new_time = jnp.where(is_single, 0.0, time_since_single_contact + self.ctrl_dt)
            # If zero command, reset grace timer so standing isn't penalized.
            new_time = jnp.where(is_zero, self.grace_period, new_time)
            return new_time, new_time

        carry, time_since_single_contact = jax.lax.scan(_body, reward_carry, (single, is_zero_cmd))
        within_grace = time_since_single_contact < self.grace_period
        reward = jnp.where(is_zero_cmd, 0.0, within_grace[:, 0])
        return reward, carry


@attrs.define(frozen=True, kw_only=True)
class FeetAirtimeReward(ksim.StatefulReward):
    """Encourages reasonable step frequency by rewarding long swing phases.

    Pays out at the moment a foot first contacts the ground (after being airborne)
    a value of (airtime - touchdown_penalty). With touchdown_penalty=0.4s:
    - airtime < 0.4s (e.g. marching in place): NEGATIVE reward at touchdown
    - airtime > 0.4s (real walking step): POSITIVE reward at touchdown

    Disabled during zero-velocity commands (so standing isn't penalized).
    Adapted from kscalelabs/ksim/examples/kbot/train.py to use our two-vector
    `feet_contact_observation` and split linear/angular velocity commands.
    """

    ctrl_dt: float = 0.02
    touchdown_penalty: float = 0.4
    feet_contact_obs_name: str = attrs.field(default="feet_contact_observation")
    contact_threshold: float = 0.1
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        # Flat shape (4,): [airtime_left, airtime_right, prev_contact_left, prev_contact_right]
        # The installed ksim version doesn't handle tuple carries in jnp.where, so we pack
        # everything into a single float array. Contacts stored as 0.0/1.0.
        return jnp.array([0.0, 0.0, 1.0, 1.0])

    def _compute_airtime(self, initial_airtime: Array, contact_bool: Array, done: Array) -> tuple[Array, Array]:
        def _body(time_since_liftoff: Array, is_contact: Array) -> tuple[Array, Array]:
            new_time = jnp.where(is_contact, 0.0, time_since_liftoff + self.ctrl_dt)
            return new_time, new_time

        contact_or_done = jnp.logical_or(contact_bool, done[:, None])
        carry, airtime = jax.lax.scan(_body, initial_airtime, contact_or_done)
        return carry, airtime

    def _compute_first_contact(self, contact_carry: Array, contact_bool: Array) -> Array:
        prev_contact = jnp.concatenate([contact_carry[None, :], contact_bool[:-1]], axis=0)
        first_contact = jnp.logical_and(contact_bool, jnp.logical_not(prev_contact))
        return first_contact

    def get_reward_stateful(self, traj: ksim.Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        airtime_carry = reward_carry[:2]
        contact_carry = reward_carry[2:] > 0.5  # bool (2,)

        feet_contact = traj.obs[self.feet_contact_obs_name]  # (T, 2): [left, right]
        contact = feet_contact > self.contact_threshold  # bool, shape (T, 2)

        new_airtime_carry, airtime = self._compute_airtime(airtime_carry, contact, traj.done)
        first_contact = self._compute_first_contact(contact_carry, contact) * ~traj.done[:, None]
        # Shift airtime by 1 to match touchdowns with previous step's airtime.
        airtime_shifted = jnp.concatenate([airtime_carry[None, :], airtime], axis=0)[:-1, :]
        reward = jnp.sum(
            (airtime_shifted - self.touchdown_penalty) * first_contact.astype(jnp.float32),
            axis=-1,
        )

        # Disable when there is no velocity command.
        lin_cmd = traj.command[self.linear_velocity_cmd_name]
        ang_cmd = traj.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([lin_cmd, ang_cmd], axis=-1), axis=-1)
        is_zero_cmd = cmd_norm < 1e-3
        reward = jnp.where(is_zero_cmd, 0.0, reward)

        # Pack new carry: (airtime_l, airtime_r, contact_l, contact_r) as float
        new_reward_carry = jnp.concatenate([new_airtime_carry, contact[-1, :].astype(jnp.float32)])
        return reward, new_reward_carry


@attrs.define(frozen=True, kw_only=True)
class MarchInPlacePenalty(ksim.Reward):
    """Penalize lifting feet when commanded to translate but not actually translating.

    Computes: penalty ∝ (max foot height) × (1 - velocity_match) × (cmd is active)
    Returns a positive value (use with negative scale).

    - When standing still and commanded to stand: 0 (cmd inactive)
    - When walking and tracking velocity well: ~0 (velocity_match ≈ 1)
    - When marching in place (feet up but body still): high penalty
    """

    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    linvel_obs_name: str = attrs.field(default="sensor_observation_local_linvel_origin")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    velocity_match_sensitivity: float = attrs.field(default=0.25)
    foot_default_height: float = attrs.field(default=0.04)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        foot_pos = trajectory.obs[self.feet_pos_obs_name]
        # Max foot height above default — how much the policy is "lifting feet"
        foot_left_z = foot_pos[..., 2]
        foot_right_z = foot_pos[..., 5]
        max_lift = jnp.maximum(foot_left_z, foot_right_z) - self.foot_default_height
        max_lift = jnp.maximum(max_lift, 0.0)

        # How well actual velocity matches commanded velocity
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        actual_xy_vel = trajectory.obs[self.linvel_obs_name][..., :2]
        vel_err_sq = jnp.sum(jnp.square(vel_cmd - actual_xy_vel), axis=-1)
        velocity_match = jnp.exp(-vel_err_sq / self.velocity_match_sensitivity)

        # Only active when a velocity command is being given
        lin_cmd_norm = jnp.linalg.norm(vel_cmd, axis=-1)
        cmd_active = lin_cmd_norm > 1e-3

        # Penalty: feet lifted × velocity not matching × command active
        penalty = max_lift * (1.0 - velocity_match) * cmd_active
        return penalty


@attrs.define(frozen=True, kw_only=True)
class NoContactPenalty(ksim.Reward):
    """Penalty for having no foot in contact with the ground while walking (i.e., flying)."""

    contact_threshold: float = 0.1
    feet_contact_obs_name: str = attrs.field(default="feet_contact_observation")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        feet_contact = traj.obs[self.feet_contact_obs_name]
        left_contact = feet_contact[..., 0] > self.contact_threshold
        right_contact = feet_contact[..., 1] > self.contact_threshold
        any_contact = jnp.logical_or(left_contact, right_contact)

        lin_cmd = traj.command[self.linear_velocity_cmd_name]
        ang_cmd = traj.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([lin_cmd, ang_cmd], axis=-1), axis=-1)
        is_zero_cmd = cmd_norm < 1e-3

        # Penalty (positive value) when both feet are airborne and command is non-zero.
        # Returns 0 when zero command or when at least one foot is in contact.
        return jnp.where(is_zero_cmd, 0.0, jnp.where(any_contact, 0.0, 1.0))


@attrs.define(frozen=True, kw_only=True)
class TerminationPenalty(ksim.Reward):
    """Penalty for termination."""

    scale: float = attrs.field(default=-1.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        reward_value = trajectory.done
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class XYPositionPenalty(ksim.Reward):
    """Penalty for deviation from a target (x, y) position."""

    target_x: float = attrs.field()
    target_y: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l2")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        current_pos = trajectory.qpos[..., :2]
        target_pos = jnp.array([self.target_x, self.target_y])
        diff = current_pos - target_pos
        reward_value = xax.get_norm(diff, self.norm).sum(axis=-1)
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class FarFromOriginTerminationReward(ksim.Reward):
    """Reward for being far from the origin."""

    max_dist: float = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        reward_value = jnp.linalg.norm(trajectory.qpos[..., :2], axis=-1) > self.max_dist
        return reward_value


@attrs.define(frozen=True, kw_only=True)
class KsimLinearVelocityTrackingReward(ksim.Reward):
    """Penalty for deviating from the linear velocity command."""

    index: int = attrs.field()
    command_name: str = attrs.field()
    norm: xax.NormType = attrs.field(default="l1")
    temp: float = attrs.field(default=1.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        dim = self.index
        lin_vel_cmd = trajectory.command[self.command_name].squeeze(-1)
        lin_vel = trajectory.qvel[..., dim]
        norm = xax.get_norm(lin_vel - lin_vel_cmd, self.norm)
        reward_value = 1.0 / (norm / self.temp + 1.0)
        return reward_value

    def get_name(self) -> str:
        return f"{self.index}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class JointPositionLimitPenalty(ksim.Reward):
    """Penalty for joint position limits."""

    lower_limits: xax.HashableArray = attrs.field()
    upper_limits: xax.HashableArray = attrs.field()

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        *,
        soft_limit_factor: float = 0.95,
        scale: float = -1.0,
    ) -> Self:
        # Note: First joint is freejoint.
        lowers, uppers = physics_model.jnt_range[1:].T
        center = (lowers + uppers) / 2
        range = uppers - lowers
        soft_lowers = center - 0.5 * range * soft_limit_factor
        soft_uppers = center + 0.5 * range * soft_limit_factor

        return cls(
            scale=scale,
            lower_limits=xax.hashable_array(soft_lowers),
            upper_limits=xax.hashable_array(soft_uppers),
        )

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        penalty = -jnp.clip(trajectory.qpos[..., 7:] - self.lower_limits.array, None, 0.0)
        penalty += jnp.clip(trajectory.qpos[..., 7:] - self.upper_limits.array, 0.0, None)
        return jnp.sum(penalty, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class ContactForcePenalty(ksim.Reward):
    """Penalty for too high contact force."""

    max_contact_force: float = attrs.field(default=350.0)
    sensor_names: tuple[str, ...] = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        for sensor_name in self.sensor_names:
            if sensor_name not in trajectory.obs:
                raise ValueError(f"{sensor_name} not found in trajectory.obs")

        forces_t3b = jnp.stack([trajectory.obs[name] for name in self.sensor_names], axis=-1)
        cost = jnp.clip(jnp.abs(forces_t3b[..., 2, :]) - self.max_contact_force, min=0.0)
        cost = jnp.sum(cost, axis=-1)
        return cost


@attrs.define(frozen=True, kw_only=True)
class StandStillReward(ksim.Reward):
    """Reward for standing still upright at target joint pose.

    Includes an orientation gate: when the robot is leaning, the reward drops
    proportionally. This prevents the large stand-still reward from competing
    against recovery foot steps — when leaning, the reward falls away and the
    OrientationPenalty gradient takes over, teaching the robot to step to recover.
    """

    scale: float = 1.0
    sensitivity: float = 0.01
    norm: xax.NormType = attrs.field(default="l1")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    joint_targets: tuple[float, ...] = attrs.field()
    stand_still_threshold: float = attrs.field(default=0.0)
    # Orientation gate: reward is multiplied by exp(-lean_error / orientation_sensitivity).
    # At 0° lean: gate=1.0 (full reward). At ~15° lean: gate≈0.1 (reward drops to 10%).
    # Set to 0.0 to disable.
    orientation_sensitivity: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)

        error = jnp.sum(
            jnp.square(trajectory.qpos[..., 7:] - jnp.array(self.joint_targets)),
            axis=-1,
        )
        reward = jnp.exp(-error / self.sensitivity)
        reward *= cmd_norm < self.stand_still_threshold

        # Orientation gate: drop reward when leaning so recovery steps aren't fought.
        if self.orientation_sensitivity > 0.0:
            quat = trajectory.qpos[..., 3:7]
            up = jnp.array([0.0, 0.0, 1.0])
            rot_up = Rotation(quat).apply(up)
            lean_error = jnp.sum(jnp.square(rot_up[..., :2]), axis=-1)
            orientation_gate = jnp.exp(-lean_error / self.orientation_sensitivity)
            reward *= orientation_gate

        return reward


@attrs.define(frozen=True, kw_only=True)
class FeetPhaseReward(ksim.Reward):
    """Reward for tracking the desired foot height.

    If `translation_gated=True`, the reward is multiplied by a Gaussian gate of the
    body's actual XY velocity tracking error (in body frame). This means the policy
    only earns the gait-clock reward when it is *also* translating in the commanded
    direction — preventing the "marching in place" failure mode.
    """

    scale: float = 1.0
    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    gait_freq_cmd_name: str = attrs.field(default="gait_frequency_command")
    max_foot_height: float = attrs.field(default=0.12)
    ctrl_dt: float = attrs.field(default=0.02)
    sensitivity: float = attrs.field(default=0.01)
    foot_default_height: float = attrs.field(default=0.0)
    stand_still_threshold: float = attrs.field(default=0.0)
    # Translation gate options:
    translation_gated: bool = attrs.field(default=False)
    translation_gate_sensitivity: float = attrs.field(default=0.25)
    linvel_obs_name: str = attrs.field(default="sensor_observation_local_linvel_origin")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.feet_pos_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.feet_pos_obs_name} not found; add it as an observation in your task.")
        if self.gait_freq_cmd_name not in trajectory.command:
            raise ValueError(f"Command {self.gait_freq_cmd_name} not found; add it as a command in your task.")

        # generate phase values
        gait_freq_n = trajectory.command[self.gait_freq_cmd_name]

        phase_dt = 2 * jnp.pi * gait_freq_n * self.ctrl_dt
        steps = jnp.int32(trajectory.timestep / self.ctrl_dt)
        steps = jnp.repeat(steps[:, None], 2, axis=1)

        start_phase = jnp.broadcast_to(jnp.array([0.0, jnp.pi]), (steps.shape[0], 2))
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        # batch reward over the time dimension
        foot_pos = trajectory.obs[self.feet_pos_obs_name]

        foot_z = jnp.array([foot_pos[..., 2], foot_pos[..., 5]]).T
        ideal_z = self.gait_phase(phase, swing_height=jnp.array(self.max_foot_height))
        error = jnp.sum(jnp.square(foot_z - ideal_z), axis=-1)
        reward = jnp.exp(-error / self.sensitivity)

        # no movement for small velocity command
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        command_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        reward *= command_norm > self.stand_still_threshold

        # Optional translation gate: suppress reward when commanded but not actually translating.
        # Multiplies the foot-phase reward by exp(-‖cmd_vel - actual_xy_vel‖² / gate_sensitivity).
        # Only active when a non-zero velocity command is present (otherwise pass through).
        if self.translation_gated:
            actual_xy_vel = trajectory.obs[self.linvel_obs_name][..., :2]
            vel_err_sq = jnp.sum(jnp.square(vel_cmd - actual_xy_vel), axis=-1)
            translation_match = jnp.exp(-vel_err_sq / self.translation_gate_sensitivity)
            # If no linear velocity command, don't gate (let yaw-only commands still earn the reward).
            lin_cmd_norm = jnp.linalg.norm(vel_cmd, axis=-1)
            cmd_active = lin_cmd_norm > 1e-3
            gate = jnp.where(cmd_active, translation_match, 1.0)
            reward *= gate

        return reward

    def gait_phase(
        self,
        phi: Array | float,
        swing_height: Array = jnp.array(0.08),
    ) -> Array:
        """Interpolation logic for the gait phase.

        Original implementation:
        https://arxiv.org/pdf/2201.00206
        https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/gait.py#L33
        """
        x = (phi + jnp.pi) / (2 * jnp.pi)
        x = jnp.clip(x, 0, 1)
        stance = xax.cubic_bezier_interpolation(jnp.array(0), swing_height, 2 * x)
        swing = xax.cubic_bezier_interpolation(swing_height, jnp.array(0), 2 * x - 1)
        return jnp.where(x <= 0.5, stance, swing)


@attrs.define(frozen=True, kw_only=True)
class FootSwingClearancePenalty(ksim.Reward):
    """Penalize foot dragging during swing phase.

    When the gait clock expects a foot to be in the air (ideal height > min_clearance),
    but the actual foot is below min_clearance, apply a penalty proportional to
    how much it's dragging. Prevents the policy from shuffling feet along the ground.

    Only active when command is non-zero.
    """

    min_clearance: float = attrs.field(default=0.08)  # ~3 inches
    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    feet_endpoints_obs_name: str = attrs.field(default="feet_endpoints_observation")
    gait_freq_cmd_name: str = attrs.field(default="gait_frequency_command")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    max_foot_height: float = attrs.field(default=0.12)
    ctrl_dt: float = attrs.field(default=0.02)
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # Compute gait phase clock (same as FeetPhaseReward)
        gait_freq_n = trajectory.command[self.gait_freq_cmd_name]
        phase_dt = 2 * jnp.pi * gait_freq_n * self.ctrl_dt
        steps = jnp.int32(trajectory.timestep / self.ctrl_dt)
        steps = jnp.repeat(steps[:, None], 2, axis=1)
        start_phase = jnp.broadcast_to(jnp.array([0.0, jnp.pi]), (steps.shape[0], 2))
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        # Expected foot height from gait clock
        x = (phase + jnp.pi) / (2 * jnp.pi)
        x = jnp.clip(x, 0, 1)
        swing_h = jnp.array(self.max_foot_height)
        stance = xax.cubic_bezier_interpolation(jnp.array(0.0), swing_h, 2 * x)
        swing  = xax.cubic_bezier_interpolation(swing_h, jnp.array(0.0), 2 * x - 1)
        ideal_z = jnp.where(x <= 0.5, stance, swing)  # shape (T, 2)

        # Actual foot heights — use minimum across center + heel + toe per foot.
        # Prevents tilt exploit: heel or toe touching ground = foot not clear.
        foot_pos = trajectory.obs[self.feet_pos_obs_name]
        center_z = jnp.stack([foot_pos[..., 2], foot_pos[..., 5]], axis=-1)
        ep = trajectory.obs[self.feet_endpoints_obs_name]
        left_min_z  = jnp.minimum(ep[..., 2],  ep[..., 5])   # min(left_heel_z,  left_toe_z)
        right_min_z = jnp.minimum(ep[..., 8],  ep[..., 11])  # min(right_heel_z, right_toe_z)
        foot_z = jnp.minimum(center_z, jnp.stack([left_min_z, right_min_z], axis=-1))

        # How much the gait clock expects the foot above min_clearance
        expected_above = jnp.maximum(ideal_z - self.min_clearance, 0.0)
        # How much the foot is actually below min_clearance (dragging)
        actual_below = jnp.maximum(self.min_clearance - foot_z, 0.0)

        # Penalty = product: only fires when BOTH gait says "up" AND foot is dragging
        penalty = jnp.sum(expected_above * actual_below, axis=-1)

        # Only active when commanded to move
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        penalty *= cmd_norm > self.stand_still_threshold

        return penalty


@attrs.define(frozen=True, kw_only=True)
class WalkingPostureReward(ksim.Reward):
    """Reward minimum knee bend AND foot clearance when commanded to walk.

    Requires knees to be bent at least min_knee_bend radians, but does NOT
    prescribe a specific target angle — the policy is free to discover the
    natural varying bend through the gait cycle.

    Reward = knee_bend_reward × foot_clearance_gate × is_walking

    - knee_bend_reward: 1.0 when both knees exceed min_knee_bend, decays
      smoothly to 0 when knees are straight. No penalty for bending MORE.
    - foot_clearance_gate: mean compliance across feet during swing phase.
      For each foot in swing (ideal_z > 0), compliance = sigmoid of how far
      foot_z is above min_clearance. During stance, contributes 1.0.
    """

    # Minimum required knee bend — reward is full above this, decays below.
    min_knee_bend: float = attrs.field(default=0.4)   # ~23°, must be meaningfully bent when walking
    sensitivity: float = attrs.field(default=0.05)   # how sharply reward falls below min_bend
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    stand_still_threshold: float = attrs.field(default=0.1)
    # qpos indices for knees (7-offset already removed — these are indices into qpos[7:])
    right_knee_idx: int = attrs.field(default=13)  # dof_right_knee_04
    left_knee_idx: int = attrs.field(default=18)   # dof_left_knee_04
    # Foot clearance gate parameters
    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    feet_endpoints_obs_name: str = attrs.field(default="feet_endpoints_observation")
    gait_freq_cmd_name: str = attrs.field(default="gait_frequency_command")
    min_clearance: float = attrs.field(default=0.08)   # ~3 inches
    max_foot_height: float = attrs.field(default=0.12)
    ctrl_dt: float = attrs.field(default=0.02)
    clearance_sensitivity: float = attrs.field(default=0.02)  # sigmoid steepness for clearance gate

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        is_walking = cmd_norm > self.stand_still_threshold

        # Right knee bends negative, left knee bends positive (robot convention).
        # Use absolute value — we only care that the knee IS bent, not how much.
        r_knee = trajectory.qpos[..., 7 + self.right_knee_idx]
        l_knee = trajectory.qpos[..., 7 + self.left_knee_idx]

        # Shortfall = how far below min_bend each knee is (0 if already bent enough).
        r_shortfall = jnp.maximum(0.0, self.min_knee_bend - jnp.abs(r_knee))
        l_shortfall = jnp.maximum(0.0, self.min_knee_bend - jnp.abs(l_knee))
        knee_match = jnp.exp(-(r_shortfall + l_shortfall) / self.sensitivity)

        # Foot clearance gate: only give reward if feet are also being lifted.
        # Compute gait clock phase for each foot (left=0, right=π offset).
        gait_freq_n = trajectory.command[self.gait_freq_cmd_name]
        phase_dt = 2 * jnp.pi * gait_freq_n * self.ctrl_dt
        steps = jnp.int32(trajectory.timestep / self.ctrl_dt)
        steps = jnp.repeat(steps[:, None], 2, axis=1)
        start_phase = jnp.broadcast_to(jnp.array([0.0, jnp.pi]), (steps.shape[0], 2))
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi
        x = (phase + jnp.pi) / (2 * jnp.pi)
        x = jnp.clip(x, 0, 1)
        swing_h = jnp.array(self.max_foot_height)
        stance_curve = xax.cubic_bezier_interpolation(jnp.array(0.0), swing_h, 2 * x)
        swing_curve  = xax.cubic_bezier_interpolation(swing_h, jnp.array(0.0), 2 * x - 1)
        ideal_z = jnp.where(x <= 0.5, stance_curve, swing_curve)

        # For feet expected to be in swing (ideal_z > 0), gate by clearance compliance.
        # Use minimum z across center + heel + toe to prevent tilt exploits.
        # endpoints obs: [left_heel_xyz, left_toe_xyz, right_heel_xyz, right_toe_xyz]
        foot_pos = trajectory.obs[self.feet_pos_obs_name]
        center_z = jnp.stack([foot_pos[..., 2], foot_pos[..., 5]], axis=-1)  # left, right
        ep = trajectory.obs[self.feet_endpoints_obs_name]
        left_min_z  = jnp.minimum(ep[..., 2],  ep[..., 5])   # min(left_heel_z,  left_toe_z)
        right_min_z = jnp.minimum(ep[..., 8],  ep[..., 11])  # min(right_heel_z, right_toe_z)
        endpoint_min_z = jnp.stack([left_min_z, right_min_z], axis=-1)
        # Most conservative: min of center, heel, and toe — all three must clear
        foot_z = jnp.minimum(center_z, endpoint_min_z)

        in_swing = ideal_z > 0.0
        clearance_diff = (foot_z - self.min_clearance) / self.clearance_sensitivity
        clearance_compliance = jax.nn.sigmoid(clearance_diff)
        gate_per_foot = jnp.where(in_swing, clearance_compliance, 1.0)
        foot_clearance_gate = jnp.mean(gate_per_foot, axis=-1)

        return knee_match * foot_clearance_gate * is_walking


@attrs.define(frozen=True, kw_only=True)
class FeetPhasePenalty(ksim.Reward):
    """Penalty for NOT following the gait clock when commanded to move.

    Complement of FeetPhaseReward: while FeetPhaseReward gives a carrot for
    correct foot phasing, this gives a stick for incorrect phasing.
    Returns (1 - phase_match) when command is active → 0 when perfect, 1 when terrible.
    Use with negative scale.
    """

    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    gait_freq_cmd_name: str = attrs.field(default="gait_frequency_command")
    max_foot_height: float = attrs.field(default=0.12)
    ctrl_dt: float = attrs.field(default=0.02)
    sensitivity: float = attrs.field(default=0.01)
    foot_default_height: float = attrs.field(default=0.0)
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        gait_freq_n = trajectory.command[self.gait_freq_cmd_name]
        phase_dt = 2 * jnp.pi * gait_freq_n * self.ctrl_dt
        steps = jnp.int32(trajectory.timestep / self.ctrl_dt)
        steps = jnp.repeat(steps[:, None], 2, axis=1)
        start_phase = jnp.broadcast_to(jnp.array([0.0, jnp.pi]), (steps.shape[0], 2))
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        foot_pos = trajectory.obs[self.feet_pos_obs_name]
        foot_z = jnp.array([foot_pos[..., 2], foot_pos[..., 5]]).T
        ideal_z = self._gait_phase(phase, swing_height=jnp.array(self.max_foot_height))
        error = jnp.sum(jnp.square(foot_z - ideal_z), axis=-1)
        phase_match = jnp.exp(-error / self.sensitivity)

        # Penalty is (1 - match): 0 when perfect, 1 when completely off.
        penalty = 1.0 - phase_match

        # Only active when command is non-zero.
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        command_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        penalty *= command_norm > self.stand_still_threshold

        return penalty

    def _gait_phase(self, phi: Array, swing_height: Array) -> Array:
        x = (phi + jnp.pi) / (2 * jnp.pi)
        x = jnp.clip(x, 0, 1)
        stance = xax.cubic_bezier_interpolation(jnp.array(0), swing_height, 2 * x)
        swing = xax.cubic_bezier_interpolation(swing_height, jnp.array(0), 2 * x - 1)
        return jnp.where(x <= 0.5, stance, swing)


@attrs.define(frozen=True, kw_only=True)
class ArmConstraintReward(ksim.Reward):
    """Penalty for deviating from a commanded arm pose, only when constrained.

    Reads the ArmConstraintCommand (11 floats: is_constrained + 10 target joints).
    When is_constrained = 1, returns an exp-shaped reward that peaks at 1.0 when
    arms exactly match the target pose and decays to 0 as deviation grows.
    When is_constrained = 0, returns 0 (no contribution either way).

    Trains the policy to keep arms still on demand (carrying tasks) using only
    legs/torso for balance, rather than relying on free arm swings.
    """

    command_name: str = attrs.field(default="arm_constraint_command")
    sensitivity: float = attrs.field(default=0.5)  # how fast reward decays with deviation
    # qpos indices for arm joints (offset of 7 for freejoint root already excluded —
    # these are indices into qpos[7:]). Right arm 0..4, left arm 5..9.
    arm_qpos_start: int = attrs.field(default=0)
    arm_qpos_count: int = attrs.field(default=10)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        cmd = trajectory.command[self.command_name]
        is_constrained = cmd[..., 0]
        target_pose = cmd[..., 1:1 + self.arm_qpos_count]

        # Current arm joint angles.
        arm_qpos = trajectory.qpos[..., 7 + self.arm_qpos_start : 7 + self.arm_qpos_start + self.arm_qpos_count]

        # Sum of squared deviations across all arm joints.
        sq_dev = jnp.sum(jnp.square(arm_qpos - target_pose), axis=-1)
        reward = jnp.exp(-sq_dev / self.sensitivity)
        return reward * is_constrained


@attrs.define(frozen=True)
class TargetLinearVelocityReward(ksim.Reward):
    """Reward for forward motion."""

    index: Literal["x", "y", "z"] = attrs.field(default="x")
    target_vel: float = attrs.field(default=0.0)
    norm: xax.NormType = attrs.field(default="l1")
    monotonic_fn: Literal["exp", "inv"] = attrs.field(default="inv")
    temp: float = attrs.field(default=1.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        vel = trajectory.qvel[..., ksim.cartesian_index_to_dim(self.index)]
        error = xax.get_norm(vel - self.target_vel, self.norm)
        return ksim.norm_to_reward(error, temp=self.temp, monotonic_fn=self.monotonic_fn)

    def get_name(self) -> str:
        return f"{self.index}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class TargetHeightReward(ksim.Reward):
    """Reward for reaching a target height."""

    target_height: float = attrs.field(default=1.0)
    norm: xax.NormType = attrs.field(default="l1")
    temp: float = attrs.field(default=1.0)
    monotonic_fn: Literal["exp", "inv"] = attrs.field(default="inv")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        qpos = trajectory.qpos
        error = qpos[..., 2] - self.target_height
        reward_value = ksim.norm_to_reward(
            xax.get_norm(error, self.norm), temp=self.temp, monotonic_fn=self.monotonic_fn
        )
        return reward_value
