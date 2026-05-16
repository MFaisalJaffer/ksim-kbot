"""Common utilities for K-Bot 2.

If some utilities will become more general, we can move them to ksim or xax.
"""

from typing import Collection, Self

import attrs
import jax
import jax.numpy as jnp
import ksim
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.utils.mujoco import (
    get_sensor_data_idxs_by_name,
    get_site_data_idx_from_name,
    slice_update,
    update_data_field,
)
from ksim.utils.priors import (
    MotionReferenceData,
    get_local_xpos,
)
from mujoco import mjx


class TargetPositionMITActuators(ksim.PositionVelocityActuator):
    """MIT-mode actuator controller operating on position."""

    def __init__(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata,
        default_targets: tuple[float, ...] = (),
        *,
        pos_action_noise: float = 0.0,
        pos_action_noise_type: ksim.actuators.NoiseType = "none",
        vel_action_noise: float = 0.0,
        vel_action_noise_type: ksim.actuators.NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: ksim.actuators.NoiseType = "none",
        ctrl_clip: list[float] | None = None,
        action_scale: float = 1.0,
        freejoint_first: bool = True,
    ) -> None:
        super().__init__(
            physics_model=physics_model,
            metadata=metadata,
            pos_action_noise=pos_action_noise,
            pos_action_noise_type=pos_action_noise_type,
            vel_action_noise=vel_action_noise,
            vel_action_noise_type=vel_action_noise_type,
            torque_noise=torque_noise,
            torque_noise_type=torque_noise_type,
        )
        if ctrl_clip is not None:
            self.ctrl_clip = jnp.array(ctrl_clip)
        self.action_scale = action_scale
        self.default_targets = jnp.array(default_targets)

    def get_ctrl(self, action: Array, physics_data: ksim.PhysicsData, rng: PRNGKeyArray) -> Array:
        """Get the control signal from the (position and velocity) action vector."""
        pos_rng, vel_rng, tor_rng = jax.random.split(rng, 3)

        current_pos = physics_data.qpos[7:]  # First 7 are always root pos (freejoint).
        current_vel = physics_data.qvel[6:]  # First 6 are always root vel.

        # Adds position and velocity noise.
        target_position = action[: len(current_pos)] * self.action_scale + self.default_targets
        target_velocity = action[len(current_pos) :] * self.action_scale
        target_position = self.add_noise(self.action_noise, self.action_noise_type, target_position, pos_rng)
        target_velocity = self.add_noise(self.vel_action_noise, self.vel_action_noise_type, target_velocity, vel_rng)

        pos_delta = target_position - current_pos
        vel_delta = target_velocity - current_vel
        ctrl = self.kps * pos_delta + self.kds * vel_delta
        return jnp.clip(
            self.add_noise(self.torque_noise, self.torque_noise_type, ctrl, tor_rng),
            -self.ctrl_clip,
            self.ctrl_clip,
        )


# Torque–velocity curves for the K-Bot 2 BLDC actuators.
# omega in rad/s, tau in Nm. Sourced from the actuator datasheets.
# At a given joint speed |omega|, the motor can deliver at most tau(|omega|) Nm.
# Above the no-load speed, tau = 0 (motor cannot produce torque at all).
#
# ALL TAU VALUES SCALED BY 0.85 from the original datasheet to bake in a
# 15% sim-to-real safety margin. The policy will never see more torque than
# 85% of the spec'd peak, so when deployed on real motors (where degradation
# from heat/wear can easily exceed 15%) it has known headroom. Combined with
# the per-step `tv_curve_randomization` of ±15%, effective torque seen during
# training is in [0.72×, 0.85×] of the cold-motor spec.
TV_CURVES: dict[str, dict[str, tuple[float, ...]]] = {
    # GIM_8108_8 — used for robstride_04. Spec peak 22 Nm; 0.85× = 18.7 Nm.
    "04": {
        "omega": (0.0,    7.9,   9.9,   10.5,  11.5,  12.0,  13.1,  14.1,
                  15.2,   15.7,  16.8,  17.8,  18.3,  18.8,  19.4,  19.9,
                  20.9,   21.5),
        "tau":   (18.70, 17.85, 17.00, 16.15, 14.45, 12.75, 11.90, 10.20,
                   8.50,  7.65,  6.375, 5.10,  4.25,  3.40,  2.55,  1.70,
                   0.85,  0.0),
    },
    # GIM_6010_8 — robstride_03 (shoulder yaw, hip yaw). Spec peak 11 Nm; 0.85× = 9.35 Nm.
    "03": {
        "omega": (0.0,    3.1,   4.7,   12.0,  14.1,  16.2,  18.3,  19.9,
                  21.5,   23.0,  24.6,  25.7,  27.2,  29.8),
        "tau":   ( 9.35,  8.925, 8.50,  8.245, 7.65,  6.80,  5.95,  5.10,
                   4.25,  3.40,  2.55,  1.70,  0.935, 0.0),
    },
    # Same motor as "03" — robstride_02 (ankle).
    "02": {
        "omega": (0.0,    3.1,   4.7,   12.0,  14.1,  16.2,  18.3,  19.9,
                  21.5,   23.0,  24.6,  25.7,  27.2,  29.8),
        "tau":   ( 9.35,  8.925, 8.50,  8.245, 7.65,  6.80,  5.95,  5.10,
                   4.25,  3.40,  2.55,  1.70,  0.935, 0.0),
    },
    # "00" wrist motor — TV curve unavailable, 5 Nm constant × 0.85 = 4.25 Nm.
    "00": {
        "omega": (0.0,  100.0),
        "tau":   (4.25,   4.25),
    },
}


def _build_tv_curve_arrays(motor_types: tuple[str, ...]) -> tuple[Array, Array]:
    """Build padded per-joint (omega, tau) curve arrays from motor type labels."""
    max_len = max(len(TV_CURVES[m]["omega"]) for m in motor_types)
    omegas = []
    taus = []
    for m in motor_types:
        omg = list(TV_CURVES[m]["omega"])
        tau = list(TV_CURVES[m]["tau"])
        # Pad with monotonically-increasing omega and tau=last (constant extrapolation).
        while len(omg) < max_len:
            omg.append(omg[-1] + 1.0)
            tau.append(tau[-1])
        omegas.append(omg)
        taus.append(tau)
    return jnp.array(omegas), jnp.array(taus)


class TVCurveMITActuators(TargetPositionMITActuators):
    """MIT-mode actuator with velocity-dependent torque limit (T-V curve).

    Real BLDC motors cannot deliver peak motoring torque at high speeds — back-EMF
    reduces available torque as the motor spins in the same direction the torque is
    applied. Braking torque (opposite direction to ω) is NOT limited by back-EMF;
    the motor regenerates instead and can apply nearly full peak torque to brake.

    Per-joint behavior (per-step):
      max_tau_motoring = interp(|qvel|, omega_curve, tau_curve)
      max_tau_braking  = ctrl_clip                         # constant (thermal/current)
      effective_limit  = where(sign(ctrl) == sign(qvel), max_tau_motoring, max_tau_braking)
      ctrl_clipped     = clip(ctrl, -effective_limit, +effective_limit)

    This direction-aware clipping lets the policy use strong braking torque (which
    real hardware can deliver) while still respecting the motoring-side T-V curve.

    Randomization: at every step, each joint's max_tau_motoring is scaled by a
    random factor in [1 - tv_curve_randomization, 1.0]. Real motors degrade when
    hot (torque dips), so we only ever scale DOWN, never up. Regularizes against
    over-reliance on exact peak torque.
    """

    def __init__(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata,
        default_targets: tuple[float, ...] = (),
        *,
        motor_types: tuple[str, ...],
        pos_action_noise: float = 0.0,
        pos_action_noise_type: ksim.actuators.NoiseType = "none",
        vel_action_noise: float = 0.0,
        vel_action_noise_type: ksim.actuators.NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: ksim.actuators.NoiseType = "none",
        ctrl_clip: list[float] | None = None,
        action_scale: float = 1.0,
        tv_curve_randomization: float = 0.15,
        freejoint_first: bool = True,
    ) -> None:
        super().__init__(
            physics_model=physics_model,
            metadata=metadata,
            default_targets=default_targets,
            pos_action_noise=pos_action_noise,
            pos_action_noise_type=pos_action_noise_type,
            vel_action_noise=vel_action_noise,
            vel_action_noise_type=vel_action_noise_type,
            torque_noise=torque_noise,
            torque_noise_type=torque_noise_type,
            ctrl_clip=ctrl_clip,
            action_scale=action_scale,
            freejoint_first=freejoint_first,
        )
        self.tv_omega_curves, self.tv_tau_curves = _build_tv_curve_arrays(motor_types)
        self.tv_curve_randomization = float(tv_curve_randomization)

    def _max_tau(self, qvel: Array, rng: PRNGKeyArray) -> Array:
        """Compute per-joint max allowed torque given current joint velocity.

        Uses linear interpolation on the per-joint T-V curve. Returns shape (num_joints,).
        """
        # |qvel| → max allowed tau by interpolating each joint's curve.
        speed = jnp.abs(qvel)
        max_tau = jax.vmap(jnp.interp)(speed, self.tv_omega_curves, self.tv_tau_curves)
        # Random scale in [1 - rand, 1.0] — motors only get weaker, never stronger.
        if self.tv_curve_randomization > 0.0:
            scale = jax.random.uniform(
                rng,
                shape=max_tau.shape,
                minval=1.0 - self.tv_curve_randomization,
                maxval=1.0,
            )
            max_tau = max_tau * scale
        return max_tau

    def get_ctrl(self, action: Array, physics_data: ksim.PhysicsData, rng: PRNGKeyArray) -> Array:
        pos_rng, vel_rng, tor_rng, tv_rng = jax.random.split(rng, 4)

        current_pos = physics_data.qpos[7:]
        current_vel = physics_data.qvel[6:]

        target_position = action[: len(current_pos)] * self.action_scale + self.default_targets
        target_velocity = action[len(current_pos) :] * self.action_scale
        target_position = self.add_noise(self.action_noise, self.action_noise_type, target_position, pos_rng)
        target_velocity = self.add_noise(self.vel_action_noise, self.vel_action_noise_type, target_velocity, vel_rng)

        pos_delta = target_position - current_pos
        vel_delta = target_velocity - current_vel
        ctrl = self.kps * pos_delta + self.kds * vel_delta
        ctrl = self.add_noise(self.torque_noise, self.torque_noise_type, ctrl, tor_rng)

        # Direction-aware T-V limit:
        #   motoring (sign(ctrl)==sign(qvel)): limit by interpolated T-V curve
        #   braking  (opposite signs):         no back-EMF limit, use full ctrl_clip
        max_tau_motoring = self._max_tau(current_vel, tv_rng)  # (num_joints,)
        is_motoring = ctrl * current_vel > 0.0  # both nonzero and same sign
        effective_limit = jnp.where(is_motoring, max_tau_motoring, self.ctrl_clip)
        ctrl = jnp.clip(ctrl, -effective_limit, effective_limit)
        # Hard safety clip on top — never exceed the constant ctrl_clip in either direction.
        ctrl = jnp.clip(ctrl, -self.ctrl_clip, self.ctrl_clip)
        return ctrl


class ScaledTorqueActuators(ksim.Actuators):
    """Direct torque control."""

    def __init__(
        self,
        default_targets: Array,
        action_scale: float = 0.5,
        noise: float = 0.0,
        noise_type: ksim.actuators.NoiseType = "none",
    ) -> None:
        super().__init__()

        self._action_scale = action_scale
        self.noise = noise
        self.noise_type = noise_type
        self.default_targets = default_targets

    def get_ctrl(self, action: Array, physics_data: ksim.PhysicsData, rng: PRNGKeyArray) -> Array:
        """Use the scaled action as the torque."""
        action = self.default_targets + action * self._action_scale
        return self.add_noise(self.noise, self.noise_type, action, rng)


@attrs.define(frozen=True)
class JointPositionObservation(ksim.Observation):
    default_targets: tuple[float, ...] = attrs.field()
    noise: float = attrs.field(default=0.0)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        qpos = state.physics_state.data.qpos[7:]  # (N,)
        diff = qpos - jnp.array(self.default_targets)
        return diff


@attrs.define(frozen=True)
class ProjectedGravityObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        gvec = xax.get_projected_gravity_vector_from_quat(state.physics_state.data.qpos[3:7])
        return gvec


@attrs.define(frozen=True)
class LocalProjectedGravityObservation(ksim.Observation):
    sensor_idx_range: tuple[int, int | None] = attrs.field()
    noise: float = attrs.field(default=0.0)
    sensor_name: str = attrs.field(default="base_link_quat")

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        quat = state.physics_state.data.sensordata[self.sensor_idx_range[0] : self.sensor_idx_range[1]].ravel()
        return xax.get_projected_gravity_vector_from_quat(quat)

    @classmethod
    def create(cls, physics_model: ksim.PhysicsModel, sensor_name: str, noise: float = 0.0) -> Self:
        if sensor_idx_range := get_sensor_data_idxs_by_name(physics_model)[sensor_name]:
            return cls(sensor_name=sensor_name, sensor_idx_range=sensor_idx_range, noise=noise)

        raise ValueError(f"Sensor {sensor_name} not found in physics model")

    def get_name(self) -> str:
        return f"{self.sensor_name}_{super().get_name()}"


@attrs.define(frozen=True)
class LastActionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.most_recent_action


@attrs.define(frozen=True)
class TrueHeightObservation(ksim.Observation):
    """Observation of the true height of the body."""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return jnp.atleast_1d(state.physics_state.data.qpos[2])


@attrs.define(frozen=True, kw_only=True)
class TimestepPhaseObservation(ksim.TimestepObservation):
    """Observation of the phase of the timestep."""

    ctrl_dt: float = attrs.field(default=0.02)
    stand_still_threshold: float = attrs.field(default=0.0)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        gait_freq = state.commands["gait_frequency_command"]
        timestep = super().observe(state, curriculum_level, rng)
        steps = timestep / self.ctrl_dt
        phase_dt = 2 * jnp.pi * gait_freq * self.ctrl_dt
        start_phase = jnp.array([0, jnp.pi])  # trotting gait
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        # Stand still case
        vel_cmd = state.commands["linear_velocity_command"]
        ang_vel_cmd = state.commands["angular_velocity_command"]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        phase = jnp.where(
            cmd_norm < self.stand_still_threshold,
            jnp.array([jnp.pi / 2, jnp.pi]),  # stand still position
            phase,
        )

        return jnp.array([jnp.cos(phase), jnp.sin(phase)]).flatten()


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()
    floor_threshold: float = attrs.field(default=0.0)

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        foot_left_site_name: str,
        foot_right_site_name: str,
        floor_threshold: float = 0.0,
    ) -> Self:
        foot_left_idx = get_site_data_idx_from_name(physics_model, foot_left_site_name)
        foot_right_idx = get_site_data_idx_from_name(physics_model, foot_right_site_name)
        return cls(
            foot_left=foot_left_idx,
            foot_right=foot_right_idx,
            floor_threshold=floor_threshold,
        )

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        foot_left_pos = state.physics_state.data.site_xpos[self.foot_left] + jnp.array([0.0, 0.0, self.floor_threshold])
        foot_right_pos = state.physics_state.data.site_xpos[self.foot_right] + jnp.array(
            [0.0, 0.0, self.floor_threshold]
        )
        return jnp.concatenate([foot_left_pos, foot_right_pos], axis=-1)


@attrs.define(frozen=True, kw_only=True)
class AppliedTorqueObservation(ksim.Observation):
    """Per-step applied joint torque, read from data.ctrl.

    ksim's built-in ActuatorForceObservation reads data.actuator_force which
    may not be reliably populated in MJX for all actuator types. Our custom
    actuators (TargetPositionMITActuators / TVCurveMITActuators) write the
    computed torque directly to data.ctrl, so reading data.ctrl gives the
    true applied torque per joint per step.
    """

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.ctrl


@attrs.define(frozen=True, kw_only=True)
class FeetEndpointsObservation(ksim.Observation):
    """World positions of heel and toe sites for each foot.

    Returns a flat array of shape (12,):
      [left_heel_xyz, left_toe_xyz, right_heel_xyz, right_toe_xyz]

    Z-indices: left_heel=2, left_toe=5, right_heel=8, right_toe=11.
    Used for multi-point foot clearance checking to prevent tilt exploits.
    """

    left_heel: int = attrs.field()
    left_toe: int = attrs.field()
    right_heel: int = attrs.field()
    right_toe: int = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        left_heel_site_name: str = "left_foot_heel",
        left_toe_site_name: str = "left_foot_toe",
        right_heel_site_name: str = "right_foot_heel",
        right_toe_site_name: str = "right_foot_toe",
    ) -> "FeetEndpointsObservation":
        return cls(
            left_heel=get_site_data_idx_from_name(physics_model, left_heel_site_name),
            left_toe=get_site_data_idx_from_name(physics_model, left_toe_site_name),
            right_heel=get_site_data_idx_from_name(physics_model, right_heel_site_name),
            right_toe=get_site_data_idx_from_name(physics_model, right_toe_site_name),
        )

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        lh = state.physics_state.data.site_xpos[self.left_heel]
        lt = state.physics_state.data.site_xpos[self.left_toe]
        rh = state.physics_state.data.site_xpos[self.right_heel]
        rt = state.physics_state.data.site_xpos[self.right_toe]
        return jnp.concatenate([lh, lt, rh, rt], axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetContactObservation(ksim.FeetContactObservation):
    """Observation of the feet contact."""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        feet_contact_12 = super().observe(state, curriculum_level, rng)
        return feet_contact_12.flatten()


@attrs.define(frozen=True, kw_only=True)
class GVecTermination(ksim.Termination):
    """Terminates the episode if the robot is facing down."""

    sensor_idx_range: tuple[int, int | None] = attrs.field()
    min_z: float = attrs.field(default=0.0)

    def __call__(self, state: ksim.PhysicsData, curriculum_level: Array) -> Array:
        start, end = self.sensor_idx_range
        return jnp.where(state.sensordata[start:end][-1] < self.min_z, -1, 0)

    @classmethod
    def create(cls, physics_model: ksim.PhysicsModel, sensor_name: str) -> Self:
        sensor_idx_range = get_sensor_data_idxs_by_name(physics_model)[sensor_name]
        return cls(sensor_idx_range=sensor_idx_range)


@attrs.define(frozen=True, kw_only=True)
class ResetDefaultJointPosition(ksim.Reset):
    """Resets the joint positions of the robot to random values."""

    default_targets: tuple[float, ...] = attrs.field()

    def __call__(self, data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> ksim.PhysicsData:
        qpos = data.qpos
        match type(data):
            case mujoco.MjData:
                qpos[:] = self.default_targets
            case mjx.Data:
                qpos = qpos.at[:].set(self.default_targets)
        return ksim.utils.mujoco.update_data_field(data, "qpos", qpos)


@attrs.define(frozen=True, kw_only=True)
class FarFromOriginTermination(ksim.Termination):
    """Terminates the episode if the robot is too far from the origin.

    This is treated as a positive termination.
    """

    max_dist: float = attrs.field()

    def __call__(self, state: ksim.PhysicsData, curriculum_level: Array) -> Array:
        return jnp.where(jnp.linalg.norm(state.qpos[..., :2], axis=-1) > self.max_dist, -1, 0)


@attrs.define(frozen=True)
class GaitFrequencyCommand(ksim.Command):
    """Command to set the gait frequency of the robot."""

    gait_freq_lower: float = attrs.field(default=1.2)
    gait_freq_upper: float = attrs.field(default=1.5)

    def initial_command(
        self,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """Returns (1,) array with gait frequency."""
        return jax.random.uniform(rng, (1,), minval=self.gait_freq_lower, maxval=self.gait_freq_upper)

    def __call__(
        self,
        prev_command: Array,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return prev_command


@attrs.define(frozen=True)
class LinearVelocityCommand(ksim.Command):
    """Command to move the robot in a straight line.

    By convention, X is forward and Y is left. The switching probability is the
    probability of resampling the command at each step. The zero probability is
    the probability of the command being zero - this can be used to turn off
    any command.
    """

    x_range: tuple[float, float] = attrs.field()
    y_range: tuple[float, float] = attrs.field()
    x_zero_prob: float = attrs.field(default=0.0)
    y_zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=1.0)
    vis_scale: float = attrs.field(default=0.05)

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_x, rng_y, rng_zero_x, rng_zero_y = jax.random.split(rng, 4)
        (xmin, xmax), (ymin, ymax) = self.x_range, self.y_range
        x = jax.random.uniform(rng_x, (1,), minval=xmin, maxval=xmax)
        y = jax.random.uniform(rng_y, (1,), minval=ymin, maxval=ymax)
        x_zero_mask = jax.random.bernoulli(rng_zero_x, self.x_zero_prob)
        y_zero_mask = jax.random.bernoulli(rng_zero_y, self.y_zero_prob)
        return jnp.concatenate(
            [
                jnp.where(x_zero_mask, 0.0, x),
                jnp.where(y_zero_mask, 0.0, y),
            ]
        )

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def get_markers(self) -> Collection[ksim.vis.Marker]:
        return []


@attrs.define(frozen=True)
class AngularVelocityCommand(ksim.Command):
    """Command to turn the robot."""

    scale: float = attrs.field()
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        """Returns (1,) array with angular velocity."""
        rng_a, rng_b = jax.random.split(rng)
        zero_mask = jax.random.bernoulli(rng_a, self.zero_prob)
        cmd = jax.random.uniform(rng_b, (1,), minval=-self.scale, maxval=self.scale)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)


# Per-arm-joint sampling ranges for ArmConstraintCommand.
# Conservative subset (~80%) of the MJCF joint ranges to avoid jamming poses
# right up against the joint limits. Order matches JOINT_TARGETS arm slots (0..9).
ARM_SAMPLE_RANGES: tuple[tuple[float, float], ...] = (
    # right arm — shoulder_pitch (-3.14 to 1.40), shoulder_roll (-1.66 to 0.35),
    #             shoulder_yaw (-1.66 to 1.66), elbow (0 to 2.48), wrist (-1.75 to 1.75)
    (-2.50,  1.10),
    (-1.30,  0.25),
    (-1.30,  1.30),
    ( 0.00,  2.00),
    (-1.40,  1.40),
    # left arm — mirror of right for pitch/roll/elbow, same for yaw/wrist
    (-1.10,  2.50),
    (-0.25,  1.30),
    (-1.30,  1.30),
    (-2.00,  0.00),
    (-1.40,  1.40),
)


@attrs.define(frozen=True)
class ArmConstraintCommand(ksim.Command):
    """Per-episode arm-pose constraint command.

    With probability `constraint_prob`, the episode is "constrained": the
    arms must hold a randomly-sampled target pose throughout the episode.
    Otherwise (`is_constrained` = 0), the arms are free and the constraint
    reward contributes nothing.

    This trains the policy to balance using legs/torso when the arms are
    locked (e.g. carrying an object), instead of relying on arm swings.

    Command layout (11 floats):
        [0]       is_constrained ∈ {0, 1}
        [1..11]   target_arm_joint_pose (10 joints: right arm 5, left arm 5)

    Switch probability is 0 — the command is fixed for the whole episode so
    the policy experiences a consistent constraint signal.
    """

    constraint_prob: float = attrs.field(default=0.3)
    sample_ranges: tuple[tuple[float, float], ...] = attrs.field(default=ARM_SAMPLE_RANGES)
    # Curriculum-gating: effective constraint prob = constraint_prob * curriculum_level.
    use_curriculum: bool = attrs.field(default=True)
    # Delayed activation: the constraint is scheduled at episode start but the
    # arm override only fires after t > activation_delay. Lets the policy
    # establish steady walking BEFORE the arms suddenly swing to their
    # target — so the legs experience a clear "arm motion" perturbation during
    # active walking, which is what we want them to learn to counteract.
    # Set to 0.0 to disable (constraint active from t=0 if sampled).
    activation_delay: float = attrs.field(default=2.0)

    # Slot 0 encoding:
    #   0.0  = no constraint this episode (no override, ever)
    #  -1.0  = constraint scheduled, currently in delay (no override yet)
    #  +1.0  = constraint currently active (actor applies arm override)
    # The actor's gate `is_constrained > 0.5` treats {0.0, -1.0} as off and +1.0 as on.

    def initial_command(
        self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_flag, rng_pose = jax.random.split(rng)
        effective_prob = jnp.where(
            self.use_curriculum,
            self.constraint_prob * curriculum_level,
            self.constraint_prob,
        )
        will_constrain = jax.random.bernoulli(rng_flag, effective_prob).astype(jnp.float32)
        # Encode: -1 if will activate (currently in delay), 0 if no constraint this episode.
        marker_init = jnp.where(will_constrain > 0.5, -1.0, 0.0)
        mins = jnp.array([r[0] for r in self.sample_ranges])
        maxs = jnp.array([r[1] for r in self.sample_ranges])
        u = jax.random.uniform(rng_pose, shape=(len(self.sample_ranges),))
        target_pose = mins + u * (maxs - mins)
        return jnp.concatenate([marker_init[None], target_pose], axis=-1)

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        # Promote scheduled (-1) → active (+1) once t > activation_delay.
        # Other states stay as-is (no constraint → stays 0; active → stays +1).
        marker = prev_command[0]
        target_pose = prev_command[1:11]
        is_scheduled = marker < -0.5
        delay_elapsed = physics_data.time > self.activation_delay
        promote = is_scheduled & delay_elapsed
        new_marker = jnp.where(promote, 1.0, marker)
        return jnp.concatenate([new_marker[None], target_pose], axis=-1)


@attrs.define(frozen=True, kw_only=True)
class XYPushEvent(ksim.Event):
    """Randomly push the robot after some interval."""

    interval_range: tuple[float, float] = attrs.field()
    force_range: tuple[float, float] = attrs.field()
    curriculum_scale: float = attrs.field(default=1.0)

    def __call__(
        self,
        model: ksim.PhysicsModel,
        data: ksim.PhysicsData,
        event_state: Array,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PhysicsData, Array]:
        # Decrement by physics timestep.
        dt = jnp.float32(model.opt.timestep)
        time_remaining = event_state - dt

        # Update the data if the time remaining is less than 0.
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_random_force(data, curriculum_level, rng),
            lambda: (data, time_remaining),
        )

        return updated_data, time_remaining

    def _apply_random_force(
        self, data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> tuple[ksim.PhysicsData, Array]:
        push_theta = jax.random.uniform(rng, maxval=2 * jnp.pi)
        push_magnitude = (
            jax.random.uniform(
                rng,
                minval=self.force_range[0],
                maxval=self.force_range[1],
            )
            * curriculum_level
            * self.curriculum_scale
        )
        push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])
        random_forces = push * push_magnitude + data.qvel[:2]
        new_qvel = slice_update(data, "qvel", slice(0, 2), random_forces)
        updated_data = update_data_field(data, "qvel", new_qvel)

        # Chooses a new remaining interval.
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(rng, (), minval=minval, maxval=maxval)

        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)


@attrs.define(frozen=True, kw_only=True)
class TorquePushEvent(ksim.Event):
    """Randomly push the robot with torque (angular velocity) after some interval."""

    interval_range: tuple[float, float] = attrs.field()
    ang_vel_range: tuple[float, float] = attrs.field()  # Min/max push angular velocity per axis
    curriculum_scale: float = attrs.field(default=1.0)

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

        # Update the data if the time remaining is less than 0.
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_random_angular_velocity_push(data, curriculum_level, rng),
            lambda: (data, time_remaining),
        )

        return updated_data, time_remaining

    def _apply_random_angular_velocity_push(
        self, data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> tuple[ksim.PhysicsData, Array]:
        """Applies a random angular velocity push to the root body."""
        rng_push, rng_interval = jax.random.split(rng)

        # Sample angular velocity push components
        min_ang_vel, max_ang_vel = self.ang_vel_range
        push_ang_vel = jax.random.uniform(
            rng_push,
            shape=(3,),  # Angular velocity is 3D (wx, wy, wz)
            minval=min_ang_vel,
            maxval=max_ang_vel,
        )
        scaled_push_ang_vel = push_ang_vel * curriculum_level * self.curriculum_scale

        # Apply the push to angular velocity (qvel indices 3:6 for free joint)
        ang_vel_indices = slice(3, 6)

        # Add the push to the current angular velocity
        current_ang_vel = data.qvel[ang_vel_indices]
        new_ang_vel_val = current_ang_vel + scaled_push_ang_vel
        new_qvel = slice_update(data, "qvel", ang_vel_indices, new_ang_vel_val)
        updated_data = update_data_field(data, "qvel", new_qvel)

        minval_interval, maxval_interval = self.interval_range
        time_remaining = jax.random.uniform(rng_interval, (), minval=minval_interval, maxval=maxval_interval)

        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)


@attrs.define(frozen=True, kw_only=True)
class ReferenceQposObservation(ksim.Observation):
    """Observation for the reference joint positions."""

    reference_motion_data: MotionReferenceData
    speed: float = attrs.field(default=1.0)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        physics_state = state.physics_state
        effective_time = physics_state.data.time * self.speed
        reference_qpos_at_time = self.reference_motion_data.get_qpos_at_time(effective_time)
        return reference_qpos_at_time[..., 7:]


@attrs.define(frozen=True, kw_only=True)
class ReferenceLocalXposObservation(ksim.Observation):
    """Observation for the reference local cartesian positions of tracked bodies."""

    reference_motion_data: MotionReferenceData
    tracked_body_ids: tuple[int, ...]

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        physics_state = state.physics_state
        target_pos_dict = self.reference_motion_data.get_cartesian_pose_at_time(physics_state.data.time)
        target_pos_list = [target_pos_dict[body_id] for body_id in self.tracked_body_ids]
        return jnp.concatenate(target_pos_list, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class TrackedLocalXposObservation(ksim.Observation):
    """Observation for the current local cartesian positions of tracked bodies."""

    tracked_body_ids: tuple[int, ...]
    mj_base_id: int

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        physics_state = state.physics_state
        tracked_positions_list: list[Array] = []
        for body_id in self.tracked_body_ids:
            body_pos = get_local_xpos(physics_state.data.xpos, body_id, self.mj_base_id)
            tracked_positions_list.append(jnp.array(body_pos))
        return jnp.concatenate(tracked_positions_list, axis=-1)
