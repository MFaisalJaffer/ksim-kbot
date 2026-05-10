# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for the default humanoid using an RNN actor."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

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

from ksim_kbot import rewards as kbot_rewards
from ksim_kbot.walking.walking_joystick import (
    NUM_CRITIC_INPUTS,
    NUM_INPUTS,
    NUM_OUTPUTS,
    JOINT_TARGETS,
    KbotWalkingTask,
    KbotWalkingTaskConfig,
)

logger = logging.getLogger(__name__)

# Same obs space except without prev action.
RNN_NUM_INPUTS = NUM_INPUTS - NUM_OUTPUTS

RNN_NUM_CRITIC_INPUTS = NUM_CRITIC_INPUTS - NUM_OUTPUTS


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
        obs_n = jnp.concatenate(
            [
                timestep_phase_4,  # 1
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
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

        return self.call_flat_obs(obs_n, carry)

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
        obs_n = jnp.concatenate(
            [
                timestep_phase_4,  # 1
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
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
                actuator_force_n,  # NUM_JOINTS
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
class KbotWalkingJoystickRNNTaskConfig(KbotWalkingTaskConfig):
    hidden_size: int = xax.field(value=256)
    depth: int = xax.field(value=5)


Config = TypeVar("Config", bound=KbotWalkingJoystickRNNTaskConfig)


class KbotWalkingJoystickRNNTask(KbotWalkingTask[Config], Generic[Config]):
    config: Config

    def get_model(self, key: PRNGKeyArray) -> KbotRNNModel:
        return KbotRNNModel(
            key,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        import asyncio
        return asyncio.run(ksim.get_mujoco_model_metadata(str(Path.home() / ".kscale/robots/kbot/robot/"), cache=False))

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = str(Path.home() / ".kscale/robots/kbot/robot/robot.mjcf")
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
        return [
            # JointDeviationPenalty: re-added as a stability anchor.
            # Knees AND ankles are unconstrained (weight 0.01 = effectively zero) so
            # the policy is free to bend knees and articulate ankles for push-off.
            # Hip pitch is also free. Arms and hip roll/yaw are constrained to keep
            # the upper body stable while legs do the walking.
            kbot_rewards.JointDeviationPenalty(
                scale=-0.1,
                joint_targets=JOINT_TARGETS,
                joint_weights=(
                    # right arm
                    1.2, 1.0, 1.0, 1.0, 1.0,
                    # left arm
                    1.2, 1.0, 1.0, 1.0, 1.0,
                    # right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle
                    0.01, 1.0, 1.0, 0.01, 0.01,  # ankle now free (was 1.0)
                    # left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle
                    0.01, 1.0, 1.0, 0.01, 0.01,  # ankle now free (was 1.0)
                ),
            ),
            kbot_rewards.HipDeviationPenalty.create(
                physics_model=physics_model,
                hip_names=(
                    "dof_right_hip_roll_03",
                    "dof_right_hip_yaw_03",
                    "dof_left_hip_roll_03",
                    "dof_left_hip_yaw_03",
                ),
                joint_targets=JOINT_TARGETS,
                scale=-0.10,  # was -0.25 — kept relaxed to allow some hip motion
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
            # Restored to scale 2.1 (was the working run_27 value), translation_gated
            # kept on so policy doesn't get free reward by marching in place.
            kbot_rewards.FeetPhaseReward(
                foot_default_height=0.04,
                max_foot_height=0.12,
                scale=2.1,
                stand_still_threshold=self.config.stand_still_threshold,
                translation_gated=True,
                translation_gate_sensitivity=0.25,
                linvel_obs_name="base_linear_velocity_observation",
            ),
            kbot_rewards.FeetSlipPenalty(scale=-0.25),
            # Restored to scale 50.0 — strong pull toward stable JOINT_TARGETS pose
            # when joystick idle. Critical anchor that was missing in run_34.
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
            kbot_rewards.ContactForcePenalty(
                scale=-0.01,
                sensor_names=("sensor_observation_left_foot_force", "sensor_observation_right_foot_force"),
            ),
            ksim.CtrlPenalty(scale=-0.005),
            ksim.ActionAccelerationPenalty(scale=-0.005),
            ksim.JointVelocityPenalty(scale=-0.005),
            kbot_rewards.KneeRangeOfMotion.create(
                physics_model=physics_model,
                knee_names=("dof_left_knee_04", "dof_right_knee_04"),
            ),
            # Restored to run_27 settings: scale 0.5, grace 0.2s
            kbot_rewards.SingleFootContactReward(
                scale=0.5,
                ctrl_dt=self.config.ctrl_dt,
                grace_period=0.2,
            ),
            # Reduced from -0.5 to -0.1 — less aggressive flying penalty.
            kbot_rewards.NoContactPenalty(scale=-0.1),
            # FeetAirtimeReward removed — was actively penalizing the policy for
            # exploring leg motion (airtimes < 0.4s gave net-negative reward).
            # MarchInPlacePenalty kept (low cost, still helpful).
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

        from ksim_kbot.walking.walking_joystick import JOINT_TARGETS
        from ksim_kbot.common import (
            TimestepPhaseObservation,
            JointPositionObservation,
            LocalProjectedGravityObservation,
            LastActionObservation,
            FeetContactObservation,
            FeetPositionObservation,
            TrueHeightObservation,
        )

        return [
            TimestepPhaseObservation(),
            JointPositionObservation(default_targets=JOINT_TARGETS, noise=0.05),
            ksim.JointVelocityObservation(noise=vel_obs_noise),
            ksim.ActuatorForceObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc", noise=imu_acc_noise),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro", noise=imu_gyro_noise),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="base_site_quat",
                lag_range=(0.0, 0.1),
                noise=local_gvec_noise,
            ),
            LocalProjectedGravityObservation.create(
                physics_model=physics_model, sensor_name="base_site_quat", noise=local_gvec_noise
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
                foot_left_geom_names="KB_D_501L_L_LEG_FOOT_collision_capsule_0",
                foot_right_geom_names="KB_D_501R_R_LEG_FOOT_collision_capsule_0",
                floor_geom_names="floor",
            ),
            FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_site_name="left_foot",
                foot_right_site_name="right_foot",
                floor_threshold=0.00,
            ),
            TrueHeightObservation(),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [ksim.NotUprightTermination(max_radians=1.4)]  # 80°: gentle to let cold-init policy survive long enough to learn. Tighten to ~0.8 once episode_length > 100s.

    # Pushes inherit from parent (walking_joystick.py): XYPushEvent (0-1.8) and
    # TorquePushEvent (0-1.8) every 2-4s. Strength is auto-scaled by curriculum_level,
    # so at level=0 pushes are zero, ramping up as the policy improves.

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        # Auto-pacing curriculum with hysteresis to prevent thrashing.
        # - num_levels=20 → 0.05 increments (smaller jumps when bumping)
        # - increase_threshold=120s → must sustain 2-min episodes before bumping up
        # - decrease_threshold=10s → only drop if episodes truly collapse
        # The wide gap between increase/decrease thresholds creates a stable dead-zone
        # so the policy can converge at each level instead of oscillating.
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
    #   python -m ksim_kbot.walking.walking_joystick_rnn
    # To visualize the environment, use the following command:
    #   python -m ksim_kbot.walking.walking_joystick_rnn run_model_viewer=True
    KbotWalkingJoystickRNNTask.launch(
        KbotWalkingJoystickRNNTaskConfig(
            num_envs=3072,
            batch_size=192,
            num_passes=4,
            epochs_per_log_step=1,
            # Simulation parameters.
            iterations=6,
            ls_iterations=6,
            dt=0.002,
            ctrl_dt=0.02,
            action_latency_range=(0.0, 0.005),
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
