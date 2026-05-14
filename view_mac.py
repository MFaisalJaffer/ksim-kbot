"""Standalone macOS-compatible policy viewer with keyboard joystick + arm control.

Usage:
    mjpython view_mac.py --ckpt /path/to/ckpt.bin

Joystick controls:
    W / S       — forward / backward
    A / D       — turn left / right
    Q / E       — strafe left / right
    SPACE       — stop (zero all velocity commands)
    R           — reset episode
    ESC         — quit

Arm controls (cycle through preset poses while the policy walks):
    T           — toggle arm constraint on / off
    1           — preset 1: neutral default (right=+1.4 elbow, left=-1.4 elbow)
    2           — preset 2: both arms forward, elbows bent (carry box)
    3           — preset 3: both arms down at sides
    4           — preset 4: both arms raised overhead
    5           — preset 5: right hand up (waving)
    6           — preset 6: arms at chest (holding tray)

Status line shows: velocity command, gait freq, arm constraint flag, current pose preset.

Uses mujoco.viewer.launch_passive which is designed for mjpython on macOS.
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import itertools
import time
from pathlib import Path

import attrs
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from jaxtyping import Array, PRNGKeyArray

# ── Arm pose presets ─────────────────────────────────────────────────────────
# Each entry is a 10-vector: right arm (shoulder_pitch, shoulder_roll, shoulder_yaw,
# elbow, wrist) then left arm in the same order.
ARM_PRESETS = {
    1: ("neutral default",       ( 0.0, 0.0, 0.0,  1.4, 0.0,    0.0, 0.0, 0.0, -1.4, 0.0)),
    2: ("carry box (arms fwd)",  ( 1.2, 0.3, 0.0,  1.8, 0.0,    1.2,-0.3, 0.0, -1.8, 0.0)),
    3: ("arms down at sides",    ( 0.0, 0.0, 0.0,  0.2, 0.0,    0.0, 0.0, 0.0, -0.2, 0.0)),
    4: ("arms overhead",         (-2.0, 0.0, 0.0,  0.4, 0.0,   -2.0, 0.0, 0.0, -0.4, 0.0)),
    5: ("right hand up (wave)",  (-1.8, 0.3, 0.0,  1.2, 0.0,    0.0, 0.0, 0.0, -1.4, 0.0)),
    6: ("arms at chest (tray)",  ( 0.6, 0.2, 0.0,  2.0, 0.0,    0.6,-0.2, 0.0, -2.0, 0.0)),
}

# ── Global keyboard command state ────────────────────────────────────────────
# Updated by key callback; read by KeyboardCommand objects inside the step loop.
_CMD: dict = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
    "reset": False,
    # Arm constraint state
    "is_constrained": 0.0,
    "arm_preset": 1,
    "arm_target": ARM_PRESETS[1][1],
}

# Step sizes per keypress
VX_STEP = 0.1   # m/s
VY_STEP = 0.1   # m/s
WZ_STEP = 0.1   # rad/s

# GLFW key codes
KEY_W, KEY_S, KEY_A, KEY_D = 87, 83, 65, 68
KEY_Q, KEY_E = 81, 69
KEY_SPACE = 32
KEY_R = 82
KEY_T = 84
KEY_ESC = 256
KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6 = 49, 50, 51, 52, 53, 54

GLFW_PRESS   = 1
GLFW_REPEAT  = 2


def _print_status() -> None:
    vx, vy, wz = _CMD["vx"], _CMD["vy"], _CMD["wz"]
    cflag = "ON " if _CMD["is_constrained"] else "off"
    pset = _CMD["arm_preset"]
    name = ARM_PRESETS[pset][0]
    print(
        f"\r  cmd: vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}   "
        f"arm[{cflag}] preset {pset}={name:<24s}",
        end="", flush=True,
    )


def _set_preset(preset_idx: int) -> None:
    if preset_idx not in ARM_PRESETS:
        return
    _CMD["arm_preset"] = preset_idx
    _CMD["arm_target"] = ARM_PRESETS[preset_idx][1]


def key_callback(keycode: int, scancode: int = 0, action: int = 1, mods: int = 0) -> None:
    if action not in (GLFW_PRESS, GLFW_REPEAT):
        return
    if keycode == KEY_W:
        _CMD["vx"] = min(_CMD["vx"] + VX_STEP, 0.7)
    elif keycode == KEY_S:
        _CMD["vx"] = max(_CMD["vx"] - VX_STEP, -0.3)
    elif keycode == KEY_Q:
        _CMD["vy"] = min(_CMD["vy"] + VY_STEP, 0.3)
    elif keycode == KEY_E:
        _CMD["vy"] = max(_CMD["vy"] - VY_STEP, -0.3)
    elif keycode == KEY_A:
        _CMD["wz"] = min(_CMD["wz"] + WZ_STEP, 0.5)
    elif keycode == KEY_D:
        _CMD["wz"] = max(_CMD["wz"] - WZ_STEP, -0.5)
    elif keycode == KEY_SPACE:
        _CMD["vx"] = _CMD["vy"] = _CMD["wz"] = 0.0
    elif keycode == KEY_R:
        _CMD["vx"] = _CMD["vy"] = _CMD["wz"] = 0.0
        _CMD["reset"] = True
    elif keycode == KEY_T:
        # Toggle arm constraint
        _CMD["is_constrained"] = 1.0 - _CMD["is_constrained"]
    elif keycode == KEY_1:
        _set_preset(1)
    elif keycode == KEY_2:
        _set_preset(2)
    elif keycode == KEY_3:
        _set_preset(3)
    elif keycode == KEY_4:
        _set_preset(4)
    elif keycode == KEY_5:
        _set_preset(5)
    elif keycode == KEY_6:
        _set_preset(6)

    _print_status()


# ── Keyboard-controlled command classes ─────────────────────────────────────

import ksim  # noqa: E402  (import after env vars)


@attrs.define(frozen=True, kw_only=True)
class KeyboardLinearVelocityCommand(ksim.Command):
    """Linear velocity command driven by keyboard state."""

    def get_name(self) -> str:
        return "linear_velocity_command"

    def initial_command(
        self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        return jnp.array([_CMD["vx"], _CMD["vy"]])

    def __call__(
        self,
        prev_command: Array,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return jnp.array([_CMD["vx"], _CMD["vy"]])


@attrs.define(frozen=True, kw_only=True)
class KeyboardAngularVelocityCommand(ksim.Command):
    """Angular velocity command driven by keyboard state."""

    def get_name(self) -> str:
        return "angular_velocity_command"

    def initial_command(
        self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        return jnp.array([_CMD["wz"]])

    def __call__(
        self,
        prev_command: Array,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return jnp.array([_CMD["wz"]])


@attrs.define(frozen=True, kw_only=True)
class KeyboardArmConstraintCommand(ksim.Command):
    """Arm constraint command driven by keyboard state.

    Reads `is_constrained` flag and the current `arm_target` 10-vector from the
    global _CMD dict and returns the 11-dim command the policy expects.
    """

    def get_name(self) -> str:
        return "arm_constraint_command"

    def _build_cmd(self) -> Array:
        return jnp.concatenate(
            [jnp.array([_CMD["is_constrained"]]), jnp.array(_CMD["arm_target"])]
        )

    def initial_command(
        self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        return self._build_cmd()

    def __call__(
        self,
        prev_command: Array,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return self._build_cmd()


# ── Task subclass with keyboard commands ─────────────────────────────────────

def run_viewer_with_policy(ckpt_path: str) -> None:
    from ksim_kbot.walking.walking_joystick_rnn import (
        KbotWalkingJoystickRNNTask,
        KbotWalkingJoystickRNNTaskConfig,
    )
    from ksim_kbot.common import GaitFrequencyCommand

    class KeyboardTask(KbotWalkingJoystickRNNTask):
        def get_commands(self, physics_model):
            return [
                KeyboardLinearVelocityCommand(),
                KeyboardAngularVelocityCommand(),
                GaitFrequencyCommand(
                    gait_freq_lower=self.config.gait_freq_lower,
                    gait_freq_upper=self.config.gait_freq_upper,
                ),
                KeyboardArmConstraintCommand(),
            ]

    cfg = KbotWalkingJoystickRNNTaskConfig(
        run_mode="view",
        load_from_ckpt_path=ckpt_path,
        disable_multiprocessing=True,
        viewer_argmax_action=True,
        num_envs=1,
        batch_size=1,
        num_passes=4,
        epochs_per_log_step=1,
        iterations=6,
        ls_iterations=6,
        dt=0.002,
        ctrl_dt=0.02,
        action_latency_range=(0.0, 0.005),
        rollout_length_seconds=5.0,
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
        domain_randomize=False,
        gait_freq_lower=1.25,
        gait_freq_upper=1.5,
        reward_clip_min=0.0,
        reward_clip_max=1000.0,
    )
    task = KeyboardTask(cfg)

    with task, jax.disable_jit():
        rng = task.prng_key()
        task.set_loggers()

        mj_model = task.get_mujoco_model()
        mj_model = task.set_mujoco_model_opts(mj_model)
        task.get_mujoco_model_metadata(mj_model)
        randomizers = task.get_physics_randomizers(mj_model)

        rng, model_rng = jax.random.split(rng)
        models, _ = task.load_initial_state(model_rng, load_optimizer=False)

        import equinox as eqx
        model_arrs, model_statics = (
            tuple(ms)
            for ms in zip(
                *(eqx.partition(m, task.model_partition_fn) for m in models),
                strict=True,
            )
        )

        def make_state(rng):
            constants = task._get_constants(
                mj_model=mj_model,
                physics_model=mj_model,
                model_statics=model_statics,
                argmax_action=cfg.viewer_argmax_action,
            )
            env_states = task._get_env_state(
                rng=rng,
                rollout_constants=constants,
                mj_model=mj_model,
                physics_model=mj_model,
                randomizers=randomizers,
            )
            shared_state = task._get_shared_state(
                mj_model=mj_model,
                physics_model=mj_model,
                model_arrs=model_arrs,
            )
            return constants, env_states, shared_state

        rng, init_rng = jax.random.split(rng)
        constants, env_states, shared_state = make_state(init_rng)

        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = np.array(env_states.physics_state.data.qpos)
        mj_data.qvel[:] = np.array(env_states.physics_state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)

        print("\nLaunching viewer with keyboard control:")
        print("  Joystick:  W/S = fwd/back   A/D = turn   Q/E = strafe   SPACE = stop   R = reset")
        print("  Arms:      T = toggle constraint   1-6 = preset pose")
        print()
        print("  Arm presets:")
        for i, (name, _) in ARM_PRESETS.items():
            print(f"    {i} — {name}")
        print()

        with mujoco.viewer.launch_passive(
            mj_model, mj_data, key_callback=key_callback
        ) as v:
            v.cam.distance = 3.5
            v.cam.elevation = -15
            v.cam.azimuth = 135

            for _ in itertools.count():
                if not v.is_running():
                    break

                # Handle episode reset
                if _CMD["reset"]:
                    _CMD["reset"] = False
                    rng, reset_rng = jax.random.split(rng)
                    constants, env_states, shared_state = make_state(reset_rng)
                    print("\n  [episode reset]")

                step_start = time.time()

                # Propagate viewer push forces into the JAX physics state
                env_states.physics_state.data.xfrc_applied[:] = mj_data.xfrc_applied
                mj_data.xfrc_applied[:] = 0  # clear so forces don't accumulate

                transition, env_states = task.step_engine(
                    constants=constants,
                    env_states=env_states,
                    shared_state=shared_state,
                )

                mj_data.qpos[:] = np.array(env_states.physics_state.data.qpos)
                mj_data.qvel[:] = np.array(env_states.physics_state.data.qvel)
                mujoco.mj_forward(mj_model, mj_data)
                v.sync()

                elapsed = time.time() - step_start
                remaining = cfg.ctrl_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)


def main():
    parser = argparse.ArgumentParser(description="macOS K-Bot policy viewer")
    parser.add_argument(
        "--ckpt",
        default=str(Path.home() / "kbot/ckpt.bin"),
        help="Path to checkpoint .bin file",
    )
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        return

    print(f"Loading checkpoint: {ckpt}")
    run_viewer_with_policy(str(ckpt))


if __name__ == "__main__":
    main()
