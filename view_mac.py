"""Standalone macOS-compatible policy viewer using mujoco.viewer.launch_passive.

Usage:
    mjpython view_mac.py --ckpt /path/to/ckpt.bin

Uses mujoco.viewer.launch_passive which is designed for mjpython on macOS,
bypassing ksim's GlfwMujocoViewer which crashes due to macOS thread restrictions.
The full policy (RNN + observations + rewards) runs via ksim's step_engine.
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import itertools
import logging
import time
from pathlib import Path

import jax
import mujoco
import mujoco.viewer
import numpy as np

logger = logging.getLogger(__name__)


def run_viewer_with_policy(ckpt_path: str) -> None:
    """Mirror of ksim's run_model_viewer but using mujoco.viewer.launch_passive."""
    # Import after env vars are set
    from ksim_kbot.walking.walking_joystick_rnn import KbotWalkingJoystickRNNTask

    from ksim_kbot.walking.walking_joystick_rnn import KbotWalkingJoystickRNNTaskConfig

    # Build config — run_mode=view disables training-specific setup
    cfg = KbotWalkingJoystickRNNTaskConfig(
        run_mode="view",
        load_from_ckpt_path=ckpt_path,
        disable_multiprocessing=True,
        viewer_argmax_action=True,
        # Match training config exactly
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
        domain_randomize=False,  # disable randomization for clean viewing
        gait_freq_lower=1.25,
        gait_freq_upper=1.5,
        reward_clip_min=0.0,
        reward_clip_max=1000.0,
    )
    task = KbotWalkingJoystickRNNTask(cfg)

    with task, jax.disable_jit():
        rng = task.prng_key()
        task.set_loggers()

        # Load MuJoCo model (with scene/ground)
        mj_model = task.get_mujoco_model()
        mj_model = task.set_mujoco_model_opts(mj_model)
        metadata = task.get_mujoco_model_metadata(mj_model)
        randomizers = task.get_physics_randomizers(mj_model)

        # Load policy checkpoint
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

        # Set up ksim constants (commands, observations, rewards config)
        constants = task._get_constants(
            mj_model=mj_model,
            physics_model=mj_model,
            model_statics=model_statics,
            argmax_action=cfg.viewer_argmax_action,
        )

        # Set up initial environment state (physics + RNN hidden state)
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

        # Create a CPU-side MjData for the viewer (policy runs on JAX/MJX side)
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = np.array(env_states.physics_state.data.qpos)
        mj_data.qvel[:] = np.array(env_states.physics_state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)

        print("Launching viewer — policy running from checkpoint")
        print("Close window or Ctrl-C to quit")

        with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
            v.cam.distance = 3.5
            v.cam.elevation = -15
            v.cam.azimuth = 135

            for _ in itertools.count():
                if not v.is_running():
                    break

                step_start = time.time()

                # Run one step: policy inference + physics
                transition, env_states = task.step_engine(
                    constants=constants,
                    env_states=env_states,
                    shared_state=shared_state,
                )

                # Copy JAX physics state → CPU viewer data
                mj_data.qpos[:] = np.array(env_states.physics_state.data.qpos)
                mj_data.qvel[:] = np.array(env_states.physics_state.data.qvel)
                mujoco.mj_forward(mj_model, mj_data)
                v.sync()

                # Pace to ctrl_dt
                elapsed = time.time() - step_start
                remaining = cfg.ctrl_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)


def main():
    parser = argparse.ArgumentParser(description="macOS policy viewer for K-Bot")
    parser.add_argument(
        "--ckpt",
        default=str(Path.home() / "kbot/ckpt.bin"),
        help="Path to checkpoint .bin file",
    )
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found at {ckpt}")
        print("Run: scp faisal@192.168.68.130:/path/to/ckpt.bin ~/kbot/ckpt.bin")
        return

    print(f"Loading checkpoint: {ckpt}")
    run_viewer_with_policy(str(ckpt))


if __name__ == "__main__":
    main()
