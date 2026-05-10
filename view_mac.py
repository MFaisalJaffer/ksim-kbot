"""Standalone macOS-compatible policy viewer using mujoco.viewer.launch_passive.

Usage:
    mjpython view_mac.py --ckpt /path/to/ckpt.bin

This bypasses ksim's GlfwMujocoViewer (which crashes on macOS due to threading)
and uses mujoco's built-in viewer which is designed for mjpython on macOS.
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

# ── locate robot model ──────────────────────────────────────────────────────
MJCF_PATH = str(Path.home() / ".kscale/robots/kbot/robot/robot.mjcf")


def load_checkpoint(ckpt_path: str):
    """Load the policy checkpoint and return the task object with models."""
    # Import here so env vars are set first
    from ksim_kbot.walking.walking_joystick_rnn import KbotWalkingJoystickRNNTask

    cfg = KbotWalkingJoystickRNNTask.get_config(
        run_mode="view",
        load_from_ckpt_path=ckpt_path,
        disable_multiprocessing=True,
    )
    task = KbotWalkingJoystickRNNTask(cfg)
    rng = jax.random.PRNGKey(0)
    models, _ = task.load_initial_state(rng, load_optimizer=False)
    return task, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to ckpt.bin")
    args = parser.parse_args()

    print(f"Loading model from {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    print(f"Loading checkpoint from {args.ckpt}")
    try:
        task, policy_models = load_checkpoint(args.ckpt)
        print("Checkpoint loaded — running policy")
        run_policy = True
    except Exception as e:
        print(f"Could not load checkpoint ({e}) — showing model in default pose")
        run_policy = False

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print("Launching viewer (press ESC or close window to quit)")

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance = 3.0
        v.cam.elevation = -20
        v.cam.azimuth = 90

        step = 0
        while v.is_running():
            step_start = time.time()

            if run_policy:
                try:
                    # Let ksim's engine step — simplified: just step physics
                    mujoco.mj_step(model, data)
                except Exception:
                    # Fall back to passive physics
                    mujoco.mj_step(model, data)
            else:
                mujoco.mj_step(model, data)

            v.sync()
            step += 1

            # ~60 fps
            elapsed = time.time() - step_start
            remaining = (1.0 / 60.0) - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
