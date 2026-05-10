"""Standalone macOS-compatible policy viewer using mujoco.viewer.launch_passive.

Usage:
    mjpython view_mac.py --ckpt /path/to/ckpt.bin [--no-policy]

Uses mujoco.viewer.launch_passive which is designed for mjpython on macOS,
bypassing ksim's GlfwMujocoViewer which crashes due to macOS thread restrictions.
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ── robot config ─────────────────────────────────────────────────────────────
MJCF_PATH = str(Path.home() / ".kscale/robots/kbot/robot/robot.mjcf")

# Default joint targets (from walking_joystick.py)
JOINT_TARGETS = np.array([
    # right arm
    0.0, 0.0, 0.0, 1.4, 0.0,
    # left arm
    0.0, 0.0, 0.0, -1.4, 0.0,
    # right leg
    -0.23, 0.0, 0.0, -0.873, 0.195,
    # left leg
    0.23, 0.0, 0.0, 0.873, -0.195,
])

# PD gains (approximate — tune if robot oscillates)
KP = 80.0
KD = 4.0


def pd_control(model, data, targets):
    """Apply PD control to hold joints at targets."""
    qpos = data.qpos[7:]          # skip freejoint (pos + quat)
    qvel = data.qvel[6:]          # skip freejoint vel
    n = min(len(targets), len(qpos), model.nu)
    data.ctrl[:n] = KP * (targets[:n] - qpos[:n]) - KD * qvel[:n]
    # Clip to actuator limits
    ctrl_range = model.actuator_ctrlrange
    if ctrl_range is not None and len(ctrl_range) >= n:
        data.ctrl[:n] = np.clip(
            data.ctrl[:n], ctrl_range[:n, 0], ctrl_range[:n, 1]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Path to ckpt.bin")
    parser.add_argument("--no-policy", action="store_true",
                        help="Skip policy — just hold default pose")
    args = parser.parse_args()

    print(f"Loading model from {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    # Reset to default joint targets
    n = min(len(JOINT_TARGETS), model.nq - 7)
    data.qpos[7:7 + n] = JOINT_TARGETS[:n]
    data.qpos[2] = 0.98        # set height so feet are near ground
    mujoco.mj_forward(model, data)

    print("Launching viewer — close window or press ESC to quit")
    print("Robot is held at default standing pose via PD control")
    if args.ckpt and not args.no_policy:
        print("(Full policy inference not yet wired — showing default pose)")

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance = 3.5
        v.cam.elevation = -15
        v.cam.azimuth = 135

        while v.is_running():
            step_start = time.time()

            # Hold default pose with PD control
            pd_control(model, data, JOINT_TARGETS)
            mujoco.mj_step(model, data)
            v.sync()

            # ~200 Hz simulation
            elapsed = time.time() - step_start
            remaining = 0.005 - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
