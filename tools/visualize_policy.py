"""Headless policy visualizer for kbot-v2-legs.

Runs a policy checkpoint forward for a few seconds under a fixed command,
then writes diagnostic plots so the gait can be inspected without TB.

    python tools/visualize_policy.py --ckpt path/to/ckpt.bin --vx 0.3 --duration 5
    # outputs to /tmp/viz_<unix_ts>/  (or --output-dir <path>)
    #     filmstrip.png       — 8 side-view snapshots across the rollout
    #     joint_traces.png    — hip / knee / ankle angles over time
    #     foot_pattern.png    — left + right foot center z over time
    #     base_trajectory.png — top-down xy path of the base, vs commanded
    # plus printed numerical fingerprints (knee bend, foot lift, translation)

Designed to share the existing checkpoint-loading pipeline (KbotLegsWalkingRNNTask
+ ksim) so the observation/action mapping is identical to training.  Push
events are disabled by default so we see policy intent, not recovery noise.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Default: let the script use CPU so it doesn't fight an active training run
# for GPU memory. Override with --gpu if you want.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import time
from pathlib import Path

import attrs
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from jaxtyping import Array, PRNGKeyArray

import ksim


# ── Fixed-value commands for deterministic eval ─────────────────────────────

@attrs.define(frozen=True, kw_only=True)
class FixedLinearCommand(ksim.Command):
    vx: float = attrs.field()
    vy: float = attrs.field()

    def get_name(self) -> str:
        return "linear_velocity_command"

    def initial_command(
        self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        return jnp.array([self.vx, self.vy], dtype=jnp.float32)

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData,
        curriculum_level: Array, rng: PRNGKeyArray,
    ) -> Array:
        return prev_command  # never changes during rollout


@attrs.define(frozen=True, kw_only=True)
class FixedAngularCommand(ksim.Command):
    wz: float = attrs.field()

    def get_name(self) -> str:
        return "angular_velocity_command"

    def initial_command(self, physics_data, curriculum_level, rng) -> Array:
        return jnp.array([self.wz], dtype=jnp.float32)

    def __call__(self, prev_command, physics_data, curriculum_level, rng) -> Array:
        return prev_command


@attrs.define(frozen=True, kw_only=True)
class FixedGaitFreq(ksim.Command):
    freq: float = attrs.field()

    def get_name(self) -> str:
        return "gait_frequency_command"

    def initial_command(self, physics_data, curriculum_level, rng) -> Array:
        return jnp.array([self.freq], dtype=jnp.float32)

    def __call__(self, prev_command, physics_data, curriculum_level, rng) -> Array:
        return prev_command


# ── Rollout ─────────────────────────────────────────────────────────────────

def run_rollout(
    ckpt_path: str,
    vx: float,
    vy: float,
    wz: float,
    gait_freq: float,
    duration: float,
    enable_pushes: bool,
    n_snapshots: int,
) -> dict:
    from ksim_kbot.walking.walking_legs_rnn import (
        KbotLegsWalkingRNNTask,
        KbotLegsWalkingRNNTaskConfig,
    )

    class EvalTask(KbotLegsWalkingRNNTask):
        def get_commands(self, physics_model):
            return [
                FixedLinearCommand(vx=vx, vy=vy),
                FixedAngularCommand(wz=wz),
                FixedGaitFreq(freq=gait_freq),
            ]

        def get_events(self, physics_model):
            if enable_pushes:
                return super().get_events(physics_model)
            return []

    cfg = KbotLegsWalkingRNNTaskConfig(
        run_mode="view",
        load_from_ckpt_path=ckpt_path,
        disable_multiprocessing=True,
        # Use stochastic sample instead of argmax: this distrax version
        # doesn't have .mode() on Normal (would equal mean anyway).
        viewer_argmax_action=False,
        num_envs=1, batch_size=1, num_passes=4, epochs_per_log_step=1,
        iterations=6, ls_iterations=6,
        dt=0.002, ctrl_dt=0.02,
        action_latency_range=(0.0, 0.005),
        rollout_length_seconds=max(duration, 5.0),
        action_scale=1.0,
        gamma=0.97, lam=0.95, entropy_coef=0.005,
        learning_rate=1e-4, clip_param=0.3, max_grad_norm=0.3,
        valid_every_n_steps=25, save_every_n_steps=25,
        export_for_inference=False, only_save_most_recent=True,
        domain_randomize=False,
        gait_freq_lower=gait_freq, gait_freq_upper=gait_freq,
        reward_clip_min=0.0, reward_clip_max=1000.0,
    )
    task = EvalTask(cfg)

    with task, jax.disable_jit():
        rng = task.prng_key()
        task.set_loggers()

        mj_model = task.get_mujoco_model()
        mj_model = task.set_mujoco_model_opts(mj_model)
        task.get_mujoco_model_metadata(mj_model)
        randomizers = task.get_physics_randomizers(mj_model)

        rng, model_rng = jax.random.split(rng)
        models, _ = task.load_initial_state(model_rng, load_optimizer=False)

        model_arrs, model_statics = (
            tuple(ms)
            for ms in zip(
                *(eqx.partition(m, task.model_partition_fn) for m in models),
                strict=True,
            )
        )

        constants = task._get_constants(
            mj_model=mj_model, physics_model=mj_model,
            model_statics=model_statics, argmax_action=False,
        )
        rng, init_rng = jax.random.split(rng)
        env_states = task._get_env_state(
            rng=init_rng, rollout_constants=constants,
            mj_model=mj_model, physics_model=mj_model, randomizers=randomizers,
        )
        shared_state = task._get_shared_state(
            mj_model=mj_model, physics_model=mj_model, model_arrs=model_arrs,
        )

        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = np.array(env_states.physics_state.data.qpos)
        mj_data.qvel[:] = np.array(env_states.physics_state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)

        n_steps = int(duration / cfg.ctrl_dt)
        qpos_log = np.zeros((n_steps, mj_model.nq))
        qvel_log = np.zeros((n_steps, mj_model.nv))
        time_log = np.zeros(n_steps)
        foot_left_z = np.zeros(n_steps)
        foot_right_z = np.zeros(n_steps)
        frames: list[tuple[float, np.ndarray]] = []

        snapshot_idxs = np.linspace(0, n_steps - 1, n_snapshots).astype(int)
        snapshot_set = set(int(i) for i in snapshot_idxs)

        # Headless renderer (640x480 side view).  Best-effort: if GPU is
        # exclusively held by training (EGL not available) or OSMesa isn't
        # installed, we skip the filmstrip but still produce the data plots,
        # which carry most of the diagnostic signal.
        renderer = None
        cam = None
        try:
            renderer = mujoco.Renderer(mj_model, height=480, width=640)
            cam = mujoco.MjvCamera()
            cam.azimuth = 90
            cam.elevation = -10
            cam.distance = 2.5
            cam.lookat[:] = [0.0, 0.0, 0.5]
        except Exception as e:
            print(f"[warn] Renderer unavailable ({type(e).__name__}: {e}); skipping filmstrip.")
            print("[warn] Data plots (joint_traces, foot_pattern, base_trajectory) still produced.")
            snapshot_set = set()  # disable snapshot collection

        left_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "KB_D_501L_L_LEG_FOOT")
        right_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "KB_D_501R_R_LEG_FOOT")
        # Foot tilt detection: read the 3 site z's per foot (center, heel, toe).
        # tilt_range = max - min reveals foot orientation; near 0 = flat,
        # several cm = rolled edge. Stance-phase tilt is the real tell because
        # during swing the foot is allowed to angle for toe-off / heel-strike.
        site_l_center = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        site_l_heel = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_heel")
        site_l_toe = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_toe")
        site_r_center = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        site_r_heel = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_heel")
        site_r_toe = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_toe")
        foot_left_3pts = np.zeros((n_steps, 3))   # columns: [center_z, heel_z, toe_z]
        foot_right_3pts = np.zeros((n_steps, 3))
        # Direct foot-body orientation: foot's local Y axis points DOWN when flat.
        # We record the world Z component of body-local Y; flat → -1, tilted → > -1.
        # tilt_angle_rad = arccos(-y_world_z). Catches ALL tilt (pitch + roll).
        foot_left_y_world_z = np.zeros(n_steps)
        foot_right_y_world_z = np.zeros(n_steps)
        # Foot contact force from site-attached force sensors (3D vector each).
        # We log the magnitude of the full force vector; vertical component
        # alone misses oblique impact. Robot weight ~13 kg → static dual stance
        # ~64 N/foot, normal walking peak 130–190 N, hard slam >300 N.
        sensor_l_force = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_force")
        sensor_r_force = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_force")
        l_force_adr = int(mj_model.sensor_adr[sensor_l_force])
        l_force_dim = int(mj_model.sensor_dim[sensor_l_force])
        r_force_adr = int(mj_model.sensor_adr[sensor_r_force])
        r_force_dim = int(mj_model.sensor_dim[sensor_r_force])
        foot_left_force = np.zeros(n_steps)   # magnitude of force vector
        foot_right_force = np.zeros(n_steps)
        foot_left_force_z = np.zeros(n_steps) # vertical component (signed)
        foot_right_force_z = np.zeros(n_steps)

        print(f"Rolling out {n_steps} steps ({duration}s at {cfg.ctrl_dt}s control)...")
        for t in range(n_steps):
            time_log[t] = t * cfg.ctrl_dt
            transition, env_states = task.step_engine(
                constants=constants, env_states=env_states, shared_state=shared_state,
            )
            qpos_log[t] = np.array(env_states.physics_state.data.qpos)
            qvel_log[t] = np.array(env_states.physics_state.data.qvel)
            mj_data.qpos[:] = qpos_log[t]
            mj_data.qvel[:] = qvel_log[t]
            mujoco.mj_forward(mj_model, mj_data)
            foot_left_z[t] = mj_data.xpos[left_foot_id, 2]
            foot_right_z[t] = mj_data.xpos[right_foot_id, 2]
            foot_left_3pts[t] = [
                mj_data.site_xpos[site_l_center, 2],
                mj_data.site_xpos[site_l_heel, 2],
                mj_data.site_xpos[site_l_toe, 2],
            ]
            foot_right_3pts[t] = [
                mj_data.site_xpos[site_r_center, 2],
                mj_data.site_xpos[site_r_heel, 2],
                mj_data.site_xpos[site_r_toe, 2],
            ]
            # xmat is shape (nbody, 9), row-major 3x3 per body. Columns of R
            # are body-local axes in world frame; body-local Y in world = R[:, 1].
            # Z-component of that = R[2, 1] = xmat[body_id, 7].
            foot_left_y_world_z[t] = mj_data.xmat[left_foot_id, 7]
            foot_right_y_world_z[t] = mj_data.xmat[right_foot_id, 7]
            # Force sensor vectors (3 components: Fx, Fy, Fz). Site-attached
            # force sensors report contact force on the foot body expressed in
            # the site's frame. We log both magnitude and signed Fz.
            l_f = mj_data.sensordata[l_force_adr : l_force_adr + l_force_dim]
            r_f = mj_data.sensordata[r_force_adr : r_force_adr + r_force_dim]
            foot_left_force[t] = float(np.linalg.norm(l_f))
            foot_right_force[t] = float(np.linalg.norm(r_f))
            foot_left_force_z[t] = float(l_f[2])
            foot_right_force_z[t] = float(r_f[2])

            if renderer is not None and t in snapshot_set:
                try:
                    cam.lookat[0] = float(mj_data.qpos[0])
                    cam.lookat[1] = float(mj_data.qpos[1])
                    renderer.update_scene(mj_data, camera=cam)
                    pixels = renderer.render()
                    frames.append((float(time_log[t]), pixels.copy()))
                except Exception as e:
                    print(f"[warn] Render failed at step {t}: {e}")
                    renderer = None  # stop trying

        return {
            "qpos": qpos_log,
            "qvel": qvel_log,
            "time": time_log,
            "frames": frames,
            "foot_left_z": foot_left_z,
            "foot_right_z": foot_right_z,
            "foot_left_3pts": foot_left_3pts,
            "foot_right_3pts": foot_right_3pts,
            "foot_left_y_world_z": foot_left_y_world_z,
            "foot_right_y_world_z": foot_right_y_world_z,
            "foot_left_force": foot_left_force,
            "foot_right_force": foot_right_force,
            "foot_left_force_z": foot_left_force_z,
            "foot_right_force_z": foot_right_force_z,
            "ctrl_dt": cfg.ctrl_dt,
        }


# ── Plotting ────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    "R hip pitch", "R hip roll", "R hip yaw", "R knee", "R ankle",
    "L hip pitch", "L hip roll", "L hip yaw", "L knee", "L ankle",
]


def make_filmstrip(data: dict, out: Path, title: str) -> None:
    frames = data["frames"]
    if not frames:
        return
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (t, pixels) in zip(axes, frames):
        ax.imshow(pixels)
        ax.set_title(f"t={t:.2f}s", fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11, y=0.97)
    plt.savefig(out / "filmstrip.png", dpi=110, bbox_inches="tight")
    plt.close()


def make_joint_traces(data: dict, out: Path, title: str) -> None:
    time = data["time"]
    qpos = data["qpos"]
    fig, (ax_hip, ax_knee) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for i, name in enumerate(JOINT_NAMES):
        col = "C0" if "R " in name else "C1"
        if "hip" in name.lower():
            ls = "-" if "pitch" in name else (":" if "roll" in name else "--")
            ax_hip.plot(time, qpos[:, 7 + i], color=col, ls=ls, label=name, alpha=0.8)
        else:  # knee or ankle
            ls = "-" if "knee" in name else "--"
            ax_knee.plot(time, qpos[:, 7 + i], color=col, ls=ls, label=name, alpha=0.8)
    for ax, ylabel in [(ax_hip, "hip joint angle (rad)"), (ax_knee, "knee / ankle (rad)")]:
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax_knee.set_xlabel("time (s)")
    fig.suptitle(title, fontsize=11)
    plt.savefig(out / "joint_traces.png", dpi=110, bbox_inches="tight")
    plt.close()


def make_foot_pattern(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    time = data["time"]
    ax.plot(time, data["foot_left_z"], color="C0", label="left foot z")
    ax.plot(time, data["foot_right_z"], color="C1", label="right foot z")
    ax.axhline(0.05, color="gray", ls="--", lw=0.5, alpha=0.6, label="standing rest (~5 cm)")
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("foot center z (m)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Foot center heights over time")
    plt.savefig(out / "foot_pattern.png", dpi=110, bbox_inches="tight")
    plt.close()


def make_foot_force(data: dict, out: Path) -> None:
    """Left/right foot contact force magnitude over time."""
    time = data["time"]
    lf = data["foot_left_force"]
    rf = data["foot_right_force"]
    body_weight_n = 13.06 * 9.81
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, lf, color="C0", lw=1.3, label="left foot |F|")
    ax.plot(time, rf, color="C1", lw=1.3, label="right foot |F|")
    ax.axhline(body_weight_n, color="gray", ls=":", lw=0.7, label=f"1× BW ({body_weight_n:.0f} N)")
    ax.axhline(2 * body_weight_n, color="orange", ls=":", lw=0.7, label="2× BW")
    ax.axhline(350.0, color="red", ls="--", lw=0.7, alpha=0.7, label="ContactForcePenalty (350 N)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("foot contact force magnitude (N)")
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_title("Foot contact force — single stance ~64 N, walking peak ~130–190 N")
    plt.savefig(out / "foot_force.png", dpi=110, bbox_inches="tight")
    plt.close()


def make_foot_tilt(data: dict, out: Path) -> None:
    """Per-foot heel/center/toe z over time + tilt range. Catches the
    foot-roll exploit: during stance, all three lines should converge."""
    time = data["time"]
    l3 = data["foot_left_3pts"]
    r3 = data["foot_right_3pts"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, three, label in ((axes[0], l3, "left"), (axes[1], r3, "right")):
        ax.plot(time, three[:, 1], color="C2", lw=1.3, label=f"{label} heel z")
        ax.plot(time, three[:, 2], color="C3", lw=1.3, label=f"{label} toe z")
        ax.plot(time, three[:, 0], color="C0", lw=1.0, alpha=0.7, label=f"{label} center z")
        tilt = three.max(axis=1) - three.min(axis=1)
        ax.fill_between(time, three.min(axis=1), three.max(axis=1), color="gray", alpha=0.15,
                        label=f"tilt range (max−min)")
        ax.axhline(0.0, color="black", lw=0.5)
        ax.set_ylabel(f"{label} foot z (m)")
        ax.legend(fontsize=8, ncol=4, loc="upper right")
        ax.grid(alpha=0.3)
    axes[0].set_title("Per-site foot heights — shaded band = max−min spread (flat foot → narrow band)")
    axes[-1].set_xlabel("time (s)")
    plt.savefig(out / "foot_tilt.png", dpi=110, bbox_inches="tight")
    plt.close()


def make_base_trajectory(data: dict, out: Path, vx: float, vy: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    qpos = data["qpos"]
    base_x, base_y = qpos[:, 0], qpos[:, 1]
    duration = float(data["time"][-1])
    ax.plot(base_x, base_y, color="C0", lw=1.5)
    ax.scatter([base_x[0]], [base_y[0]], color="green", s=80, label="start", zorder=5)
    ax.scatter([base_x[-1]], [base_y[-1]], color="red", s=80, label="end", zorder=5)
    expected_dx, expected_dy = vx * duration, vy * duration
    if abs(expected_dx) + abs(expected_dy) > 1e-3:
        ax.annotate(
            "", xy=(base_x[0] + expected_dx, base_y[0] + expected_dy),
            xytext=(base_x[0], base_y[0]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=2, alpha=0.5),
        )
        ax.text(
            base_x[0] + expected_dx * 0.5, base_y[0] + expected_dy * 0.5 + 0.02,
            f"commanded\n{duration:.1f}s @ {np.hypot(vx, vy):.2f} m/s",
            fontsize=9, color="gray", ha="center",
        )
    ax.set_aspect("equal")
    ax.set_xlabel("base x (m)")
    ax.set_ylabel("base y (m)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Base trajectory: actual vs commanded")
    plt.savefig(out / "base_trajectory.png", dpi=110, bbox_inches="tight")
    plt.close()


def print_fingerprints(data: dict, vx: float, vy: float) -> None:
    qpos = data["qpos"]
    base_x, base_y = qpos[:, 0], qpos[:, 1]
    duration = float(data["time"][-1])
    actual_dx = base_x[-1] - base_x[0]
    actual_dy = base_y[-1] - base_y[0]
    actual_dist = float(np.hypot(actual_dx, actual_dy))
    expected_dist = float(np.hypot(vx * duration, vy * duration))
    pct = 100 * actual_dist / max(expected_dist, 1e-6)

    r_knee = qpos[:, 10]
    l_knee = qpos[:, 15]
    r_hip_pitch = qpos[:, 7]
    l_hip_pitch = qpos[:, 12]

    fl, fr = data["foot_left_z"], data["foot_right_z"]
    l_lift = fl.max() - fl.min()
    r_lift = fr.max() - fr.min()
    symmetry = min(l_lift, r_lift) / max(l_lift, r_lift, 1e-6)

    print()
    print("════════ Numerical fingerprints ════════")
    print(f"  Duration:                 {duration:.2f} s")
    print(f"  Commanded distance:       {expected_dist:.3f} m  (vx={vx}, vy={vy})")
    print(f"  Actual distance:          {actual_dist:.3f} m  ({pct:.0f}% of commanded)")
    print(f"  Drift orthogonal to cmd:  {abs(actual_dy - vy * duration) if abs(vx) > abs(vy) else abs(actual_dx - vx * duration):.3f} m")
    print()
    print(f"  Knee bend MAX  (R/L):     {abs(r_knee).max():.3f} / {abs(l_knee).max():.3f} rad")
    print(f"  Knee bend STD  (R/L):     {r_knee.std():.4f} / {l_knee.std():.4f}    ← gait-shape oscillation")
    print(f"  Knee bend MEAN (R/L):     {r_knee.mean():+.3f} / {l_knee.mean():+.3f} rad")
    print()
    print(f"  Hip pitch MAX  (R/L):     {abs(r_hip_pitch).max():.3f} / {abs(l_hip_pitch).max():.3f} rad")
    print(f"  Hip pitch STD  (R/L):     {r_hip_pitch.std():.4f} / {l_hip_pitch.std():.4f}    ← stride amplitude")
    print()
    print(f"  Foot z MAX   (L/R):       {fl.max():.3f} / {fr.max():.3f} m")
    print(f"  Foot z LIFT  (L/R):       {l_lift:.3f} / {r_lift:.3f} m   ← actual lift amplitude")
    print(f"  Foot z STD   (L/R):       {fl.std():.4f} / {fr.std():.4f}")
    print(f"  Stepping symmetry:        {symmetry:.2f}   (1.0 = perfectly symmetric, 0 = one foot planted)")
    print()
    # Foot-flatness fingerprints. Two complementary signals:
    #   1. Pitch tilt: |heel_z - toe_z| — front/back rocking (heel-strike, toe-off).
    #   2. Orientation tilt: angle between foot body's bottom normal and world up.
    #      Catches ALL tilt axes including the inner/outer roll the user observed.
    #      Foot's local +Y is the bottom normal; flat → Y_world = (0, 0, -1).
    l3 = data["foot_left_3pts"]
    r3 = data["foot_right_3pts"]
    # heel = column 1, toe = column 2 (column 0 is center, dropped — has 3.6cm baseline offset)
    l_pitch = np.abs(l3[:, 1] - l3[:, 2])
    r_pitch = np.abs(r3[:, 1] - r3[:, 2])
    # Orientation tilt from horizontal, in degrees
    l_y_wz = data["foot_left_y_world_z"]
    r_y_wz = data["foot_right_y_world_z"]
    l_tilt_deg = np.degrees(np.arccos(np.clip(-l_y_wz, -1.0, 1.0)))
    r_tilt_deg = np.degrees(np.arccos(np.clip(-r_y_wz, -1.0, 1.0)))
    # Stance = min(heel, toe) z within 3cm of floor (sites are at foot bottom, near 0 when flat)
    stance_thresh = 0.03
    l_min_ht = np.minimum(l3[:, 1], l3[:, 2])
    r_min_ht = np.minimum(r3[:, 1], r3[:, 2])
    l_stance = l_min_ht < stance_thresh
    r_stance = r_min_ht < stance_thresh

    def _mean(arr, mask):
        return float(arr[mask].mean()) if mask.any() else float("nan")

    print(f"  Pitch tilt MEAN (L/R):    {l_pitch.mean()*100:.2f} / {r_pitch.mean()*100:.2f} cm   (|heel_z − toe_z|)")
    print(f"  Pitch tilt STANCE (L/R):  {_mean(l_pitch, l_stance)*100:.2f} / {_mean(r_pitch, r_stance)*100:.2f} cm   ← flat≈0, rocking>2cm")
    print(f"  Orient tilt MEAN (L/R):   {l_tilt_deg.mean():.1f}° / {r_tilt_deg.mean():.1f}°   (angle of foot bottom from horizontal)")
    print(f"  Orient tilt STANCE (L/R): {_mean(l_tilt_deg, l_stance):.1f}° / {_mean(r_tilt_deg, r_stance):.1f}°   ← flat<5°, rolled edge>15°")
    print(f"  Stance fraction (L/R):    {100.0 * l_stance.mean():.0f}% / {100.0 * r_stance.mean():.0f}%")
    print()
    # Foot strike force. Robot ~13 kg → static dual stance ~64 N/foot, normal
    # walking peak 130–190 N (~1.5× body weight), hard slam >300 N.
    # ContactForcePenalty (training) only fires above 350 N — anything below
    # is invisible there but visible here.
    lf = data["foot_left_force"]
    rf = data["foot_right_force"]
    body_weight_n = 13.06 * 9.81  # ~128 N total
    # Stance-only force (filter out swing-phase numerical noise)
    l_stance_force = lf[l_stance]
    r_stance_force = rf[r_stance]
    l_stance_mean = float(l_stance_force.mean()) if l_stance_force.size else float("nan")
    r_stance_mean = float(r_stance_force.mean()) if r_stance_force.size else float("nan")
    # Landing peaks: rising edges of contact = first stance step after swing,
    # peak force in following 100 ms (5 steps at 20 ms ctrl_dt) per landing.
    def _landing_peaks(stance_mask, force_arr, window=5):
        rising = np.where((~stance_mask[:-1]) & stance_mask[1:])[0] + 1
        peaks = [float(force_arr[i : min(i + window, len(force_arr))].max()) for i in rising]
        return peaks
    l_peaks = _landing_peaks(l_stance, lf)
    r_peaks = _landing_peaks(r_stance, rf)
    l_peak_mean = float(np.mean(l_peaks)) if l_peaks else float("nan")
    r_peak_mean = float(np.mean(r_peaks)) if r_peaks else float("nan")
    l_peak_max = float(np.max(l_peaks)) if l_peaks else float("nan")
    r_peak_max = float(np.max(r_peaks)) if r_peaks else float("nan")
    print(f"  Stance force MEAN (L/R):  {l_stance_mean:.0f} / {r_stance_mean:.0f} N   ({l_stance_mean/body_weight_n:.2f}× / {r_stance_mean/body_weight_n:.2f}× body weight, {body_weight_n:.0f} N)")
    print(f"  Landing peak MEAN (L/R):  {l_peak_mean:.0f} / {r_peak_mean:.0f} N   (over {len(l_peaks)} / {len(r_peaks)} landings)")
    print(f"  Landing peak MAX  (L/R):  {l_peak_max:.0f} / {r_peak_max:.0f} N   ← ContactForcePenalty fires >350 N")
    print()


def make_velocity_tracking(data: dict, out: Path, vx: float, vy: float) -> None:
    """Commanded vs actual base velocity over time."""
    qpos = data["qpos"]
    time = data["time"]
    dt = data["ctrl_dt"]
    # Compute actual velocity by finite differences on base position.
    vx_actual = np.gradient(qpos[:, 0], dt)
    vy_actual = np.gradient(qpos[:, 1], dt)
    # Smooth slightly for readability.
    win = min(11, len(vx_actual) // 5 * 2 + 1)
    if win >= 3:
        kernel = np.ones(win) / win
        vx_actual = np.convolve(vx_actual, kernel, mode="same")
        vy_actual = np.convolve(vy_actual, kernel, mode="same")

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax_x.plot(time, vx_actual, color="C0", label="actual vx")
    ax_x.axhline(vx, color="gray", ls="--", label=f"commanded vx={vx:+.2f}")
    ax_x.set_ylabel("vx (m/s)")
    ax_x.legend(fontsize=9, loc="upper right")
    ax_x.grid(alpha=0.3)
    ax_y.plot(time, vy_actual, color="C1", label="actual vy")
    ax_y.axhline(vy, color="gray", ls="--", label=f"commanded vy={vy:+.2f}")
    ax_y.set_xlabel("time (s)")
    ax_y.set_ylabel("vy (m/s)")
    ax_y.legend(fontsize=9, loc="upper right")
    ax_y.grid(alpha=0.3)
    fig.suptitle("Base velocity — actual vs commanded", fontsize=11)
    plt.savefig(out / "velocity_tracking.png", dpi=110, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Visualize a kbot-v2-legs policy checkpoint")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .bin")
    p.add_argument("--vx", type=float, default=0.3, help="Linear x velocity command (m/s)")
    p.add_argument("--vy", type=float, default=0.0, help="Linear y velocity command (m/s)")
    p.add_argument("--wz", type=float, default=0.0, help="Angular yaw rate command (rad/s)")
    p.add_argument("--gait-freq", type=float, default=1.4, help="Gait frequency (Hz)")
    p.add_argument("--duration", type=float, default=5.0, help="Rollout duration (s)")
    p.add_argument("--snapshots", type=int, default=8, help="Filmstrip frame count")
    p.add_argument("--pushes", action="store_true", default=False,
                   help="Enable training-time push events (default: off, so we see policy intent)")
    p.add_argument("--gpu", action="store_true", default=False,
                   help="Run on GPU (default: CPU so it doesn't fight training)")
    p.add_argument("--output-dir", default=None, help="Where to write PNGs (default: /tmp/viz_<ts>)")
    args = p.parse_args()

    if args.gpu:
        os.environ["JAX_PLATFORMS"] = "gpu"

    if args.output_dir is None:
        args.output_dir = f"/tmp/viz_{int(time.time())}"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = run_rollout(
        args.ckpt, args.vx, args.vy, args.wz,
        args.gait_freq, args.duration, args.pushes, args.snapshots,
    )

    title = (
        f"Policy rollout — cmd vx={args.vx:+.2f}  vy={args.vy:+.2f}  wz={args.wz:+.2f}, "
        f"gait {args.gait_freq} Hz, {'+ pushes' if args.pushes else 'no pushes'}"
    )
    make_filmstrip(data, out, title)
    make_joint_traces(data, out, title)
    make_foot_pattern(data, out)
    make_foot_tilt(data, out)
    make_foot_force(data, out)
    make_base_trajectory(data, out, args.vx, args.vy)
    make_velocity_tracking(data, out, args.vx, args.vy)
    print_fingerprints(data, args.vx, args.vy)

    print(f"Outputs:  {out.resolve()}/")
    print(f"  filmstrip.png         — visual gait (8 side-view snapshots)")
    print(f"  joint_traces.png      — joint angles vs time")
    print(f"  foot_pattern.png      — foot heights vs time")
    print(f"  foot_tilt.png         — heel/center/toe z + tilt range per foot")
    print(f"  foot_force.png        — left/right contact force magnitude over time")
    print(f"  base_trajectory.png   — top-down xy path")
    print(f"  velocity_tracking.png — commanded vs actual vx, vy")


if __name__ == "__main__":
    main()
