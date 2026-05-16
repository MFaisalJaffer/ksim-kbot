"""Post-hoc analysis: TV-curve saturation from a saved checkpoint.

Loads a policy, runs a deterministic rollout in headless mode under a chosen
velocity command, records applied torques and joint velocities, and reports
how close the policy operates to the actuator T-V curve limits.

Usage:
    conda run -n humanoid python tools/analyze_tv_saturation.py <ckpt_path> \\
        [--vx 0.5] [--vy 0.0] [--wz 0.0] [--steps 500]

Reads `data.ctrl` (true applied torque) and `data.qvel` from each sim step,
then computes saturation = |ctrl| / max_tau_motoring(|qvel|) per joint
(only counts joints currently motoring — same-sign ctrl × qvel).

Output:
  - Per-joint breakdown: mean, max, % time > 0.9 saturation
  - Per-group summary: arms vs legs, by motor type
  - Overall mean saturation
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import sys
from pathlib import Path

import attrs
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

import ksim

# Per-joint motor types (must match the order in walking_joystick.py JOINT_TARGETS)
MOTOR_TYPES = (
    "04", "04", "03", "04", "00",   # right arm
    "04", "04", "03", "04", "00",   # left arm
    "04", "04", "03", "04", "02",   # right leg
    "04", "04", "03", "04", "02",   # left leg
)

JOINT_NAMES = (
    "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow",  "R_wrist",
    "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow",  "L_wrist",
    "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee",   "R_ankle",
    "L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee",   "L_ankle",
)


def build_constant_command(value: jnp.ndarray, name: str) -> "type":
    """Make a ksim.Command that always returns `value`."""

    @attrs.define(frozen=True, kw_only=True)
    class _Const(ksim.Command):
        _name: str = attrs.field()
        _value: tuple = attrs.field()

        def get_name(self) -> str:
            return self._name

        def initial_command(self, physics_data, curriculum_level, rng):
            return jnp.array(self._value)

        def __call__(self, prev_command, physics_data, curriculum_level, rng):
            return jnp.array(self._value)

    return _Const(_name=name, _value=tuple(np.array(value).tolist()))


def analyze(ckpt_path: str, vx: float, vy: float, wz: float, n_steps: int) -> None:
    from ksim_kbot.walking.walking_joystick_rnn import (
        KbotWalkingJoystickRNNTask, KbotWalkingJoystickRNNTaskConfig)
    from ksim_kbot.common import GaitFrequencyCommand, _build_tv_curve_arrays

    # Build curves once for analysis.
    omega_curves, tau_curves = _build_tv_curve_arrays(MOTOR_TYPES)
    omega_np = np.array(omega_curves)
    tau_np = np.array(tau_curves)

    print(f"\nAnalyzing TV saturation under cmd (vx={vx}, vy={vy}, wz={wz})")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Steps: {n_steps}\n")

    # Forced constant-command task subclass.
    class _ConstTask(KbotWalkingJoystickRNNTask):
        def get_commands(self, physics_model):
            from ksim_kbot.common import ArmConstraintCommand
            return [
                build_constant_command(jnp.array([vx, vy]), "linear_velocity_command"),
                build_constant_command(jnp.array([wz]),     "angular_velocity_command"),
                GaitFrequencyCommand(
                    gait_freq_lower=self.config.gait_freq_lower,
                    gait_freq_upper=self.config.gait_freq_upper,
                ),
                # No arm constraint for this run
                ArmConstraintCommand(constraint_prob=0.0, use_curriculum=False),
            ]

    cfg = KbotWalkingJoystickRNNTaskConfig(
        run_mode="view",
        load_from_ckpt_path=ckpt_path,
        disable_multiprocessing=True,
        viewer_argmax_action=True,
        num_envs=1, batch_size=1,
        num_passes=4, epochs_per_log_step=1,
        iterations=6, ls_iterations=6,
        dt=0.002, ctrl_dt=0.02,
        action_latency_range=(0.0, 0.0),
        rollout_length_seconds=5.0,
        action_scale=1.0,
        gamma=0.97, lam=0.95, entropy_coef=0.005,
        learning_rate=1e-4, clip_param=0.3, max_grad_norm=0.3,
        valid_every_n_steps=25, save_every_n_steps=25,
        export_for_inference=False, only_save_most_recent=True,
        domain_randomize=False,
        gait_freq_lower=1.25, gait_freq_upper=1.5,
        reward_clip_min=0.0, reward_clip_max=1000.0,
    )
    task = _ConstTask(cfg)

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

        rng, init_rng = jax.random.split(rng)
        constants = task._get_constants(
            mj_model=mj_model, physics_model=mj_model,
            model_statics=model_statics, argmax_action=True,
        )
        env_states = task._get_env_state(
            rng=init_rng, rollout_constants=constants,
            mj_model=mj_model, physics_model=mj_model,
            randomizers=randomizers,
        )
        shared_state = task._get_shared_state(
            mj_model=mj_model, physics_model=mj_model, model_arrs=model_arrs,
        )

        ctrls = []
        qvels = []
        for step in range(n_steps):
            transition, env_states = task.step_engine(
                constants=constants, env_states=env_states, shared_state=shared_state,
            )
            ctrl = np.array(env_states.physics_state.data.ctrl)
            qvel = np.array(env_states.physics_state.data.qvel[6:])  # skip freejoint
            ctrls.append(ctrl)
            qvels.append(qvel)

    ctrls = np.stack(ctrls)   # (T, 20)
    qvels = np.stack(qvels)   # (T, 20)
    T, N = ctrls.shape
    print(f"Rollout collected: {T} steps × {N} joints\n")

    # Per-joint, per-step max torque from TV curve.
    speed = np.abs(qvels)
    max_tau = np.zeros_like(speed)
    for t in range(T):
        for j in range(N):
            max_tau[t, j] = np.interp(speed[t, j], omega_np[j], tau_np[j])

    is_motoring = ctrls * qvels > 0.0
    sat = np.where(is_motoring, np.abs(ctrls) / (max_tau + 1e-6), np.nan)

    # Per-joint stats.
    print(f"{'#':<3} {'Joint':<11} {'Mot':<5} {'mean':>6} {'p50':>6} {'p90':>6} {'p99':>6} {'max':>6} {'% > 0.9':>8}")
    print("-" * 72)
    for j in range(N):
        s = sat[:, j]
        s = s[~np.isnan(s)]
        if len(s) == 0:
            print(f"{j:<3} {JOINT_NAMES[j]:<11} {MOTOR_TYPES[j]:<5}  (no motoring)")
            continue
        mean = s.mean(); p50 = np.percentile(s, 50); p90 = np.percentile(s, 90)
        p99 = np.percentile(s, 99); mx = s.max()
        pct_pegged = (s > 0.9).mean() * 100.0
        print(f"{j:<3} {JOINT_NAMES[j]:<11} {MOTOR_TYPES[j]:<5} {mean:>6.3f} {p50:>6.3f} {p90:>6.3f} {p99:>6.3f} {mx:>6.3f} {pct_pegged:>7.2f}%")

    # Overall summary.
    overall = sat[~np.isnan(sat)]
    print()
    print(f"OVERALL: mean={overall.mean():.3f}  p90={np.percentile(overall, 90):.3f}  max={overall.max():.3f}  "
          f"% > 0.9: {(overall > 0.9).mean()*100:.2f}%")

    # Per-group breakdown.
    print("\nPer-group (mean saturation):")
    groups = [
        ("arms", list(range(0, 10))),
        ("legs", list(range(10, 20))),
        ("knees", [13, 18]),
        ("hips_pitch", [10, 15]),
        ("ankles", [14, 19]),
    ]
    for name, idxs in groups:
        s = sat[:, idxs]
        s = s[~np.isnan(s)]
        if len(s) == 0:
            print(f"  {name:<12}  (no motoring)")
        else:
            print(f"  {name:<12}  mean={s.mean():.3f}  max={s.max():.3f}  % > 0.9: {(s > 0.9).mean()*100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help="Path to checkpoint .bin")
    parser.add_argument("--vx", type=float, default=0.5, help="Forward velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Side velocity command (m/s)")
    parser.add_argument("--wz", type=float, default=0.0, help="Yaw velocity command (rad/s)")
    parser.add_argument("--steps", type=int, default=500, help="Number of sim steps (1 step = 20ms)")
    args = parser.parse_args()
    analyze(args.ckpt, args.vx, args.vy, args.wz, args.steps)


if __name__ == "__main__":
    main()
