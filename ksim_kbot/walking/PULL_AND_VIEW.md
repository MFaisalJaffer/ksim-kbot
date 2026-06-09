# Pull latest checkpoint and view on Mac

Quick reference for grabbing the latest training checkpoint from the
desktop and running it interactively in `view_mac_legs.py` on macOS.

## Prerequisites (one-time)

- macOS with `mjpython` available (comes with `mujoco` Python package)
- `ksim-kbot` checked out locally on the Mac (the viewer needs the same
  task code + MJCF that produced the checkpoint)
- Network access to the desktop — either:
  - **LAN**: `192.168.68.130`
  - **Tailscale**: `100.95.25.109`
- SSH key already trusted on the desktop (otherwise you'll be prompted
  for the password on every scp)

## Where checkpoints live

On the desktop:

```
/home/faisal/ksim-kbot/ksim_kbot/walking/kbot_legs_walking_rnntask/run_<N>/checkpoints/
```

Each run directory has:
- `ckpt.bin` — symlink to the latest checkpoint, always safe to grab
- `ckpt.<step>.bin` — individual snapshots (e.g. `ckpt.4325.bin`)

The symlink is updated atomically by training, so copying `ckpt.bin`
mid-training is safe; you may get either the previous snapshot or the
new one, but never a half-written file.

## Pull the latest checkpoint

From the Mac, pick the run number you want and copy the symlink target.
`scp` will resolve the symlink automatically:

```bash
# LAN (faster if you're on the same network)
scp faisal@192.168.68.130:ksim-kbot/ksim_kbot/walking/kbot_legs_walking_rnntask/run_27/checkpoints/ckpt.bin ~/kbot/ckpt.bin

# Tailscale (works from anywhere)
scp faisal@100.95.25.109:ksim-kbot/ksim_kbot/walking/kbot_legs_walking_rnntask/run_27/checkpoints/ckpt.bin ~/kbot/ckpt.bin
```

Replace `run_27` with the run you want.

## Find the actual step number (optional)

The symlink hides the step count. To check which step you just pulled:

```bash
ssh faisal@192.168.68.130 readlink ksim-kbot/ksim_kbot/walking/kbot_legs_walking_rnntask/run_27/checkpoints/ckpt.bin
# → ckpt.13671.bin
```

Or list all available snapshots:

```bash
ssh faisal@192.168.68.130 'ls -lt ksim-kbot/ksim_kbot/walking/kbot_legs_walking_rnntask/run_27/checkpoints/ckpt.*.bin | head'
```

## Run the viewer on Mac

From the ksim-kbot repo root on the Mac:

```bash
mjpython view_mac_legs.py --ckpt ~/kbot/ckpt.bin
```

A MuJoCo viewer window opens. The robot starts standing at the origin.

### Keyboard controls

| Key | Action |
|---|---|
| `W` / `S` | forward / backward (±0.1 m/s per press) |
| `A` / `D` | turn left / right (±0.1 rad/s per press) |
| `Q` / `E` | strafe left / right (±0.1 m/s per press) |
| `SPACE` | stop — zero all velocity commands |
| `R` | reset episode (robot back to origin, joints to default) |
| `ESC` | quit |

The current commanded velocity stays active until you change it. Holding
`W` doesn't continuously accelerate — each press adds 0.1 m/s. Use
`SPACE` to halt before sending the opposite command.

### Camera controls (MuJoCo defaults)

- Right-click + drag: rotate camera
- Middle-click + drag (or two-finger drag on trackpad): pan
- Scroll wheel: zoom
- Double-click on a body: track it

## One-liner: pull + view

```bash
RUN=27 && \
scp faisal@192.168.68.130:ksim-kbot/ksim_kbot/walking/kbot_legs_walking_rnntask/run_${RUN}/checkpoints/ckpt.bin ~/kbot/ckpt.bin && \
mjpython view_mac_legs.py --ckpt ~/kbot/ckpt.bin
```

## Troubleshooting

**`shape mismatch` or `expected N inputs, got M`** — the viewer's task
code is out of sync with the checkpoint. Pull the latest `ksim-kbot`
repo on the Mac (`git pull` in the repo dir). Checkpoints from runs
that used different observation/architecture shapes can't be loaded by
the new code.

**`FileNotFoundError: robot.mjcf`** — the viewer can't find the MJCF.
The viewer expects `~/.kscale/robots/kbot/robot/robot.mjcf` (or the
configured path). Same MJCF as the desktop; check the README for setup.

**Viewer launches but robot stands stiff and never moves** — you forgot
to press a movement key. `W` once gets you `vx=+0.1`; press `W` three
times for `vx=+0.3` which matches the training command range.

**Robot falls immediately on a checkpoint that worked before** — likely
the policy was trained with a different reward/task config and the
viewer config doesn't match. Diff `view_mac_legs.py`'s task config
against `walking_legs_rnn.py`'s defaults.

**`distrax.Normal has no .mode()` error** — fixed long ago; if you see
this, your local viewer is out of date. `git pull`.
