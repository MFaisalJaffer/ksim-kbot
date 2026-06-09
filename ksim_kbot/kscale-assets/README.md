# kscale-assets

Vendored robot asset packages — physically committed to this repo so
training is reproducible on a fresh clone.

## `kbot-v2-legs/`

The model this branch's `kbot_legs_walking_rnntask` trains against:
torso + both legs, no arms, mass-scaled to the physical robot
(13.06 kg / 28.8 lb). See `kbot-v2-legs/README.md` for details on
which upstream commit + strip script produced it.

## `kbot-v2-feet/` (optional, local-only)

The joystick task (`kbot_walking_joystick_rnntask`) uses the full kbot
(legs + arms). On this machine, the user has a local symlink:

```
kbot-v2-feet → /home/faisal/.kscale/robots/kbot/robot
```

This symlink is **ignored by git** (see `.gitignore`) because it points
to an absolute local path and isn't portable. If you need the
full-body assets on a fresh checkout, populate `kbot-v2-feet/` from
`kscalelabs/kscale-assets` or the equivalent local source.

## History

Originally this directory was a git submodule pointing at
`kscalelabs/kscale-assets`. That was bypassed locally with symlinks to
external asset edits (the `odrive-mit-control` repo had foot site,
hip_roll motor, and root body name corrections). For reproducibility
the legs assets are now vendored directly — anyone cloning this branch
gets exactly the model that training ran against.
