# LQR 2WD→4WD Balancing Transition Fix

## Context
This note documents the 2WD→4WD transition instability fix in `tribot_sim.py` when using `CONTROLLER='lqr'`.

Observed failure pattern:
- Robot switched from stable 2WD to 4WD.
- During/after transition, wheel torque direction was effectively wrong for velocity damping.
- Robot accelerated instead of braking, then tipped and fell.

## Main symptoms from logs
- During transition (`act=0`), non-zero wheel commands could inject motion.
- In settled 4WD (`act=1`), `verr` and `drv` signs were inconsistent (example: `verr<0` while `drv>0`), causing runaway.
- Transition stabilizer could also over-inject forward speed if too aggressive.

## Root causes
1. **Transition torque leakage / injection**
   - Wheel commands during geometry change (triplet rotating 60°→0°) caused unwanted acceleration.
2. **Settled 4WD sign ambiguity**
   - Mixed sign paths (LQR conventions + URDF wheel axis inversion + follow remap) made correction direction unreliable.
3. **Over-aggressive transition stabilizer**
   - Velocity-coupled transition term could push the robot instead of just damping tilt.

## Implemented fix

### 1) Add targeted debug instrumentation
Added `_debug` fields and telemetry so the full sign chain is visible:
- Raw controller outputs
- 4WD follow internals (`pos_error`, `v_ref`, `speed_error`, `drive_torque`, `yaw_torque`)
- Transition torque (`fourwd_transition_torque`)
- Final per-side command, motor command, wheel velocity

Also added compact console line:
- `4WDDBG ...`

### 2) Transition phase control (act=0)
In `update()` when `drive_mode == '4wd'` and transition/not-settled:
- Apply a **limited transition stabilizer** based on LQR pitch/rate terms only:
  - `u_trans = -(K2*x_pitch + K3*x_pitch_rate)`
  - clamp with `FOURWD_TRANSITION_MAX_TORQUE`
- Removed velocity coupling from transition stabilizer to avoid forward injection.

### 3) Settled 4WD follow control (act=1)
When transition is complete and triplet is settled:
- Use explicit follow controller (not LQR K decomposition):
  - `v_ref = clip(FOURWD_POS_KP * pos_error, ±FOURWD_MAX_SPEED)`
  - `speed_error = v_ref - fwd_vel`
  - `drive_torque = FOURWD_DRIVE_SIGN * FOURWD_SPEED_KP * speed_error`
  - clamp by `FOURWD_MAX_TORQUE`
- Yaw correction only if yaw command is non-zero.

This removes sign ambiguity and keeps behavior interpretable from logs.

## Key config values (current)
- `FOURWD_DRIVE_SIGN: 1.0`
- `FOURWD_POS_KP: 1.2`
- `FOURWD_MAX_SPEED: 0.6`
- `FOURWD_SPEED_KP: 1.0`
- `FOURWD_MAX_TORQUE: 0.35`
- `FOURWD_YAW_RATE_KP: 0.35`
- `FOURWD_MAX_YAW_TORQUE: 0.35`
- `FOURWD_TRANSITION_MAX_TORQUE: 0.15`
- `FOURWD_SETTLE_ANGLE_TOL: 2 deg`
- `FOURWD_SETTLE_RATE_TOL: 1 rad/s`

## Validation checklist for another branch
1. Ensure `CONTROLLER='lqr'`.
2. Reproduce switch test with neutral sticks.
3. Confirm in logs:
   - During transition (`act=0`): `utr` present but moderate; no large wheel acceleration spikes.
   - After settle (`act=1`): if `v>0` and `v_ref≈0`, then `drv<0` (braking).
   - No sustained `drv` saturation in direction that increases `|v|`.
4. If unstable, tune in this order:
   1. Lower `FOURWD_TRANSITION_MAX_TORQUE`
   2. Lower `FOURWD_MAX_TORQUE`
   3. Lower `FOURWD_SPEED_KP`
   4. Lower `FOURWD_POS_KP`

## Minimal patch areas to re-apply
- `CONFIG` 4WD parameters block.
- `TribotBalanceBot.__init__`: `_debug` dict fields.
- `TribotBalanceBot.update()`:
  - transition stabilizer branch
  - settled 4WD follow branch
  - debug writes
- PlotJuggler payload additions (`dbg_*` fields).
- console `4WDDBG` print block.

## Quick sanity command
```bash
python3 -m py_compile tribot_sim.py
```
