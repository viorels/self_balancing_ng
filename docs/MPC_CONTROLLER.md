# MPC Balance Controller

Implementation of `docs/plans/DYNAMIC_CONTROLLER_RECOMMENDATION.md`: a
mode-scheduled, constrained, two-input model predictive controller that
balances the tribot in 4WD and 2WD, drives it, executes 4WD/2WD
transitions and carries an emergency-flip reflex. It is the default
controller (`config.sim.controller = 'mpc'`).

## Files

| File | Role |
|------|------|
| `controllers/control_mpc.py` | `MPCBalanceController`: state assembly, model scheduling, QP call, yaw PI, hub torque split, safety cut-off |
| `controllers/mpc_plant.py` | Planar Lagrangian model of the robot, linearisation, discretisation, hub-angle geometry helpers |
| `controllers/mpc_qp.py` | Sparse QP with a fixed sparsity pattern, solved by OSQP with warm start |
| `controllers/triplet_planner.py` | Event layer: per-side hub references, asymmetric mode transitions, flip state machine |
| `tools/validate_mpc_plant.py` | Compares model accelerations and sign conventions against MuJoCo |
| `tools/run_headless.py` | Scripted scenarios without the viewer (flat ground and stairs); exit code 1 if the robot falls |
| `robot.py` (`_probe_terrain`) | Forward-looking terrain probe (ToF model): ground height change ahead of the hub, from ray casts against terrain geoms (group 3) |

## Plant model

State `z = [s, ṡ, θ, θ̇, λ, λ̇]`, inputs `u = [u_d, u_t]` (total drive
torque, total hub torque, both sides summed).

- `s` forward position of the grounded wheel centre (odometry).
- `θ` body pitch, **forward lean positive**. The simulator's measured
  pitch is backward-positive (rotation about body +Y with forward = body
  −X), so `θ = −pitch`. This was checked numerically, not assumed.
- `λ` leg angle: direction from the grounded wheel centre to the hub
  axis, hub-forward positive. It is derived from the encoders by
  `α = wrap120(φ − θ + 60°)`, `λ = −α`, where `wrap120` folds the angle
  into ±60° (the triplet's 120° symmetry). A hub joint at an odd multiple
  of 60° gives `α ≈ 0` (one wheel down, 2WD); an even multiple gives
  `α = ±60°` (two wheels tied, 4WD).

Two contact modes share the state layout:

- **Single contact** (2WD and any intermediate hub angle): double inverted
  pendulum on a rolling wheel. Link 1 is the leg (length R = 0.12 m,
  carries the hubs and the two free wheels), link 2 is the body. The
  belt-coupled wheel spin adds an equivalent mass of 6·I_w/r² ≈ 0.12 kg.
  The drive torque acts between wheel and body (the hub is transparent to
  it), the hub torque acts on the hub joint `φ = θ − λ + const`.
- **4WD** (leg locked): cart-pendulum about the hub axis. The hub torque
  is a direct body torque. The unilateral ground contacts add the tipping
  constraint `|u_t + (z_w/r) u_d| ≤ m g d` (3.46 Nm), applied with a
  safety factor.

The mass matrix comes from point-mass Jacobians, gravity from the
potential, and the model is linearised numerically at the current
`(λ, θ)` each time the operating point moves by more than 3°. Divergence
rates: 7.9 rad/s in 4WD, 16.9 rad/s for the 2WD leg mode (the light leg
falls fast and is what the hub motor spends most of its effort on).

Validation against MuJoCo (`tools/validate_mpc_plant.py`), instantaneous
accelerations for unit inputs and small displacements, agree to within a
few percent in 4WD and within roughly 10% in 2WD.

## QP

Horizon N = 20 steps of 20 ms (0.4 s), re-solved every control tick
(200 Hz). Cost: weighted state error to a per-stage reference, input
error to the model's equilibrium input, an input-rate penalty, a DARE
terminal cost on the controllable states, and a soft pitch limit (±20°)
with linear plus quadratic slack. Hard bounds on both inputs with margin
reserved for the yaw loop and the leg PD. OSQP solves in 0.3 to 0.7 ms
typically, with rare spikes to a few milliseconds under extreme pushes.

## Event layer

- **Mode transitions** are asymmetric: the left hub rotates backward and
  the right hub forward by 60° on a minimum-jerk profile over 0.8 s, so
  one front and one rear wheel stay grounded and the support line passes
  under the hub axis throughout. The MPC only sees the symmetric leg; a PD
  with gravity feedforward tracks the antisymmetric part.
- **Flip**: armed when the MPC's own predicted trajectory (or the measured
  pitch) exceeds `flip_theta_trigger` while falling in that direction, and
  fired when the DCM time-to-crash drops under the flip budget. Both hubs
  then rotate 120° toward the fall on a 0.25 s profile under a stiff PD
  while the MPC keeps balancing with the drive only. Position reference is
  frozen during the flip and re-latched on landing.
- **Re-latch**: if the MPC rolls the triplets onto another wheel by itself
  (it does this under hard pushes), the planner re-snaps its references and
  mode once the hubs rest near a new 60° multiple.

## Stairs

A 5 cm riser is 86% of the wheel radius, so no drive torque rolls a wheel
up it; the cluster has to step. The planner runs a step manoeuvre in 4WD
whenever the terrain probe reports a climbable riser just in front of the
front wheel while driving forward:

1. **Approach**: the velocity command is capped (`step_approach_speed`)
   within `step_approach_distance` of a riser so the wheel meets it gently.
2. **Lean**: the pitch reference ramps to the lean that puts the composite
   CoG just behind the front axle (`lean_for_pivot`, margin
   `step_lean_margin`). The lean is computed from the measured leg angle to
   the front wheel, which is 60° behind vertical on level ground and about
   74° in the oblique stance on a step. The drive is pinned to zero so the
   pivot wheel rolls freely up to the riser; the 4WD tipping constraint
   and the pitch soft limit are relaxed.
3. **Roll**: the MPC switches to the single-contact model with the front
   wheel as pivot and a leg reference that sweeps forward. Past the top the
   reference is kept a fixed lead ahead of the measurement, and the pitch
   reference follows the leg geometry with a rate limit, so the MPC never
   brakes the leg or arrests the body with hub torque before the wheel is
   down (either torque rolls the cluster back off the tread). Odometry is
   pinned while the pivot wheel is blocked.
4. **Landing**: detected from the hub encoders, whose rate collapses within
   about 20 ms when the upper wheel hits the tread, with the geometric
   landing angle and a stall detector as fallbacks. The MPC returns to the
   locked-leg model and settles the body upright in the new, oblique,
   two-contact stance.

Descending uses the same manoeuvre with a negative step height
(`step_down_enabled`); the position is then held during the lean, because
there is no riser to keep the pivot wheel at the edge. Drops deeper than
`step_max_drop`, and any obstacle in 2WD or mid-transition, make the
planner hold position in front of it instead.

Headless results on the configured two-step staircase (5 cm risers, 20 cm
treads, 10 cm drop at the far end), sensor noise on, 14 s runs:

| Case | Outcome |
|------|---------|
| 4WD, 0.15 to 0.5 m/s, step-down on | climbs both steps, rolls down the far edge, drives on; 12 of 12 runs, peak pitch 32° (the deliberate lean) |
| 4WD, step-down off (`step_down_enabled=False`) | climbs both steps and holds at the top edge; 3 of 3 |
| 2WD into the riser | stops in front of it |

Each step is one 120° cluster roll, so the hub encoders read exactly three
rolls after a full traverse. On the real robot the terrain probe
corresponds to the forward-looking ToF sensor in the hardware list.

## Sign conventions for outputs

Positive drive torque drives forward; positive hub torque increases the
hub joint angle (bottom wheel moves forward). Yaw follows the existing
convention: `left = u_d/2 − corr`, `right = u_d/2 + corr`.

## Results (headless, sensor noise on, 8 s runs)

| Scenario | Outcome |
|----------|---------|
| Balance 4WD / 2WD | peak pitch 1.7° (initial offset), no saturation |
| Drive 0.5 m/s + yaw 1 rad/s, 4WD / 2WD | tracks, yaw rate reaches setpoint |
| 4WD → 2WD → drive → 4WD | peak pitch 3.7°, ends at hub angle 0 |
| Pushes 1.5 to 2.5 N·s (4WD), 0.8 to 1.5 N·s (2WD) | peak pitch ≤ 6° |
| Forward push in 2WD, flip disabled | recovers up to 14 N·s (LQR baseline fails at 8 N·s); beyond 8 N·s it settles into a 4WD stance by rolling the leg |
| Same with flip enabled | flip fires only above the trigger; it did not rescue a 15 N·s hit |

The explicit flip is therefore a last-resort reflex; on flat ground the
MPC's use of the leg covers every push tested before the flip becomes
relevant. Stairs are handled by the planned step manoeuvre described
below, not by the flip.

## Not implemented from the plan

- The EKF for pitch and velocity: the sim's IMU complementary filter and
  wheel odometry are used directly. Wheel odometry was corrected to use the
  absolute wheel spin (encoder + hub rate + pitch rate), which matters
  during transitions and flips.
- Slope handling. Stairs with risers up to `step_max_height` (9 cm) and
  drops up to `step_max_drop` (12 cm) are handled; taller obstacles stop
  the robot.
- Code generation for an embedded target; the OSQP problem is the
  reference formulation to port.

## Tuning knobs (`config.py`, `MPCConfig`)

`q_diag`, `r_diag`, `rd_diag` set the actuator allocation; raising the
leg weight makes the hub hold the leg stiffer, raising `r_diag[1]` pushes
pitch correction onto the wheels. `theta_soft_limit` bounds the lean the
MPC will plan. `tipping_safety` limits hub torque in 4WD.
`transition_time`, `flip_time`, `flip_theta_trigger` shape the event layer.

Note on solver failures: a stair traversal logs roughly 15 to 35 QP
failures (out of ~2800 solves) during the rolls, where the linearisation
point moves quickly. Each one is covered by the previous plan shifted one
step, and none coincided with a fall in the runs above.
