# Lean Transition Trajectory Planner

## Problem

When the operator commands a lean change (e.g. 0° → 21°), the LQR pitch
reference steps instantly to the new value.  The body cannot follow a step —
it must **fall** under gravity to reach the new angle.  During the free-fall:

- The LQR sees a large pitch error and commands the wheels **backward**
  (opposing the fall), but the body *needs* to fall to reach the target.
- The position term commands wheels **forward** (to stay under the CoG),
  but the pitch term overrides it because |K_pitch| >> |K_pos|.
- Net result: **0.3 m position excursion** per lean transition, with
  overshoot and settling oscillation.

## Failed Approaches

### 1. Follow-the-Triplet (EMA-filtered supported lean)
Compute the lean angle the actual triplet position supports via inverse
sine-theorem geometry, low-pass filter it (EMA, α = 0.05), and feed
that to the LQR as the pitch reference.

**Why it failed:** The free-spinning hub picks up motor reaction torque,
causing high-frequency triplet wobble.  Even with filtering, this
creates a positive-feedback loop: triplet wobble → LQR reference shift →
torque swing → more hub wobble.  The EMA filter also introduces 400 ms
lag during transitions, leaving the LQR reference far behind the actual
body during any commanded lean change.

### 2. Feedforward Torque Compensation
Estimate triplet angular acceleration via finite differences on the
encoder, multiply by hub inertia (I_eff ≈ 0.0035 kg·m²), and add the
result to the LQR command to cancel the hub's reaction on the body.

**Why it failed:** Finite-difference acceleration on a low-inertia joint
at 500 Hz produces ±1.3 Nm spikes — **exceeding motor capacity** (1.0 Nm).
The actuators saturate every tick, creating a sustained oscillation that
grows until the robot falls.

### 3. Faster EMA Filter (α = 0.5)
Reduce filter lag to ~4 ms.

**Why it failed:** The fundamental problem isn't filter delay — it's that
a step reference is physically infeasible.  The body falls under gravity
at ~9.81 m/s² × sin(θ), and the LQR pitch term fights this motion the
entire way because the reference is already at the final value.

## Solution: Minimum-Jerk Trajectory Planner

Instead of a step reference, generate a **smooth, physically feasible
trajectory** that the LQR can track with small errors throughout the
transition.

### Architecture

```
Gamepad → set_lean(θ_new)
            │
            ▼
    ┌──────────────────┐
    │  LeanTrajectory   │  minimum-jerk quintic profile
    │  s(τ)=10τ³-15τ⁴+6τ⁵  │
    └──────┬───────────┘
           │  [x_ref, ẋ_ref, θ_ref, θ̇_ref]
           ▼
    ┌──────────────────┐
    │  LQR Controller   │  state error = measured − ref
    │  u = −K·(x−x_ref) │  (including velocity and pitch_rate refs)
    └──────┬───────────┘
           │
           ├──→ Wheel motors (torque command)
           │
           ▼
    ┌──────────────────┐
    │  Triplet PD       │  target = compute_triplet_from_pitch(θ_ref)
    │  (tracks θ_ref)   │  (moves contact point smoothly during transition)
    └──────────────────┘
```

### Trajectory Profile

The minimum-jerk (quintic) profile has zero velocity and acceleration at
both endpoints:

```
s(τ)    = 10τ³ − 15τ⁴ + 6τ⁵         position interpolant
ṡ(τ)    = 30τ² − 60τ³ + 30τ⁴         velocity profile (bell-shaped)
```

where τ = t/T is normalised time.

**State references during transition:**

| Channel       | Formula                                      |
|---------------|----------------------------------------------|
| Position      | x₀ + Δx · s(τ), where Δx = h·(sin θ₁ − sin θ₀) |
| Velocity      | Δx · ṡ(τ)/T                                  |
| Pitch         | θ₀ + Δθ · s(τ)                               |
| Pitch rate    | Δθ · ṡ(τ)/T                                  |

The position reference accounts for the geometric CoG shift — wheels must
move Δx to stay under the body's centre of gravity at the new lean angle.

### Duration Auto-Scaling

Transition duration scales with lean magnitude:

```
T = 0.5 s × |Δθ| / 15°
```

Clamped to [0.3 s, 2.0 s].  Small lean nudges complete in 0.3 s; the
full 21° lean takes about 0.7 s.

### Key Properties

- **No position excursion**: wheels move only ~2 cm (vs 30 cm before)
- **No actuator saturation**: LQR sees small errors throughout, never
  needs to command full torque
- **Smooth triplet motion**: PD tracks the same trajectory pitch, so
  triplet and body move together
- **Zero residual**: trajectory endpoints match the new equilibrium
  exactly — no settling oscillation
- **Instant response**: trajectory starts immediately on lean command;
  1° dead-zone prevents retriggering from noise

## Files

| File | Role |
|------|------|
| `lean_trajectory.py` | `LeanTrajectory` class — quintic profile generator |
| `controllers/control_lqr.py` | Integrates trajectory into LQR state error |
| `robot.py` | Feeds trajectory pitch to triplet PD during transitions |

## Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `h_cog` | 0.19 m | CoG height — scales position reference Δx |
| `min_duration` | 0.3 s | Fastest allowed transition |
| `max_duration` | 2.0 s | Slowest allowed transition |
| `lean_per_sec_factor` | 0.5 s/15° | Duration scaling with lean magnitude |
| Dead-zone (in `set_lean`) | 1° | Minimum lean change to trigger trajectory |
