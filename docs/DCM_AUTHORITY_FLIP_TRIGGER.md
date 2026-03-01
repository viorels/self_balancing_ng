# DCM Authority-Gated Flip Trigger (DAFT)

## A Predictive Recovery Strategy for Wheeled Inverted Pendulums with Discrete Support Transitions

---

## 1. What This Is (and What It Isn't)

### 1.1 Name

**DCM Authority-Gated Flip Trigger (DAFT)**

- **DCM**: Divergent Component of Motion — the exponentially-unstable part of
  the inverted pendulum state
- **Authority-Gated**: the trigger fires only when the DCM exceeds what the
  drive actuator can arrest, not at a fixed angle threshold
- **Flip Trigger**: commands a discrete support-point transition (triplet
  rotation, leg step, etc.)

### 1.2 Relationship to ZMP / Capture Point Theory

| Concept | Classical ZMP / Capture Point | DAFT |
|---------|-------------------------------|------|
| **Origin** | Bipedal walking robots (Honda, Boston Dynamics) | Wheeled inverted pendulums with discrete support changes |
| **Support** | Flat foot (static polygon) | Rolling wheel (zero-area, moving contact) |
| **Placement** | Foot placed exactly at capture point ξ | Next support is at a fixed geometric position (discrete) |
| **Trigger** | ξ exits support polygon → step | ξ exceeds drive authority envelope → flip |
| **Post-landing** | Passive capture (ξ inside new polygon → stable) | Active balance required (wheel keeps rolling) |
| **Degrees of freedom** | Step distance + step time | Only step time (distance is fixed by geometry) |

Classical capture-point theory assumes the robot can **choose where to place
the foot**. Our robot cannot — the next wheel position is dictated by triplet
geometry (120° apart on a fixed radius). We can only choose **when** to
initiate the transition.

Classical theory also assumes the foot is a **static contact point** after
landing. Our ground contact is a **rolling wheel** that continues to move
under drive torque. There is no passive capture — the balance controller must
actively stabilise after every transition.

DAFT answers a different question: not "where should I step?" but **"when has
the fall exceeded what my drive motor can arrest, requiring a support
transition to survive?"**

### 1.3 Key Insight

The inverted pendulum's drive actuator (wheel motor) can arrest falls up to
some angle θ_eq that depends on available torque. Falls within this envelope
are handled by normal balance control. Falls beyond it are unrecoverable
without changing the support point. DAFT computes the boundary between these
two regimes in real time and triggers the support transition at the last
responsible moment.

---

## 2. Physics Derivation

### 2.1 System Model

Consider a rigid-body inverted pendulum (the robot body) pivoting about a
wheel contact point on the ground.

**Parameters:**

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| m_b | Body mass | 2.72 kg |
| L | CoG height above wheel axis | 0.247 m |
| I_b | Body pitch inertia about CoG | 0.0564 kg·m² |
| I_w | Wheel/triplet rotational inertia | 0.00238 kg·m² |
| R_w | Wheel radius | 0.058 m |
| R_tri | Triplet circumradius | 0.12 m |
| τ_max | Max drive motor torque | 1.0 Nm |
| τ_trip | Max triplet motor torque | 5.0 Nm |
| g | Gravitational acceleration | 9.81 m/s² |
| b_pd | Effective PD gain (pitch → drive torque) | ~49.5 Nm/rad |

**Effective inertia** about the contact point (parallel axis theorem):

```
I_eff = I_b + m_b × L²
      = 0.0564 + 2.72 × 0.247²
      = 0.0564 + 0.1659
      = 0.2224 kg·m²
```

**Rigid-body natural frequency** (NOT the point-mass approximation):

```
ω₀ = √(m_b × g × L / I_eff)
   = √(2.72 × 9.81 × 0.247 / 0.2224)
   = √(6.591 / 0.2224)
   = √29.64
   = 5.44 rad/s
```

Note: the point-mass LIPM gives ω₀ = √(g/L) = 6.30 rad/s, which is 15.8%
too fast. This matters for timing — a 16% error in ω₀ means the trigger fires
100+ ms too early or too late.

### 2.2 Divergent Component of Motion (DCM)

The linearised equation of motion for the inverted pendulum is:

```
I_eff × θ̈ = m_b × g × L × sin(θ) − τ_drive
```

Without control input (τ_drive = 0), the state-space has one stable and one
unstable eigenvalue at ±ω₀. The DCM isolates the unstable component:

```
ξ = L × sin(θ) + (L × cos(θ) × θ̇) / ω₀
```

Under free fall (no drive torque), the DCM evolves as:

```
ξ(t) = ξ(0) × exp(ω₀ × t)
```

This is the "point of no return" position — if a static support were placed
at position ξ, the pendulum would exactly converge to upright. Since our
support is a rolling wheel (not static), ξ is instead the boundary between
"drive can handle it" and "drive cannot handle it."

### 2.3 Drive Authority Envelope

The balance controller (whether PD, LQR, or MPC) produces drive torque
proportional to pitch error. At maximum authority, it can statically hold
the pendulum at angle θ_eq where gravitational torque equals max drive
torque:

```
m_b × g × L × sin(θ_eq) = η × b_pd × θ_eq
```

where:
- b_pd is the effective pitch-to-torque gain of the balance controller
- η ∈ [0, 1] is the **control authority fraction** — how much of the
  theoretical max torque is actually available during a dynamic fall

For small angles, sin(θ) ≈ θ, so:

```
θ_eq = η × b_pd / (m_b × g × L)
```

But we want the large-angle version. The exact solution requires solving
the transcendental equation numerically, but a good closed-form
approximation uses:

```
θ_eq ≈ min(η × τ_eff / (m_b × g × L), θ_crash)
```

where:
- τ_eff = b_pd × 1 rad = effective torque at 1 radian of pitch error
- θ_crash is the mechanical limit beyond which recovery is impossible
  regardless of torque (typically 40–50° for an inverted pendulum due to
  geometric/contact constraints)

The **DCM authority boundary** is then:

```
ξ_max = L × sin(θ_eq)
```

This is the maximum DCM magnitude that the drive actuator can arrest. Any
|ξ| > ξ_max requires a support transition.

### 2.4 Authority-Shifted DCM

The raw DCM ξ is always nonzero during normal balance (the robot sways).
To avoid false triggers, we define the **excess DCM**:

```
ξ_excess = |ξ| − ξ_max
```

The trigger arms when ξ_excess > 0 (the fall has exceeded drive authority).

### 2.5 Time-to-Capture Prediction

Given ξ_excess > 0, we predict how long until the full DCM reaches the
crash limit:

```
ξ_crash = L × sin(θ_crash)

t_capture = (1/ω₀) × ln(ξ_crash / |ξ|)
```

The flip must complete before t_capture reaches zero.

### 2.6 Flip Time Budget

The total time budget for the support transition is:

```
T_budget = T_flip_nominal + T_flip_margin
```

where:
- T_flip_nominal: measured time for the triplet to rotate 120° under max
  torque (depends on motor, inertia, friction)
- T_flip_margin: safety margin for motor lag, belt compliance, etc.

The trigger fires when:

```
t_capture ≤ T_budget
```

### 2.7 Stair Climbing Correction

When climbing stairs, the next wheel lands on a surface that is h_step
higher than the current contact. This changes the effective pendulum
length:

```
L_eff = L + h_step / 2     (approximate — the CoG is higher relative
                             to the new contact point)
```

And modifies ω₀ accordingly:

```
I_eff_stair = I_b + m_b × L_eff²
ω₀_stair = √(m_b × g × L_eff / I_eff_stair)
```

The stair correction makes the pendulum slower (lower ω₀), which means
the fall develops more slowly and the trigger can fire slightly later —
giving the triplet motor more time to complete the rotation.

---

## 3. Finite State Machine

DAFT operates as a 4-phase FSM layered on top of the balance controller:

```
                    ┌──────────┐
                    │  NORMAL  │ ← normal 2WD balance
                    └────┬─────┘
                         │ t_capture < 1.5 × T_budget
                         │ AND |pitch_rate| > rate_gate
                         ▼
                    ┌──────────┐
                    │  ARMED   │ ← flip imminent, pre-lean if desired
                    └────┬─────┘
                         │ t_capture ≤ T_budget
                         ▼
                    ┌──────────┐
                    │ FLIPPING │ ← triplet rotating, aggressive tracking
                    └────┬─────┘
                         │ |pitch| < recover_threshold
                         │ AND triplet rotation > min_rotation
                         │  ── OR ──
                         │ triplet deviation < tolerance
                         ▼
                    ┌──────────┐
                    │ SETTLING │ ← hold new position, wait for oscillations
                    └────┬─────┘
                         │ settle_timer expires
                         │ → set cooldown_until = now + cooldown
                         ▼
                    ┌──────────┐
                    │  NORMAL  │
                    └──────────┘
```

### 3.1 Phase Details

**NORMAL**: Balance controller runs normally. DAFT computes ξ, ξ_excess,
and t_capture every cycle but takes no action. Transition to ARMED is
blocked during the cooldown window after a previous flip.

**ARMED**: The flip direction is locked (sign of ξ at arming time). The
pre-flip equilibrium is snapshot. The balance controller may optionally
add a small pitch bias in the fall direction to build momentum (reduces
flip time). Transitions to FLIPPING when t_capture ≤ T_budget.

**FLIPPING**: The triplet target is set to current_equilibrium + flip_dir
× 120°. Controller cost weights are reshaped:
- Triplet tracking weight increased (Q_trip: 40 → 120)
- Triplet torque cost decreased (R_trip: 1 → 0.05)
- Pitch tracking weight increased (Q_pitch: 50 → 120)
- Forward position tracking frozen (don't chase old target)

Exit conditions (checked every cycle):
1. **Early landing**: |pitch| < recover_threshold AND |triplet_rotation|
   > min_rotation — the new wheel touched down before the full 120°
2. **Full rotation**: |triplet_deviation_from_target| < tolerance

**SETTLING**: Equilibrium is advanced to the new position (snapped to
nearest valid 2WD angle). Target position is reset to current position
(prevents "run away" toward old target). Controller returns to normal
weights. A timer runs for T_settle seconds, then transitions to NORMAL
with a cooldown guard.

### 3.2 Cooldown and Rate Gate

Two mechanisms prevent oscillatory flip-flip-flip behaviour:

1. **Cooldown**: After SETTLING exits, re-arming is blocked for
   `cooldown` seconds (typically 0.8s). This gives the balance controller
   time to fully stabilise on the new wheel.

2. **Pitch rate gate**: NORMAL→ARMED requires |pitch_rate| >
   `min_fall_rate` (typically 15°/s). Normal balance sway rarely exceeds
   5–7°/s; genuine obstacle impacts produce 15–50°/s. This filters out
   slow sway that might momentarily push ξ_excess > 0.

---

## 4. Controller-Agnostic Interface

DAFT is **not tied to MPC**. It can work with any balance controller (PID,
LQR, MPC, RL policy) through a simple interface:

### 4.1 Required Inputs (every control cycle)

```python
@dataclass
class DAFTInputs:
    pitch: float          # body pitch angle (rad), forward positive
    pitch_rate: float     # body pitch angular velocity (rad/s)
    triplet_angle_L: float  # left triplet angle (rad)
    triplet_angle_R: float  # right triplet angle (rad)
    sim_time: float       # current time (s)
```

### 4.2 Outputs

```python
@dataclass
class DAFTOutputs:
    phase: int            # 0=NORMAL, 1=ARMED, 2=FLIPPING, 3=SETTLING
    should_flip: bool     # True on the cycle the flip is triggered
    flip_dir: float       # +1.0 or -1.0 (forward/backward)
    triplet_target: float # target triplet angle (rad) — only during FLIPPING
    urgency: float        # 0.0–1.0 scalar for telemetry / torque shaping
    freeze_position: bool # True during FLIPPING (don't chase old target)
    reset_position: bool  # True on SETTLING entry (latch current position)
    new_equilibrium: float  # new triplet equilibrium angle (rad)
```

### 4.3 Integration Pattern

```python
# --- Generic integration (works with PID, LQR, MPC, or any controller) ---

class BalanceControllerWithDAFT:
    def __init__(self, config):
        # Your existing balance controller
        self.inner_controller = YourBalanceController(config)

        # DAFT trigger (controller-agnostic)
        self.daft = DAFTrigger(config)

        # Triplet equilibrium (the "zero" angle for the triplet)
        self.triplet_equilibrium = config['INITIAL_TRIPLET_ANGLE']

    def update(self, pitch, pitch_rate, trip_L, trip_R, position, sim_time, dt):
        # 1. Run DAFT trigger
        daft_out = self.daft.update(
            pitch, pitch_rate, trip_L, trip_R, sim_time
        )

        # 2. Handle DAFT outputs
        if daft_out.reset_position:
            self.inner_controller.set_target_position(position)

        if daft_out.phase == SETTLING:
            self.triplet_equilibrium = daft_out.new_equilibrium

        # 3. Compute balance control (pitch → drive torque)
        if daft_out.freeze_position:
            # During flip: don't chase old position target
            drive_torque = self.inner_controller.update_pitch_only(
                pitch, pitch_rate, dt
            )
        else:
            drive_torque = self.inner_controller.update(
                pitch, pitch_rate, position, dt
            )

        # 4. Compute triplet torque
        if daft_out.phase == FLIPPING:
            # Aggressive PD to the flip target
            trip_error = daft_out.triplet_target - (trip_L + trip_R) / 2
            triplet_torque = KP_TRIP_FLIP * trip_error  # high gain
        elif daft_out.phase == SETTLING:
            # Gentle PD to hold new position
            trip_error = self.triplet_equilibrium - (trip_L + trip_R) / 2
            triplet_torque = KP_TRIP_NORMAL * trip_error
        else:
            # Normal: hold equilibrium
            trip_error = self.triplet_equilibrium - (trip_L + trip_R) / 2
            triplet_torque = KP_TRIP_NORMAL * trip_error

        return drive_torque, triplet_torque
```

### 4.4 What Changes Between Controllers

| Aspect | PID / LQR | MPC |
|--------|-----------|-----|
| Drive authority estimate (η) | Use controller gains directly: η = K_p × θ_eq / τ_max | MPC can estimate from dual variables, or use same fixed η |
| Cost reshaping during flip | N/A — use separate PD gains for flip | Rebuild QP with flip-mode Q/R matrices |
| Pitch bias during ARMED | Add offset to target_pitch | Add offset to x_ref[pitch] |
| Position freeze during FLIPPING | Disable outer position loop | Set Q_position = 0 in horizon |
| Triplet tracking | Separate PD loop on triplet error | Triplet states already in MPC state vector |

The **DAFT trigger itself** (DCM computation, authority envelope, FSM
transitions, timing) is identical regardless of which controller is used.

---

## 5. Tunable Parameters

### 5.1 Primary (must be set for each robot)

| Parameter | Description | How to Determine |
|-----------|-------------|------------------|
| `T_FLIP_NOMINAL` | Time for 120° triplet rotation (s) | Measure on hardware: command 120° step, record arrival time |
| `T_FLIP_MARGIN` | Safety margin on flip time (s) | Start at 0.05s, reduce if robot tilts too far before new wheel lands |
| `CTRL_AUTHORITY` (η) | Fraction of max drive torque available during fall | Start at 0.20. Raise toward 0.30 if false triggers during normal driving. Lower toward 0.10 if the robot crashes without triggering. |
| `THETA_CRASH` | Maximum recoverable angle (rad) | ~π/4 (45°) for most inverted pendulums. Lower if the robot has a high CoG or weak motors. |

### 5.2 Secondary (tune for behaviour quality)

| Parameter | Description | Default | Effect of Raising |
|-----------|-------------|---------|-------------------|
| `MIN_FALL_RATE_DEG_S` | Pitch rate gate for arming (°/s) | 15 | Fewer false triggers, but may miss slow-onset falls |
| `FLIP_COOLDOWN` | Post-flip rearm lockout (s) | 0.8 | More stable settling, but slower stair climbing |
| `T_SETTLE` | Settling window duration (s) | 0.4 | More time to stabilise, but slower response to next obstacle |
| `FLIP_MIN_ROTATION` | Minimum triplet travel for early exit (°) | 40 | Prevents premature exit, but delays transition if the geometry allows <40° landings |
| `PITCH_RECOVER_THRESHOLD` | Pitch threshold for early landing exit (rad) | 0.12 (~7°) | More conservative early exit |
| `STAIR_HEIGHT` | Expected step height (m) | 0.0 | Adjusts ω₀ for stair climbing; 0 = flat ground |

### 5.3 Tuning Procedure

1. **Start on flat ground.** Set `CTRL_AUTHORITY = 0.20`, push the robot
   by hand. It should NOT trigger a flip for pushes that the drive motor
   can handle. If it false-triggers, raise η to 0.25.

2. **Add obstacles.** Place a small obstacle (~2 cm) in the path. The
   robot should flip and recover. If it crashes without flipping, lower η
   to 0.15 or lower `MIN_FALL_RATE_DEG_S` to 10.

3. **Time the flip.** Watch the triplet rotation in telemetry. If the
   new wheel lands with significant delay, reduce `T_FLIP_MARGIN`. If
   it's still rotating when the robot has already recovered, the trigger
   is too early — raise η.

4. **Test stair climbing.** Set `STAIR_HEIGHT` to the step height. The
   corrected ω₀ will be slightly lower, giving the trigger more time.
   Reduce `FLIP_COOLDOWN` if the robot needs to take multiple steps in
   quick succession.

---

## 6. Comparison to Alternative Approaches

### 6.1 Fixed Pitch Threshold Trigger

```
if |pitch| > 15°: flip()
```

**Problem**: Ignores pitch rate. A robot at 14° with 50°/s pitch rate is
about to crash; a robot at 16° with 0°/s pitch rate might recover.
DAFT handles both cases correctly because DCM includes velocity.

### 6.2 Fixed DCM Threshold (Classical Capture Point)

```
if |ξ| > ξ_max: flip()
```

**Problem**: ξ_max must be hand-tuned. Too conservative → false triggers
during aggressive driving. Too aggressive → crashes. The threshold
depends on controller gains, motor health, battery voltage, and surface
friction — all of which change during operation. DAFT computes ξ_max
from first principles using the authority model.

### 6.3 Free-Fall LIPM (Point Mass)

```
ω₀ = √(g/L),  ξ = L×θ + L×θ̇/ω₀
```

**Problem**: Overestimates ω₀ by ~16% for typical robots
(L=0.247m → 6.30 vs 5.44 rad/s). This means the DCM appears to grow
faster than reality, causing the trigger to fire too early. DAFT uses the
rigid-body ω₀ which includes rotational inertia.

### 6.4 Pure MPC with Triplet in State Vector

The MPC could in principle decide when to flip by including a binary
"flip/don't flip" decision in the optimisation. This requires:
- Mixed-integer programming (NP-hard, not real-time on ESP32)
- Or a very long horizon to "see" the obstacle coming
- Or a pre-planned trajectory that assumes obstacle timing is known

DAFT is computationally trivial (one exp, one log, a few multiplies per
cycle) and works as a reactive layer that interrupts the MPC when needed.
The MPC handles smooth balance; DAFT handles discrete emergencies.

---

## 7. Applicability to Other Robots

DAFT applies to any robot that:

1. **Balances as an inverted pendulum** (wheeled or legged)
2. **Has discrete support transitions** (triplet flips, leg steps,
   training-wheel deployment)
3. **Cannot choose the landing position** freely (or has a small
   finite set of positions)

### 7.1 Examples

| Robot Type | Support Transition | What DAFT Controls |
|------------|-------------------|-------------------|
| Tri-wheel balancer (this robot) | 120° triplet rotation | When to rotate |
| Segway with deployable kickstand | Kickstand extension | When to deploy |
| Bicycle robot with outrigger | Outrigger swing-out | When to extend |
| Hexapod in bipedal stance | Leg placement | When to step (but not where — that's capture point theory) |

### 7.2 What to Change for a New Robot

1. **Compute I_eff and ω₀** from your robot's mass, CoG height, and
   rotational inertia.
2. **Measure T_flip_nominal** for your specific actuator and mechanism.
3. **Estimate η** from your balance controller's effective gain and
   actuator limits.
4. **Set θ_crash** based on your robot's geometry (when does it
   physically hit the ground?).
5. **Implement the FSM** with your controller's interface for cost
   reshaping / gain switching.

---

## 8. References

1. **Pratt, J. et al.** "Capture Point: A Step toward Humanoid Push
   Recovery." IEEE-RAS Humanoids, 2006. — Original capture point theory
   for bipeds with static feet.

2. **Takenaka, T. et al.** "Zero Moment Point." Honda R&D Technical
   Review, 2009. — ZMP criterion for walking stability.

3. **Kajita, S. et al.** "The 3D Linear Inverted Pendulum Mode."
   IROS 2001. — LIPM derivation and DCM concept.

4. **Englsberger, J. et al.** "Three-Dimensional Bipedal Walking Control
   Based on Divergent Component of Motion." IEEE TRO, 2015. — DCM-based
   walking control that inspired the authority-gated extension.

5. This work extends DCM theory to **rolling-contact inverted pendulums**
   where the support point is not static and the landing position is
   constrained to a discrete set. The authority-gating concept (η) and
   the rigid-body ω₀ correction are novel contributions specific to this
   class of robot.

---

## Appendix A: Quick-Reference Equations

```
# Rigid-body natural frequency
I_eff = I_body + m_body × L²
ω₀ = √(m_body × g × L / I_eff)

# DCM (exact, not small-angle)
ξ = L × sin(θ) + L × cos(θ) × θ̇ / ω₀

# Drive authority envelope
b_pd = K_p_pitch + K_d_pitch × ω₀        (effective PD gain at ω₀)
τ_eff = η × b_pd × 1 rad                  (effective max torque)
θ_eq = min(τ_eff / (m_b × g × L), θ_crash)
ξ_max = L × sin(θ_eq)

# Excess DCM (positive = beyond authority)
ξ_excess = |ξ| − ξ_max

# Time to crash (only valid when |ξ| > 0)
ξ_crash = L × sin(θ_crash)
t_capture = (1/ω₀) × ln(ξ_crash / |ξ|)

# Trigger condition
FLIP when t_capture ≤ T_budget = T_flip_nominal + T_flip_margin

# Stair correction
L_eff = L + h_step / 2
I_eff_stair = I_body + m_body × L_eff²
ω₀_stair = √(m_body × g × L_eff / I_eff_stair)
```

## Appendix B: Numerical Example (Tribot)

```
Given:
  m_b       = 2.7167 kg
  L         = 0.247 m
  I_b       = 0.0564 kg·m²
  g         = 9.81 m/s²
  τ_max     = 1.0 Nm (drive motor)
  K_p_pitch = 8.0 Nm/rad (MPC cross-drive gain)
  K_d_pitch = 0.5 Nm/(rad/s)
  η         = 0.20

Computed:
  I_eff   = 0.0564 + 2.7167 × 0.247² = 0.2222 kg·m²
  ω₀      = √(2.7167 × 9.81 × 0.247 / 0.2222) = 5.44 rad/s

  b_pd    = 8.0 + 0.5 × 5.44 = 10.72 Nm/rad
  τ_eff   = 0.20 × 10.72 = 2.14 Nm
  θ_eq    = min(2.14 / (2.7167 × 9.81 × 0.247), π/4)
           = min(2.14 / 6.59, 0.785)
           = min(0.325, 0.785) = 0.325 rad (18.6°)
  ξ_max   = 0.247 × sin(0.325) = 0.0789 m (78.9 mm)

During normal balance:
  θ ≈ 3°, θ̇ ≈ 5°/s
  ξ = 0.247 × sin(0.052) + 0.247 × cos(0.052) × 0.087 / 5.44
    = 0.0129 + 0.00395 = 0.0168 m (16.8 mm)
  ξ_excess = 16.8 − 78.9 = −62.1 mm → NORMAL (no trigger)

During obstacle impact:
  θ ≈ 12°, θ̇ ≈ 40°/s
  ξ = 0.247 × sin(0.209) + 0.247 × cos(0.209) × 0.698 / 5.44
    = 0.0512 + 0.0308 = 0.0820 m (82.0 mm)
  ξ_excess = 82.0 − 78.9 = 3.1 mm → ARM (exceeds authority)

  ξ_crash = 0.247 × sin(0.785) = 0.1747 m
  t_capture = (1/5.44) × ln(0.1747 / 0.0820) = 0.184 × 0.757 = 0.139 s

  T_budget = 0.18 + 0.05 = 0.23 s
  t_capture (0.139) < T_budget (0.23) → FLIP NOW
```