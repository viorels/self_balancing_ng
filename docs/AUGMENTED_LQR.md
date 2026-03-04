# Augmented LQR Controller (`control_lqr_aug.py`)

## Overview

The augmented LQR controller extends the standard single-input inverted-pendulum LQR to produce **two independent torque commands**: one for the wheel motors and one for the triplet motors. Both act on the same 4-state plant, but through physically different mechanisms. The LQR cost function naturally determines the optimal split between them at every instant.

---

## Classical LQR — Single Input

The standard controller (`control_lqr.py`) uses the textbook inverted-pendulum-on-cart model:

**State:** $\mathbf{x} = [x,\ \dot{x},\ \theta,\ \dot{\theta}]^\top$

**Input:** $u = \tau_w$ (total wheel torque, both sides combined, Nm)

**Equation of motion:**

$$\begin{bmatrix} M_{tot} & -m_b l \\ -m_b l & I_{eff} \end{bmatrix} \begin{bmatrix} \ddot{x} \\ \ddot{\theta} \end{bmatrix} = \begin{bmatrix} 0 \\ m_b g l \end{bmatrix} \begin{bmatrix} x \\ \theta \end{bmatrix} + \begin{bmatrix} 1/r \\ 1 \end{bmatrix} \tau_w$$

The wheel motor torque $\tau_w$ creates:
- A **ground reaction force** $F = \tau_w / r$ that translates the cart
- A **stator reaction torque** on the body (Newton's 3rd law) via the motor mount

**LQR gain:** $K \in \mathbb{R}^{1\times4}$, solving $u = -K\mathbf{x}$

---

## Augmented LQR — Two Inputs

The augmented controller adds a second input $\tau_t$ (triplet motor torque).

**State:** same $\mathbf{x} = [x,\ \dot{x},\ \theta,\ \dot{\theta}]^\top$

**Input:** $\mathbf{u} = [\tau_w,\ \tau_t]^\top$

### How the triplet torque acts

The triplet assembly rests on the ground. When the triplet motor applies torque $\tau_t$ between the body and the triplet:

- The **grounded triplet** cannot rotate freely — ground friction holds it
- The reaction goes entirely into the **body**: the body gets $-\tau_t$ on its pitch axis
- There is **no horizontal force** on the cart ($Q_x = 0$ for $\tau_t$)

This is fundamentally different from a reaction wheel. A reaction wheel spins freely in air, so its torque is purely a momentum exchange. Here, the triplet is grounded, so the torque path is: _motor → triplet → ground reaction → body_. The effect is equivalent to an external torque applied directly to the body pitch axis.

### Extended equation of motion

$$\underbrace{\begin{bmatrix} M_{tot} & -m_b l \\ -m_b l & I_{eff} \end{bmatrix}}_{M} \begin{bmatrix} \ddot{x} \\ \ddot{\theta} \end{bmatrix} = \begin{bmatrix} 0 \\ m_b g l \end{bmatrix} \begin{bmatrix} x \\ \theta \end{bmatrix} + \underbrace{\begin{bmatrix} 1/r & 0 \\ 1 & -1 \end{bmatrix}}_{B_{gf}} \begin{bmatrix} \tau_w \\ \tau_t \end{bmatrix}$$

The generalised force matrix $B_{gf}$ has column structure:

| | $\tau_w$ | $\tau_t$ |
|---|---|---|
| $Q_x$ (horizontal) | $+1/r$ | $0$ |
| $Q_\theta$ (pitch) | $+1$ | $-1$ |

The sign of the triplet column is $-1$ because positive joint torque (child in +Y) produces negative pitch reaction on the body.

### State-space B matrix

$$B = M^{-1} B_{gf} = \begin{bmatrix} 0 & 0 \\ b_{1w} & b_{1t} \\ 0 & 0 \\ b_{3w} & b_{3t} \end{bmatrix}$$

With the tribot parameters (`r = 0.058 m`, `m_b = 2.717 kg`, etc.):

```
B (4×2):
[[  0.         0.      ]
 [ 14.900     -2.221   ]   ← ẍ per Nm
 [  0.         0.      ]
 [ 49.501    -11.209   ]]   ← θ̈ per Nm
```

**Key ratio:** one Nm of wheel torque produces $49.5\ \text{rad/s}^2$ pitch acceleration; one Nm of triplet torque produces $11.2\ \text{rad/s}^2$. Wheels are ~**4.4× more effective per Nm** for pitch correction, but the triplet motor has a much higher torque budget (5 Nm vs 1 Nm), so its total angular impulse is comparable.

**LQR gain:** $K \in \mathbb{R}^{2\times4}$

$$\mathbf{u} = -K\mathbf{x} = -\begin{bmatrix} K_{w,1} & K_{w,2} & K_{w,3} & K_{w,4} \\ K_{t,1} & K_{t,2} & K_{t,3} & K_{t,4} \end{bmatrix} \mathbf{x}$$

Row 0 drives the wheels, row 1 drives the triplets.

---

## Cost Function and Tuning

The LQR minimises:

$$J = \int_0^\infty \left( \mathbf{x}^\top Q \mathbf{x} + \mathbf{u}^\top R \mathbf{u} \right) dt$$

with:

$$Q = \text{diag}(q_x,\ q_{\dot{x}},\ q_\theta,\ q_{\dot\theta}), \qquad R = \text{diag}(R_w,\ R_t)$$

**$R$ is the primary tuning knob for torque allocation:**

| $R_w / R_t$ ratio | Effect |
|---|---|
| $R_w \ll R_t$ | LQR prefers wheels; triplet barely used |
| $R_w \gg R_t$ | LQR offloads balance to triplet; wheels reserved for locomotion |
| Equal | Allocation based purely on physical effectiveness |

Default setting: `ALQR_R_DIAG = [4.0, 0.5]` → triplet is 8× "cheaper" in the cost.

Resulting gains with default settings:

```
K_wheels  = [ 0.9054,  0.6337,  3.7322,  0.7702]
K_triplet = [ 4.1764,  3.8573, -13.0556, -2.7836]
            [  pos      vel      pitch    pitch_rate ]
```

Note the **opposite sign** on the triplet pitch gain ($-13.06$) vs the wheel pitch gain ($+3.73$): this reflects the $-1$ in $B_{gf}$ — to correct a positive pitch error, you command positive wheel torque but negative joint torque on the triplet (which produces a positive corrective body torque).

### R_triplet sweep

```
R_trip   K_w_pitch   K_t_pitch
  0.1      1.829     -28.018    ← triplet dominates completely
  0.3      3.058     -16.683
  0.5      3.732     -13.056    ← default
  1.0      4.774      -9.261
  2.0      5.949      -6.353
  4.0      7.135      -4.097
 10.0      8.399      -2.043    ← nearly equivalent to standard LQR
```

---

## Practical Considerations

### Triplet friction requirement

The triplet torque path relies on ground friction to hold the triplet stationary. The required friction force is:

$$F_{friction} \geq \frac{\tau_t}{R_{triplet}} \approx \frac{5\ \text{Nm}}{0.12\ \text{m}} = 41.7\ \text{N per side}$$

With $\mu = 1.2$ and normal force $\approx 13\ \text{N}$ per contact wheel, maximum static friction is $\approx 15.6\ \text{N}$ — **less than required at full triplet torque**. In practice, triplet slippage acts as a natural torque limiter and degrades gracefully (reduced authority rather than instability).

This means `ALQR_R_DIAG[1]` should be tuned to keep the triplet torque command well below the friction limit during normal balance, reserving higher commands only for transient disturbances.

### Triplet angle drift

Unlike a rolling wheel, a slipping or actively-driven triplet accumulates an angle offset. The existing `TripletTransitionController` handles deliberate 60° rotations (4WD↔2WD), but small drift from balance corrections is acceptable since the joint has full rotation freedom and the LQR has no angle state for the triplet.

If drift becomes a concern, add a slow integral reset term:

```python
# Soft reset: gentle restoring torque toward nominal angle
triplet_reset = -k_reset * (triplet_angle - nominal_angle)
triplet_cmd += triplet_reset  # add before clamping
```

### Hill climbing

On a slope, gravity creates a constant pitch disturbance. The standard LQR compensates entirely with wheel torque, leaving no margin for additional traction force. With the augmented controller, set a higher `R_wheels` value to shift steady-state pitch compensation to the triplet, freeing wheel torque for propulsion up the slope:

```python
# Aggressive hill-climbing mode
'ALQR_R_DIAG': [10.0, 0.3],  # triplet handles balance, wheels handle climbing
```

The physics: triplet torque controls pitch *without* exerting a net forward force, while wheel torque both corrects pitch *and* propels the robot. On flat ground this coupling is irrelevant; on a hill it's critical.

---

## Implementation Notes

- `control_lqr_aug.py`: `AugmentedLQRController` class, `build_augmented_state_space()`
- `control_lqr.py`: `compute_lqr_gain()` is reused (no scipy dependency)
- `tribot_sim.py`: `CONTROLLER: 'lqr_aug'` activates the augmented controller. The `triplet_torque_L` / `triplet_torque_R` attributes are read by `TribotBalanceBot.update()` via `getattr(..., 0.0)`, so the interface is backward-compatible with standard LQR and PID.
- PlotJuggler signals: `triplet_torque_cmd`, `triplet_torque_L`, `triplet_torque_R`, `K_trip_pos`, `K_trip_vel`, `K_trip_pitch`, `K_trip_pitch_rate`
- `scripts/lqr_augmented.py`: standalone analysis tool — prints A, B, K matrices and the R_triplet sweep table
- `scripts/derive_eom.py`: symbolic Lagrangian derivation — Part 1 verifies the B matrix analytically
