# Tribot — Triplet-Wheel Self-Balancing Robot

## Purpose

An inverted-pendulum robot that balances on wheel clusters ("triplets"). It can operate in **4WD** (two wheels per side grounded — stable polygon) or **2WD** (one wheel per side — active balance required), and transition between them by rotating the triplet assemblies 60°. It must also survive falls by executing emergency 120° "flips" to plant a fresh wheel before it crashes.

## Physical Structure

- **Body** (`c_body`): 2.717 kg, I_yy = 0.056 kg·m² (pitch, about CoG), CoG at (0, 0, 0.247) m above hub axis.
- **Triplet hubs** (L at y = −0.0825 m, R at y = +0.0825 m): each 0.254 kg, continuous joint about **+Y axis**. Circumradius R = 0.12 m.
- **Drive wheels** (3 per side): 0.027 kg each, radius 0.058 m, belt-coupled (1:1 gear ratio — all 3 spin together). Wheel 1 at (0, ±0.07, +0.12), wheel 2/3 at (∓0.104, ±0.07, −0.060).

Total mass ≈ 3.39 kg. Ground height in 4WD ≈ 0.118 m.

## Coordinate Frame & Sign Conventions

| Axis | Direction | Positive motion |
|------|-----------|-----------------|
| **X** | Forward | Robot drives forward |
| **Y** | Left | — |
| **Z** | Up | — |

- **Pitch (θ)**: rotation about Y. Positive = lean forward (destabilising).
- **Triplet angle (φ)**: rotation about Y relative to body. Both joints share `<axis xyz="0 1 0"/>`.
- **Yaw**: rotation about Z. Controlled via differential wheel torque.

## Actuators

| Motor | Location | Limit | Action |
|-------|----------|-------|--------|
| **Drive motor** (×2) | One per side, on body, drives all 3 wheels through belt | 1.0 Nm stall | Balances pitch, controls position and yaw |
| **Triplet motor** (×2) | One per side, rotates the triplet hub relative to body | 5.0 Nm | Mode transitions (4WD↔2WD) and 2WD triplet hold |

Drive motors have realistic lag (τ = 3 ms), back-EMF (0.005 Nm/(rad/s)), cogging (14-pole, 0.005 Nm), and deadband (0.01 Nm).

## Torque Application — The Free-Hub Trick

The URDF chain is `body → triplet_hub → wheels`. The drive motor stator is on the body. To make the hub free-spinning (transparent to drive torque), the sim applies:

```
wheel joints:    +motor_torque / 3  (split across 3 belt-coupled wheels)
triplet joint:   -motor_torque + triplet_cmd
```

`-motor_torque` cancels the wheel reaction, so the hub only feels `triplet_cmd`. The body receives the drive motor reaction (−motor_torque) plus the triplet motor reaction (−triplet_cmd). This yields the generalised-force vectors:

```
τ_drive:   Q = [+1/r, +1,  0]   (ground force + body reaction, nothing on hub)
τ_triplet: Q = [ 0,   −1, +1]   (body reaction + hub drive)
```

## L/R Triplet Geometry — Symmetric Convention

Both triplet joints share the **same +Y axis**, and the two triplets are **identical in the sagittal (X–Z) plane**: the wheels sit at the same `(x, z)` offsets on each side and differ only in the `y` (hub) offset — L hub at y = −0.0825, R at y = +0.0825. Because the geometry is identical (**not** mirrored) in the plane of rotation, a **same-sign** command produces the **same** physical rotation on both sides.

- **Same torque sign on both sides** → synchronized rotation. No R sign flip.
- Combined angle: **φ = (angle_L + angle_R) / 2**
- 4WD target: L = 0°, R = 0° → φ = 0°
- 2WD target: L = R = +60° (or both −60°) → φ = ±60°

The only genuinely anti-symmetric term is **yaw steering**: the drive command is split `left = cmd − yaw_corr`, `right = cmd + yaw_corr` for differential turning. That is layered on top of the symmetric balance/drive command and is unrelated to triplet synchronization.

## Balancing Modes

### 4WD Mode (φ = 0°)

Two wheels per side touch the ground (wheels 2 & 3 at x = ±0.104 m from hub axis) — a wide support polygon. The robot is passively stable in roll.

4WD admits three distinct control sub-modes for pitch:

#### 4WD-A: Triplet-Only Balancing

The triplet motors tilt the body relative to the grounded wheel base. The drive wheels receive **speed commands only** (velocity control, no balance feedback). The body is balanced by keeping its CoG projection within the support polygon.

**Acceleration limit** — The maximum sustainable acceleration before the robot tips off the support polygon edge is:

```
a_max = g · d / h

where:
    d = support polygon half-length in drive direction  (0.104 m)
    h = total CoG height above ground                   (≈ 0.365 m)

a_max ≈ 9.81 × 0.104 / 0.365 ≈ 2.80 m/s²
```

This comes from the moment balance at the tipping edge: the gravitational restoring moment $m g d$ must exceed the inertial overturning moment $m a h$. Shorter robots with wider bases can accelerate harder.

The triplet motors (5.0 Nm) have ample torque to hold the body lean. At steady acceleration, the required hold torque is $τ_{hold} ≈ m_b · a · L_{CoG} ≈ 2.72 \times 2.80 \times 0.247 ≈ 1.88$ Nm — well within budget.

**Advantages**: Simple wheel control (just velocity), high torque authority from triplet motors, no balance instability. **Disadvantage**: acceleration is hard-limited by the static tipping geometry.

#### 4WD-B: Wheel-Only Balancing (Cart–Pendulum)

The triplet motors are **idle**. The drive wheels perform LQR balance. The system becomes a classical cart–pendulum problem:

```
                ┌───────┐
                │ c_body│  ← pendulum (m_b = 2.72 kg)
                │       │     pivots about hub axis
                └───┬───┘
                    │  L_p = 0.247 m (CoG above hub axis)
              ──────┼──────
              │  triplet  │  ← "cart" (hub + 3 wheels)
              │  (stable) │     rolls on 2 ground contacts
              ▼▼▼▼▼▼▼▼▼▼▼▼
            ──────────────────  ground
```

The critical insight: the **effective pendulum length is measured from the hub (triplet) axis**, not from the ground. In 4WD-B the pole is just the body ($L_p = 0.247$ m, $I_{eff} = 0.222$ kg·m²), whereas in 2WD the pole is the entire robot minus the grounded wheels ($L_p = 0.381$ m composite CoG, $I_{eff} = 0.574$ kg·m²). The shorter, lighter pole in 4WD-B means:

- **Faster natural frequency**: $ω_0 = \sqrt{m g L / I}$ → 5.45 rad/s (4WD-B) vs 4.66 rad/s (2WD) — 17% faster divergence, but also 17% more responsive to corrections.
- **Higher angular acceleration per unit torque**: the wheel torque acts through the ground contact, creating a horizontal force on the cart. The moment arm to the pendulum pivot is smaller (just the hub height 0.118 m vs 0 in 2WD), changing the effective B matrix.
- **Drive motor is the bottleneck**: only 1.0 Nm available, but the shorter pendulum needs less torque for the same angular correction: $τ_{balance} ∝ m_b · L_p · \ddot{θ}$.

The cart mass is: $m_{cart} = m_{hub} + 3 \cdot m_{wheel} = 0.254 + 3 \times 0.027 = 0.335$ kg per side, 0.670 kg total.

**Advantages**: No triplet motor power consumption, standard LQR, unlimited dynamic acceleration (can lean beyond the static base). **Disadvantage**: drive motor saturation limits recovery authority (same 1.0 Nm budget as 2WD), and the system is actively unstable.

#### 4WD-C: Hybrid (Triplet + Wheel) — Authority-Weighted Blend (continuous, adaptive)

Both actuators contribute to pitch control simultaneously.
A single LQR over the full state, but with an **acceleration-dependent weighting** that smoothly shifts authority between actuators:

```
α = clamp(|a_demand| / a_static_max, 0, 1)

At α ≈ 0  (low accel):  Triplet dominates  → quasi-static lean, wheels track speed
At α ≈ 1  (high accel): Wheels dominate    → full cart-pendulum LQR
Intermediate:            Both contribute    → shared pitch authority

On slopes: α biased higher (static lean partially consumed by gravity),
           triplet handles gravity offset, wheels handle dynamic margin
```

The weighting can also account for slope angle: on a slope, part of the triplet's lean budget is consumed compensating gravity, so the wheels must engage earlier. The transition is smooth — no mode-switching discontinuities.

**Good for**: all-terrain operation, particularly mixed terrain with varying slopes and acceleration demands. Best position tracking because the controller always has dynamic authority beyond the static limit. Most complex to tune (full MIMO LQR with 2 actuator channels).

### 2WD Mode (φ = 60°)

One wheel per side touches the ground. The system is an actively-unstable inverted pendulum (see 2WD Cart–Pendulum parameters below).

**Two concurrent control loops run simultaneously:**

1. **Pitch balance (LQR on drive wheels)** — identical algorithm to 4WD-B, but with a different plant: the pole is the entire robot above ground contact (composite CoG = 0.381 m, I_eff = 0.574 kg·m²), and the pivot is the ground contact, not the hub axis.

2. **Triplet hold (PD on triplet motors)** — a separate PD controller holds each hub at ±60° with a gravity-compensation feedforward. The feedforward cancels the coupling torque that body pitch creates at the hub joint: when the body pitches by θ, the body weight (acting at CoG = 0.247 m above hub) creates a moment ≈ m_body·g·L·sin(θ) ≈ 6.58·θ Nm at the hub, which the triplet motor must resist. The triplet's own mass contributes zero gravity torque about the hub axis (three wheels at 120° spacing + hub CoG at center = net offset zero).

**Triplet–pitch coupling**: any triplet rotation shifts the ground contact point (see Ground-Contact Physics), which pitches the body. The current PD controller treats this as a disturbance. A future MIMO LQR could instead use triplet rotation as a **second balance actuator** — the triplet motor's 5.0 Nm budget is 5× the drive motor's, giving much greater recovery authority at extreme pitch angles.

### Asymetric Mode Transition (4WD → 2WD or reverse)

The triplet reference ramps from 0° to 60° (or back).
1. Left triplet rotates backward, right rotates forward (mirrored on X axis). The new supporting wheels come from opposite directions towards the base center (in symetry around Y axis) without creating any disturbance in the robot pitch.
2. Simultaneously compensate the yaw disturbance from supporting wheels moving in opposite directinos by adding extra torque in the direction towards the center of the robot (on X axis)


### Symetric Mode Transition (4WD → 2WD or reverse)

Both triplets could rotate same direction to transition, but that requires a lean of the robot, to bring the total CoG above the supporting wheel, so the transition is smooth.

### Emergency Flip (120° triplet rotation)

When a fall is detected (via DCM/ZMP analysis), the triplet executes a fast 120° rotation to plant the next wheel before the robot crashes. Theoretical minimum ~63 ms (bang-bang), configured ~230 ms including margins. After landing, the balance controller arrests residual pitch velocity.

## Ground-Contact Physics

Both actuators create pitch **indirectly**, by moving the support point relative to the body's centre of gravity. Gravity does the rest.

**Drive wheels**: Spinning the wheels accelerates the lower part of the robot horizontally. The body's inertia resists, creating a pitch torque as a side effect. This is the standard inverted-pendulum mechanism — it works identically in 4WD and 2WD, differing only in the pivot point (hub axis vs ground contact) and therefore the effective pendulum length and inertia.

**Triplet rotation**: In 4WD, as the triplet hub rotates the front OR back grounded wheels push more against the ground, shifting a *virtual support point* along the contact line (between the two grounded wheels per side). When the support point moves forward of the CoG projection, gravity pitches the body backward, and vice versa. In 2WD, triplet rotation physically relocates the single ground contact point forward or backward under the robot — same effect, but the shift is the actual contact position rather than a virtual point within a polygon.

Both mechanisms reduce to the same abstraction: **any actuator motion that displaces the support point relative to the CoG projection creates a gravitational moment that pitches the body**. The drive wheels do this by accelerating the base (inertial reaction), and the triplet does this by geometrically repositioning where the ground pushes up.

## Terrain

| Type | Description |
|------|-------------|
| `flat` | Uniform plane, friction 1.0 |
| `heightfield` | Procedural, flat spawn zone, ascending stairs (+X) and descending (−X), 15 mm steps |
| `box_stairs` | Sharp box primitives, 100–150 mm steps, 200 mm tread depth |

Stair negotiation changes the flip dynamics: shorter rotation needed (wheel lands on the step edge sooner) and shorter effective pendulum length.

## Sensors

- **IMU**: MEMS model with complementary filter fusion. Gyro drift (0.001 rad/s/√s), accel vibration noise (0.15 m/s²), 16-bit quantisation, 500 Hz sample rate.
- **Triplet encoders**: joint angle and velocity from each hub.
- **Wheel encoders**: joint angle and velocity (forward position estimated by integration).
- (Optional) **ToF/depth sensor**: obstacle hint flag, allows earlier flip trigger arming.

## Key Physical Parameters

### General

```
Body mass:          2.717 kg        Body I_yy (about CoG):  0.056 kg·m²
Hub mass (each):    0.254 kg        Hub I_yy (about CoG):   0.0021 kg·m²
Wheel mass (each):  0.027 kg        Wheel radius:           0.058 m
Total mass:         3.39 kg         Triplet circumradius:   0.12 m
Body CoG above hub: 0.247 m         Joint damp:             0.05 Nm·s/rad
Drive motor limit:  1.0 Nm          Trip motor:             5.0 Nm
```

### 4WD Cart–Pendulum (pivot = hub axis)

The cart is the entire triplet assembly rolling on two ground contacts.
The pole is the body only, pivoting about the hub axis.

```
Cart = 2×(hub + 3 wheels)   = 0.670 kg
Pole = body (c_body)        = 2.717 kg

Hub above ground:              0.118 m   (wheel_z=0.060 + wheel_r=0.058)
Body CoG above ground:         0.365 m   (0.247 + 0.118)
Pole CoG above pivot:          0.247 m
Pole I_yy about pivot:         0.222 kg·m²  (0.056 + 2.717×0.247²)
ω₀ = √(m·g·L / I):              5.45 rad/s
```

### 2WD Cart–Pendulum (pivot = ground contact)

The cart is two grounded wheels (one per side) — nearly massless.
The pole is the entire robot above the contact: body + both hubs + 4 non-grounded wheels.

```
Cart = 2 grounded wheels     = 0.054 kg
Pole = body + 2 hubs + 4 whl = 3.333 kg

Hub above ground:              0.178 m   (wheel_z=0.120 + wheel_r=0.058)
Body CoG above ground:         0.425 m   (0.247 + 0.178)
Pole composite CoG above GND:  0.381 m   (mass-weighted: body@0.425, hubs@0.178, wheels@0.238)
Pole I_yy about pivot:         0.574 kg·m²  (body: 0.547, hubs: 0.020, wheels: 0.006)
ω₀ = √(m·g·L / I):              4.66 rad/s
```

Note on 2WD geometry: at φ = 60°, the grounded wheel (originally at 210°) rotates to 270° on the triplet circle, placing its center at z = −0.12 m below hub. Adding wheel radius (0.058 m) gives hub height = 0.178 m. The non-grounded wheels (at 150° and 30°) sit at z = +0.06 m above hub = 0.238 m above ground.

## Common Pitfalls for Controller Development

1. **Torque budget is tiny** — 1.0 Nm drive motors on a 0.247 m pendulum. High gains saturate instantly; any K_pitch > ~12 or K_rate > ~2 will bang-bang.
2. **Triplet is NOT a reaction wheel** — it pushes on the ground. The mass matrix must include rolling inertia (β·R² terms) or the model underestimates how hard the triplet is to rotate.
3. **L/R triplets are symmetric, not mirrored** — same +Y axis and identical sagittal (X–Z) geometry, so the same torque sign rotates both the same way. Do **not** flip the R sign for triplet angle or drive torque; the combined triplet angle is `(L + R) / 2`. The only differential term is yaw steering (`left = cmd − yaw_corr`, `right = cmd + yaw_corr`).
4. **Drive motor cancellation on hub** — the sim explicitly zeroes the wheel motor reaction on the triplet hub (`-motor_torque + triplet_cmd`). The B_gf matrix must match this convention, not textbook "motor-on-joint" physics.
5. **4WD ↔ 2WD transitions are low-friction** — the grounded wheels roll freely in the direction of triplet rotation (X axis), so there is no Coulomb friction to overcome. The transition torque is dominated by inertia and gravity (weight shifting between wheels), not sliding friction.
6. **Gravity coupling on triplet** — zero in 4WD (bilateral support), nonzero in 2WD (≈0.39 Nm/rad). The plant model should switch or schedule this.
