# Tribot — Triplet-Wheel Self-Balancing Robot

## Purpose

An inverted-pendulum robot that balances on wheel clusters ("triplets"). It can operate in **4WD** (two wheels per side grounded — stable polygon) or **2WD** (one wheel per side — active balance required), and transition between them by rotating the triplet assemblies 60°. It must also survive falls by executing emergency 120° "flips" to plant a fresh wheel before it crashes.

## Physical Structure

```
        ┌─────────┐
        │ c_body  │   ← main body, CoG at z = +0.247 m above axle
        │ 2.72 kg │
        └───┬─┬───┘
       Y−   │ │   Y+
    ┌───────┘ └───────┐
    │ L triplet       │ R triplet       ← hub joints, axis +Y
    │ 0.254 kg        │ 0.254 kg
    │  ▲  ▲  ▲        │  ▲  ▲  ▲       ← 3 wheels each, belt-coupled
    │ w1 w2 w3        │ w1 w2 w3         (0.027 kg per wheel, r = 0.058 m)
    └─────────────────┘
```

- **Body** (`c_body`): 2.717 kg, I_yy = 0.167 kg·m² (pitch), CoG at (0, 0, 0.247) m.
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

## L/R Mirror Geometry — Critical Gotcha

Both triplet joints use the **same +Y axis**, but L sits at y = −0.0825 and R at y = +0.0825. Physically mirrored. To produce the **same** flip direction on both sides:

- **L gets +τ**, **R gets −τ** (anti-symmetric torque split)
- Combined angle: **φ = (angle_L − angle_R) / 2**
- 4WD target: L = 0°, R = 0° → φ = 0°
- 2WD target: L = +60°, R = −60° → φ = +60°

Forgetting the sign flip produces opposite rotations instead of synchronized ones.

## Balancing Modes

### 4WD Mode (φ = 0°)

Two wheels per side touch the ground — a wide support polygon. The robot is passively stable in roll. Balance is only needed in pitch. Ground friction on both grounded wheels resists triplet rotation (~3.7 Nm/side of friction torque to overcome during transitions).

### 2WD Mode (φ = 60°)

One wheel per side touches the ground. The triplet motor must actively hold the hub angle against gravity (restoring torque ≈ m_trip·g·R ≈ 0.39 Nm/rad). Both pitch balance and triplet-angle hold run simultaneously. Triplet torque couples into body pitch: pushing the triplet tilts the body the opposite way.

### Mode Transition (4WD → 2WD or reverse)

The triplet reference ramps from 0° to 60° (or back). During the ramp, grounded wheels drag against the floor creating large friction loads. The controller must:
1. Apply enough triplet torque to overcome ground friction (~3.7 Nm/side)
2. Simultaneously compensate the pitch disturbance from triplet reaction torque
3. Allow intentional body lean (relaxed pitch gains) while keeping the robot recoverable

### Emergency Flip (120° triplet rotation)

When a fall is detected (via DCM/ZMP analysis), the triplet executes a fast 120° rotation to plant the next wheel before the robot crashes. Theoretical minimum ~63 ms (bang-bang), configured ~230 ms including margins. After landing, the balance controller arrests residual pitch velocity.

## Ground-Contact Physics

The triplet wheels are **not** reaction wheels — they push on the ground. When the triplet rotates:

- Ground wheel(s) roll, creating horizontal force on the robot
- The rolling constraint adds effective inertia: I_eff_trip = I_hub + β·R² where β ≈ 3·m_wheel (belt coupling spins all 3 wheels per side)
- This makes the triplet ~25% harder to rotate than a free-spinning flywheel
- The mass matrix has cross-coupling M_xφ ≠ 0 (triplet rotation creates horizontal acceleration)

In 4WD, **two** grounded wheels per side create bilateral support — no gravity restoring torque on φ. In 2WD, **one** grounded wheel creates a restoring/destabilising tendency depending on the lean direction.

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

```
Body mass:          2.717 kg        Body I_yy:    0.167 kg·m²
Total wheel mass:   0.670 kg        Wheel radius: 0.058 m
CoG height:         0.247 m         Triplet R:    0.12 m
Hub inertia (each): 0.00238 kg·m²   Joint damp:   0.05 Nm·s/rad
Drive motor limit:  1.0 Nm          Trip motor:   5.0 Nm
```

## Common Pitfalls for Controller Development

1. **Torque budget is tiny** — 1.0 Nm drive motors on a 0.247 m pendulum. High gains saturate instantly; any K_pitch > ~12 or K_rate > ~2 will bang-bang.
2. **Triplet is NOT a reaction wheel** — it pushes on the ground. The mass matrix must include rolling inertia (β·R² terms) or the model underestimates how hard the triplet is to rotate.
3. **L/R anti-symmetry** — same URDF axis but mirrored geometry. Every torque split and angle combination must flip the R sign.
4. **Drive motor cancellation on hub** — the sim explicitly zeroes the wheel motor reaction on the triplet hub (`-motor_torque + triplet_cmd`). The B_gf matrix must match this convention, not textbook "motor-on-joint" physics.
5. **Ground friction during 4WD transitions** — ~3.7 Nm/side. The linear model doesn't include Coulomb friction, so feedforward is needed.
6. **Gravity coupling on triplet** — zero in 4WD (bilateral support), nonzero in 2WD (≈0.39 Nm/rad). The plant model should switch or schedule this.
