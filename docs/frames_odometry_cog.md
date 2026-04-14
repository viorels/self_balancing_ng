# Frames, Coordinates & Odometry

## Frame Definitions

| Frame | Origin | Orientation | Notes |
|-------|--------|-------------|-------|
| **base_link** | Hub axis midpoint (between L/R triplet joints) | Body-fixed, pitches/rolls with body | Height above ground: 0.118 m (4WD) to 0.178 m (2WD) |
| **base_footprint** | base_link projected onto ground (z=0) | Yaw only, no pitch/roll | Odometry frame — the robot's 2D pose |
| **imu_link** | Co-located with base_link (IMU mounted on body) | Same as base_link | Provides pitch, pitch_rate, yaw_rate |

Total CoG is 0.198 m above base_link in body Z (mass-weighted: body@0.247 m, hubs@0, wheels@variable). Fixed in body frame — triplet rotation doesn't shift it because 3 equal wheels at 120° always centroid at hub center.

## Global & Axis Coordinate System (REP-103 / FLU)

```
       +X (Forward)
        ↑
        │
+Y ←────┼ (Left)
        │
        +Z = Up (out of page)
```

Right-hand rule. Matches ROS REP-103 (Forward-Left-Up).

## Rotational Conventions (Euler Angles)

| Angle | Axis | Positive direction | Sensor |
|-------|------|--------------------|--------|
| **Roll** | X | Right side down | IMU (not actively controlled) |
| **Pitch (θ)** | Y | Lean forward (destabilising) | IMU complementary filter |
| **Yaw (ψ)** | Z | Turn left (CCW from above) | Integrated from differential wheel velocity |

Euler order: extrinsic XYZ (roll → pitch → yaw). Small-angle regime for pitch during balance (< 10°).

## Triplet Joint Angles & Zero States

Both triplet joints rotate about **+Y axis** (same as pitch).

| Joint | Zero state (φ=0°) | Positive rotation | 2WD target |
|-------|-------------------|-------------------|------------|
| **Left** (`c_body_to_l_triplet`) | 4WD — wheels 2&3 grounded | Forward (top wheel tips forward) | +60° |
| **Right** (`c_body_to_r_triplet`) | 4WD — wheels 2&3 grounded | Forward (same +Y axis) | +60° |

Combined triplet angle: **φ = (φ_L + φ_R) / 2**. Triplet motors are mirrored so right motor receives inverted torque (L gets +τ, R gets −τ). But the LQR controller sends symmetric torque commands, the sign is abstracthed away in motor controller.

Three wheels per side at base angles on the triplet circle (circumradius R=0.12 m, XZ plane):
- W1: 90° (top) — W2: 210° (bottom-rear) — W3: 330° (bottom-front)

## Odometry

### Principle

Wheel encoders measure rolling velocity at the contact patch. Triplet encoders measure hub rotation relative to body. When the triplet rotates, the hub (and body) translates relative to the grounded contact — a geometric correction that must be added to wheel-only odometry.

### Equations

Per-side velocity (wheel + triplet correction), computed for L and R independently:
```
v = -r·ω + R·sin(θ_ref + φ)·dφ/dt
```
where r=0.058 m (wheel radius), R=0.12 m (circumradius), θ_ref = base angle of grounded wheel, φ = triplet angle, ω = wheel angular velocity. Each side has its own φ, ω, and θ_ref.

θ_ref is selected by which wheel is lowest (three 120° sectors of φ mod 360°). The correction is continuous at all sector boundaries.

Forward velocity, yaw, and 2D pose:
```
v_fwd    = (v_L + v_R) / 2
ω_yaw    = (v_R − v_L) / track_width       track_width = 0.305 m
yaw     += ω_yaw · dt
x       += v_fwd · cos(yaw) · dt
y       += v_fwd · sin(yaw) · dt
```

Base-link height (continuous through mode transitions):
```
z_base = r − R·sin(θ_ref + φ)              averaged over L/R
```

### Key property

For asymmetric transitions (L→+60°, R→−60°), the L and R corrections cancel — wheel-only odometry is already correct. The correction matters for symmetric transitions, lean operations, and emergency flips (~0.2 m displacement per 120° flip).
