# Hybrid MPC + PD Stair Climbing Controller Architecture

## Document Purpose

This document specifies the full control architecture for the Tribot stair-climbing robot.
It is intended as a reference for implementing the FSM, MPC solver, PD tracking loop, and
all supporting infrastructure (state estimation, trajectory buffering, safety).

---

## 1. Robot Description

### 1.1 Physical Layout

- **Body** (`c_body`): Main chassis, mass 2.716678 kg, CoG at z=0.247 m above wheel axis
- **Left triplet** (`l_triplet`): 3-wheel cluster, mass 0.253511 kg, connected to body via continuous joint at y=-0.0825 m
- **Right triplet** (`r_triplet`): 3-wheel cluster, mass 0.253511 kg, connected to body via continuous joint at y=+0.0825 m
- **Wheels**: 6 total (3 per side), each 0.027132 kg, radius ~0.058 m
- **Triplet circumradius**: 0.12 m (wheel centers are 120° apart on this radius)
- **Total mass**: ~3.39 kg
- **Target stair step height**: ~17 cm

### 1.2 Wheel Positions on Each Triplet (relative to triplet center)

```
Wheel 1 (top):        ( 0.0,     ±0.07, +0.12)       — 0° position
Wheel 2 (bottom-left): (-0.10414, ±0.07, -0.060125)  — 240° position
Wheel 3 (bottom-right):(+0.10414, ±0.07, -0.060125)  — 120° position
```

When the triplet rotates 120° around its axis (body Y-axis), the next wheel in the triangle takes the ground-contact position.

### 1.3 Actuators (4 motors)

| Motor | URDF Joint | Axis | Function |
|-------|-----------|------|----------|
| Drive Left | `l_triplet_to_l_wheel_*` (belted) | Y | Drives all 3 left wheels together |
| Drive Right | `r_triplet_to_r_wheel_*` (belted) | Y | Drives all 3 right wheels together |
| Triplet Left | `c_body_to_l_triplet` | Y | Rotates left triplet assembly relative to body |
| Triplet Right | `c_body_to_r_triplet` | Y | Rotates right triplet assembly relative to body |

**Motor torque limits**: Drive motors 1.0 Nm, Triplet motors 5.0 Nm (see Section 3.3).

### 1.4 Sensors

| Sensor | Rate | Output |
|--------|------|--------|
| IMU (accelerometer + gyroscope) | 500 Hz | Pitch angle, pitch rate (fused via complementary/Madgwick filter) |
| Motor encoders (4×) | Per motor driver | Joint angle, angular velocity for each motor |
| Wheel odometry (derived) | Same as encoder rate | Forward position and velocity estimate |

### 1.5 URDF Joint/Link Hierarchy

```
c_body (root, floating base)
├── c_body_to_l_triplet (continuous, axis Y) → l_triplet
│   ├── l_triplet_to_l_wheel_1 (continuous, axis Y) → l_wheel_1
│   ├── l_triplet_to_l_wheel_2 (continuous, axis Y) → l_wheel_2
│   └── l_triplet_to_l_wheel_3 (continuous, axis Y) → l_wheel_3
└── c_body_to_r_triplet (continuous, axis Y) → r_triplet
    ├── r_triplet_to_r_wheel_1 (continuous, axis Y) → r_wheel_1
    ├── r_triplet_to_r_wheel_2 (continuous, axis Y) → r_wheel_2
    └── r_triplet_to_r_wheel_3 (continuous, axis Y) → r_wheel_3
```

All 3 wheels on each side are belt-coupled (same angular velocity).

---

## 2. Control Architecture Overview

### 2.1 Two-Rate Dual-Core Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  ESP32-S3 (or simulation equivalent)                                 │
│                                                                      │
│  ┌─────────────────────────────────┐  shared   ┌──────────────────┐ │
│  │ CORE 0: MPC Solver (30-50 Hz)  │  memory   │ CORE 1: Fast PD  │ │
│  │                                 │ (double   │ Loop (200-500 Hz) │ │
│  │ • Read latest state estimate    │ buffered) │                  │ │
│  │ • Linearize dynamics            │ ───────►  │ • IMU read+filter│ │
│  │ • Build QP                      │ x_ref[]   │ • Encoder read   │ │
│  │ • Solve QP (SIMD-accelerated)   │ u_ff[]    │ • Interpolate    │ │
│  │ • Write trajectory to buffer    │           │   MPC trajectory │ │
│  │                                 │ ◄───────  │ • PD feedback    │ │
│  │                                 │ x_actual  │ • Safety clamp   │ │
│  │                                 │           │ • Motor commands │ │
│  │                                 │           │ • FSM transitions│ │
│  └─────────────────────────────────┘           └──────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Design Principle

The **MPC** plans optimal trajectories (what should happen over the next ~200-330 ms).
The **PD loop** tracks those trajectories at high rate and rejects fast disturbances.
The **FSM** sequences the climbing phases and modifies MPC cost weights/constraints.

---

## 3. Operating Modes

### 3.1 Mode Definitions

The triplet angle is a continuous variable that determines the robot's operating mode.
Three named postures serve as reference points, but the MPC cost weights adapt
continuously as the triplet angle changes (see Section 4).

| Mode | Triplet Angle | Body Pitch | Wheels on Ground | Balance Type |
|------|--------------|------------|------------------|-------------|
| **4WD** | ~0° | ~0° | 4 (2 per side) | Statically stable |
| **Lean** | ~30° | ~15° | 2 (1 per side) | Dynamic, offset equilibrium |
| **2WD** | ~60° | ~0° | 2 (1 per side) | Dynamic, inverted pendulum |

```
Side view at each mode:

  4WD (triplet ≈ 0°)         Lean (triplet ≈ 30°)      2WD (triplet ≈ 60°)

       ┌───┐                      ┌───┐                    ┌───┐
       │   │  CoG at center       │  ╱│  CoG shifted       │   │  CoG above
       │   │  of support          │ ╱ │  forward            │   │  single point
       │   │  polygon             │╱  │                    │   │
       └───┘                      └───┘                    └───┘
      ○─────○                       ○                        ○
   rear    front              single contact           single contact
   ├─support─┤                (lean mode pitch ≈ 15°)   (pitch ≈ 0°)
```

### 3.2 Mode Characteristics

| Property | 4WD | Lean | 2WD |
|----------|-----|------|-----|
| Triplet angle | ~0° | ~30° | ~60° |
| Pitch setpoint | 0° (center of support polygon) | ~15° (offset equilibrium) | 0° (above contact) |
| Turning capability | Poor (skid steer, high lateral friction) | Good | Excellent (single contact) |
| Obstacle clearance | Low | Medium | High (17 cm stairs) |
| Energy consumption | Low (no active balance) | Medium | High (constant balance) |
| Triplet motor torque | Low (hold at 0°) | Medium (hold at 30° against gravity) | High (hold at 60° + balance) |
| Shock absorption | Poor (rigid 4-point) | Good (compliant lean) | Poor (single point) |
| Lateral lean | Not possible | Yes (asymmetric L/R triplet angles) | Not useful |
| Max speed | High | Medium | Low (stability-limited) |

### 3.3 Motor Torque Limits per Mode

The triplet motor has a higher torque limit (5 Nm) than the drive motor (1 Nm),
since it must maintain 2WD/lean posture against gravity and handle disturbances.

| Motor | Torque Limit | Rationale |
|-------|-------------|----------|
| Drive (left/right) | 1.0 Nm | Wheel traction is the bottleneck |
| Triplet (left/right) | 5.0 Nm | Must hold 60° against gravity (~1.5 Nm static) + dynamic margin |

### 3.4 Reactive Mode FSM

Mode transitions are driven by terrain sensing (multipoint TOF) or remote control
command. The FSM uses hysteresis to prevent chatter.

```
              obstacle detected           step detected
              height > 3cm                height > 8cm
  ┌──────┐  ──────────────────►  ┌──────┐  ──────────────────►  ┌──────┐
  │ 4WD  │                       │ LEAN │                       │ 2WD  │
  │      │  ◄──────────────────  │      │  ◄──────────────────  │      │
  └──────┘  flat for >2s         └──────┘  obstacle cleared     └──────┘
            no obstacles                   height < 5cm
            ahead (TOF)                    (hysteresis)
```

**Transition logic:**

```python
class ReactiveModeFSM:
    """Mode FSM with hysteresis thresholds and remote override."""

    def __init__(self):
        self.mode = '4WD'             # Current mode
        self.mode_timer = 0.0         # Time in current mode (s)
        self.remote_override = None   # Set by remote control for testing

    def update(self, obstacle_height, terrain_slope, flat_distance, dt):
        # Remote override takes priority (for testing)
        if self.remote_override is not None:
            self.mode = self.remote_override
            return self.mode

        if self.mode == '4WD':
            if obstacle_height > 0.03 or abs(terrain_slope) > radians(10):
                self.mode = 'LEAN'
                self.mode_timer = 0.0

        elif self.mode == 'LEAN':
            if obstacle_height > 0.08:
                self.mode = '2WD'
                self.mode_timer = 0.0
            elif flat_distance > 0.3 and self.mode_timer > 2.0:
                self.mode = '4WD'
                self.mode_timer = 0.0

        elif self.mode == '2WD':
            if obstacle_height < 0.05 and self.mode_timer > 0.5:
                self.mode = 'LEAN'
                self.mode_timer = 0.0

        self.mode_timer += dt
        return self.mode
```

The mode FSM sets the triplet angle target. The stair-climbing FSM (Section 9)
operates within the 2WD mode to sequence the actual climb phases.

### 3.5 Remote Control Override

For testing, the remote control can force any mode:
- **Button A**: Force 4WD
- **Button B**: Force Lean
- **Button C**: Force 2WD
- **Button D**: Return to automatic (Reactive FSM)

This allows each mode to be tested independently before enabling automatic transitions.

---

## 4. Support Polygon and Adaptive Pitch Reference

### 4.1 The Fundamental Problem

In **2WD**, balance has a single equilibrium: the CoG must be directly above the
contact point. The pitch reference is a single angle.

In **4WD**, the CoG can be anywhere above the support polygon (the convex hull
of ground contact points). There is a **range** of valid pitch angles, not a
single equilibrium. The controller must choose an optimal point within that range.

As the triplet angle rotates from 0° to 60°, the support polygon **continuously
shrinks** from a rectangle to a line to a point. The MPC cost weights must adapt
to this shrinking margin.

### 4.2 Contact Point Geometry

Wheel positions on each triplet in the side-view (x-z) plane, relative to the
triplet hub center:

```python
# Wheel base angles from URDF geometry (measured from +x axis, CCW)
wheel_angles_base = [
    atan2(+0.12,     0.0),      # wheel_1:  90° (top in default pose)
    atan2(-0.060125, -0.10414),  # wheel_2: ~210° (bottom-rear)
    atan2(-0.060125, +0.10414),  # wheel_3: ~330° (bottom-front)
]

# Rotated by triplet_angle:
for each wheel i:
    theta = wheel_angles_base[i] + triplet_angle
    x_i = R_triplet * cos(theta)    # forward position (m)
    z_i = R_triplet * sin(theta)    # vertical position (m)

# R_triplet = 0.12 m
# A wheel is "in contact" when it is one of the two lowest (4WD)
# or the single lowest (2WD).
```

### 4.3 Support Polygon Width

The forward-direction width of the support polygon depends on the triplet angle:

```
Triplet angle:  0°        15°        30°        45°        60°

Side view:     ○          ○           ○           ○          ○
              / \        / \         / \         / \        / \
             ○   ○      ○   ○       ○   ○       ○   ○      ○   ○
             ▓   ▓       ▓  ▓        ▓ ▓         ▓▓          ▓
             ▓▓▓▓▓       ▓▓▓▓        ▓▓▓         ▓▓          ▓

Support      |←─L──→|  |←─L─→|    |←L→|       |L|          •
width (L):   ~0.208m    ~0.18m     ~0.12m     ~0.05m       0

Contact:     4WD         4WD        Transitional  ~2WD       2WD
Stable:      HIGH        HIGH       MEDIUM        LOW        NONE (pitch)
```

```python
def support_polygon_width(triplet_angle, R=0.12):
    """Forward-direction width of support polygon (m)."""
    contacts = get_ground_contact_wheels(triplet_angle, R)
    if len(contacts) < 2:
        return 0.0  # 2WD — point contact
    x_front = max(c.x for c in contacts)
    x_rear  = min(c.x for c in contacts)
    return x_front - x_rear
```

### 4.4 Pitch Reference Computation

#### 4.4.1 In 2WD (single contact point)

```python
def pitch_reference_2wd(triplet_angle, R=0.12, cog_z=0.247):
    """The single equilibrium pitch that places CoG above contact."""
    contact = get_lowest_wheel(triplet_angle, R)
    # contact.x = forward offset of contact point from hub
    return asin(clip(contact.x / cog_z, -1, 1))
```

This is a **single value** — the robot must track it precisely or it falls.

#### 4.4.2 In 4WD (support polygon)

```python
def pitch_reference_4wd(triplet_angle, R=0.12, cog_z=0.247,
                         strategy='center'):
    """Optimal pitch angle within the support polygon.

    Returns (pitch_ref, pitch_min, pitch_max).
    pitch_min/max are the tipping limits.
    """
    contacts = get_two_lowest_wheels(triplet_angle, R)
    x_front = max(c.x for c in contacts)
    x_rear  = min(c.x for c in contacts)

    # Tipping limits
    pitch_max = asin(clip(x_front / cog_z, -1, 1))  # tip forward
    pitch_min = asin(clip(x_rear  / cog_z, -1, 1))  # tip backward

    if strategy == 'center':
        # Maximum stability margin — equal distance to both edges
        x_center = (x_front + x_rear) / 2.0
        pitch_ref = asin(clip(x_center / cog_z, -1, 1))

    elif strategy == 'energy':
        # Minimum combined motor torque — gravity centred
        pitch_ref = asin(clip((x_front + x_rear) / 2.0 / cog_z, -1, 1))

    elif strategy == 'front':
        # Biased 70% toward front edge — ready for lean/lunge
        x_biased = x_rear + 0.7 * (x_front - x_rear)
        pitch_ref = asin(clip(x_biased / cog_z, -1, 1))

    elif strategy == 'rear':
        # Biased 70% toward rear edge — ready for acceleration
        x_biased = x_rear + 0.3 * (x_front - x_rear)
        pitch_ref = asin(clip(x_biased / cog_z, -1, 1))

    return pitch_ref, pitch_min, pitch_max
```

#### 4.4.3 Unified function (works for any triplet angle)

```python
def compute_pitch_reference(triplet_angle, strategy='center',
                             R=0.12, cog_z=0.247, wheel_r=0.058):
    """Compute pitch reference for any triplet angle (4WD, lean, or 2WD).

    Automatically detects how many wheels are in contact based on
    the vertical gap between the two lowest wheels.
    """
    positions = get_all_wheel_positions(triplet_angle, R)
    sorted_by_z = sorted(positions, key=lambda p: p.z)

    z_gap = sorted_by_z[1].z - sorted_by_z[0].z
    two_in_contact = (z_gap < wheel_r * 0.5)

    if two_in_contact:
        return pitch_reference_4wd(triplet_angle, R, cog_z, strategy)
    else:
        ref = pitch_reference_2wd(triplet_angle, R, cog_z)
        return ref, ref, ref  # min = max = ref (zero margin)
```

### 4.5 Continuous Q_pitch Adaptation

The MPC pitch cost weight `q_pitch` scales **continuously** with the support
polygon width. This is the key mechanism that unifies 4WD and 2WD control
under a single MPC formulation.

```python
def compute_adaptive_q_pitch(triplet_angle, R=0.12):
    """Compute MPC pitch weight based on support polygon width.

    Wide polygon (4WD) → LOW weight (any pitch in range is fine, save energy)
    Narrow polygon      → INCREASING weight (tighter tracking needed)
    Zero polygon (2WD)  → HIGH weight (survival — must track precisely)

    Uses quadratic scaling so weight increases faster as margin shrinks.
    """
    Q_PITCH_MIN = 10.0    # 4WD: gentle, energy-saving
    Q_PITCH_MAX = 80.0    # 2WD: aggressive, survival mode
    MAX_WIDTH   = 0.208   # theoretical max support polygon width at triplet=0°

    width = support_polygon_width(triplet_angle, R)

    if width < 0.001:
        return Q_PITCH_MAX

    t = 1.0 - (width / MAX_WIDTH)  # 0 at 4WD, 1 at 2WD
    t = clip(t, 0.0, 1.0)

    # Quadratic scaling — weight increases faster as margin shrinks
    q_pitch = Q_PITCH_MIN + (Q_PITCH_MAX - Q_PITCH_MIN) * t * t

    return q_pitch
```

```
  Q_pitch
   80 ┤                                                      ●  2WD
      │                                                   ╱╱
      │                                                ╱╱
   60 ┤                                             ╱╱
      │                                          ╱
      │                                       ╱
   40 ┤                                    ╱
      │                                 ╱           ← quadratic
      │                             ╱╱               (increases faster
   20 ┤                         ╱╱                     near 2WD)
      │                  ╱╱╱╱
   10 ●─────────────╱╱╱╱                              4WD
      ├──────┬──────┬──────┬──────┬──────┬──────┤
     0°     10°    20°    30°    40°    50°    60°
                     triplet_angle
```

### 4.6 How It All Feeds Into the MPC

Every MPC cycle, before building the QP:

```python
def update_mpc_references(triplet_angle_L, triplet_angle_R, fsm_state):
    """Compute pitch reference and Q_pitch from current triplet angles."""

    # Use average triplet angle for pitch adaptation
    avg_triplet = (triplet_angle_L + triplet_angle_R) / 2.0

    # Pitch reference depends on FSM state + support polygon
    if fsm_state == 'APPROACH':
        strategy = 'front'    # ready to lean
    elif fsm_state in ('PREPARE', 'LEAN_FORWARD'):
        strategy = 'front'    # preparing to lunge
    elif fsm_state in ('LUNGE', 'LAND', '2WD_BALANCE'):
        strategy = 'center'   # single contact — only one option
    elif fsm_state == 'PULL_UP':
        strategy = 'center'   # recovering stability
    else:
        strategy = 'center'   # 4WD default — max margin

    pitch_ref, pitch_min, pitch_max = compute_pitch_reference(
        avg_triplet, strategy=strategy)

    # Continuous Q_pitch from support polygon
    q_pitch = compute_adaptive_q_pitch(avg_triplet)

    # Override: FSM can further adjust (e.g., LUNGE reduces q_pitch)
    if fsm_state == 'LUNGE':
        q_pitch = 2.0   # override — allow free fall

    return pitch_ref, q_pitch, pitch_min, pitch_max
```

This means the MPC **automatically adapts** as the triplet rotates:
- Starting in 4WD (triplet=0°): q_pitch=10, pitch_ref = center of wide polygon
- Transitioning to Lean (triplet=30°): q_pitch≈35, pitch_ref≈15°, polygon shrinking
- Arriving at 2WD (triplet=60°): q_pitch=80, pitch_ref=0°, zero margin

No discrete switching is needed — the cost weights change smoothly with the
triplet angle, tracked continuously by the encoder.

### 4.7 Summary Table

| Triplet Angle | Support Width | Pitch Range | Pitch Ref (center) | Q_pitch | Mode |
|--------------|---------------|-------------|--------------------|---------|---------|
| 0° | ~0.208 m | −4.8° to +4.8° | 0.0° | ~10 | 4WD |
| 15° | ~0.18 m | −4.1° to +4.1° | ~0.0° | ~13 | 4WD |
| 30° | ~0.12 m | −2.8° to +2.8° | ~0.0° | ~35 | Lean |
| 45° | ~0.05 m | −1.2° to +1.2° | ~0.0° | ~60 | near-2WD |
| 60° | 0 m | (single point) | 0.0° | ~80 | 2WD |

---

## 5. State Vector and Inputs

### 5.1 Medium MPC Model (Recommended)

```
State vector x (8 elements):
  x[0] = pitch            # Body pitch angle (rad), positive = leaning forward
  x[1] = pitch_rate       # Body pitch angular velocity (rad/s)
  x[2] = triplet_angle_L  # Left triplet angle relative to body (rad)
  x[3] = triplet_angle_R  # Right triplet angle relative to body (rad)
  x[4] = triplet_rate_L   # Left triplet angular velocity (rad/s)
  x[5] = triplet_rate_R   # Right triplet angular velocity (rad/s)
  x[6] = forward_pos      # Horizontal position (m), from wheel odometry
  x[7] = forward_vel      # Horizontal velocity (m/s)

Input vector u (4 elements):
  u[0] = tau_triplet_L    # Left triplet motor torque (Nm)
  u[1] = tau_triplet_R    # Right triplet motor torque (Nm)
  u[2] = tau_drive_L      # Left drive motor torque (Nm)
  u[3] = tau_drive_R      # Right drive motor torque (Nm)
```

### 5.2 Horizon

- **N = 10** steps
- **dt_mpc** = 1 / MPC_rate ≈ 20-33 ms (at 30-50 Hz)
- **Lookahead** = N × dt_mpc ≈ 200-330 ms

### 5.3 Constraints

```
Input constraints (box):
  -TAU_TRIP_MAX  ≤ u[0], u[1] ≤ TAU_TRIP_MAX     # Triplet motor limits
  -TAU_DRIVE_MAX ≤ u[2], u[3] ≤ TAU_DRIVE_MAX     # Drive motor limits

State constraints (optional, phase-dependent):
  pitch_min      ≤ x[0]       ≤ pitch_max          # Pitch safety bounds
  triplet_rate_min ≤ x[4], x[5] ≤ triplet_rate_max # Triplet rotation speed limits
```

---

## 6. MPC Formulation

### 6.1 Linearized Dynamics

At each MPC cycle, linearize the nonlinear dynamics around the current state:

```
x[k+1] = A @ x[k] + B @ u[k]

where A and B are obtained by linearizing the equations of motion of the
planar model (inverted pendulum body + rotating triplet hubs + driven wheels)
around the current operating point.
```

The continuous-time dynamics are derived from the Euler-Lagrange equations of the system.
The key physical couplings are:

1. **Body pitch** is affected by gravity (body CoG above wheel axis), drive motor reaction torque, and triplet motor reaction torque.
2. **Triplet angle** is directly driven by the triplet motor, with gravity coupling through the offset wheel masses.
3. **Forward position** is driven by wheel torques, coupled to pitch through the contact point geometry.

### 6.2 QP Formulation

```
minimize    Σ_{k=0}^{N-1} [ (x[k] - x_ref[k])ᵀ Q (x[k] - x_ref[k])
                           + (u[k] - u_ref[k])ᵀ R (u[k] - u_ref[k]) ]
          + (x[N] - x_ref[N])ᵀ Q_f (x[N] - x_ref[N])

subject to  x[k+1] = A x[k] + B u[k]        (linearized dynamics)
            u_min ≤ u[k] ≤ u_max              (input bounds)
            x_min ≤ x[k] ≤ x_max              (state bounds, phase-dependent)
```

### 6.3 Cost Matrices

Q, R, and Q_f are **diagonal** and **phase-dependent** (set by the FSM):

```
Q = diag(q_pitch, q_pitch_rate, q_tripL, q_tripR, q_tripL_rate, q_tripR_rate, q_fwd, q_fwd_vel)
R = diag(r_tripL, r_tripR, r_driveL, r_driveR)
Q_f = Q (or a terminal cost from discrete-time algebraic Riccati, DARE)
```

### 6.4 Solver

Recommended: **Dense QP solver** (qpOASES-style active set, or custom) for this small problem size.
OSQP (sparse, ADMM-based) is also viable but dense is faster for N=10, nx=8, nu=4.

**QP dimensions:**
- Decision variables: N × (nx + nu) = 10 × 12 = 120
- Constraints: ~10 × 16 + bounds ≈ ~250

**ESP32-S3 estimated solve time**: 15-45 ms (single core, SIMD via ESP-DSP).

---

## 7. MPC Output: Trajectory Buffer

### 7.1 Data Structure

```c
typedef struct {
    float t_solve;           // Timestamp when this MPC solution was computed
    float dt_mpc;            // Time step between trajectory knot points
    int   N;                 // Horizon length (10)

    float x_ref[11][8];     // Optimal state trajectory x*[0..N] (N+1 points)
    float u_ff[10][4];      // Optimal feedforward inputs u*[0..N-1] (N points)

    bool  valid;             // Set true after solve completes, false if solver fails
} mpc_trajectory_t;
```

### 7.2 Double Buffering

```
mpc_trajectory_t trajectory[2];      // Two buffers
volatile int active_buffer = 0;      // Index read by PD loop (Core 1)

// Core 0 writes to trajectory[!active_buffer], then:
//   active_buffer = !active_buffer;  // Atomic swap (single word write)
```

Core 1 always reads from `trajectory[active_buffer]`. Core 0 always writes to the other.
No locks needed — the swap is a single-word atomic write.

---

## 8. Fast PD Tracking Loop (Core 1, 200-500 Hz)

### 8.1 Interpolation

At each PD cycle (every 2-5 ms), the loop determines its position within the MPC trajectory:

```c
float t_elapsed = t_now - traj->t_solve;
float segment_float = t_elapsed / traj->dt_mpc;
int k = (int)segment_float;             // Trajectory segment index
float alpha = segment_float - (float)k; // Interpolation fraction [0, 1)

// Clamp: if MPC is late, hold the last trajectory point
if (k >= traj->N - 1) { k = traj->N - 1; alpha = 0.0f; }

// Linearly interpolate reference state
for (int i = 0; i < 8; i++)
    x_ref[i] = (1.0f - alpha) * traj->x_ref[k][i]
             + alpha * traj->x_ref[k + 1][i];

// Linearly interpolate feedforward input
for (int i = 0; i < 4; i++)
    u_ff[i] = (1.0f - alpha) * traj->u_ff[k][i]
            + alpha * traj->u_ff[min(k + 1, traj->N - 1)][i];
```

### 8.2 PD Feedback + Feedforward Computation

```c
// Tracking errors
float e_pitch     = x_ref[0] - pitch_actual;
float e_pitch_d   = x_ref[1] - pitch_rate_actual;
float e_tripL     = x_ref[2] - tripL_angle_actual;
float e_tripR     = x_ref[3] - tripR_angle_actual;
float e_tripL_d   = x_ref[4] - tripL_rate_actual;
float e_tripR_d   = x_ref[5] - tripR_rate_actual;
float e_fwd       = x_ref[6] - fwd_pos_actual;
float e_fwd_d     = x_ref[7] - fwd_vel_actual;

// --- Left triplet motor ---
float tau_tripL = u_ff[0]
    + Kp_trip * e_tripL + Kd_trip * e_tripL_d
    + Kp_pitch_trip * e_pitch + Kd_pitch_trip * e_pitch_d;

// --- Right triplet motor ---
float tau_tripR = u_ff[1]
    + Kp_trip * e_tripR + Kd_trip * e_tripR_d
    + Kp_pitch_trip * e_pitch + Kd_pitch_trip * e_pitch_d;

// --- Left drive motor ---
float tau_driveL = u_ff[2]
    + Kp_drive * e_fwd + Kd_drive * e_fwd_d
    + Kp_pitch_drive * e_pitch + Kd_pitch_drive * e_pitch_d;

// --- Right drive motor ---
float tau_driveR = u_ff[3]
    + Kp_drive * e_fwd + Kd_drive * e_fwd_d
    + Kp_pitch_drive * e_pitch + Kd_pitch_drive * e_pitch_d;
```

### 8.3 Contribution Breakdown

| Component | What it does | Typical share of total torque |
|-----------|-------------|-------------------------------|
| `u_ff` (feedforward from MPC) | Gravity compensation, planned acceleration, anticipated forces | 70-90% |
| PD on triplet error | Tracks planned triplet rotation trajectory | 5-15% |
| PD on pitch error | Fast disturbance rejection on body tilt | 5-20% |
| PD on forward error | Position/velocity tracking corrections | 2-10% |

### 8.4 PD Gains (Initial Estimates, Need Tuning)

```
Kp_trip         = 5.0       # Nm/rad — triplet position tracking
Kd_trip         = 0.3       # Nm/(rad/s) — triplet velocity damping
Kp_pitch_trip   = 2.0       # Nm/rad — pitch→triplet cross-coupling
Kd_pitch_trip   = 0.1       # Nm/(rad/s)

Kp_drive        = 3.0       # Nm/m — forward position tracking
Kd_drive        = 1.0       # Nm/(m/s) — forward velocity damping
Kp_pitch_drive  = 10.0      # Nm/rad — pitch→drive (Segway-style balance)
Kd_pitch_drive  = 0.5       # Nm/(rad/s)
```

---

## 9. Stair Climbing FSM

This FSM operates **within 2WD mode** (see Section 3.4). It sequences the phases
of climbing a single stair step. The Reactive Mode FSM (Section 3.4) handles
switching between 4WD/Lean/2WD; this FSM handles the climb once 2WD is active.

Pitch reference and Q_pitch values listed below are **base values** from the FSM.
The continuous adaptation from Section 4.5 is applied on top — during phases where
the triplet angle changes (PREPARE, PULL_UP), Q_pitch transitions smoothly rather
than jumping.

### 9.1 State Diagram

```
                    ┌──────────────────────────────────────────────┐
                    │                                              │
                    ▼                                              │
             ┌──────────┐    at stair    ┌──────────┐             │
  ───────►   │  BALANCE  │ ──────────►   │ APPROACH  │             │
  (startup)  │ (flat gnd)│   detected    │           │             │
             └──────────┘               └─────┬─────┘             │
                    ▲                         │                    │
                    │              at edge,    │                    │
                    │              aligned     ▼                    │
                    │                   ┌──────────┐               │
                    │                   │  PREPARE  │               │
                    │                   │ (lean fwd)│               │
                    │                   └─────┬─────┘               │
                    │                         │                    │
                    │              past tipping│                    │
                    │              point       ▼                    │
                    │                   ┌──────────┐               │
                    │                   │  LUNGE   │               │
                    │                   │ (ctrl'd  │               │
                    │                   │  fall)   │               │
                    │                   └─────┬─────┘               │
                    │                         │                    │
                    │              impact      │                    │
                    │              detected    ▼                    │
                    │                   ┌──────────┐               │
                    │                   │  LAND    │               │
                    │                   │ (absorb  │               │
                    │                   │ +transfer)│               │
                    │                   └─────┬─────┘               │
                    │                         │                    │
                    │              weight on   │                    │
                    │              new wheels  ▼                    │
                    │                   ┌──────────┐               │
                    │                   │ PULL_UP  │               │
                    │                   │          │ ──────────────┘
                    │                   └─────┬─────┘  (more stairs
                    │                         │         → APPROACH)
                    │              stable on   │
                    │              new step    │
                    └─────────────────────────┘
                                   (no more stairs → BALANCE)
```

### 9.2 State Definitions

Each FSM state sets: **MPC cost weights (Q, R)**, **reference targets (x_ref)**, **constraints**, and **transition conditions**.

---

#### 9.2.1 BALANCE (Default — Flat Ground)

**Operating mode**: 4WD (triplet ≈ 0°). Q_pitch from Section 4.5 ≈ 10 (relaxed).
Pitch reference from `compute_pitch_reference(0°, strategy='center')` ≈ 0°.

**Goal**: Self-balance on flat ground, drive to target position.

```
MPC cost weights:
  q_pitch       = HIGH (80.0)    — stay upright
  q_pitch_rate  = MEDIUM (5.0)   — smooth
  q_tripL/R     = LOW (1.0)      — hold current triplet angle loosely
  q_tripL/R_rate= LOW (0.5)      — no rapid triplet rotation
  q_fwd         = MEDIUM (10.0)  — track target position
  q_fwd_vel     = LOW (1.0)      — smooth velocity
  r_trip        = MEDIUM (5.0)   — don't waste triplet torque
  r_drive       = LOW (1.0)      — allow drive authority

MPC reference:
  pitch_ref       = 0.0
  triplet_ref_L/R = current_contact_angle   (whichever wheel is on ground)
  forward_ref     = target_position

Drive mode: All 4 wheels on ground → 4WD.

Transition to APPROACH:
  - Stair detected ahead (distance sensor, vision, or pre-programmed)
  - forward_pos within approach_start_distance of stair edge
```

---

#### 9.2.2 APPROACH

**Operating mode**: 4WD (triplet ≈ 0°). Q_pitch ≈ 10.
Pitch reference from `compute_pitch_reference(0°, strategy='front')` — biased
toward the front edge of the support polygon to prepare for the lean.

**Goal**: Drive to the stair edge, align precisely. Still balancing normally.

```
MPC cost weights:
  q_pitch       = HIGH (80.0)
  q_fwd         = HIGH (50.0)    — precise positioning matters
  q_fwd_vel     = MEDIUM (5.0)   — approach slowly
  (others same as BALANCE)

MPC reference:
  pitch_ref       = 0.0
  forward_ref     = stair_edge_x - approach_offset   (e.g., 2 cm from edge)

Transition to PREPARE:
  - |forward_pos - stair_edge| < position_threshold (e.g., 5 mm)
  - |forward_vel| < velocity_threshold (e.g., 0.02 m/s)
  - Robot is ~stationary at the edge
```

---

#### 9.2.3 PREPARE (Lean Forward)

**Operating mode**: Transitioning 4WD → 2WD (triplet rotates 0° → 60°).
Q_pitch adapts continuously via Section 4.5 as support polygon shrinks.
Pitch reference tracks `compute_pitch_reference(current_triplet, strategy='front')`.

**Goal**: Tilt the body forward to move CoG past the tipping point. Switch from 4WD to 2WD (rear pair of each triplet provides traction, front pair lifts off).

```
MPC cost weights:
  q_pitch       = MEDIUM (20.0)  — follow the lean trajectory, don't fight it
  q_pitch_rate  = HIGH (10.0)    — lean slowly and smoothly
  q_tripL/R     = HIGH (50.0)    — triplet must hold position firmly
  q_fwd         = LOW (2.0)      — forward position is now less important
  r_trip        = LOW (1.0)      — allow triplet torque to hold firmly
  r_drive       = MEDIUM (5.0)

MPC reference:
  pitch_ref       = LEAN_TARGET (ramp from 0.0 to ~0.3-0.5 rad over ~300 ms)
  triplet_ref_L/R = current_contact_angle (locked)

Drive mode: 2WD — only rear-most wheels of triplet maintain ground contact.
  (Triplet motor holds angle firmly while body tilts)

Transition to LUNGE:
  - pitch_actual > lunge_trigger_angle (e.g., 0.35 rad / ~20°)
  - OR pitch_rate > lunge_trigger_rate (body is accelerating forward)
```

---

#### 9.2.4 LUNGE (Controlled Fall)

**Operating mode**: 2WD (triplet ≈ 60°). Q_pitch **overridden to LOW (2.0)**
despite Section 4.5 wanting 80 — the FSM intentionally allows free fall.

**Goal**: Let the robot fall forward so the front wheels land on the next step. MPC allows and assists the fall rather than fighting it.

```
MPC cost weights:
  q_pitch       = LOW (2.0)      — ALLOW large pitch excursion (this is the key change)
  q_pitch_rate  = MEDIUM (5.0)   — don't spin out of control, but allow forward rotation
  q_tripL/R     = HIGH (50.0)    — triplet locked for impact preparation
  q_tripL/R_rate= HIGH (10.0)    — absolutely no triplet rotation during fall
  q_fwd         = LOW (1.0)      — not relevant during ballistic phase
  r_trip        = LOW (1.0)      — triplet motor can use full torque to hold
  r_drive       = LOW (0.5)      — drive motor can assist the lunge

MPC reference:
  pitch_ref       = lunge_target_pitch (e.g., 0.5-0.8 rad — lean far forward)
  triplet_ref_L/R = current_contact_angle (locked)

Physics of this phase:
  - Duration: ~150-300 ms (dominated by gravity: t ≈ sqrt(2h/g) for 17cm ≈ 186 ms)
  - The MPC feedforward will output positive drive torque (assisting fall)
    and strong triplet holding torque
  - The PD loop's pitch error term will naturally reduce, since MPC's pitch_ref
    is ahead of the actual pitch (the MPC is "asking" the robot to lean)

Transition to LAND:
  - Impact detected: sudden change in pitch_rate (deceleration spike)
  - OR IMU acceleration spike exceeds threshold
  - OR forward_pos indicates wheel has reached next step surface
  - OR timeout from entering LUNGE state
```

---

#### 9.2.5 LAND (Impact Absorption + Weight Transfer)

**Operating mode**: 2WD (triplet ≈ 60°, then rotating 120° to next wheel pair).
Q_pitch from Section 4.5 ≈ 80 (aggressive stabilization).

**Goal**: Absorb landing impact, rotate triplets 120° to bring the next wheel pair into ground contact on the new step.

```
MPC cost weights:
  q_pitch       = HIGH (80.0)    — aggressively stabilize after impact
  q_pitch_rate  = HIGH (20.0)    — damp pitch oscillations
  q_tripL/R     = HIGH (50.0)    — track the 120° rotation trajectory
  q_tripL/R_rate= MEDIUM (5.0)   — controlled rotation speed
  q_fwd         = MEDIUM (10.0)  — don't slide off the step
  r_trip        = LOW (0.5)      — full triplet torque authority for rotation
  r_drive       = MEDIUM (3.0)

MPC reference:
  pitch_ref       = 0.0 (return to upright)
  triplet_ref_L/R = previous_contact_angle + 2π/3   (next wheel pair, 120° rotation)
                    Ramp over ~200-300 ms, not instant
  forward_ref     = stair_surface_center (enough on the step to not fall back)

This phase includes:
  1. Initial impact absorption (first ~50-100 ms): high damping, hold position
  2. Triplet rotation (next ~200-300 ms): controlled 120° rotation to
     bring new wheel pair to ground contact on the upper step
  3. Weight transfer: body pitch returns to vertical as new wheels bear load

Transition to PULL_UP:
  - |triplet_angle - target_angle| < rotation_threshold (e.g., 5°)
  - |pitch_rate| < stability_threshold (oscillations damped)
  - Body is approximately upright on the new step
```

---

#### 9.2.6 PULL_UP

**Operating mode**: Transitioning 2WD → 4WD (triplet settling to new contact angle).
Q_pitch adapts continuously via Section 4.5 as support polygon widens.

**Goal**: Drive the robot fully onto the new step, stabilize for next stair or flat ground.

```
MPC cost weights:
  q_pitch       = HIGH (80.0)    — stay upright
  q_pitch_rate  = MEDIUM (5.0)
  q_tripL/R     = MEDIUM (10.0)  — hold new contact angle
  q_fwd         = HIGH (50.0)    — drive forward onto step surface
  q_fwd_vel     = MEDIUM (5.0)   — moderate speed
  r_drive       = LOW (1.0)      — drive authority needed

MPC reference:
  pitch_ref       = 0.0
  triplet_ref_L/R = new_contact_angle (post-rotation position)
  forward_ref     = step_center + safety_margin

Drive mode: Back to 4WD (all wheels on step surface).

Transition to APPROACH (if more stairs):
  - Robot stable on step
  - Next stair detected ahead

Transition to BALANCE (if no more stairs):
  - Robot stable on step
  - No further stairs detected
```

---

### 9.3 FSM Timing Summary

| Phase | Typical Duration | MPC Updates (at 40Hz) | Critical Axis |
|-------|------------------|-----------------------|---------------|
| BALANCE | Indefinite | Continuous | Pitch, Forward |
| APPROACH | 500-2000 ms | 20-80 | Forward position |
| PREPARE | 200-400 ms | 8-16 | Pitch (lean) |
| LUNGE | 150-300 ms | 6-12 | Pitch (free fall) |
| LAND | 200-400 ms | 8-16 | Triplet rotation, Pitch |
| PULL_UP | 300-500 ms | 12-20 | Forward, Pitch |
| **Total per step** | **~1.5-3.5 s** | | |

---

## 10. Torque Application Model

The motor-to-joint torque mapping follows the URDF chain and the physical motor mounting:

```
Motor stator is mounted on BODY. Motor rotor drives WHEEL shaft through
the free-spinning TRIPLET hub bearing.

For DRIVE motors (apply wheel torque, reaction goes to body):
  - Apply +τ to wheel joints (drives wheels forward)
  - Apply -τ to triplet joint (reaction, keeps triplet free-spinning)
  - Net: wheels +τ, triplet 0 (free hub), body -τ (motor reaction)

For TRIPLET motors (rotate triplet relative to body):
  - Apply +τ to triplet joint (rotates triplet)
  - Reaction: body experiences -τ (must be handled by balance controller)
```

In the simulation (PyBullet):
```python
# Drive motor torque (per side):
p.setJointMotorControl2(body_id, triplet_joint, TORQUE_CONTROL, force=-motor_torque)
for wj in wheel_joints:
    p.setJointMotorControl2(body_id, wj, TORQUE_CONTROL, force=-motor_torque / 3.0)

# Triplet motor torque (per side):
p.setJointMotorControl2(body_id, triplet_joint, TORQUE_CONTROL, force=triplet_torque)
# (reaction on body is automatic via the joint constraint)
```

---

## 11. Safety and Fallback

### 11.1 Safety Checks (every PD cycle)

```c
// Emergency abort: pitch exceeds safe limit → cut all torques, go limp
if (fabs(pitch_actual) > PITCH_EMERGENCY_LIMIT)  // e.g., 60° = 1.05 rad
    enter_state(EMERGENCY_STOP);

// Triplet rate limit: if triplet spinning too fast, clamp
if (fabs(tripL_rate) > TRIPLET_RATE_MAX)
    tau_tripL = -sign(tripL_rate) * TRIPLET_BRAKE_TORQUE;  // active braking

// Motor current/temperature protection (if available from driver)
```

### 11.2 MPC Solver Fallback

```
If MPC solver does not converge within allocated time:
  1. Reuse previous trajectory (shifted forward in time) — valid for 1-2 missed cycles
  2. If 3+ consecutive misses: fall back to PD-only mode using last known x_ref[end] as
     static reference + gravity compensation feedforward
  3. If in LUNGE state and solver fails: hold triplet firmly, let lunge complete
     ballistically, rely on PD pitch tracking for landing
```

### 11.3 Communication Loss with Motor Drivers

```
If motor driver does not acknowledge command within timeout:
  - Hold last command for 1 cycle (motor driver internal loop continues)
  - After 2 missed cycles: zero all torques → robot falls safely
  - Never hold nonzero torque indefinitely on communication loss
```

---

## 12. Computational Performance Budget (ESP32-S3)

### 12.1 Core 0 — MPC Solver

| Item | Time Budget |
|------|-------------|
| State estimation read | < 0.1 ms |
| Dynamics linearization (A, B matrices) | ~1-2 ms |
| QP construction | ~1-2 ms |
| QP solve (dense, SIMD) | ~10-30 ms |
| Trajectory buffer write | < 0.1 ms |
| **Total** | **~15-35 ms → 30-65 Hz** |

### 12.2 Core 1 — Fast Loop

| Item | Time Budget |
|------|-------------|
| IMU read + Madgwick filter | ~0.2 ms |
| Encoder reads (4 motors) | ~0.1 ms |
| Trajectory interpolation | ~0.05 ms |
| PD computation | ~0.1 ms |
| Safety checks | ~0.05 ms |
| Motor command send | ~0.1 ms |
| FSM logic | ~0.05 ms |
| **Total** | **~0.65 ms → up to ~1500 Hz** |

### 12.3 Memory Budget (512 KB SRAM)

| Component | Estimate |
|-----------|----------|
| QP workspace (N=10, dense) | ~50-80 KB |
| Trajectory double buffer | ~2 KB |
| State estimator | ~5 KB |
| ESP-DSP scratch | ~10-20 KB |
| FreeRTOS (2 tasks + stacks) | ~30-50 KB |
| Application code + data | ~50-80 KB |
| **Total** | **~150-235 KB of 512 KB** |

All MPC data must reside in **internal SRAM**, not external PSRAM (which has 3-10× higher latency).

---

## 13. Implementation Order

Recommended implementation sequence:

### Phase 1: Simulation Foundation
1. Implement the planar dynamics model (equations of motion) in Python
2. Verify model against PyBullet simulation (compare state evolution)
3. Implement a basic QP solver (use `osqp` or `scipy.optimize` for prototyping)
4. Run MPC in simulation with static BALANCE state — verify the robot balances

### Phase 2: FSM + Stair Climbing in Simulation
5. Implement the FSM with all 6 states
6. Create stair terrain in PyBullet (already exists: `box_stairs` in `terrain.py`)
7. Tune MPC cost weights per FSM state until the robot climbs one step in sim
8. Test consecutive stair climbing
9. Add PD inner loop (two-rate architecture) and verify improvement

### Phase 3: Embedded Port
10. Port dynamics model to C (float32, ESP-DSP matrix ops)
11. Port QP solver to C with SIMD
12. Implement FreeRTOS dual-core task structure
13. Integrate with motor drivers and IMU hardware
14. Tune PD gains on real hardware (start with robot on flat ground)
15. Tune MPC cost weights on real stairs

### Phase 4: Robustness
16. Add impact detection logic for LUNGE → LAND transition
17. Add MPC solver fallback and timeout handling
18. Test with varying stair heights (14-20 cm)
19. Test with slight yaw misalignment

---

## 14. Key Design Parameters (Tuning Reference)

```python
# --- System identification (from URDF + PyBullet) ---
BODY_MASS           = 2.7167    # kg
WHEEL_MASS_TOTAL    = 0.6698    # kg (2 triplets + 6 wheels)
COG_HEIGHT          = 0.247     # m (body CoG above wheel axis)
BODY_INERTIA_YY     = 0.056436  # kg·m² (pitch axis, PyBullet-computed)
TRIPLET_RADIUS      = 0.12      # m (wheel center distance from triplet axis)
WHEEL_RADIUS        = 0.058     # m
TRIPLET_HALF_WIDTH  = 0.0825    # m (lateral offset from body center)
STAIR_STEP_HEIGHT   = 0.17      # m

# --- MPC parameters ---
MPC_RATE_HZ         = 40        # MPC solve rate (target)
MPC_HORIZON_N       = 10        # Prediction horizon steps
MPC_DT              = 1.0 / MPC_RATE_HZ

# --- PD loop parameters ---
PD_RATE_HZ          = 500       # PD inner loop rate
PD_DT               = 1.0 / PD_RATE_HZ

# --- Motor limits ---
TAU_TRIP_MAX        = 5.0       # Nm — triplet motor torque limit
TAU_DRIVE_MAX       = 1.0       # Nm — drive motor torque limit

# --- Safety limits ---
PITCH_EMERGENCY_LIMIT = 1.05    # rad (~60°) — abort threshold
TRIPLET_RATE_MAX      = 10.0    # rad/s — triplet rotation speed limit

# --- FSM transition thresholds ---
APPROACH_OFFSET       = 0.02    # m — stop distance from stair edge
POSITION_THRESHOLD    = 0.005   # m — "at edge" position tolerance
VELOCITY_THRESHOLD    = 0.02    # m/s — "stopped" velocity tolerance
LEAN_TARGET           = 0.40    # rad (~23°) — forward lean for prepare phase
LUNGE_TRIGGER_ANGLE   = 0.35   # rad (~20°) — pitch angle to trigger lunge
LUNGE_TRIGGER_RATE    = 1.0     # rad/s — pitch rate to trigger lunge
IMPACT_ACCEL_THRESH   = 3.0     # g — IMU acceleration spike for landing detection
ROTATION_THRESHOLD    = 0.09    # rad (~5°) — triplet rotation completion tolerance
STABILITY_THRESHOLD   = 0.5     # rad/s — pitch rate for "stable" determination

# --- Reactive Mode FSM thresholds (Section 3.4) ---
MODE_OBSTACLE_4WD_TO_LEAN  = 0.03   # m — obstacle height triggering 4WD→Lean
MODE_OBSTACLE_LEAN_TO_2WD  = 0.08   # m — obstacle height triggering Lean→2WD
MODE_OBSTACLE_2WD_TO_LEAN  = 0.05   # m — obstacle height (hysteresis) for 2WD→Lean
MODE_FLAT_DISTANCE_LEAN_4WD= 0.30   # m — flat ground ahead to de-escalate Lean→4WD
MODE_LEAN_DWELL_TIME       = 2.0    # s — minimum time in Lean before de-escalating
MODE_2WD_DWELL_TIME        = 0.5    # s — minimum time in 2WD before de-escalating
MODE_SLOPE_THRESHOLD       = 0.175  # rad (~10°) — slope triggering 4WD→Lean

# --- Support polygon / adaptive Q_pitch (Section 4.5) ---
Q_PITCH_MIN         = 10.0     # 4WD: relaxed pitch tracking
Q_PITCH_MAX         = 80.0     # 2WD: aggressive pitch tracking
SUPPORT_WIDTH_MAX   = 0.208    # m — theoretical max at triplet=0°
```

---

## 15. Notation / Conventions

- **Pitch positive** = body leaning forward (toward stair)
- **Triplet angle positive** = counterclockwise rotation when viewed from the left side (following right-hand rule on Y-axis)
- **Forward positive** = direction the robot faces (−X in URDF world frame)
- **Drive torque positive** = drives robot forward
- **All angles in radians**, all torques in Nm, all positions in meters
- **Coordinate frame**: Y-axis is the wheel axle direction; pitch rotation is around Y; forward motion is along X (negated from URDF convention)
