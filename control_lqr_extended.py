"""
Extended-State LQR Balance Controller with Triplet Angle Reference

6-state full-state feedback LQR that explicitly models triplet dynamics,
enabling smooth mode transitions (4WD ↔ 2WD) within the optimal controller.

Both triplets rotate in the SAME direction, so their reaction torques on
the body ADD (producing body lean during transitions).  The coupled LQR
naturally coordinates wheel + triplet torques to maintain balance through
the lean/flip phase — no separate transition controller needed.

State:  x = [position, velocity, pitch, pitch_rate, φ_trip, φ̇_trip]
Input:  u = [τ_wheels, τ_triplet]

Gain scheduling (three modes):
  Normal:     Steady-state 4WD/2WD balance.  High pitch weight, moderate
              triplet hold.
  Transition: Active during the φ_ref ramp + settling.  Relaxed pitch
              (allows intentional lean), very high triplet weight (overcomes
              ground friction in 4WD).  Paired friction feedforward provides
              base torque; the LQR adds incremental corrections.
  Aggressive: Position-tracking mode when far from target.

Physical model (3-DOF linearised inverted pendulum with triplet):

  Generalised coordinates: q = [x, θ, φ]
    x — forward position of the robot (positive = forward)
    θ — body pitch angle (positive = lean forward, destabilising)
    φ — triplet angle relative to body (both sides, same direction)

  Ground-coupled mass matrix (includes belt-coupled wheel rolling):

           [M_tot+β          −(m_b·l+β·R)       −β·R          ]
      M =  [−(m_b·l+β·R)     I_eff+I_h+β·R²     I_h+β·R²     ]
           [−β·R              I_h+β·R²            I_h+β·R²     ]

      G = [0,  m_b·g·l·θ + g_trip·(θ+φ),  g_trip·(θ+φ)]'

  where β      = I_spin_total / r² ≈ 3·m_wheel (belt-coupled rolling inertia),
        I_h    = 2 × single-side triplet inertia (both sides move together),
        I_eff  = I_b + m_b·l²  (parallel-axis theorem),
        M_tot  = m_b + m_w,
        R      = triplet radius (hub to wheel centre),
        g_trip = triplet gravity coupling (0 for 4WD, m_trip·g·R for 2WD).

  Unlike a reaction wheel (M_xφ=0, M_φφ=I_t), the grounded triplet
  couples to horizontal motion through the rolling constraint, and its
  effective φ inertia is I_h + β·R² ≫ I_h alone.

  Generalised forces:
    τ_w: Q = [1/r, +1,  0]'   (ground reaction + motor-stator reaction on body)
    τ_t: Q = [ 0,  -1, +1]'   (body reaction + triplet drive)

  Note on triplet DOF:  The drive motor reaction torque on the triplet is
  explicitly cancelled in the sim (triplet_total = -motor_torque + triplet_cmd),
  so the triplet link only feels +triplet_cmd.  The body receives +motor_torque
  (from the triplet joint reaction) and -triplet_cmd (triplet motor reaction).
  Hence Q_θ = +1 for τ_w and Q_θ = -1 for τ_t, Q_φ = 0 for τ_w.

Tuning guide:
  ELQR_Q_DIAG = [pos, vel, pitch, pitch_rate, trip_angle, trip_rate]
  ELQR_R_DIAG = [R_wheels, R_triplet]
  ELQR_TRANSITION_Q_DIAG = [...] — during mode transitions
  ELQR_TRANSITION_R_DIAG = [...]
  ELQR_FRICTION_FF        — Nm total paired feedforward during ramp
  ELQR_REF_RAMP_RATE      — rad/s ramp speed for φ_ref

Inputs:  measured pitch, gyro rate, forward position, yaw rate
Outputs: per-side wheel torques (left, right)
         per-side triplet torques (triplet_torque_L, triplet_torque_R)
"""

import math
import numpy as np

from control_lqr import compute_lqr_gain


# ============================================================================
# Extended plant model (6 states × 2 inputs)
# ============================================================================

def build_extended_state_space(config):
    """
    Build the continuous-time A (6×6), B (6×2) matrices for the linearised
    inverted-pendulum with explicit triplet dynamics and GROUND CONTACT
    coupling.

    State:  x = [position, velocity, pitch, pitch_rate, φ_trip, φ̇_trip]
    Input:  u = [τ_wheels, τ_triplet]

    GROUND-COUPLED MODEL
    --------------------
    The triplet wheels sit on the ground — they are NOT free-spinning
    reaction wheels.  When the triplet rotates, grounded wheel(s) push
    against the ground, creating horizontal forces and moments on the
    robot body.

    The kinetic energy includes the rolling constraint of belt-coupled
    wheels: when any ground wheel rolls, all 3 per side spin due to the
    belt, contributing an effective rolling inertia  β = I_spin_total / r².

    Linearised ground-wheel contact position:
        x_gw ≈ x − R·(θ + φ)

    Rolling KE:  T_roll = ½ β (ẋ − R(θ̇ + φ̇))²

    Full mass matrix (derived from Lagrangian KE with rolling):

           [M_tot+β         −(m_b·l + β·R)      −β·R          ]
      M =  [−(m_b·l + β·R)   I_eff + I_h + β·R²  I_h + β·R²  ]
           [−β·R              I_h + β·R²           I_h + β·R²  ]

    Key differences from the old reaction-wheel model (M_xφ=0, M_φφ=I_t):
      • M_xφ = −β·R ≠ 0 : triplet rotation creates horizontal acceleration
      • M_φφ = I_h + β·R² ≫ I_h : ground rolling greatly increases effective
        triplet inertia → LQR computes appropriately larger gains
      • M_θφ = I_h + β·R² : pitch–triplet coupling increases

    Generalised forces (unchanged, matched to sim torque application):
      τ_w: Q = [1/r, +1,  0]'   (ground reaction + motor-stator reaction)
      τ_t: Q = [ 0,  −1, +1]'   (body reaction + triplet drive)

    Optional parameters:
      ELQR_ROLLING_BETA     – effective rolling mass β (kg).
                              Default ≈ 6 × ½ × m_wheel ≈ 0.081 kg.
      ELQR_TRIPLET_GRAVITY  – effective gravity coupling on φ (Nm/rad).
                              0 = 4WD (bilateral ground support, stable).
                              Positive = destabilising (2WD single contact).
                              Typical 2WD value: m_trip·g·R ≈ 0.39 Nm/rad.
    """
    m_b = config['LQR_BODY_MASS']
    m_h = config['LQR_WHEEL_MASS']       # total hub + wheel mass (both sides)
    l   = config['LQR_COG_HEIGHT']
    I_b = config['LQR_BODY_INERTIA']
    r   = config['WHEEL_RADIUS']
    g   = abs(config['GRAVITY'])
    R   = config.get('TRIPLET_RADIUS', 0.12)

    # Combined hub inertia of both triplet assemblies about hub axis
    I_h = 2.0 * config.get('MPC_TRIPLET_INERTIA', 0.00238)
    # Combined joint damping (both sides)
    d_trip = 2.0 * config.get('TRIPLET_JOINT_DAMPING', 0.05)

    # --- Ground-contact rolling inertia β = I_spin_total / r² ---
    # When any ground wheel rolls, belt coupling spins all 3 per side.
    # For solid-cylinder wheels: I_spin = ½ m_w r²
    # 6 wheels total:  β = 6 × ½ × m_per_wheel = 3 × m_per_wheel
    # With m_per_wheel ≈ 0.027 kg (from URDF):  β ≈ 0.081 kg
    beta = config.get('ELQR_ROLLING_BETA', 3.0 * 0.027)

    I_eff = I_b + m_b * l**2       # body inertia about wheel axis

    # --- 3×3 ground-coupled mass matrix ---
    #
    # Derived from Lagrangian KE (linearised around θ=0, φ=0):
    #   T = ½ m_b (ẋ − l·θ̇)² + ½ I_b θ̇²          (body)
    #     + ½ m_h ẋ²                                 (hub + wheels at axle)
    #     + ½ I_h (θ̇ + φ̇)²                          (hub rotation)
    #     + ½ β (ẋ − R(θ̇ + φ̇))²                     (wheel rolling)
    #
    M = np.array([
        [ m_b + m_h + beta,
         -(m_b * l + beta * R),
         -beta * R],
        [-(m_b * l + beta * R),
          I_eff + I_h + beta * R**2,
          I_h + beta * R**2],
        [-beta * R,
          I_h + beta * R**2,
          I_h + beta * R**2],
    ])

    Mi = np.linalg.inv(M)

    # --- Gravity coupling ---
    # From Euler–Lagrange:
    #   G_x = 0
    #   G_θ = m_b·g·l·sin(θ) ≈ m_b·g·l · θ   (body inverted pendulum)
    #   G_φ = g_trip·sin(θ+φ) ≈ g_trip·(θ+φ)  (triplet ground-contact effect)
    #
    # g_trip = 0 for 4WD (bilateral ground support prevents triplet toppling).
    # g_trip ≈ m_trip·g·R for 2WD (single contact, inverted-pendulum-like).
    g_trip = config.get('ELQR_TRIPLET_GRAVITY', 0.0)

    # Acceleration contributions from θ and φ displacements:
    grav_per_theta = np.array([0.0, m_b * g * l + g_trip, g_trip])
    grav_per_phi   = np.array([0.0, g_trip,                g_trip])

    accel_from_theta = Mi @ grav_per_theta   # [ẍ, θ̈, φ̈] per unit θ
    accel_from_phi   = Mi @ grav_per_phi     # [ẍ, θ̈, φ̈] per unit φ

    # --- Damping: joint friction ∝ −d_trip · φ̇ ---
    damp_force = np.array([0.0, 0.0, -d_trip])
    accel_from_phidot = Mi @ damp_force

    # --- A matrix (6×6): ẋ = A·x ---
    A = np.zeros((6, 6))
    A[0, 1] = 1.0                          # ẋ = v
    A[1, 2] = accel_from_theta[0]          # v̇ ← θ  (gravity coupling)
    A[1, 4] = accel_from_phi[0]            # v̇ ← φ  (gravity + ground contact)
    A[1, 5] = accel_from_phidot[0]         # v̇ ← φ̇  (triplet damping → x)
    A[2, 3] = 1.0                          # θ̇ = ω
    A[3, 2] = accel_from_theta[1]          # ω̇ ← θ  (gravity, unstable pole)
    A[3, 4] = accel_from_phi[1]            # ω̇ ← φ  (gravity + ground contact)
    A[3, 5] = accel_from_phidot[1]         # ω̇ ← φ̇  (triplet damping → pitch)
    A[4, 5] = 1.0                          # φ̇ = φ_rate
    A[5, 2] = accel_from_theta[2]          # φ̈ ← θ  (gravity coupling)
    A[5, 4] = accel_from_phi[2]            # φ̈ ← φ  (gravity + ground contact)
    A[5, 5] = accel_from_phidot[2]         # φ̈ ← φ̇  (damping)

    # --- B matrix (6×2): B_accel = M⁻¹ · B_gf ---
    # Generalised-force input matrix (matched to sim torque application):
    #   τ_w → [1/r, +1, 0]'   (ground force + motor-body reaction)
    #   τ_t → [0,   -1, +1]'  (body reaction + triplet drive)
    # Note: B_gf is independent of the mass matrix.  The grounded-contact
    # physics enter through M (inertial coupling), not through Q (forces).
    B_gf = np.array([
        [1.0 / r,  0.0],
        [1.0,     -1.0],
        [0.0,      1.0],
    ])
    B_accel = Mi @ B_gf   # 3×2

    B = np.zeros((6, 2))
    B[1, :] = B_accel[0, :]   # v̇
    B[3, :] = B_accel[1, :]   # ω̇
    B[5, :] = B_accel[2, :]   # φ̈

    return A, B


# ============================================================================
# Extended LQR Balance Controller
# ============================================================================

class ExtendedLQRController:
    """
    6-state LQR with integrated triplet angle reference tracking and
    gain-scheduled transitions.

    Three gain sets:
      K_normal      – steady-state 4WD / 2WD balance
      K_transition  – active during mode-switch ramp (high triplet Q,
                      relaxed pitch Q so the robot can lean through)
      K_aggressive  – position tracking when far from target

    During transitions, a paired friction feedforward adds a constant triplet
    push (to overcome ground friction) together with a proportional wheel push
    (to cancel the resulting body-pitch disturbance at the linear-model level).

    Config keys (in addition to standard LQR plant keys):
        ELQR_Q_DIAG              – 6-element diagonal Q (normal)
        ELQR_R_DIAG              – 2-element diagonal R (normal)
        ELQR_TRANSITION_Q_DIAG   – 6-element diagonal Q (transition)
        ELQR_TRANSITION_R_DIAG   – 2-element diagonal R (transition)
        ELQR_FRICTION_FF         – Nm total paired feedforward during ramp
        ELQR_TRANSITION_SETTLE   – s post-ramp settling time
        ELQR_PITCH_SAFETY_LIMIT  – rad — reduce trip torque above this pitch
        ELQR_REF_RAMP_RATE       – rad/s ramp speed for mode transitions
        MAX_TORQUE                – per-side wheel torque limit (Nm)
        MAX_TRIPLET_TORQUE        – per-side triplet torque limit (Nm)
    """

    ROTATION_STEP = math.pi / 3.0   # 60° per mode change

    def __init__(self, config):
        self.cfg = config

        # Build plant model
        A, B = build_extended_state_space(config)
        self._A, self._B = A, B

        # --- Normal LQR gain (2×6) ---
        Q = np.diag(config['ELQR_Q_DIAG'])
        R = np.diag(config['ELQR_R_DIAG'])
        self.K_normal = compute_lqr_gain(A, B, Q, R)

        print("  Extended LQR (6-state, 2-input, ground-coupled):")
        _beta = config.get('ELQR_ROLLING_BETA', 3.0 * 0.027)
        _R = config.get('TRIPLET_RADIUS', 0.12)
        _Ih = 2.0 * config.get('MPC_TRIPLET_INERTIA', 0.00238)
        _g_trip = config.get('ELQR_TRIPLET_GRAVITY', 0.0)
        print(f"    Ground coupling: β={_beta:.4f} kg, βR²={_beta*_R**2:.6f}, "
              f"I_hub={_Ih:.6f}, I_eff_trip={_Ih+_beta*_R**2:.6f}")
        print(f"    Triplet gravity: g_trip={_g_trip:.3f} Nm/rad")
        print(f"    K_wheels  = [{', '.join(f'{k:.4f}' for k in self.K_normal[0])}]")
        print(f"    K_triplet = [{', '.join(f'{k:.4f}' for k in self.K_normal[1])}]")
        print(f"    Q_diag = {config['ELQR_Q_DIAG']}")
        print(f"    R_diag = {config['ELQR_R_DIAG']}")

        # Closed-loop eigenvalue check
        A_cl = A - B @ self.K_normal
        eigvals = np.linalg.eigvals(A_cl)
        stable = all(e.real < 0 for e in eigvals)
        print(f"    Eigenvalues: "
              f"{[f'{e.real:.2f}{e.imag:+.2f}j' for e in eigvals]}")
        print(f"    Stable: {'✓' if stable else '✗ WARNING: UNSTABLE!'}")

        # --- Transition LQR gain (2×6) ---
        if 'ELQR_TRANSITION_Q_DIAG' in config:
            Q_trans = np.diag(config['ELQR_TRANSITION_Q_DIAG'])
            R_trans = np.diag(config['ELQR_TRANSITION_R_DIAG'])
            self.K_transition = compute_lqr_gain(A, B, Q_trans, R_trans)
            A_cl_t = A - B @ self.K_transition
            eigvals_t = np.linalg.eigvals(A_cl_t)
            stable_t = all(e.real < 0 for e in eigvals_t)
            print(f"    K_trans_wheels  = "
                  f"[{', '.join(f'{k:.4f}' for k in self.K_transition[0])}]")
            print(f"    K_trans_triplet = "
                  f"[{', '.join(f'{k:.4f}' for k in self.K_transition[1])}]")
            print(f"    Trans Q = {config['ELQR_TRANSITION_Q_DIAG']}")
            print(f"    Trans R = {config['ELQR_TRANSITION_R_DIAG']}")
            print(f"    Trans eigenvalues: "
                  f"{[f'{e.real:.2f}{e.imag:+.2f}j' for e in eigvals_t]}")
            print(f"    Trans stable: "
                  f"{'✓' if stable_t else '✗ WARNING: UNSTABLE!'}")
        else:
            self.K_transition = None

        # --- Gain-scheduled aggressive mode ---
        if 'ELQR_AGGRESSIVE_Q_DIAG' in config:
            Q_agg = np.diag(config['ELQR_AGGRESSIVE_Q_DIAG'])
            R_agg = np.diag(config['ELQR_AGGRESSIVE_R_DIAG'])
            self.K_aggressive = compute_lqr_gain(A, B, Q_agg, R_agg)
            self.switch_threshold = config.get('ELQR_SWITCH_THRESHOLD', 0.20)
            self.switch_hysteresis = config.get('ELQR_SWITCH_HYSTERESIS', 0.05)
            print(f"    K_agg_wheels  = "
                  f"[{', '.join(f'{k:.4f}' for k in self.K_aggressive[0])}]")
            print(f"    K_agg_triplet = "
                  f"[{', '.join(f'{k:.4f}' for k in self.K_aggressive[1])}]")
        else:
            self.K_aggressive = None
            self.switch_threshold = 0.0
            self.switch_hysteresis = 0.0

        self.K = self.K_normal
        self.aggressive_active = False

        # --- Friction feedforward (paired wheel + triplet) ---
        self._friction_ff = config.get('ELQR_FRICTION_FF', 0.0)
        # Compute wheel compensation ratio to cancel pitch disturbance
        # from the triplet feedforward at the linear-model level.
        # Body pitch acc from τ_t: B[3,1];  from τ_w: B[3,0].
        # To cancel:  ff_wheel = -B[3,1]/B[3,0] × ff_triplet
        if abs(B[3, 0]) > 1e-6:
            self._wheel_ff_ratio = -B[3, 1] / B[3, 0]
        else:
            self._wheel_ff_ratio = 0.0
        print(f"    Friction FF: {self._friction_ff:.1f} Nm  "
              f"(wheel comp ratio: {self._wheel_ff_ratio:.3f})")

        # --- Transition timing ---
        self._trip_ref_rate = config.get('ELQR_REF_RAMP_RATE', 0.5)
        self._transition_settle = config.get('ELQR_TRANSITION_SETTLE', 0.5)
        self._pitch_safety = config.get('ELQR_PITCH_SAFETY_LIMIT', 0.35)
        self._transitioning = False     # True during ramp + settling
        self._transition_end_time = 0.0 # sim_time when settling ends

        # --- Triplet reference tracking ---
        self._trip_ref = 0.0            # current ramped reference (rad)
        self._trip_ref_target = 0.0     # destination reference (rad)
        self._mode = 0                  # 0 = 4WD, 1 = 2WD

        # --- Triplet state (fed from sim via set_triplet_state) ---
        self._trip_angle = 0.0
        self._trip_rate = 0.0

        # --- Standard interface (backward-compatible with LQR / Aug LQR) ---
        self.target_position = 0.0
        self.target_pitch = 0.0         # always 0
        self.control_torque = 0.0       # wheel torque command (total)
        self.triplet_torque_cmd = 0.0   # triplet torque command (total)
        self.triplet_torque_L = 0.0
        self.triplet_torque_R = 0.0

        # Per-state contributions for PlotJuggler (backward compat: 4-element)
        self.K_contributions = np.zeros(4)
        self.K_contributions_trip = np.zeros(4)
        self.state_error = np.zeros(4)

        # Full 6-state contributions (new telemetry)
        self.K_contributions_full = np.zeros(6)
        self.K_contributions_trip_full = np.zeros(6)
        self.state_error_full = np.zeros(6)

        # --- Velocity estimation ---
        self.velocity = 0.0
        self.prev_position = 0.0
        self.prev_vel_time = 0.0
        self.vel_filter_alpha = 0.1

        # --- Control loop timing ---
        self.control_period = 1.0 / config['CONTROL_RATE_HZ']
        self.next_control_time = 0.0

        # --- Sensor-to-actuator delay buffer ---
        delay_steps = config['SENSOR_TO_ACTUATOR_DELAY_STEPS']
        self.torque_delay_buffer = [(0.0, 0.0, 0.0)] * (delay_steps + 1)

        # --- Torque limits ---
        self.max_wheel_torque = config['MAX_TORQUE']
        self.max_triplet_torque = config.get('MAX_TRIPLET_TORQUE', 5.0)

        # --- Yaw ---
        self.yaw_rate_setpoint = 0.0

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    @property
    def mode(self) -> int:
        """Current destination mode: 0 = 4WD, 1 = 2WD."""
        return self._mode

    @property
    def transition_active(self) -> bool:
        """True while the triplet reference is ramping or settling."""
        return self._transitioning

    def set_mode(self, mode: int):
        """Set target mode.  φ_ref ramps smoothly toward the target.

        CRITICAL: start the ramp from the CURRENT measured triplet angle,
        not from zero.  In 4WD the triplet drifts to ~pitch_angle (stays
        world-vertical while body leans), so if we ramp from 0° we get a
        large positive initial trip_err → massive NEGATIVE torque pushing
        the triplet backward during the first ~20 s of transition.
        """
        self._mode = mode
        self._trip_ref = self._trip_angle   # ← START FROM ACTUAL POSITION
        self._trip_ref_target = self.ROTATION_STEP if mode == 1 else 0.0
        self._transitioning = True

    def toggle_mode(self) -> int:
        """Toggle 4WD ↔ 2WD.  Returns the new destination mode."""
        self.set_mode(1 - self._mode)
        return self._mode

    # ------------------------------------------------------------------
    # Standard setters
    # ------------------------------------------------------------------

    def set_target_position(self, position):
        """Set the desired forward position (m)."""
        self.target_position = position

    def set_yaw_rate(self, yaw_rate):
        """Set desired yaw rate (rad/s). 0 = drive straight."""
        self.yaw_rate_setpoint = yaw_rate

    def set_triplet_state(self, angle_L, angle_R, rate_L, rate_R):
        """
        Receive per-side triplet encoder readings.

        GEOMETRY NOTE: L and R triplets are physically mirrored.  Both joints
        share the same +Y axis in the URDF body frame, but the L arm sits at
        y=-0.0825 m and the R arm at y=+0.0825 m.  On the LEFT side, a positive
        joint angle corresponds to the wheel assembly rotating counterclockwise
        (viewed from the robot front).  On the RIGHT side, the geometry is
        mirrored, so a NEGATIVE joint angle is the equivalent motion.

        Combined (anti-symmetric) metric:
            φ = (angle_L − angle_R) / 2

        This has the property that:
          • In 4WD, both joints drift by the same amount as the body pitches
            (both track pitch as gravity keeps them world-vertical), so
            (L − R) / 2 ≈ 0 → no spurious LQR triplet correction.
          • In 2WD, target is L=+60°, R=−60°, so φ_target = +60°.
          • Compatible with the B-matrix model (φ represents same-direction
            wheel-flip on both sides).
        """
        self._trip_angle = (angle_L - angle_R) / 2.0
        self._trip_rate  = (rate_L  - rate_R)  / 2.0

    # ------------------------------------------------------------------
    # Control update
    # ------------------------------------------------------------------

    def update(self, measured_pitch, measured_pitch_rate,
               position, yaw_rate, sim_time, dt):
        """
        Run one controller tick.

        Returns:
            (left_torque, right_torque): commanded WHEEL motor torques (Nm)

        Side effects:
            Sets self.triplet_torque_L, self.triplet_torque_R for the
            triplet motor commands (read by tribot_sim.py).
        """
        # --- Ramp triplet reference toward target ---
        ramp_active = False
        ref_err = self._trip_ref_target - self._trip_ref
        if abs(ref_err) > 1e-4:
            step = self._trip_ref_rate * dt
            self._trip_ref += math.copysign(min(step, abs(ref_err)), ref_err)
            ramp_active = True
            # While ramping, keep pushing out the settle deadline
            self._transition_end_time = sim_time + self._transition_settle

        # --- Transition state machine ---
        if self._transitioning:
            if not ramp_active and sim_time >= self._transition_end_time:
                self._transitioning = False

        # --- LQR update at CONTROL_RATE_HZ ---
        jitter = (np.random.normal(0, self.cfg['CONTROL_JITTER_STD'])
                  if self.cfg.get('ADD_SENSOR_NOISE', False) else 0)

        if sim_time >= self.next_control_time:
            self.next_control_time = sim_time + self.control_period + jitter

            # Velocity estimation (finite difference, low-pass filtered)
            vel_dt = (sim_time - self.prev_vel_time
                      if self.prev_vel_time > 0 else self.control_period)
            if vel_dt > 0:
                raw_vel = (position - self.prev_position) / vel_dt
                self.velocity += self.vel_filter_alpha * (raw_vel - self.velocity)
            self.prev_position = position
            self.prev_vel_time = sim_time

            # --- 6-state error vector ---
            x = np.array([
                position - self.target_position,     # position error
                self.velocity,                        # forward velocity
                measured_pitch,                       # pitch (ref = 0)
                measured_pitch_rate,                  # pitch rate
                self._trip_angle - self._trip_ref,   # triplet angle error
                self._trip_rate,                      # triplet angular rate
            ])

            # Store for logging
            self.state_error = x[:4].copy()
            self.state_error_full = x.copy()

            # --- Gain scheduling: transition > aggressive > normal ---
            if self._transitioning and self.K_transition is not None:
                K_use = self.K_transition
            else:
                pos_err = abs(x[0])
                if self.K_aggressive is not None:
                    if (not self.aggressive_active
                            and pos_err > self.switch_threshold):
                        self.aggressive_active = True
                    elif (self.aggressive_active
                          and pos_err < self.switch_threshold
                                        - self.switch_hysteresis):
                        self.aggressive_active = False
                K_use = (self.K_aggressive if self.aggressive_active
                         else self.K_normal)
            self.K = K_use

            # --- u = -K @ x → [τ_wheels, τ_triplet] ---
            u = -self.K @ x

            # --- Paired friction feedforward during ramp ---
            # Adds a constant triplet push to overcome ground friction in 4WD,
            # paired with a wheel torque that cancels the resulting pitch
            # disturbance at the linear-model level.
            if ramp_active and self._friction_ff > 0:
                ramp_dir = math.copysign(1.0, ref_err)
                ff_trip = ramp_dir * self._friction_ff
                ff_wheel = self._wheel_ff_ratio * ff_trip
                u[0] += ff_wheel
                u[1] += ff_trip

            # --- Pitch safety: scale back triplet torque if pitch is large ---
            # Prevents the transition from driving the robot past recoverable
            # angles.  Smoothly fades triplet torque to zero between
            # PITCH_SAFETY_LIMIT and 2×PITCH_SAFETY_LIMIT.
            abs_pitch = abs(measured_pitch)
            if abs_pitch > self._pitch_safety:
                fade = max(0.0, 1.0 - (abs_pitch - self._pitch_safety)
                                       / self._pitch_safety)
                u[1] *= fade

            # Wheel torque
            wheel_torque = float(np.clip(
                u[0], -self.max_wheel_torque, self.max_wheel_torque))
            self.control_torque = wheel_torque
            self.K_contributions = self.K[0, :4] * x[:4]
            self.K_contributions_full = self.K[0] * x

            # Triplet torque (total for both sides)
            triplet_total = float(np.clip(
                u[1],
                -2.0 * self.max_triplet_torque,
                 2.0 * self.max_triplet_torque))
            self.triplet_torque_cmd = triplet_total
            self.K_contributions_trip = self.K[1, :4] * x[:4]
            self.K_contributions_trip_full = self.K[1] * x

            # Yaw damping (differential wheel torque)
            yaw_correction = (self.cfg['YAW_DAMPING_K']
                              * (yaw_rate - self.yaw_rate_setpoint))

            self.torque_delay_buffer.append(
                (wheel_torque, yaw_correction, triplet_total))

        # === Pop delayed commands ===
        delay_depth = self.cfg['SENSOR_TO_ACTUATOR_DELAY_STEPS'] + 1
        if len(self.torque_delay_buffer) > delay_depth:
            delayed_wheel, delayed_yaw, delayed_trip = \
                self.torque_delay_buffer.pop(0)
        else:
            delayed_wheel, delayed_yaw, delayed_trip = \
                self.torque_delay_buffer[0]

        # Per-side wheel torques (with yaw correction)
        left_torque = delayed_wheel - delayed_yaw
        right_torque = delayed_wheel + delayed_yaw

        # Per-side triplet torques — ANTI-SYMMETRIC split (mirrored geometry).
        # L and R triplets share the +Y joint axis but are physically mirrored.
        # A positive combined torque must drive:
        #   L: +torque → positive joint angle (arm rotates CCW from front)
        #   R: -torque → negative joint angle (arm rotates CCW from front too)
        # This produces the SAME physical wheel-flip direction on both sides.
        self.triplet_torque_L =  delayed_trip / 2.0
        self.triplet_torque_R = -delayed_trip / 2.0

        return left_torque, right_torque
