"""
LQR Balance Controller for Self-Balancing Robots

Full-state feedback using the linearised inverted-pendulum-on-wheels model.
State:  x = [position, velocity, pitch, pitch_rate]
Input:  u = total wheel torque (Nm, both sides combined)

The continuous-time plant is derived from first principles and the gain
matrix K is computed by solving the continuous algebraic Riccati equation
(CARE).  At runtime, the controller simply computes:

    u = -K @ (x - x_ref)

with torque clamping and yaw damping identical to the PID controller,
plus the same sensor-to-actuator delay pipeline for realism.

Inputs:  measured pitch, gyro rate, forward position, yaw rate
Outputs: per-side commanded torques (left, right)
"""

import math
import numpy as np

from .base import BalanceControllerBase, StateReference


# ============================================================================
# Algebraic Riccati solver (no scipy dependency)
# ============================================================================

def _solve_care(A, B, Q, R):
    """
    Solve the continuous-time algebraic Riccati equation:
        A'P + PA - PBR^{-1}B'P + Q = 0
    using the Schur / eigendecomposition method on the Hamiltonian matrix.

    Returns P (symmetric positive-definite solution).
    """
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    BR_inv_BT = B @ R_inv @ B.T

    # Hamiltonian matrix
    H = np.block([
        [A, -BR_inv_BT],
        [-Q, -A.T]
    ])

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eig(H)

    # Select the n eigenvectors with negative real part (stable subspace)
    idx = np.argsort(eigvals.real)
    stable_idx = idx[:n]
    U = eigvecs[:, stable_idx]

    U1 = U[:n, :]
    U2 = U[n:, :]

    # P = U2 @ inv(U1)
    P = np.real(U2 @ np.linalg.inv(U1))
    # Ensure symmetry
    P = (P + P.T) / 2.0
    return P


def compute_lqr_gain(A, B, Q, R):
    """
    Compute LQR gain K such that u = -K x minimises
    J = ∫ (x'Qx + u'Ru) dt.

    Returns K = R^{-1} B' P.
    """
    P = _solve_care(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


# ============================================================================
# Linearised plant model
# ============================================================================

def build_state_space(config):
    """
    Build the continuous-time A, B matrices for the linearised
    inverted-pendulum-on-wheels system.

    State:  x = [position, velocity, pitch, pitch_rate]
    Input:  u = total motor torque (Nm)

    Physical parameters (from config):
        BODY_MASS       – mass of the body above the wheel axis (kg)
        WHEEL_MASS      – total wheel/triplet mass (kg)
        COG_HEIGHT      – distance from wheel axis to body CoG (m)
        BODY_INERTIA    – body pitch inertia about its CoG (kg·m²)
        WHEEL_RADIUS    – effective wheel radius (m)
    """
    m_b = config.plant.body_mass
    m_w = config.plant.wheel_mass
    l   = config.plant.cog_height
    I_b = config.plant.body_inertia
    r   = config.robot.wheel_radius
    g   = abs(config.sim.gravity)

    # Effective rotational inertia about the wheel contact point
    I_eff = I_b + m_b * l**2       # parallel-axis theorem
    M_tot = m_b + m_w              # total translational mass

    # Coupled mass matrix:
    #   [M_tot   m_b*l] [x_ddot ]   [  0  ] [x  ]   [+1/r]
    #   [m_b*l   I_eff] [θ_ddot ] = [m_b*g*l] [θ  ] + [ +1 ] u
    #
    # x is the robot's forward distance (positive = forward).
    # In the simulation, positive commanded torque → negative wheel joint
    # torque (URDF axis negation) → robot moves in -X world → forward
    # distance increases.  So the effective input vector is [+1/r; +1].
    #
    # Gravity coupling: forward lean (positive θ) → forward acceleration
    # → positive x_ddot.  So a13 is positive.
    #
    # Invert the 2×2 mass matrix to get x_ddot, θ_ddot as functions of θ and u.

    det = M_tot * I_eff - (m_b * l)**2

    # Gravity terms  (only θ column is non-zero)
    # inv(M) @ [0; m_b*g*l]
    a13 =  (m_b * l) * (m_b * g * l) / det    # x_ddot from θ  (coupling)
    a33 =  M_tot     * (m_b * g * l) / det    # θ_ddot from θ

    # Input terms
    # inv(M) @ [+1/r; +1]
    b1 =  (I_eff / (det * r)) + (m_b * l) / det    # x_ddot from u
    b3 =  (m_b * l) / (det * r) + M_tot  / det     # θ_ddot from u

    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, a13, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, a33, 0.0],
    ])

    B = np.array([[0.0], [b1], [0.0], [b3]])

    return A, B


def compute_equivalent_triplet_torque(T_wheel, l, R_trip, contact_angle):
    """
    Return the triplet hub torque that produces the same body pitch effect
    as T_wheel applied to the drive wheels.

    The triplet foot pushes down on the ground asymmetrically.  Ground
    pushes back up at horizontal offset d = R_trip·sin(contact_angle) from
    the hub.  The pitch leverage is this offset relative to CoG height:

        γ = l / (R_trip · |sin(contact_angle)|)

    At contact_angle ≈ 0 the foot is directly below the hub — no horizontal
    offset, no pitch leverage (γ → ∞, clamped).  At 90° the foot is
    maximally offset and most effective.

    Args:
        T_wheel:       drive-wheel torque (Nm)
        l:             CoG height above hub axis (m)
        R_trip:        triplet hub-to-contact radius (m)
        contact_angle: angle of contact point from hub vertical in world
                       frame (rad).  0 = directly below, π/2 = horizontal.

    Returns:
        T_trip (Nm) — equivalent hub torque (before PyBullet sign flip)
    """
    sin_a = max(abs(math.sin(contact_angle)), 0.5)    # floor ≈ 30°
    return T_wheel * l / (R_trip * sin_a)


def compute_max_triplet_torque(weight_R, contact_angle):
    """
    Maximum hub torque before the triplet flips over its ground contact.

    The robot's weight pressing down through the hub creates a restoring
    moment about the foot contact point:

        T_max = (W/2) · R_trip · sin(contact_angle)

    Since weight_R = (W/2) · R_trip is precomputed:

        T_max = weight_R · sin(contact_angle)

    At small angles the foot is below the hub — tiny horizontal lever arm,
    so very little torque before flipping.  At large angles the foot is
    far to the side — weight has a long moment arm, so more torque is safe.

    Args:
        weight_R:      (W/2) · R_trip precomputed (N·m)
        contact_angle: angle of contact point from hub vertical (rad)

    Returns:
        T_max (Nm) — absolute torque limit (symmetric ±)
    """
    sin_a = abs(math.sin(contact_angle))
    return weight_R * sin_a


# ============================================================================
# LQR Balance Controller
# ============================================================================

class LQRBalanceController(BalanceControllerBase):
    """
    Full-state-feedback LQR controller for a self-balancing robot.

    Config keys used (in addition to the plant-model keys above):
        LQR_Q_DIAG       – list of 4 diagonal Q weights [pos, vel, pitch, pitch_rate]
        LQR_R             – scalar R weight (torque penalty)
        MAX_TORQUE        – saturation limit per motor (Nm)
        CONTROL_RATE_HZ   – control update rate
        CONTROL_JITTER_STD
        SENSOR_TO_ACTUATOR_DELAY_STEPS
        ADD_SENSOR_NOISE
        YAW_DAMPING_K
    """

    def __init__(self, config):
        self.cfg = config

        # Build linearised model and compute gain
        A, B = build_state_space(config)
        Q = np.diag(config.lqr.q_diag)
        R = np.array([[config.lqr.r]])
        self.K_normal = compute_lqr_gain(A, B, Q, R)

        print(f"  LQR gain K_normal    = [{', '.join(f'{k:.4f}' for k in self.K_normal[0])}]")
        print(f"  LQR Q_diag = {config.lqr.q_diag},  R = {config.lqr.r}")

        # --- Gain-scheduled aggressive mode ---
        Q_agg = np.diag(config.lqr.aggressive_q_diag)
        R_agg = np.array([[config.lqr.aggressive_r]])
        self.K_aggressive = compute_lqr_gain(A, B, Q_agg, R_agg)
        self.switch_threshold = config.lqr.switch_threshold
        self.switch_hysteresis = config.lqr.switch_hysteresis
        print(f"  LQR gain K_aggressive= [{', '.join(f'{k:.4f}' for k in self.K_aggressive[0])}]")
        print(f"  LQR Q_agg = {config.lqr.aggressive_q_diag},  R_agg = {config.lqr.aggressive_r}")
        print(f"  Switch: |err|>{self.switch_threshold}m → aggressive, "
              f"<{self.switch_threshold - self.switch_hysteresis}m → normal")

        # Active gain (start in normal mode)
        self.K = self.K_normal
        self.aggressive_active = False

        # --- Reference state ---
        self.target_position = 0.0

        # --- Velocity estimation (finite difference at control rate) ---
        self.prev_position = 0.0
        self.prev_vel_time = 0.0
        self.velocity = 0.0
        self.vel_filter_alpha = 0.1   # low-pass on velocity estimate

        # --- Control loop timing ---
        self.control_period = 1.0 / config.control.control_rate_hz
        self.next_control_time = 0.0

        # --- Sensor-to-actuator delay buffer ---
        delay_steps = config.control.sensor_to_actuator_delay_steps
        self.torque_delay_buffer = [(0.0, 0.0, 0.0)] * (delay_steps + 1)

        # --- Exposed for logging ---
        self.control_torque = 0.0
        self.target_pitch = 0.0       # mirrors target_lean for log compat

        # --- Per-state torque contributions (for PlotJuggler / debug) ---
        self.K_contributions = np.zeros(4)  # K[0]*x_pos, K[1]*x_vel, K[2]*x_pitch, K[3]*x_prate
        self.state_error = np.zeros(4)

        # --- Yaw rate setpoint (for joystick control) ---
        self.yaw_rate_setpoint = 0.0

        # --- Lean setpoints ---
        # _requested_lean: raw operator command from gamepad / remote.
        # target_lean:     effective pitch reference (from last ref passed
        #                  to update), kept for telemetry and sim-loop reads.
        self._requested_lean = 0.0
        self.target_lean = 0.0

        # --- LQR-implied desired lean (computed each control tick) ---
        # Exposed so the triplet PD can cooperate with the lean the LQR
        # needs for position tracking.
        self._desired_lean = 0.0

        # --- Lean-assist: dynamic wheel / triplet lean-acceleration split ---
        # gamma(α) = compute_equivalent_triplet_torque(1.0, ..., α) varies
        # with the triplet contact angle — called each tick in update().
        self._lean_cog_height = config.plant.cog_height
        self._lean_R_trip = config.robot.triplet_radius

        # Anti-lift: hub torque vertical component must not exceed weight.
        # T_max = (weight_per_side · R_trip) / |sin(α)|, floored at sin=0.2.
        _M = config.plant.body_mass + config.plant.wheel_mass
        self._lean_weight_R = (_M * abs(config.sim.gravity) / 2) * self._lean_R_trip

        self._triplet_lean_torque = 0.0
        self._debug_u_lean = 0.0
        self._debug_delayed_u_lean = 0.0
        self._debug_gamma = 0.0
        self._debug_lean_ff_raw = 0.0

        _gamma_0  = compute_equivalent_triplet_torque(1.0, self._lean_cog_height, self._lean_R_trip, 0.0)
        _gamma_90 = compute_equivalent_triplet_torque(1.0, self._lean_cog_height, self._lean_R_trip, math.pi / 2)
        print(f"  Lean assist: gamma(0°)={_gamma_0:.2f}, gamma(90°)={_gamma_90:.2f}, "
              f"T_max(30°)={compute_max_triplet_torque(self._lean_weight_R, math.radians(30)):.2f} Nm, "
              f"T_max(60°)={compute_max_triplet_torque(self._lean_weight_R, math.radians(60)):.2f} Nm")

    def reset(self):
        """Zero all internal state for a clean restart.

        Called by robot.reset() to ensure the controller doesn't inject
        stale torques, velocity estimates, or gain-scheduling state from
        a previous run.
        """
        self.target_position = 0.0
        self.yaw_rate_setpoint = 0.0
        self._requested_lean = 0.0
        self.target_lean = 0.0
        self.target_pitch = 0.0
        self._desired_lean = 0.0

        # Velocity estimator — stale prev_position causes a wild velocity
        # spike on the first control tick after reset.
        self.prev_position = 0.0
        self.prev_vel_time = 0.0
        self.velocity = 0.0

        # Control-loop timing — let it fire on the very next tick.
        self.next_control_time = 0.0

        # Torque delay buffer — flush pre-reset saturated torques.
        delay_steps = self.cfg.control.sensor_to_actuator_delay_steps
        self.torque_delay_buffer = [(0.0, 0.0, 0.0)] * (delay_steps + 1)

        self._triplet_lean_torque = 0.0
        self._debug_u_lean = 0.0
        self._debug_delayed_u_lean = 0.0
        self._debug_gamma = 0.0
        self._debug_lean_ff_raw = 0.0

        # Gain scheduling — return to normal mode.
        self.K = self.K_normal
        self.aggressive_active = False

        # Zero telemetry accumulators.
        self.control_torque = 0.0
        self.K_contributions = np.zeros(4)
        self.state_error = np.zeros(4)

    # ----------------------------------------------------------------
    # BalanceControllerBase interface
    # ----------------------------------------------------------------

    def set_target_position(self, position):
        """Set the desired forward position (m)."""
        self.target_position = position

    def set_yaw_rate(self, yaw_rate):
        """Set desired yaw rate (rad/s). 0 = drive straight."""
        self.yaw_rate_setpoint = yaw_rate

    def set_lean(self, lean_rad):
        """Store the raw operator lean command (rad). Positive = forward."""
        self._requested_lean = lean_rad

    @property
    def requested_lean(self) -> float:
        """Raw operator lean command (rad), before triplet coordination."""
        return self._requested_lean

    def set_triplet_state(self, angle_L, angle_R, rate_L, rate_R):
        """Update triplet encoder readings (called each tick from tribot_sim)."""
        self._triplet_angle_L = angle_L
        self._triplet_angle_R = angle_R
        self._triplet_rate_L = rate_L
        self._triplet_rate_R = rate_R

    @property
    def desired_lean(self) -> float:
        return self._desired_lean

    @property
    def triplet_lean_torque(self) -> float:
        return self._triplet_lean_torque

    def get_telemetry(self) -> dict:
        """Return LQR-specific diagnostic signals."""
        return {
            "state_err_pos":    float(self.state_error[0]),
            "state_err_vel":    float(self.state_error[1]),
            "state_err_pitch":  float(self.state_error[2]),
            "state_err_prate":  float(self.state_error[3]),
            "torque_cmd":       float(self.control_torque),
            "K_pos":            float(self.K_contributions[0]),
            "K_vel":            float(self.K_contributions[1]),
            "K_pitch":          float(self.K_contributions[2]),
            "K_pitch_rate":     float(self.K_contributions[3]),
            "desired_lean":     float(self._desired_lean),
            "requested_lean":   float(self._requested_lean),
            "target_pos":       float(self.target_position),
            "target_lean":      float(self.target_lean),
            "target_pitch":     float(self.target_pitch),
            "aggressive":       float(self.aggressive_active),
            "triplet_lean_ff":        float(self._triplet_lean_torque),
            "u_lean":                 float(self._debug_u_lean),
            "delayed_u_lean":         float(self._debug_delayed_u_lean),
            "gamma":                  float(self._debug_gamma),
            "lean_ff_raw":            float(self._debug_lean_ff_raw),
        }

    def update(self, measured_pitch, measured_pitch_rate,
               position, yaw_rate, sim_time, dt,
               ref=None):
        """
        Run one controller tick.

        Args:
            measured_pitch:      fused pitch angle (rad)
            measured_pitch_rate: gyro pitch rate (rad/s)
            position:            forward position estimate (m)
            yaw_rate:            body-frame yaw rate (rad/s)
            sim_time:            current simulation time (s)
            dt:                  physics timestep (s)
            ref:                 StateReference with target [pos, vel, pitch,
                                 pitch_rate].  Built by the robot loop which
                                 owns the trajectory planner and drive-mode
                                 knowledge.

        Returns:
            (left_torque, right_torque): commanded motor torques (Nm)
        """
        # Fallback when no ref provided (e.g. standalone use)
        if ref is None:
            ref = StateReference(
                position=self.target_position,
                pitch=self._requested_lean,
            )

        # Keep telemetry-visible attributes in sync with the ref
        self.target_lean = ref.pitch
        self.target_pitch = ref.pitch

        # --- Velocity estimation (only at control rate to avoid noise) ---
        # Estimating at 500Hz physics rate amplifies tiny position jitter.
        # Instead, update velocity only when the control loop fires.

        # --- LQR update at CONTROL_RATE_HZ ---
        jitter = (np.random.normal(0, self.cfg.control.control_jitter_std)
                  if self.cfg.imu.add_sensor_noise else 0)

        if sim_time >= self.next_control_time:
            self.next_control_time = sim_time + self.control_period + jitter

            # Velocity estimated over the control period (not physics dt)
            vel_dt = sim_time - self.prev_vel_time if self.prev_vel_time > 0 else self.control_period
            if vel_dt > 0:
                raw_vel = (position - self.prev_position) / vel_dt
                self.velocity += self.vel_filter_alpha * (raw_vel - self.velocity)
            self.prev_position = position
            self.prev_vel_time = sim_time

            # State error: measured − reference.
            # The robot loop provides ref.velocity and ref.pitch_rate
            # during trajectory transitions so the LQR doesn't fight
            # the planned motion.
            x = np.array([
                position - ref.position,
                self.velocity - ref.velocity,
                measured_pitch - ref.pitch,
                measured_pitch_rate - ref.pitch_rate,
            ])
            self.state_error = x.copy()

            # --- Gain scheduling: switch K based on position error ---
            pos_err = abs(x[0])
            if self.K_aggressive is not None:
                if not self.aggressive_active and pos_err > self.switch_threshold:
                    self.aggressive_active = True
                    self.K = self.K_aggressive
                elif self.aggressive_active and pos_err < (self.switch_threshold - self.switch_hysteresis):
                    self.aggressive_active = False
                    self.K = self.K_normal

            self.K_contributions = self.K[0] * x  # element-wise: K_i * x_i

            # --- LQR-implied desired lean angle ---
            # The position+velocity terms of K·x represent a "lean demand":
            # the pitch the LQR needs to achieve to drive position error
            # toward zero.  At steady state (pitch_rate=0, u=0):
            #   K_pos·e_pos + K_vel·v + K_pitch·θ_desired = 0
            #   θ_desired = -(K_pos·e_pos + K_vel·v) / K_pitch
            # Expose this so the triplet PD can cooperate instead of fight.
            K = self.K[0]
            if abs(K[2]) > 1e-9:
                self._desired_lean = -(K[0] * x[0] + K[1] * x[1]) / K[2]
            else:
                self._desired_lean = 0.0

            # u = -K x  (total torque for both sides)
            u_raw = float(-self.K @ x)

            # Decompose u_raw into lean demand vs balance correction.
            # u_lean:    -(K₀·x₀ + K₁·x₁) — position+velocity driven.
            #            Smooth, changes on ~0.5 s timescale.  This is the
            #            "create lean to move" signal the triplet should boost.
            # u_balance: -(K₂·x₂ + K₃·x₃) — pitch+pitch_rate corrections.
            #            High-frequency, belongs exclusively to the wheels.
            u_lean = float(-(K[0] * x[0] + K[1] * x[1]))
            u_lean = max(-self.cfg.motor.max_torque,
                         min(self.cfg.motor.max_torque, u_lean))
            self._debug_u_lean = u_lean

            u = np.clip(u_raw, -self.cfg.motor.max_torque, self.cfg.motor.max_torque)
            commanded_torque = float(u)
            self.control_torque = commanded_torque

            # Yaw damping relative to setpoint
            yaw_correction = self.cfg.control.yaw_damping_k * (yaw_rate - self.yaw_rate_setpoint)

            # Store u_lean (not u_raw) for the triplet feedforward.
            self.torque_delay_buffer.append(
                (commanded_torque, yaw_correction, u_lean))

        # === Pop delayed torque command ===
        delay_depth = self.cfg.control.sensor_to_actuator_delay_steps + 1
        if len(self.torque_delay_buffer) > delay_depth:
            delayed_torque, delayed_yaw, delayed_u_lean = self.torque_delay_buffer.pop(0)
        else:
            delayed_torque, delayed_yaw, delayed_u_lean = self.torque_delay_buffer[0]
        self._debug_delayed_u_lean = delayed_u_lean

        # Triplet lean feedforward (per-side, symmetric — no yaw component).
        # Only the lean-demand component (position+velocity) reaches the hub.
        # Balance corrections (pitch+pitch_rate) stay with the wheels — this
        # eliminates the high-frequency sign-flipping that caused hub chatter.
        #
        # Dynamic gamma: the triplet's ground-contact effectiveness depends
        # on cos(contact_angle).  Contact angle ≈ body_pitch + triplet_angle
        # in 4WD (base_angle = 0).
        _avg_trip = (self._triplet_angle_L + self._triplet_angle_R) / 2
        _alpha = measured_pitch + _avg_trip
        _gamma = compute_equivalent_triplet_torque(
            1.0, self._lean_cog_height, self._lean_R_trip, _alpha)
        # Clamp gamma: the geometric formula diverges near α=0 (foot
        # below hub, no pitch leverage).  The sin floor inside the
        # function caps it at ~12; add a safety clamp here too.
        _gamma = max(0.0, min(_gamma, 5.0))
        self._debug_gamma = _gamma

        _raw = -_gamma * delayed_u_lean
        self._debug_lean_ff_raw = _raw

        # Anti-flip clamp: hub torque must not exceed the weight's restoring
        # moment about the foot contact point.
        _T_max = 50 * compute_max_triplet_torque(self._lean_weight_R, _alpha)
        self._triplet_lean_torque = max(-_T_max, min(_T_max, _raw))

        # Per-side torques (left −yaw, right +yaw)
        # l_triplet is at -Y (robot's left from behind), r_triplet at +Y (right).
        # Positive yaw_correction → more torque on right side → turns right.
        left_torque = delayed_torque - delayed_yaw
        right_torque = delayed_torque + delayed_yaw

        return left_torque, right_torque
