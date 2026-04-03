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

        # --- Exposed for logging ---
        self.control_torque = 0.0
        self.target_pitch = 0.0       # mirrors target_lean for log compat
        self._left_torque = 0.0
        self._right_torque = 0.0

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
            u = np.clip(u_raw, -self.cfg.motor.max_torque, self.cfg.motor.max_torque)
            commanded_torque = float(u)
            self.control_torque = commanded_torque

            # Yaw damping relative to setpoint
            yaw_correction = self.cfg.control.yaw_damping_k * (yaw_rate - self.yaw_rate_setpoint)

            # Per-side torques (left −yaw, right +yaw)
            # l_triplet is at -Y (robot's left from behind), r_triplet at +Y (right).
            # Positive yaw_correction → more torque on right side → turns right.
            self._left_torque = commanded_torque - yaw_correction
            self._right_torque = commanded_torque + yaw_correction

        return self._left_torque, self._right_torque
