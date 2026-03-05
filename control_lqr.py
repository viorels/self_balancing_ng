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
    m_b = config['LQR_BODY_MASS']
    m_w = config['LQR_WHEEL_MASS']
    l   = config['LQR_COG_HEIGHT']
    I_b = config['LQR_BODY_INERTIA']
    r   = config['WHEEL_RADIUS']
    g   = abs(config['GRAVITY'])

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

class LQRBalanceController:
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
        Q = np.diag(config['LQR_Q_DIAG'])
        R = np.array([[config['LQR_R']]])
        self.K_normal = compute_lqr_gain(A, B, Q, R)

        print(f"  LQR gain K_normal    = [{', '.join(f'{k:.4f}' for k in self.K_normal[0])}]")
        print(f"  LQR Q_diag = {config['LQR_Q_DIAG']},  R = {config['LQR_R']}")

        # --- Gain-scheduled aggressive mode ---
        if 'LQR_AGGRESSIVE_Q_DIAG' in config:
            Q_agg = np.diag(config['LQR_AGGRESSIVE_Q_DIAG'])
            R_agg = np.array([[config['LQR_AGGRESSIVE_R']]])
            self.K_aggressive = compute_lqr_gain(A, B, Q_agg, R_agg)
            self.switch_threshold = config.get('LQR_SWITCH_THRESHOLD', 0.20)
            self.switch_hysteresis = config.get('LQR_SWITCH_HYSTERESIS', 0.05)
            print(f"  LQR gain K_aggressive= [{', '.join(f'{k:.4f}' for k in self.K_aggressive[0])}]")
            print(f"  LQR Q_agg = {config['LQR_AGGRESSIVE_Q_DIAG']},  R_agg = {config['LQR_AGGRESSIVE_R']}")
            print(f"  Switch: |err|>{self.switch_threshold}m → aggressive, "
                  f"<{self.switch_threshold - self.switch_hysteresis}m → normal")
        else:
            self.K_aggressive = None
            self.switch_threshold = 0.0
            self.switch_hysteresis = 0.0

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
        self.control_period = 1.0 / config['CONTROL_RATE_HZ']
        self.next_control_time = 0.0

        # --- Sensor-to-actuator delay buffer ---
        delay_steps = config['SENSOR_TO_ACTUATOR_DELAY_STEPS']
        self.torque_delay_buffer = [(0.0, 0.0)] * (delay_steps + 1)

        # --- Exposed for logging ---
        self.control_torque = 0.0
        self.target_pitch = 0.0       # mirrors target_lean for log compat

        # --- Per-state torque contributions (for PlotJuggler / debug) ---
        self.K_contributions = np.zeros(4)  # K[0]*x_pos, K[1]*x_vel, K[2]*x_pitch, K[3]*x_prate
        self.state_error = np.zeros(4)

        # --- Yaw rate setpoint (for joystick control) ---
        self.yaw_rate_setpoint = 0.0

        # --- Lean setpoint (left-joystick lean command, rad) ---
        # The controller receives (measured_pitch - target_lean) so an
        # intentional user lean is not treated as an error to correct.
        self.target_lean = 0.0

        # --- LQR-implied desired lean (computed each control tick) ---
        # Exposed so the triplet PD can cooperate with the lean the LQR
        # needs for position tracking.
        self.desired_lean = 0.0

    def set_target_position(self, position):
        """Set the desired forward position (m)."""
        self.target_position = position

    def set_yaw_rate(self, yaw_rate):
        """Set desired yaw rate (rad/s). 0 = drive straight."""
        self.yaw_rate_setpoint = yaw_rate

    def set_lean(self, lean_rad):
        """Set desired lean angle (rad). Positive = lean forward.

        The controller will see (measured_pitch - lean_rad) as the pitch
        error, so the robot leans to the requested angle without fighting it.
        """
        self.target_lean = lean_rad
        self.target_pitch = lean_rad   # keep log field in sync

    def update(self, measured_pitch, measured_pitch_rate,
               position, yaw_rate, sim_time, dt):
        """
        Run one controller tick.

        Args:
            measured_pitch:      fused pitch angle (rad)
            measured_pitch_rate: gyro pitch rate (rad/s)
            position:            forward position estimate (m)
            yaw_rate:            body-frame yaw rate (rad/s)
            sim_time:            current simulation time (s)
            dt:                  physics timestep (s)

        Returns:
            (left_torque, right_torque): commanded motor torques (Nm)
        """
        # --- Velocity estimation (only at control rate to avoid noise) ---
        # Estimating at 500Hz physics rate amplifies tiny position jitter.
        # Instead, update velocity only when the control loop fires.

        # --- LQR update at CONTROL_RATE_HZ ---
        jitter = (np.random.normal(0, self.cfg['CONTROL_JITTER_STD'])
                  if self.cfg.get('ADD_SENSOR_NOISE', False) else 0)

        if sim_time >= self.next_control_time:
            self.next_control_time = sim_time + self.control_period + jitter

            # Velocity estimated over the control period (not physics dt)
            vel_dt = sim_time - self.prev_vel_time if self.prev_vel_time > 0 else self.control_period
            if vel_dt > 0:
                raw_vel = (position - self.prev_position) / vel_dt
                self.velocity += self.vel_filter_alpha * (raw_vel - self.velocity)
            self.prev_position = position
            self.prev_vel_time = sim_time

            # State error vector
            # Subtract the user-requested lean so the controller does not
            # try to correct an intentional lean commanded via the joystick.
            x = np.array([
                position - self.target_position,
                self.velocity,
                measured_pitch - self.target_lean,
                measured_pitch_rate,
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
                self.desired_lean = -(K[0] * x[0] + K[1] * x[1]) / K[2]
            else:
                self.desired_lean = 0.0

            # u = -K x  (total torque for both sides)
            u_raw = float(-self.K @ x)
            u = np.clip(u_raw, -self.cfg['MAX_TORQUE'], self.cfg['MAX_TORQUE'])
            commanded_torque = float(u)
            self.control_torque = commanded_torque

            # Yaw damping relative to setpoint
            yaw_correction = self.cfg['YAW_DAMPING_K'] * (yaw_rate - self.yaw_rate_setpoint)

            self.torque_delay_buffer.append((commanded_torque, yaw_correction))

        # === Pop delayed torque command ===
        delay_depth = self.cfg['SENSOR_TO_ACTUATOR_DELAY_STEPS'] + 1
        if len(self.torque_delay_buffer) > delay_depth:
            delayed_torque, delayed_yaw = self.torque_delay_buffer.pop(0)
        else:
            delayed_torque, delayed_yaw = self.torque_delay_buffer[0]

        # Per-side torques (left −yaw, right +yaw)
        # l_triplet is at -Y (robot's left from behind), r_triplet at +Y (right).
        # Positive yaw_correction → more torque on right side → turns right.
        left_torque = delayed_torque - delayed_yaw
        right_torque = delayed_torque + delayed_yaw

        return left_torque, right_torque
