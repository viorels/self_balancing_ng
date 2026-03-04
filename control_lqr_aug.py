"""
Augmented LQR Balance Controller with Triplet Torque

Extends the standard inverted-pendulum LQR to output BOTH wheel torque
and triplet torque.  The triplet motor applies direct body-pitch torque
through the grounded triplet assembly (acts as a pivot).

State:  x = [position, velocity, pitch, pitch_rate]   (same 4-state model)
Input:  u = [τ_wheels, τ_triplet]                       (2 inputs)

Physical model:
    The mass matrix is the same standard cart-pole coupling:

        [M_tot   -m_b·l] [ẍ ]   [     0     ] [x]   [1/r   0] [τ_w]
        [-m_b·l   I_eff] [θ̈ ] = [m_b·g·l    ] [θ] + [ 1   -1] [τ_t]

    Generalised forces:
      τ_w:  Q_x = +τ_w/r  (wheel ground reaction)
             Q_θ = +τ_w    (motor stator reaction on body)
      τ_t:  Q_x = 0        (no horizontal force)
             Q_θ = -τ_t    (body gets −τ_t reaction: PyBullet joint convention
                            — positive joint force pushes child +Y, parent −Y)

    So positive τ_t from LQR → body pitch DECREASES.  The LQR gain
    matrix K (2×4) accounts for this automatically.

Tuning guide:
    R_DIAG = [R_wheels, R_triplet]
      • Increase R_wheels → LQR shifts balancing burden to triplet motor,
        freeing wheel torque for locomotion (good for hill climbing).
      • Increase R_triplet → LQR prefers wheels (original behaviour).
      • Equal values → LQR optimally splits based on effectiveness.

Inputs:  measured pitch, gyro rate, forward position, yaw rate
Outputs: per-side wheel torques (left, right)
         per-side triplet torques (triplet_torque_L, triplet_torque_R)
"""

import math
import numpy as np

from control_lqr import compute_lqr_gain


# ============================================================================
# Augmented plant model (4 states × 2 inputs)
# ============================================================================

def build_augmented_state_space(config):
    """
    Build the continuous-time A (4×4), B (4×2) matrices for the linearised
    inverted-pendulum with two torque inputs: wheels + triplet.

    State:  x = [position, velocity, pitch, pitch_rate]
    Input:  u = [τ_wheels, τ_triplet]

    Physical parameters are the same as the standard LQR model (from config).
    """
    m_b = config['LQR_BODY_MASS']
    m_w = config['LQR_WHEEL_MASS']
    l   = config['LQR_COG_HEIGHT']
    I_b = config['LQR_BODY_INERTIA']
    r   = config['WHEEL_RADIUS']
    g   = abs(config['GRAVITY'])

    I_eff = I_b + m_b * l**2       # parallel-axis theorem
    M_tot = m_b + m_w

    det = M_tot * I_eff - (m_b * l)**2

    # --- Gravity coupling (same as standard LQR) ---
    a13 = (m_b * l) * (m_b * g * l) / det      # ẍ from θ
    a33 = M_tot     * (m_b * g * l) / det       # θ̈ from θ

    # --- Input matrix ---
    # Mass-matrix inverse (with M = [[M_tot, -m_b*l], [-m_b*l, I_eff]]):
    #   M⁻¹ = [[I_eff, m_b*l], [m_b*l, M_tot]] / det
    #
    # Generalised force matrix B_gf:
    #   [[1/r, 0], [1, -1]]
    #
    # B_accel = M⁻¹ @ B_gf:
    #   Column 0 (τ_w): M⁻¹ @ [1/r, 1]ᵀ
    #   Column 1 (τ_t): M⁻¹ @ [0, -1]ᵀ

    # τ_w column (identical to existing LQR):
    b1_w = (I_eff / (det * r)) + (m_b * l) / det     # ẍ from τ_w
    b3_w = (m_b * l) / (det * r) + M_tot / det        # θ̈ from τ_w

    # τ_t column (new — direct body torque through grounded triplet):
    b1_t = -(m_b * l) / det                            # ẍ from τ_t
    b3_t = -(M_tot) / det                              # θ̈ from τ_t

    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, a13, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, a33, 0.0],
    ])

    B = np.array([
        [0.0,  0.0 ],
        [b1_w, b1_t],
        [0.0,  0.0 ],
        [b3_w, b3_t],
    ])

    return A, B


# ============================================================================
# Augmented LQR Balance Controller
# ============================================================================

class AugmentedLQRController:
    """
    Full-state-feedback LQR controller that outputs both wheel AND triplet
    torques for a self-balancing robot.

    The K matrix is 2×4:
        K[0,:] → wheel  torque gains for [pos, vel, pitch, pitch_rate]
        K[1,:] → triplet torque gains for [pos, vel, pitch, pitch_rate]

    Config keys (in addition to standard LQR plant keys):
        ALQR_Q_DIAG      – 4-element list: [pos, vel, pitch, pitch_rate]
        ALQR_R_DIAG      – 2-element list: [R_wheels, R_triplet]
        MAX_TORQUE        – per-side wheel torque limit (Nm)
        MAX_TRIPLET_TORQUE – per-side triplet torque limit (Nm)
    """

    def __init__(self, config):
        self.cfg = config

        # Build augmented plant
        A, B = build_augmented_state_space(config)

        # LQR cost matrices
        Q = np.diag(config['ALQR_Q_DIAG'])
        R = np.diag(config['ALQR_R_DIAG'])
        self.K_normal = compute_lqr_gain(A, B, Q, R)

        print("  Augmented LQR (4-state, 2-input):")
        print(f"    K_wheels  = [{', '.join(f'{k:.4f}' for k in self.K_normal[0])}]")
        print(f"    K_triplet = [{', '.join(f'{k:.4f}' for k in self.K_normal[1])}]")
        print(f"    Q_diag = {config['ALQR_Q_DIAG']}")
        print(f"    R_diag = {config['ALQR_R_DIAG']}")

        # Eigenvalue check
        A_cl = A - B @ self.K_normal
        eigvals = np.linalg.eigvals(A_cl)
        stable = all(e.real < 0 for e in eigvals)
        print(f"    Closed-loop eigenvalues: {[f'{e.real:.2f}{e.imag:+.2f}j' for e in eigvals]}")
        print(f"    Stable: {'✓' if stable else '✗ WARNING: UNSTABLE!'}")

        # --- Gain-scheduled aggressive mode ---
        if 'ALQR_AGGRESSIVE_Q_DIAG' in config:
            Q_agg = np.diag(config['ALQR_AGGRESSIVE_Q_DIAG'])
            R_agg = np.diag(config['ALQR_AGGRESSIVE_R_DIAG'])
            self.K_aggressive = compute_lqr_gain(A, B, Q_agg, R_agg)
            self.switch_threshold = config.get('ALQR_SWITCH_THRESHOLD', 0.20)
            self.switch_hysteresis = config.get('ALQR_SWITCH_HYSTERESIS', 0.05)
            print(f"    K_agg_wheels  = [{', '.join(f'{k:.4f}' for k in self.K_aggressive[0])}]")
            print(f"    K_agg_triplet = [{', '.join(f'{k:.4f}' for k in self.K_aggressive[1])}]")
        else:
            self.K_aggressive = None
            self.switch_threshold = 0.0
            self.switch_hysteresis = 0.0

        # Active gain (start in normal mode)
        self.K = self.K_normal
        self.aggressive_active = False

        # --- Reference state ---
        self.target_position = 0.0

        # --- Velocity estimation ---
        self.prev_position = 0.0
        self.prev_vel_time = 0.0
        self.velocity = 0.0
        self.vel_filter_alpha = 0.1

        # --- Control loop timing ---
        self.control_period = 1.0 / config['CONTROL_RATE_HZ']
        self.next_control_time = 0.0

        # --- Sensor-to-actuator delay buffer ---
        delay_steps = config['SENSOR_TO_ACTUATOR_DELAY_STEPS']
        self.torque_delay_buffer = [(0.0, 0.0, 0.0)] * (delay_steps + 1)
        #                          (wheel_torque, yaw_correction, triplet_torque)

        # --- Torque limits ---
        self.max_wheel_torque = config['MAX_TORQUE']
        self.max_triplet_torque = config.get('MAX_TRIPLET_TORQUE', 5.0)

        # --- Exposed for logging (compatible with existing LQR interface) ---
        self.control_torque = 0.0        # wheel torque command (total)
        self.target_pitch = 0.0          # always 0 for LQR
        self.triplet_torque_cmd = 0.0    # triplet torque command (total)

        # Per-side triplet torque output (read by tribot_sim.py)
        self.triplet_torque_L = 0.0
        self.triplet_torque_R = 0.0

        # Per-state torque contributions (for PlotJuggler)
        # Extended: 4 for wheel contributions + 4 for triplet contributions
        self.K_contributions = np.zeros(4)       # wheel K*x (backward compat)
        self.K_contributions_trip = np.zeros(4)  # triplet K*x
        self.state_error = np.zeros(4)

        # --- Yaw rate setpoint ---
        self.yaw_rate_setpoint = 0.0

    def set_target_position(self, position):
        """Set the desired forward position (m)."""
        self.target_position = position

    def set_yaw_rate(self, yaw_rate):
        """Set desired yaw rate (rad/s). 0 = drive straight."""
        self.yaw_rate_setpoint = yaw_rate

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
        # --- LQR update at CONTROL_RATE_HZ ---
        jitter = (np.random.normal(0, self.cfg['CONTROL_JITTER_STD'])
                  if self.cfg.get('ADD_SENSOR_NOISE', False) else 0)

        if sim_time >= self.next_control_time:
            self.next_control_time = sim_time + self.control_period + jitter

            # Velocity estimation
            vel_dt = sim_time - self.prev_vel_time if self.prev_vel_time > 0 else self.control_period
            if vel_dt > 0:
                raw_vel = (position - self.prev_position) / vel_dt
                self.velocity += self.vel_filter_alpha * (raw_vel - self.velocity)
            self.prev_position = position
            self.prev_vel_time = sim_time

            # State error vector
            x = np.array([
                position - self.target_position,
                self.velocity,
                measured_pitch,
                measured_pitch_rate,
            ])
            self.state_error = x.copy()

            # --- Gain scheduling ---
            pos_err = abs(x[0])
            if self.K_aggressive is not None:
                if not self.aggressive_active and pos_err > self.switch_threshold:
                    self.aggressive_active = True
                    self.K = self.K_aggressive
                elif self.aggressive_active and pos_err < (self.switch_threshold - self.switch_hysteresis):
                    self.aggressive_active = False
                    self.K = self.K_normal

            # u = -K @ x → [τ_w, τ_t]
            u = -self.K @ x

            # --- Wheel torque (u[0]) ---
            wheel_torque = float(np.clip(u[0], -self.max_wheel_torque,
                                                self.max_wheel_torque))
            self.control_torque = wheel_torque
            self.K_contributions = self.K[0] * x  # per-state wheel contributions

            # --- Triplet torque (u[1]) ---
            # Total triplet torque, split equally to L and R sides.
            # Positive τ_t from LQR → positive triplet_cmd in sim → body gets
            # -τ_t reaction (pitch decreases).  LQR K handles the sign.
            triplet_total = float(np.clip(u[1],
                                          -2.0 * self.max_triplet_torque,
                                           2.0 * self.max_triplet_torque))
            self.triplet_torque_cmd = triplet_total
            self.K_contributions_trip = self.K[1] * x

            # Yaw damping
            yaw_correction = self.cfg['YAW_DAMPING_K'] * (yaw_rate - self.yaw_rate_setpoint)

            self.torque_delay_buffer.append(
                (wheel_torque, yaw_correction, triplet_total))

        # === Pop delayed commands ===
        delay_depth = self.cfg['SENSOR_TO_ACTUATOR_DELAY_STEPS'] + 1
        if len(self.torque_delay_buffer) > delay_depth:
            delayed_wheel, delayed_yaw, delayed_trip = self.torque_delay_buffer.pop(0)
        else:
            delayed_wheel, delayed_yaw, delayed_trip = self.torque_delay_buffer[0]

        # Per-side wheel torques (with yaw correction)
        left_torque = delayed_wheel - delayed_yaw
        right_torque = delayed_wheel + delayed_yaw

        # Per-side triplet torques (split equally, same sign for pitch control)
        self.triplet_torque_L = delayed_trip / 2.0
        self.triplet_torque_R = delayed_trip / 2.0

        return left_torque, right_torque
