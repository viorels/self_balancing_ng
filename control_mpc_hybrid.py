"""
Hybrid MPC + PD Balance Controller for Self-Balancing Tribot

Two-rate architecture:
  - SLOW LOOP (MPC, 30-50 Hz): Linearises the dynamics around the current
    state, builds a QP, and solves it to produce an optimal state + input
    trajectory over a receding horizon (N=10).
  - FAST LOOP (PD, 200 Hz): Interpolates the MPC trajectory at the current
    time, computes tracking errors, and produces motor torques as:

        τ = u_ff(t) + Kp·(x_ref(t) − x) + Kd·(ẋ_ref(t) − ẋ)

    where u_ff is the MPC feedforward and Kp/Kd are diagonal PD gains.

State vector  (nx=8):
    x = [pitch, pitch_rate,
         triplet_angle_L, triplet_angle_R,
         triplet_rate_L,  triplet_rate_R,
         forward_pos,     forward_vel]

Input vector  (nu=4):
    u = [tau_triplet_L, tau_triplet_R, tau_drive_L, tau_drive_R]

The MPC solve time is artificially capped to simulate a realistic
embedded platform (ESP32-S3 with SIMD @ 240 MHz).

Interface is compatible with BalanceController / LQRBalanceController:
    Inputs:  measured_pitch, pitch_rate, position, yaw_rate, sim_time, dt
    Outputs: (left_torque, right_torque)

Note: In the current URDF the triplet joints are free-spinning hubs,
not actuated.  The MPC *plans* triplet trajectories but the actual
triplet torques are not applied to the simulation (they would require
separate triplet motors).  The drive-motor outputs (u[2], u[3]) are
what physically drives the robot.  Triplet feedforward terms are
kept in the code for completeness — they will become active once
triplet actuators are added to the hardware/URDF.
"""

import math
import time as _time
import numpy as np
from scipy import linalg as la


# ============================================================================
# Dense QP Solver  (no OSQP dependency — suitable for small problems)
# ============================================================================

def solve_dense_qp(H, g, C=None, d_lo=None, d_hi=None,
                   max_iter=200, tol=1e-6):
    """
    Minimise  0.5 x'Hx + g'x   subject to  d_lo <= Cx <= d_hi.

    Uses a simple projected gradient / ADMM-lite approach for box-constrained
    QPs (after reformulating inequality constraints as box constraints on
    the slack variable z = Cx).

    For the unconstrained case (C is None) falls back to a direct solve.

    Parameters
    ----------
    H : (n, n) positive-definite cost matrix
    g : (n,) linear cost vector
    C : (m, n) constraint matrix or None
    d_lo, d_hi : (m,) lower/upper constraint bounds or None
    max_iter : int
    tol : float

    Returns
    -------
    x : (n,) optimal decision variable
    """
    n = H.shape[0]

    if C is None or C.shape[0] == 0:
        # Unconstrained QP — direct Cholesky solve
        try:
            L = la.cho_factor(H)
            return la.cho_solve(L, -g)
        except la.LinAlgError:
            return np.linalg.solve(H + 1e-8 * np.eye(n), -g)

    m = C.shape[0]
    if d_lo is None:
        d_lo = np.full(m, -1e20)
    if d_hi is None:
        d_hi = np.full(m, 1e20)

    # ADMM parameters
    rho = 1.0
    sigma = 1e-6

    # Pre-factor  (H + sigma I + rho C'C)
    M = H + sigma * np.eye(n) + rho * (C.T @ C)
    try:
        L_factor = la.cho_factor(M)
    except la.LinAlgError:
        M += 1e-6 * np.eye(n)
        L_factor = la.cho_factor(M)

    x = np.zeros(n)
    z = np.zeros(m)
    y = np.zeros(m)   # dual variable

    for _ in range(max_iter):
        # x-update
        rhs = -g + sigma * x + C.T @ (rho * z - y)
        x_new = la.cho_solve(L_factor, rhs)

        # z-update (projection onto box constraints)
        Cx = C @ x_new
        z_raw = Cx + y / rho
        z_new = np.clip(z_raw, d_lo, d_hi)

        # dual update
        y = y + rho * (Cx - z_new)

        # convergence check
        primal_res = np.linalg.norm(Cx - z_new)
        dual_res = rho * np.linalg.norm(C.T @ (z_new - z))
        x = x_new
        z = z_new

        if primal_res < tol and dual_res < tol:
            break

    return x


# ============================================================================
# Linearised plant model  (extended 8-state version)
# ============================================================================

def build_mpc_state_space(cfg):
    """
    Build continuous-time A (8×8) and B (8×4) matrices for the tribot.

    State:  x = [pitch, pitch_rate,
                 trip_ang_L, trip_ang_R,
                 trip_rate_L, trip_rate_R,
                 fwd_pos, fwd_vel]

    Input:  u = [tau_trip_L, tau_trip_R, tau_drive_L, tau_drive_R]

    The sagittal (pitch / forward) dynamics reuse the same inverted-pendulum
    model as control_lqr.py.  Triplet dynamics are modelled as decoupled
    second-order rotational systems with inertia + gravity coupling.
    """
    m_b = cfg['LQR_BODY_MASS']          # body mass (kg)
    m_w = cfg['LQR_WHEEL_MASS']         # total wheel/triplet mass (kg)
    l   = cfg['LQR_COG_HEIGHT']         # CoG height above wheel axis (m)
    I_b = cfg['LQR_BODY_INERTIA']       # body pitch inertia (kg·m²)
    r   = cfg['WHEEL_RADIUS']           # effective wheel radius (m)
    g   = abs(cfg['GRAVITY'])

    # --- Sagittal dynamics (same derivation as control_lqr) ---
    I_eff = I_b + m_b * l**2
    M_tot = m_b + m_w
    det = M_tot * I_eff - (m_b * l)**2

    # pitch_ddot from pitch (gravity)
    a_pitch_pitch = M_tot * (m_b * g * l) / det
    # fwd_ddot from pitch (gravity coupling)
    a_fwd_pitch = (m_b * l) * (m_b * g * l) / det

    # pitch_ddot from drive torque
    b_pitch_drive = (m_b * l) / (det * r) + M_tot / det
    # fwd_ddot from drive torque
    b_fwd_drive = I_eff / (det * r) + (m_b * l) / det

    # --- Triplet dynamics ---
    # Each triplet is a rotating assembly (~0.33 kg, radius ~0.12 m)
    # modelled as a simple rotational inertia.
    # Inertia = 0.5 * m_trip * R_trip² (thin cylinder approx)
    m_trip = m_w / 2.0                   # mass per triplet assembly
    R_trip = cfg.get('TRIPLET_RADIUS', 0.12)
    I_trip = cfg.get('MPC_TRIPLET_INERTIA',
                     0.5 * m_trip * R_trip**2)

    # Gravity coupling on triplet: when triplet angle ≠ 0 and body is
    # pitched, gravity creates a restoring torque.  For small angles
    # this is approximately m_trip * g * R_trip * sin(trip_angle).
    # Linearised: a_trip = m_trip * g * R_trip / I_trip
    a_trip_grav = m_trip * g * R_trip / I_trip

    # Triplet torque input gain
    b_trip = 1.0 / I_trip

    # --- Assemble A (8×8) ---
    #   Indices: 0=pitch, 1=pitch_rate, 2=tripL, 3=tripR,
    #            4=tripL_rate, 5=tripR_rate, 6=fwd_pos, 7=fwd_vel
    A = np.zeros((8, 8))

    # pitch dynamics
    A[0, 1] = 1.0                          # d(pitch)/dt = pitch_rate
    A[1, 0] = a_pitch_pitch                # pitch_ddot from pitch

    # triplet L dynamics
    A[2, 4] = 1.0                          # d(trip_ang_L)/dt = trip_rate_L
    A[4, 2] = a_trip_grav                  # trip_ddot from trip_angle (gravity)

    # triplet R dynamics
    A[3, 5] = 1.0                          # d(trip_ang_R)/dt = trip_rate_R
    A[5, 3] = a_trip_grav                  # trip_ddot from trip_angle (gravity)

    # forward dynamics
    A[6, 7] = 1.0                          # d(fwd_pos)/dt = fwd_vel
    A[7, 0] = a_fwd_pitch                  # fwd_ddot from pitch

    # --- Assemble B (8×4) ---
    #   Inputs: 0=tau_trip_L, 1=tau_trip_R, 2=tau_drive_L, 3=tau_drive_R
    B = np.zeros((8, 4))

    # triplet torques → triplet accelerations
    B[4, 0] = b_trip                       # trip_L_ddot from tau_trip_L
    B[5, 1] = b_trip                       # trip_R_ddot from tau_trip_R

    # drive torques → pitch + forward accelerations
    # Both drive motors contribute equally to sagittal dynamics
    B[1, 2] = b_pitch_drive                # pitch_ddot from tau_drive_L
    B[1, 3] = b_pitch_drive                # pitch_ddot from tau_drive_R
    B[7, 2] = b_fwd_drive                  # fwd_ddot from tau_drive_L
    B[7, 3] = b_fwd_drive                  # fwd_ddot from tau_drive_R

    return A, B


def discretise_zoh(A_c, B_c, dt):
    """Exact zero-order-hold discretisation using matrix exponential."""
    nx = A_c.shape[0]
    nu = B_c.shape[1]

    # Build augmented matrix [A B; 0 0] and exponentiate
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = A_c * dt
    M[:nx, nx:] = B_c * dt
    eM = la.expm(M)

    Ad = eM[:nx, :nx]
    Bd = eM[:nx, nx:]
    return Ad, Bd


# ============================================================================
# MPC Builder + Solver
# ============================================================================

class MPCSolver:
    """
    Builds and solves a linear MPC (dense QP formulation).

    Decision variable:  z = [u_0, u_1, ..., u_{N-1}]  (N*nu variables)
    States are eliminated using the recursive prediction:
        x_{k+1} = A x_k + B u_k
        x_k = A^k x_0 + Σ_{j=0}^{k-1} A^{k-1-j} B u_j

    The QP is:
        min  Σ_{k=1}^{N} x_k' Q x_k  +  Σ_{k=0}^{N-1} u_k' R u_k
        s.t. u_min <= u_k <= u_max   for all k
    """

    def __init__(self, Ad, Bd, Q, R, Q_terminal, N,
                 u_min=None, u_max=None):
        self.nx = Ad.shape[0]
        self.nu = Bd.shape[1]
        self.N = N

        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal
        self.u_min = u_min
        self.u_max = u_max

        # Pre-compute prediction matrices:  X = Sx @ x0 + Su @ U
        # where X = [x_1; x_2; ... x_N],  U = [u_0; u_1; ... u_{N-1}]
        self._build_prediction_matrices()
        self._build_qp_matrices()

    def _build_prediction_matrices(self):
        """Construct Sx (N·nx × nx) and Su (N·nx × N·nu)."""
        nx, nu, N = self.nx, self.nu, self.N
        Ad, Bd = self.Ad, self.Bd

        Sx = np.zeros((N * nx, nx))
        Su = np.zeros((N * nx, N * nu))

        A_pow = np.eye(nx)
        for k in range(N):
            A_pow = A_pow @ Ad   # A^{k+1}
            Sx[k*nx:(k+1)*nx, :] = A_pow

            for j in range(k + 1):
                # x_{k+1} contribution from u_j:  A^{k-j} B
                row = k * nx
                col = j * nu
                if k - j == 0:
                    Su[row:row+nx, col:col+nu] = Bd
                else:
                    # Compute A^{k-j} B
                    Apow_kj = np.linalg.matrix_power(Ad, k - j)
                    Su[row:row+nx, col:col+nu] = Apow_kj @ Bd

        self.Sx = Sx
        self.Su = Su

    def _build_qp_matrices(self):
        """Build the dense QP cost:  0.5 U' H U + g' U."""
        nx, nu, N = self.nx, self.nu, self.N

        # Block-diagonal Q for stages 1..N-1, Q_terminal for stage N
        Q_bar = np.zeros((N * nx, N * nx))
        for k in range(N - 1):
            Q_bar[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = self.Q
        Q_bar[(N-1)*nx:N*nx, (N-1)*nx:N*nx] = self.Q_terminal

        # Block-diagonal R
        R_bar = np.zeros((N * nu, N * nu))
        for k in range(N):
            R_bar[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = self.R

        # H = Su' Q_bar Su + R_bar
        self.H = self.Su.T @ Q_bar @ self.Su + R_bar
        # Make sure H is exactly symmetric (numerical hygiene)
        self.H = (self.H + self.H.T) / 2.0

        # Pre-compute the part of g that depends on x0:
        #   g = Su' Q_bar Sx x0   (minus reference terms, added at solve time)
        self.Q_bar = Q_bar
        self.R_bar = R_bar

        # Constraint matrices (simple box on U)
        nU = N * nu
        if self.u_min is not None and self.u_max is not None:
            self.C = np.eye(nU)
            self.d_lo = np.tile(self.u_min, N)
            self.d_hi = np.tile(self.u_max, N)
        else:
            self.C = None
            self.d_lo = None
            self.d_hi = None

    def solve(self, x0, x_ref=None):
        """
        Solve the MPC QP for current state x0.

        Parameters
        ----------
        x0 : (nx,) current state
        x_ref : (nx,) or (N, nx) reference trajectory.  If 1-D, the same
                reference is used at every stage.

        Returns
        -------
        u_traj : (N, nu) optimal input trajectory
        x_traj : (N+1, nx) predicted state trajectory (x_traj[0] = x0)
        """
        nx, nu, N = self.nx, self.nu, self.N

        # Reference handling
        if x_ref is None:
            X_ref = np.zeros(N * nx)
        elif x_ref.ndim == 1:
            X_ref = np.tile(x_ref, N)
        else:
            X_ref = x_ref.flatten()

        # Predicted state without control: X_free = Sx @ x0
        X_free = self.Sx @ x0

        # g = Su' Q_bar (Sx x0 - X_ref)   (linear cost term)
        g = self.Su.T @ self.Q_bar @ (X_free - X_ref)

        # Solve QP
        U_opt = solve_dense_qp(self.H, g, self.C, self.d_lo, self.d_hi)

        # Reconstruct state trajectory
        X_opt = X_free + self.Su @ U_opt

        u_traj = U_opt.reshape(N, nu)
        x_traj = np.zeros((N + 1, nx))
        x_traj[0] = x0
        for k in range(N):
            x_traj[k + 1] = X_opt[k*nx:(k+1)*nx]

        return u_traj, x_traj


# ============================================================================
# Hybrid MPC + PD Controller
# ============================================================================

class MPCHybridController:
    """
    Two-rate MPC + PD balance controller for the tribot.

    Slow loop (MPC):
        Runs at ``MPC_RATE_HZ`` (default 40 Hz).  Linearises the continuous
        dynamics around the current operating point, discretises, builds
        and solves a QP, and stores the resulting trajectory.

        The solve call is artificially rate-limited to simulate the wall-
        clock time a real ESP32-S3 would need (configurable via
        ``MPC_SIMULATED_SOLVE_MS``).

    Fast loop (PD):
        Runs at ``CONTROL_RATE_HZ`` (200 Hz).  Interpolates the latest MPC
        trajectory to the current time, computes the tracking error, and
        outputs:
            τ = u_ff  +  Kp · e  +  Kd · ė

    Config keys (in addition to plant-model keys from ``build_mpc_state_space``):
        MPC_RATE_HZ               – MPC update rate (default 40)
        MPC_HORIZON               – prediction horizon N (default 10)
        MPC_Q_DIAG                – 8-element list, stage cost weights
        MPC_R_DIAG                – 4-element list, input cost weights
        MPC_Q_TERMINAL_SCALE      – scalar multiplier on Q for terminal cost
        MPC_SIMULATED_SOLVE_MS    – artificial solve time cap (ms)
        MPC_PD_KP                 – 4-element list, PD proportional gains
        MPC_PD_KD                 – 4-element list, PD derivative gains
        MPC_PITCH_PD_CROSS_DRIVE  – cross-coupling: pitch error → drive torque
        MAX_TORQUE                – per-motor torque limit (Nm)
        CONTROL_RATE_HZ           – fast PD loop rate
    """

    # State indices
    IDX_PITCH      = 0
    IDX_PITCH_RATE = 1
    IDX_TRIP_L     = 2
    IDX_TRIP_R     = 3
    IDX_TRIP_L_D   = 4
    IDX_TRIP_R_D   = 5
    IDX_FWD_POS    = 6
    IDX_FWD_VEL    = 7

    # Input indices
    IDX_U_TRIP_L  = 0
    IDX_U_TRIP_R  = 1
    IDX_U_DRIVE_L = 2
    IDX_U_DRIVE_R = 3

    def __init__(self, config):
        self.cfg = config
        self.nx = 8
        self.nu = 4

        # ---- MPC parameters ----
        self.mpc_rate = config.get('MPC_RATE_HZ', 40)
        self.mpc_period = 1.0 / self.mpc_rate
        self.N = config.get('MPC_HORIZON', 10)

        # Artificial solve-time budget (simulates ESP32-S3 wall-clock)
        self.simulated_solve_ms = config.get('MPC_SIMULATED_SOLVE_MS', 25.0)

        # Cost weights
        q_diag = config.get('MPC_Q_DIAG', [
            80.0,   # pitch
            5.0,    # pitch rate
            2.0,    # triplet angle L
            2.0,    # triplet angle R
            0.5,    # triplet rate L
            0.5,    # triplet rate R
            1.0,    # forward position
            0.5,    # forward velocity
        ])
        r_diag = config.get('MPC_R_DIAG', [
            5.0,    # tau_triplet_L
            5.0,    # tau_triplet_R
            10.0,   # tau_drive_L
            10.0,   # tau_drive_R
        ])
        q_term_scale = config.get('MPC_Q_TERMINAL_SCALE', 3.0)

        self.Q = np.diag(q_diag)
        self.R = np.diag(r_diag)
        self.Q_terminal = self.Q * q_term_scale

        # Input bounds
        tau_max = config['MAX_TORQUE']
        trip_tau_max = config.get('MPC_TRIPLET_TORQUE_MAX', tau_max)
        self.u_min = np.array([-trip_tau_max, -trip_tau_max, -tau_max, -tau_max])
        self.u_max = np.array([ trip_tau_max,  trip_tau_max,  tau_max,  tau_max])

        # ---- Build initial model and solver ----
        self.Ac, self.Bc = build_mpc_state_space(config)
        Ad, Bd = discretise_zoh(self.Ac, self.Bc, self.mpc_period)
        self.mpc_solver = MPCSolver(
            Ad, Bd, self.Q, self.R, self.Q_terminal, self.N,
            u_min=self.u_min, u_max=self.u_max
        )

        print(f"  MPC Hybrid controller initialised")
        print(f"    MPC rate: {self.mpc_rate} Hz, horizon N={self.N}, "
              f"dt_mpc={self.mpc_period*1000:.1f} ms")
        print(f"    Simulated solve time: {self.simulated_solve_ms:.0f} ms")
        print(f"    Q_diag = {q_diag}")
        print(f"    R_diag = {r_diag}")
        print(f"    u_bounds = [{self.u_min}, {self.u_max}]")

        # ---- PD tracking gains ----
        # These are applied per-output-channel in the fast loop.
        # Order: [trip_L, trip_R, drive_L, drive_R]
        self.Kp_pd = np.array(config.get('MPC_PD_KP', [2.0, 2.0, 12.0, 12.0]))
        self.Kd_pd = np.array(config.get('MPC_PD_KD', [0.3, 0.3, 0.8, 0.8]))

        # Cross-coupling: pitch error → additional drive torque
        self.Kp_pitch_cross = config.get('MPC_PITCH_PD_CROSS_DRIVE', 8.0)
        self.Kd_pitch_cross = config.get('MPC_PITCH_RATE_PD_CROSS_DRIVE', 0.5)

        print(f"    PD Kp = {self.Kp_pd.tolist()}")
        print(f"    PD Kd = {self.Kd_pd.tolist()}")
        print(f"    Pitch→drive cross-coupling: Kp={self.Kp_pitch_cross}, "
              f"Kd={self.Kd_pitch_cross}")

        # ---- Fast-loop timing ----
        self.control_period = 1.0 / config['CONTROL_RATE_HZ']
        self.next_control_time = 0.0

        # ---- MPC timing ----
        self.next_mpc_time = 0.0
        self.mpc_busy_until = 0.0   # simulates compute delay

        # ---- Trajectory buffer (most recent MPC solution) ----
        self.traj_t0 = 0.0          # sim-time when trajectory was computed
        self.traj_x = np.zeros((self.N + 1, self.nx))   # reference states
        self.traj_u = np.zeros((self.N, self.nu))        # feedforward inputs
        self.traj_valid = False

        # ---- MPC state estimate ----
        # (pitch and fwd from sensor, triplet from URDF joints — zeroed here)
        self.x_est = np.zeros(self.nx)

        # ---- Velocity estimator ----
        self.prev_position = 0.0
        self.prev_vel_time = 0.0
        self.velocity = 0.0
        self.vel_filter_alpha = 0.1

        # ---- Reference / target ----
        self.target_position = 0.0
        self.x_ref = np.zeros(self.nx)     # reference state for MPC

        # ---- Yaw control (same interface as LQR/PID controllers) ----
        self.yaw_rate_setpoint = 0.0
        self.yaw_damping_k = config.get('YAW_DAMPING_K', 0.5)

        # ---- Sensor-to-actuator delay pipeline ----
        delay_steps = config.get('SENSOR_TO_ACTUATOR_DELAY_STEPS', 0)
        self.torque_delay_buffer = [(0.0, 0.0)] * (delay_steps + 1)
        self.delay_depth = delay_steps + 1

        # ---- Triplet torque outputs (for tribot_sim to apply) ----
        self.triplet_torque_L = 0.0
        self.triplet_torque_R = 0.0

        # ---- Logging (compatible with tribot_sim PlotJuggler) ----
        self.control_torque = 0.0
        self.target_pitch = 0.0
        self.K_contributions = np.zeros(4)   # [trip_L, trip_R, drive_L, drive_R] ff
        self.state_error = np.zeros(4)       # [pos_err, vel, pitch, pitch_rate]

        # ---- Performance accounting ----
        self.mpc_solve_count = 0
        self.mpc_last_wall_ms = 0.0
        self.mpc_max_wall_ms = 0.0

    # ----------------------------------------------------------------
    # Public setters (same API as PID / LQR controllers)
    # ----------------------------------------------------------------

    def set_target_position(self, position):
        self.target_position = position

    def set_yaw_rate(self, yaw_rate):
        self.yaw_rate_setpoint = yaw_rate

    # ----------------------------------------------------------------
    # MPC slow loop
    # ----------------------------------------------------------------

    def _run_mpc(self, x0, sim_time):
        """
        Solve the MPC QP and store the resulting trajectory.

        The actual Python solve runs instantly, but we record wall-clock
        time and enforce a simulated compute budget so that MPC results
        are not used until ``mpc_busy_until``.
        """
        # Reference: drive all states to x_ref
        self.x_ref[:] = 0.0
        self.x_ref[self.IDX_FWD_POS] = self.target_position

        # Measure actual wall-clock solve time (for diagnostics)
        t_wall_start = _time.monotonic()

        u_traj, x_traj = self.mpc_solver.solve(x0, x_ref=self.x_ref)

        t_wall_end = _time.monotonic()
        self.mpc_last_wall_ms = (t_wall_end - t_wall_start) * 1000.0
        self.mpc_max_wall_ms = max(self.mpc_max_wall_ms, self.mpc_last_wall_ms)

        # Store trajectory (double-buffer would be used on real HW;
        # here we just overwrite since Python is single-threaded).
        self.traj_t0 = sim_time
        self.traj_x = x_traj.copy()
        self.traj_u = u_traj.copy()
        self.traj_valid = True

        # Mark MPC as busy until the simulated solve time has elapsed
        self.mpc_busy_until = sim_time + self.simulated_solve_ms / 1000.0

        self.mpc_solve_count += 1

    # ----------------------------------------------------------------
    # Trajectory interpolation
    # ----------------------------------------------------------------

    def _interpolate_trajectory(self, sim_time):
        """
        Given current sim_time, interpolate the stored MPC trajectory
        to obtain (x_ref, u_ff) at this instant.

        Returns
        -------
        x_ref : (nx,) interpolated reference state
        u_ff  : (nu,) interpolated feedforward input
        """
        if not self.traj_valid:
            return np.zeros(self.nx), np.zeros(self.nu)

        t_elapsed = sim_time - self.traj_t0
        seg_float = t_elapsed / self.mpc_period
        k = int(seg_float)
        alpha = seg_float - k

        # Clamp to trajectory bounds
        if k >= self.N:
            k = self.N - 1
            alpha = 0.0
        if k < 0:
            k = 0
            alpha = 0.0

        x_ref = (1.0 - alpha) * self.traj_x[k] + alpha * self.traj_x[k + 1]

        k_u = min(k, self.N - 1)
        k_u_next = min(k + 1, self.N - 1)
        u_ff = (1.0 - alpha) * self.traj_u[k_u] + alpha * self.traj_u[k_u_next]

        return x_ref, u_ff

    # ----------------------------------------------------------------
    # Fast PD loop
    # ----------------------------------------------------------------

    def _pd_step(self, x_ref, u_ff, x_actual):
        """
        PD tracking controller.

        Computes: τ = u_ff + Kp·e + Kd·ė

        The error mapping from 8-state space to 4-output space:
            e_trip_L   = x_ref[trip_L]   - x_actual[trip_L]
            e_trip_R   = x_ref[trip_R]   - x_actual[trip_R]
            e_drive_L  = x_ref[pitch]    - x_actual[pitch]    (pitch tracking)
            e_drive_R  = x_ref[pitch]    - x_actual[pitch]

        Plus cross-coupling terms and forward position feedback on drives.
        """
        # ---- Tracking errors ----
        e_pitch     = x_ref[self.IDX_PITCH]      - x_actual[self.IDX_PITCH]
        e_pitch_d   = x_ref[self.IDX_PITCH_RATE]  - x_actual[self.IDX_PITCH_RATE]
        e_trip_L    = x_ref[self.IDX_TRIP_L]      - x_actual[self.IDX_TRIP_L]
        e_trip_R    = x_ref[self.IDX_TRIP_R]      - x_actual[self.IDX_TRIP_R]
        e_trip_L_d  = x_ref[self.IDX_TRIP_L_D]   - x_actual[self.IDX_TRIP_L_D]
        e_trip_R_d  = x_ref[self.IDX_TRIP_R_D]   - x_actual[self.IDX_TRIP_R_D]
        e_fwd       = x_ref[self.IDX_FWD_POS]    - x_actual[self.IDX_FWD_POS]
        e_fwd_d     = x_ref[self.IDX_FWD_VEL]    - x_actual[self.IDX_FWD_VEL]

        # ---- Per-channel PD correction ----
        tau = np.copy(u_ff)

        # Triplet motors: PD on triplet angle tracking
        tau[self.IDX_U_TRIP_L] += (self.Kp_pd[0] * e_trip_L
                                   + self.Kd_pd[0] * e_trip_L_d)
        tau[self.IDX_U_TRIP_R] += (self.Kp_pd[1] * e_trip_R
                                   + self.Kd_pd[1] * e_trip_R_d)

        # Drive motors: PD on pitch error (primary) + forward pos (secondary)
        # Pitch cross-coupling (fast disturbance rejection — like a Segway)
        pitch_correction = (self.Kp_pitch_cross * e_pitch
                            + self.Kd_pitch_cross * e_pitch_d)

        tau[self.IDX_U_DRIVE_L] += (self.Kp_pd[2] * e_fwd
                                    + self.Kd_pd[2] * e_fwd_d
                                    + pitch_correction)
        tau[self.IDX_U_DRIVE_R] += (self.Kp_pd[3] * e_fwd
                                    + self.Kd_pd[3] * e_fwd_d
                                    + pitch_correction)

        # Clamp each channel
        tau = np.clip(tau, -self.cfg['MAX_TORQUE'], self.cfg['MAX_TORQUE'])

        return tau

    # ----------------------------------------------------------------
    # Main update  (called every physics tick from tribot_sim)
    # ----------------------------------------------------------------

    def update(self, measured_pitch, measured_pitch_rate,
               position, yaw_rate, sim_time, dt):
        """
        Run one tick of the hybrid MPC+PD controller.

        Same signature as LQRBalanceController.update() for drop-in use.

        Returns
        -------
        left_torque, right_torque : float
            Commanded motor torques (Nm) for left and right drive motors.
        """
        # ---- Velocity estimation (same approach as LQR controller) ----
        vel_dt = sim_time - self.prev_vel_time if self.prev_vel_time > 0 else dt
        if vel_dt > 0:
            raw_vel = (position - self.prev_position) / vel_dt
            self.velocity += self.vel_filter_alpha * (raw_vel - self.velocity)
        self.prev_position = position
        self.prev_vel_time = sim_time

        # ---- Build full 8-state estimate ----
        # Pitch and forward come from sensors; triplet states are zero
        # for now (triplet joints are unactuated in current URDF).
        self.x_est[self.IDX_PITCH]      = measured_pitch
        self.x_est[self.IDX_PITCH_RATE] = measured_pitch_rate
        self.x_est[self.IDX_TRIP_L]     = 0.0
        self.x_est[self.IDX_TRIP_R]     = 0.0
        self.x_est[self.IDX_TRIP_L_D]   = 0.0
        self.x_est[self.IDX_TRIP_R_D]   = 0.0
        self.x_est[self.IDX_FWD_POS]    = position
        self.x_est[self.IDX_FWD_VEL]    = self.velocity

        # ---- MPC slow loop ----
        if sim_time >= self.next_mpc_time and sim_time >= self.mpc_busy_until:
            self._run_mpc(self.x_est.copy(), sim_time)
            self.next_mpc_time = sim_time + self.mpc_period

        # ---- Fast PD loop ----
        jitter = (np.random.normal(0, self.cfg.get('CONTROL_JITTER_STD', 0))
                  if self.cfg.get('ADD_SENSOR_NOISE', False) else 0)

        if sim_time >= self.next_control_time:
            self.next_control_time = sim_time + self.control_period + jitter

            # Interpolate MPC trajectory at current sim_time
            # If MPC is still "computing" (sim_time < mpc_busy_until),
            # we use the previous trajectory — just like real hardware.
            x_ref, u_ff = self._interpolate_trajectory(sim_time)

            # PD tracking step
            tau_4 = self._pd_step(x_ref, u_ff, self.x_est)

            # The tribot_sim expects a single drive torque per side.
            # Combine: drive torque = u_drive + (triplet feedback mapped
            # through the physical coupling).
            # Since triplet actuators don't exist yet, we only use drive outputs.
            drive_torque_L = float(tau_4[self.IDX_U_DRIVE_L])
            drive_torque_R = float(tau_4[self.IDX_U_DRIVE_R])
            trip_torque_L  = float(tau_4[self.IDX_U_TRIP_L])
            trip_torque_R  = float(tau_4[self.IDX_U_TRIP_R])

            # Average for logging
            base_torque = (drive_torque_L + drive_torque_R) / 2.0
            self.control_torque = base_torque

            # Yaw damping (differential torque opposing yaw rate)
            yaw_correction = self.yaw_damping_k * (yaw_rate - self.yaw_rate_setpoint)

            # Push into delay buffer (drive L, drive R, trip L, trip R, yaw)
            self.torque_delay_buffer.append((drive_torque_L, drive_torque_R,
                                             trip_torque_L, trip_torque_R,
                                             yaw_correction))

            # Logging: fill state_error for compatibility with tribot_sim
            self.state_error[0] = position - self.target_position
            self.state_error[1] = self.velocity
            self.state_error[2] = measured_pitch
            self.state_error[3] = measured_pitch_rate

            # K_contributions: show feedforward vs PD breakdown
            self.K_contributions[0] = float(u_ff[self.IDX_U_DRIVE_L])
            self.K_contributions[1] = float(u_ff[self.IDX_U_DRIVE_R])
            self.K_contributions[2] = float(tau_4[self.IDX_U_DRIVE_L] - u_ff[self.IDX_U_DRIVE_L])
            self.K_contributions[3] = float(tau_4[self.IDX_U_DRIVE_R] - u_ff[self.IDX_U_DRIVE_R])

            self.target_pitch = float(x_ref[self.IDX_PITCH])

        # ---- Pop delayed torque command ----
        if len(self.torque_delay_buffer) > self.delay_depth:
            entry = self.torque_delay_buffer.pop(0)
        else:
            entry = self.torque_delay_buffer[0]

        if len(entry) == 5:
            delayed_L, delayed_R, delayed_trip_L, delayed_trip_R, delayed_yaw = entry
        elif len(entry) == 3:
            delayed_L, delayed_R, delayed_yaw = entry
            delayed_trip_L = 0.0
            delayed_trip_R = 0.0
        else:
            # Legacy 2-tuple in initial buffer
            delayed_L = entry[0]
            delayed_R = entry[1] if len(entry) > 1 else entry[0]
            delayed_trip_L = 0.0
            delayed_trip_R = 0.0
            delayed_yaw = 0.0

        # Apply yaw correction differentially on drive motors
        left_torque  = delayed_L - delayed_yaw
        right_torque = delayed_R + delayed_yaw

        # Expose triplet torques for tribot_sim to apply
        self.triplet_torque_L = delayed_trip_L
        self.triplet_torque_R = delayed_trip_R

        return left_torque, right_torque
