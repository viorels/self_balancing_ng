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

2WD mode: the triplet assemblies are rotated 60° so one wheel per side
touches the ground.  The MPC actively controls both triplet and drive
torques.  Triplet encoders (angles + rates) are fed back into the
state estimate.  Triplet motors have a higher torque limit (5 Nm)
than the drive motors (1 Nm) since they must maintain the 2WD pose
against gravity and disturbances.
"""

import math
import time as _time
import numpy as np
from scipy import linalg as la

from .base import BalanceControllerBase


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
    m_b = cfg.plant.body_mass          # body mass (kg)
    m_w = cfg.plant.wheel_mass         # total wheel/triplet mass (kg)
    l   = cfg.plant.cog_height         # CoG height above wheel axis (m)
    I_b = cfg.plant.body_inertia       # body pitch inertia (kg·m²)
    r   = cfg.robot.wheel_radius           # effective wheel radius (m)
    g   = abs(cfg.sim.gravity)

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
    R_trip = cfg.robot.triplet_radius
    I_trip = cfg.mpc.triplet_inertia

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
# ZMP / DCM-based Predictive Triplet Flip Trigger
# ============================================================================
#
# PHYSICS DERIVATION — from first principles
# ============================================
#
# The tribot is an inverted pendulum on rolling wheels.  The triplet flip
# rotates the wheel cluster by 120° to "catch" the robot when it's about to
# fall beyond the recovery limits of normal 2WD balance control.
#
# Unlike a biped stepping at the capture point, the tribot's ground contact
# does NOT shift laterally — every valid 2WD wheel position is directly
# below the hub (the triplet is symmetric).  So the flip is not a "step to
# the capture point" — it's a CRASH PREVENTION TRIGGER that buys time by
# placing a fresh wheel on the ground before the lean exceeds the controller's
# mechanical limits.
#
# The key improvement over a naïve free-fall DCM is accounting for the drive
# wheel torque that actively fights the fall during the flip window.
#
# MODEL
# -----
# Pitch dynamics (linearised rigid body on a wheel):
#
#   I_eff · θ̈  =  m_b·g·L · θ  −  τ_ctrl
#
# where τ_ctrl is the effective restoring torque from both drive motors.
# Rearranging:
#
#   θ̈  =  ω₀² · θ  −  a_ctrl
#
# with:
#   ω₀ = √(m_b·g·L / I_eff)            (rigid body, NOT √(g/L)!)
#   a_ctrl = b_pitch · 2 · η · τ_max    (pitch accel from drives at fraction η)
#   b_pitch = B[1,2] from the MPC A/B matrices
#
# Substituting  φ = θ − θ_eq  where  θ_eq = a_ctrl / ω₀² :
#
#   φ̈ = ω₀² · φ                        (free LIPM in shifted frame)
#
# The CONTROLLED DCM is:
#
#   ξ_ctrl = L·(θ − sgn(θ)·θ_eq) + L·θ̇/ω₀
#          = ξ_raw − sgn(ξ_raw) · L·θ_eq
#
# which we call the "DCM excess":
#
#   ξ_excess = |ξ_raw| − dcm_shift     where dcm_shift = L · θ_eq
#
# If ξ_excess ≤ 0: the drive torque can arrest this fall.  No flip needed.
# If ξ_excess > 0: it grows exponentially as ξ_excess(t) = ξ_excess · e^{ω₀t}
#
# The crash happens when |θ| reaches θ_crash (mechanical limit, ~45°):
#
#   ξ_crash = L·θ_crash − dcm_shift
#
# Time to crash:
#
#   t_crash = (1/ω₀) · ln(ξ_crash / ξ_excess)     if ξ_excess > 0
#
# The flip must be triggered when  t_crash ≤ T_budget  (flip time + margin).
#
# FLIP TIMING
# -----------
# Theoretical bang-bang minimum:  T_bb = 2·√(Δα · I_t / τ_trip_max)
# With Δα=120°, I_t=0.00238, τ_trip_max=5.0 → T_bb ≈ 63 ms.
# Config override (motor lag, belt compliance): ZMP_T_FLIP_NOMINAL ≈ 180 ms.
#
# REACTION TORQUE
# ---------------
# During the flip, the triplet motor reaction torque (±5 Nm) acts on the body.
# For a symmetric acceleration-deceleration profile, the net impulse is zero.
# We conservatively ignore this benefit (the actual MPC profile may give a
# small net deceleration in the first half).
#
# STAIR HEIGHT
# ------------
# When landing on a step of height h:
#   - Required rotation: α_land = arccos(1 − h/R_t)  (< 120°, faster landing)
#   - Effective pendulum: L_eff = L − h  (shorter, slightly easier to balance)
#   - T_flip scales as  T_bb · √(α_land / Δα)
# Default: h = 0 (flat ground, most conservative).
#
# ============================================================================

_G = 9.81

# Wheel base angles (radians, CCW from +x in the triplet side-view plane)
_WHEEL_BASE_ANGLES = [
    math.atan2(+0.12,      0.0),        # wheel_1:  90°  (top in default pose)
    math.atan2(-0.060125, -0.10414),    # wheel_2: ~210° (bottom-rear)
    math.atan2(-0.060125, +0.10414),    # wheel_3: ~330° (bottom-front)
]

# Flip phase constants (used as str tags — no Enum needed for ESP32 compat)
FLIP_PHASE_NORMAL   = 'NORMAL'     # 2WD balance, no flip pending
FLIP_PHASE_ARMED    = 'ARMED'      # DCM approaching — raise urgency, pre-spin
FLIP_PHASE_FLIPPING = 'FLIPPING'   # MPC is actively driving to +120° target
FLIP_PHASE_SETTLING = 'SETTLING'   # New wheel landed; waiting for pitch to settle


def _dcm(pitch, pitch_rate, L, omega0):
    """
    Divergent Component of Motion (DCM / capture point) for a LIPM.

        ξ = L sin θ  +  (1/ω₀) L cos θ · θ̇

    Positive ξ = falling forward.  Negative ξ = falling backward.
    """
    x_com  = L * math.sin(pitch)
    xd_com = L * math.cos(pitch) * pitch_rate
    return x_com + xd_com / omega0


def _time_to_controlled_crash(pitch, pitch_rate, L, omega0, dcm_shift, dcm_crash):
    """
    Predict time (s) until |θ| reaches θ_crash under controlled LIPM dynamics.

    Under maximum expected drive torque, the DCM excess evolves as:
        ξ_excess(t) = ξ_excess(0) · exp(ω₀ · t)

    The "crash" DCM target (in the shifted frame) is:
        ξ_crash_shifted = dcm_crash − dcm_shift

    Parameters
    ----------
    L          : pendulum height (m)
    omega0     : rigid-body natural frequency (rad/s)
    dcm_shift  : L · θ_eq from drive torque authority (m)
    dcm_crash  : L · θ_crash where θ_crash is the mechanical crash limit (m)

    Returns
    -------
    t > 0   — seconds until crash under controlled dynamics
    0.0     — already past crash point
    +inf    — drive torque can arrest this fall (ξ_excess ≤ 0)
    """
    xi_raw = _dcm(pitch, pitch_rate, L, omega0)
    xi_excess = abs(xi_raw) - dcm_shift
    dcm_crash_shifted = dcm_crash - dcm_shift

    if dcm_crash_shifted <= 0:
        # Control authority exceeds crash limit — can hold any angle
        return math.inf

    if xi_excess <= 0:
        # Control can arrest this fall
        return math.inf

    if xi_excess >= dcm_crash_shifted:
        return 0.0      # Already past crash — flip overdue

    return (1.0 / omega0) * math.log(dcm_crash_shifted / xi_excess)


class ZMPFlipTrigger:
    """
    Physics-based predictive trigger for the 120° triplet flip.

    All thresholds are **derived from the robot's physical parameters** —
    mass, inertia, torque limits, geometry — rather than hand-tuned.
    The only tunable fraction is ``ZMP_CTRL_AUTHORITY`` (how much of the
    drive torque budget is assumed available during the fall, default 20%).

    Runs inside the MPC slow loop (~30-50 Hz).  Each call to ``update()``
    returns a ``FlipDecision`` dict consumed by ``_run_mpc``.

    Parameters (from controller config dict)
    -----------------------------------------
    ZMP_CTRL_AUTHORITY      : float — fraction [0,1] of max drive torque assumed
                              during fall (0 = free-fall, 1 = full control).
                              Default 0.20 (conservative: motor lag, belt slack,
                              back-EMF at speed consume ~80% of authority).
    ZMP_THETA_CRASH         : float — mechanical crash angle (rad, default π/4 = 45°)
    ZMP_STAIR_HEIGHT        : float — expected step height at landing (m, default 0)
    ZMP_T_FLIP_NOMINAL      : float — expected 120° rotation time (s, default 0.18)
    ZMP_T_FLIP_MARGIN       : float — safety margin on top (s, default 0.05)
    ZMP_T_SETTLE            : float — post-flip settling window (s, default 0.40)
    ZMP_TRIP_TOL            : float — angle tolerance for "flip complete" (rad)
    ZMP_FLIP_COOLDOWN       : float — post-flip rearm lockout (s, default 0.8)
    ZMP_MIN_FALL_RATE_DEG_S : float — secondary fall-rate gate (°/s)
    """

    def __init__(self, config):
        g = abs(config.sim.gravity)

        # ---- Robot physical parameters ----
        m_b    = config.plant.body_mass
        m_w    = config.plant.wheel_mass
        L      = config.plant.cog_height
        I_b    = config.plant.body_inertia
        r_w    = config.robot.wheel_radius
        R_t    = config.robot.triplet_radius
        I_t    = config.mpc.triplet_inertia
        tau_d  = config.motor.max_torque         # per-motor drive torque (Nm)
        tau_t  = config.mpc.triplet_torque_max

        self.L  = L
        self.R_t = R_t

        # ---- Correct rigid-body natural frequency ----
        I_eff = I_b + m_b * L**2
        self.omega0 = math.sqrt(m_b * g * L / I_eff)

        # ---- Pitch dynamics from the linearised state-space model ----
        # (Same formulas as build_mpc_state_space — must stay in sync.)
        M_tot = m_b + m_w
        det   = M_tot * I_eff - (m_b * L)**2
        a_pp  = M_tot * (m_b * g * L) / det          # A[1,0]
        b_pd  = (m_b * L) / (det * r_w) + M_tot / det  # B[1,2]=B[1,3]

        # ---- Drive torque authority during the fall ----
        eta = config.mpc.zmp_ctrl_authority
        # Effective pitch deceleration from both motors at η of max:
        a_ctrl       = b_pd * 2 * eta * tau_d    # rad/s²
        theta_eq     = a_ctrl / a_pp             # equilibrium shift (rad)
        self.dcm_shift = L * theta_eq            # metres: the "safe zone"

        # ---- Crash / target thresholds ----
        theta_crash  = config.mpc.zmp_theta_crash  # 45° default
        self.dcm_crash = L * theta_crash         # metres

        # ---- Stair height adjustment ----
        h_stair = config.mpc.zmp_stair_height
        self.stair_height = h_stair
        if h_stair > 0 and h_stair < R_t:
            # Effective pendulum height is shorter when landing on a step
            L_eff = L - h_stair
            self.omega0_land = math.sqrt(m_b * g * L_eff / I_eff)
        else:
            self.omega0_land = self.omega0

        # ---- Flip timing ----
        delta_alpha = 2 * math.pi / 3   # 120°
        # Theoretical bang-bang minimum:
        T_bb = 2.0 * math.sqrt(delta_alpha * I_t / tau_t) if tau_t > 0 else 0.5
        self.t_flip_bb = T_bb

        # For stair: shorter rotation → faster landing
        if h_stair > 0 and h_stair < R_t:
            alpha_land = math.acos(max(-1.0, 1.0 - h_stair / R_t))
            T_bb_stair = T_bb * math.sqrt(alpha_land / delta_alpha)
        else:
            alpha_land = delta_alpha
            T_bb_stair = T_bb

        self.t_flip   = config.mpc.zmp_t_flip_nominal
        self.t_margin = config.mpc.zmp_t_flip_margin
        self.t_budget = self.t_flip + self.t_margin

        self.t_settle = config.mpc.zmp_t_settle
        self.trip_tol = config.mpc.zmp_trip_tol   # rad

        # ---- Fall-rate gate (secondary safety) ----
        self.min_fall_rate = math.radians(
            config.mpc.zmp_min_fall_rate_deg_s)

        # ---- Early-landing exit from FLIPPING ----
        self.pitch_recover_threshold = config.mpc.zmp_pitch_recover_threshold  # rad (~7°)
        self.flip_min_rotation = config.mpc.zmp_flip_min_rotation  # rad

        # ---- Post-flip cooldown ----
        self.t_cooldown   = config.mpc.zmp_flip_cooldown
        self.cooldown_until = 0.0

        # ---- State ----
        self.phase        = FLIP_PHASE_NORMAL
        self.flip_target  = 0.0
        self.flip_dir     = 0.0
        self.settle_until = 0.0
        self.obstacle_hint = False

        # ---- Diagnostics ----
        self.dcm           = 0.0
        self.dcm_target    = self.dcm_crash
        self.t_to_capture  = math.inf
        self.urgency       = 0.0

        # ---- Print derived parameters ----
        print(f"    ZMP trigger (physics-based):")
        print(f"      ω₀ = {self.omega0:.2f} rad/s  (rigid body; "
              f"point-mass would be {math.sqrt(g/L):.2f})")
        print(f"      Drive authority: η={eta:.0%} of {tau_d:.1f} Nm/motor "
              f"→ a_ctrl={a_ctrl:.1f} rad/s², θ_eq={math.degrees(theta_eq):.1f}°")
        print(f"      DCM shift (safe zone) = {self.dcm_shift*1000:.1f} mm  "
              f"(normal balance stays below this)")
        print(f"      DCM crash = {self.dcm_crash*1000:.1f} mm  "
              f"(θ_crash={math.degrees(theta_crash):.0f}°)")
        print(f"      T_flip: {T_bb*1000:.0f} ms (bang-bang) → {self.t_flip*1000:.0f} ms (config) "
              f"+ {self.t_margin*1000:.0f} ms margin = {self.t_budget*1000:.0f} ms budget")
        if h_stair > 0:
            print(f"      Stair: h={h_stair*100:.0f} cm → α_land={math.degrees(alpha_land):.0f}°, "
                  f"T_land≈{T_bb_stair*1000:.0f} ms")
        print(f"      Fall-rate gate: {math.degrees(self.min_fall_rate):.0f}°/s  "
              f"Cooldown: {self.t_cooldown:.1f}s")

    def set_obstacle_hint(self, detected: bool):
        """Set from TOF/depth sensor to arm the trigger 50 ms earlier."""
        self.obstacle_hint = detected

    def set_stair_height(self, h: float):
        """Update expected stair height (m) from sensor data."""
        self.stair_height = h

    def update(self, pitch, pitch_rate, triplet_eq_angle, trip_dev_L, trip_dev_R,
               sim_time):
        """
        Evaluate the flip condition and advance the phase FSM.

        Uses the controlled LIPM prediction: the time-to-crash accounts for
        the fraction of drive torque fighting the fall (dcm_shift), so the
        trigger fires only when the fall has exceeded the drive's recovery
        capability — not during normal balance oscillations.

        Returns dict with keys: phase, should_flip, flip_target, flip_dir,
        urgency, dcm, t_to_capture.
        """
        L  = self.L
        w0 = self.omega0

        # ---- Compute raw DCM and controlled time-to-crash ----
        self.dcm = _dcm(pitch, pitch_rate, L, w0)
        self.t_to_capture = _time_to_controlled_crash(
            pitch, pitch_rate, L, w0, self.dcm_shift, self.dcm_crash)

        # Fall direction: +1 = forward, -1 = backward
        raw_dir = math.copysign(1.0, self.dcm) if abs(self.dcm) > 1e-4 else 0.0

        # Dynamic budget: obstacle hint arms 50 ms earlier
        extra  = 0.05 if self.obstacle_hint else 0.0
        budget = self.t_budget + extra

        # Compute [0, 1] urgency
        arm_window = 1.5 * budget
        if self.t_to_capture >= arm_window:
            self.urgency = 0.0
        elif self.t_to_capture <= 0.0:
            self.urgency = 1.0
        else:
            self.urgency = float(np.clip(
                1.0 - self.t_to_capture / arm_window, 0.0, 1.0))

        should_flip = False

        # ---- Phase FSM ----
        if self.phase == FLIP_PHASE_NORMAL:
            # Arm only when:
            #   (a) DCM time-to-capture is within the 1.5× budget window, AND
            #   (b) the pitch rate in the fall direction is fast enough to be a
            #       real obstacle fall, not a slow balance oscillation.
            #   (c) the post-flip cooldown has expired (prevents re-trigger from
            #       post-landing rocking after a successful flip).
            falling_fast = abs(pitch_rate) >= self.min_fall_rate
            cooled_down  = sim_time >= self.cooldown_until
            if self.t_to_capture <= 1.5 * budget and falling_fast and cooled_down:
                self.phase    = FLIP_PHASE_ARMED
                self.flip_dir = raw_dir if raw_dir != 0.0 else 1.0

        elif self.phase == FLIP_PHASE_ARMED:
            if self.t_to_capture <= budget:
                # Time to go — rotate toward the fall to catch it.
                # +1 = forward fall → +120° rotation
                # -1 = backward fall → -120° rotation
                self.flip_target = self.flip_dir * 2.0 * math.pi / 3.0
                self.phase       = FLIP_PHASE_FLIPPING
                should_flip      = True
            elif self.t_to_capture > 1.5 * budget + 0.1:
                # DCM retreated with hysteresis (e.g. controller recovered)
                self.phase    = FLIP_PHASE_NORMAL
                self.flip_dir = 0.0

        elif self.phase == FLIP_PHASE_FLIPPING:
            # Primary completion: triplet arrived within tolerance of ±120° target.
            avg_dev = 0.5 * (abs(trip_dev_L - self.flip_target)
                             + abs(trip_dev_R - self.flip_target))
            reached_target = avg_dev < self.trip_tol

            # Early-landing completion: new wheel has already taken the robot's
            # weight (pitch returned near vertical) even though the triplet has
            # not yet rotated a full 120°.  This happens when the new wheel
            # contacts the ground at ~60° of rotation instead of 120°.
            # Conditions: pitch near zero AND triplet has moved at least 60°.
            avg_rotation = abs(0.5 * (trip_dev_L + trip_dev_R))
            early_landing = (
                abs(pitch) < self.pitch_recover_threshold
                and avg_rotation >= self.flip_min_rotation
            )

            if reached_target or early_landing:
                self.settle_until = sim_time + self.t_settle
                self.phase = FLIP_PHASE_SETTLING
            should_flip = True   # Keep targeting the flip until landed

        elif self.phase == FLIP_PHASE_SETTLING:
            if sim_time >= self.settle_until:
                # Start cooldown: block re-arming until the robot has had
                # time to settle onto the new wheel without oscillating.
                self.cooldown_until = sim_time + self.t_cooldown
                self.phase       = FLIP_PHASE_NORMAL
                self.flip_target = 0.0
                self.flip_dir    = 0.0

        return {
            'phase':       self.phase,
            'should_flip': should_flip,
            'flip_target': self.flip_target,
            'flip_dir':    self.flip_dir,
            'urgency':     self.urgency,
            'dcm':         self.dcm,
            't_to_capture': self.t_to_capture,
        }


# ============================================================================
# Hybrid MPC + PD Controller
# ============================================================================

class MPCHybridController(BalanceControllerBase):
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

    ZMP flip trigger config keys (all optional):
        ZMP_T_FLIP_NOMINAL        – expected 120° rotation time (s, default 0.18)
        ZMP_T_FLIP_MARGIN         – safety margin added to flip budget (s, default 0.05)
        ZMP_CTRL_AUTHORITY        – fraction of max drive torque during fall (default 0.20)
        ZMP_THETA_CRASH           – mechanical crash limit (rad, default π/4)
        ZMP_STAIR_HEIGHT          – expected step height at landing (m, default 0.0)
        ZMP_T_SETTLE              – post-flip pitch settling window (s, default 0.40)
        ZMP_TRIP_TOL              – angle tolerance for "flip complete" (rad, default 0.15)
        ZMP_FLIP_Q_TRIP           – Q diagonal weight for triplet during flip (default 120.0)
        ZMP_FLIP_R_TRIP           – R diagonal weight for triplet during flip (default 0.05)
        ZMP_FLIP_Q_PITCH          – Q pitch weight raised during flip (default 120.0)
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
        self.mpc_rate = config.mpc.rate_hz
        self.mpc_period = 1.0 / self.mpc_rate
        self.N = config.mpc.horizon

        # Artificial solve-time budget (simulates ESP32-S3 wall-clock)
        self.simulated_solve_ms = config.mpc.simulated_solve_ms

        # Cost weights
        q_diag = config.mpc.q_diag
        r_diag = config.mpc.r_diag
        q_term_scale = config.mpc.q_terminal_scale

        self.Q = np.diag(q_diag)
        self.R = np.diag(r_diag)
        self.Q_terminal = self.Q * q_term_scale

        # Input bounds
        tau_max = config.motor.max_torque
        trip_tau_max = config.mpc.triplet_torque_max
        self.u_min = np.array([-trip_tau_max, -trip_tau_max, -tau_max, -tau_max])
        self.u_max = np.array([ trip_tau_max,  trip_tau_max,  tau_max,  tau_max])

        # ---- Flip-mode cost weights (override normal Q/R during 120° rotation) ----
        # Higher Q on pitch + triplet = tighter tracking during the manoeuvre.
        # Lower R on triplet = allow the motor to rotate faster.
        zmp_q_trip  = config.mpc.zmp_flip_q_trip
        zmp_q_pitch = config.mpc.zmp_flip_q_pitch
        zmp_r_trip  = config.mpc.zmp_flip_r_trip
        q_flip_diag = list(q_diag)          # copy
        q_flip_diag[0] = zmp_q_pitch        # pitch
        q_flip_diag[1] = zmp_q_pitch * 0.3  # pitch rate
        q_flip_diag[2] = zmp_q_trip         # triplet angle L
        q_flip_diag[3] = zmp_q_trip         # triplet angle R
        q_flip_diag[4] = zmp_q_trip * 0.1   # triplet rate L
        q_flip_diag[5] = zmp_q_trip * 0.1   # triplet rate R
        r_flip_diag = list(r_diag)
        r_flip_diag[0] = zmp_r_trip         # tau_triplet_L
        r_flip_diag[1] = zmp_r_trip         # tau_triplet_R
        self.Q_flip = np.diag(q_flip_diag)
        self.R_flip = np.diag(r_flip_diag)
        self._flip_solver_active = False    # True while flip-mode solver is in use

        # ---- Build initial model and solver ----
        self.Ac, self.Bc = build_mpc_state_space(config)
        Ad, Bd = discretise_zoh(self.Ac, self.Bc, self.mpc_period)

        # Compute DARE (infinite-horizon LQR) terminal cost.
        # This gives the MPC proper handling of the non-minimum-phase
        # pitch/position coupling without needing an outer position loop.
        try:
            P_dare = la.solve_discrete_are(Ad, Bd, self.Q, self.R)
            self.Q_terminal = P_dare
            print(f"    Using DARE terminal cost (infinite-horizon LQR)")
        except Exception as e:
            print(f"    DARE failed ({e}), falling back to scaled Q")
            self.Q_terminal = self.Q * q_term_scale

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
        self.Kp_pd = np.array(config.mpc.pd_kp)
        self.Kd_pd = np.array(config.mpc.pd_kd)

        # Cross-coupling: pitch error → additional drive torque
        self.Kp_pitch_cross = config.mpc.pitch_pd_cross_drive
        self.Kd_pitch_cross = config.mpc.pitch_rate_pd_cross_drive

        print(f"    PD Kp = {self.Kp_pd.tolist()}")
        print(f"    PD Kd = {self.Kd_pd.tolist()}")
        print(f"    Pitch→drive cross-coupling: Kp={self.Kp_pitch_cross}, "
              f"Kd={self.Kd_pitch_cross}")

        # ---- Fast-loop timing ----
        self.control_period = 1.0 / config.control.control_rate_hz
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
        self.x_est = np.zeros(self.nx)

        # ---- Triplet equilibrium angle (2WD operating point) ----
        # State-vector triplet angles are deviations from this equilibrium.
        self.triplet_equilibrium = config.sim.initial_triplet_angle

        # Triplet encoder readings (set by set_triplet_state before each update)
        self._triplet_angle_L = self.triplet_equilibrium
        self._triplet_angle_R = self.triplet_equilibrium
        self._triplet_rate_L = 0.0
        self._triplet_rate_R = 0.0

        # ---- Velocity estimator ----
        self.prev_position = 0.0
        self.prev_vel_time = -1.0    # sentinel: first call initialises only
        self.velocity = 0.0
        self.vel_filter_alpha = 0.2   # slightly faster filter than LQR default

        # ---- Reference / target ----
        self.target_position = 0.0
        self.x_ref = np.zeros(self.nx)     # reference state for MPC

        # Position tracking is handled directly by the MPC via Q_pos
        # weight + DARE terminal cost.  No outer position loop needed —
        # the DARE terminal cost encodes the correct pitch/position
        # tradeoff for the non-minimum-phase dynamics.

        # ---- Yaw control (same interface as LQR/PID controllers) ----
        self.yaw_rate_setpoint = 0.0
        self.yaw_damping_k = config.control.yaw_damping_k

        # ---- Sensor-to-actuator delay pipeline ----
        delay_steps = config.control.sensor_to_actuator_delay_steps
        self.torque_delay_buffer = [(0.0, 0.0)] * (delay_steps + 1)
        self.delay_depth = delay_steps + 1

        # ---- Triplet torque outputs (for tribot_sim to apply) ----
        self._triplet_torque_L = 0.0
        self._triplet_torque_R = 0.0

        # ---- Logging (compatible with tribot_sim PlotJuggler) ----
        self.control_torque = 0.0
        self.target_pitch = 0.0
        self.K_contributions = np.zeros(4)   # [trip_L, trip_R, drive_L, drive_R] ff
        self.state_error = np.zeros(4)       # [pos_err, vel, pitch, pitch_rate]

        # ---- Performance accounting ----
        self.mpc_solve_count = 0
        self.mpc_last_wall_ms = 0.0
        self.mpc_max_wall_ms = 0.0

        # ---- ZMP / DCM flip trigger ----
        self.zmp_trigger = ZMPFlipTrigger(config)
        # Track whether the triplet_equilibrium was already advanced during
        # the current SETTLING window (prevents double-advance).
        self._flip_eq_updated = False
        # Snapshot of triplet_equilibrium at the moment the flip was armed
        # (before any triplet movement).  Used to snap equilibrium to the
        # nearest valid 2WD angle after landing.
        self._equil_before_flip = self.triplet_equilibrium
        # Snapshot of flip_target at the moment we entered FLIPPING phase.
        self._flip_abs_target = self.triplet_equilibrium
        # Direction of the most recent flip (+1 forward, -1 backward);
        # used to apply a small pitch bias during SETTLING.
        self._flip_dir_settled = 0.0
        # (ZMP trigger prints its own derived-parameter summary in __init__)

    # ----------------------------------------------------------------
    # BalanceControllerBase interface
    # ----------------------------------------------------------------

    @property
    def plans_triplet_torque(self) -> bool:
        return True

    @property
    def triplet_torque_L(self) -> float:
        return self._triplet_torque_L

    @property
    def triplet_torque_R(self) -> float:
        return self._triplet_torque_R

    def get_telemetry(self) -> dict:
        """Return MPC-specific diagnostic signals."""
        d = {
            "state_err_pos":      float(self.state_error[0]),
            "state_err_vel":      float(self.state_error[1]),
            "state_err_pitch":    float(self.state_error[2]),
            "state_err_prate":    float(self.state_error[3]),
            "torque_cmd":         float(self.control_torque),
            "target_pitch":       float(self.target_pitch),
            "target_pos":         float(self.target_position),
            "mpc_solve_count":    int(self.mpc_solve_count),
            "mpc_last_wall_ms":   float(self.mpc_last_wall_ms),
            "mpc_max_wall_ms":    float(self.mpc_max_wall_ms),
            "mpc_ff_drive_L":     float(self.K_contributions[0]),
            "mpc_ff_drive_R":     float(self.K_contributions[1]),
            "mpc_pd_drive_L":     float(self.K_contributions[2]),
            "mpc_pd_drive_R":     float(self.K_contributions[3]),
            "triplet_torque_L":   float(self._triplet_torque_L),
            "triplet_torque_R":   float(self._triplet_torque_R),
            "triplet_angle_L":    float(self._triplet_angle_L),
            "triplet_angle_R":    float(self._triplet_angle_R),
            "triplet_dev_L":      float(self.x_est[self.IDX_TRIP_L]),
            "triplet_dev_R":      float(self.x_est[self.IDX_TRIP_R]),
        }
        # Merge flip diagnostics
        flip_diag = self.get_flip_diagnostics()
        if flip_diag:
            d.update(flip_diag)
        return d

    # ----------------------------------------------------------------
    # Public setters (same API as PID / LQR controllers)
    # ----------------------------------------------------------------

    def set_target_position(self, position):
        self.target_position = position

    def set_yaw_rate(self, yaw_rate):
        self.yaw_rate_setpoint = yaw_rate

    def set_triplet_state(self, angle_L, angle_R, rate_L, rate_R):
        """Update triplet encoder readings (called each tick from tribot_sim)."""
        self._triplet_angle_L = angle_L
        self._triplet_angle_R = angle_R
        self._triplet_rate_L = rate_L
        self._triplet_rate_R = rate_R

    def notify_obstacle(self, detected: bool):
        """Forward TOF / depth-sensor obstacle hint to the ZMP trigger."""
        self.zmp_trigger.set_obstacle_hint(detected)

    def get_flip_diagnostics(self) -> dict:
        """
        Return a flat dict suitable for logging to PlotJuggler.

        Keys
        ----
        zmp/phase        : 0=NORMAL 1=ARMED 2=FLIPPING 3=SETTLING
        zmp/dcm          : DCM position (m)
        zmp/dcm_target   : next-wheel x-offset target (m)
        zmp/t_capture    : predicted seconds to DCM target (capped at 2.0)
        zmp/urgency      : [0, 1] flip urgency
        zmp/eq_angle_deg : current triplet equilibrium angle (degrees)
        """
        phase_map = {FLIP_PHASE_NORMAL: 0, FLIP_PHASE_ARMED: 1,
                     FLIP_PHASE_FLIPPING: 2, FLIP_PHASE_SETTLING: 3}
        zt = self.zmp_trigger
        return {
            'zmp/phase':       phase_map.get(zt.phase, -1),
            'zmp/dcm':         zt.dcm,
            'zmp/dcm_max':     zt.dcm_crash,
            'zmp/dcm_trigger': zt.dcm_shift,
            'zmp/t_capture':   min(zt.t_to_capture, 2.0),
            'zmp/urgency':     zt.urgency,
            'zmp/eq_angle_deg': math.degrees(self.triplet_equilibrium),
        }

    # ----------------------------------------------------------------
    # Solver rebuild helper
    # ----------------------------------------------------------------

    def _rebuild_solver(self, Q, R):
        """
        Rebuild the MPCSolver with new cost matrices Q and R.

        Called lazily when flip mode changes to avoid rebuilding every cycle.
        The DARE terminal cost is recomputed; if it fails, falls back to
        scaled Q.
        """
        Ad, Bd = discretise_zoh(self.Ac, self.Bc, self.mpc_period)
        try:
            Q_term = la.solve_discrete_are(Ad, Bd, Q, R)
        except Exception:
            Q_term = Q * self.cfg.mpc.q_terminal_scale

        self.mpc_solver = MPCSolver(
            Ad, Bd, Q, R, Q_term, self.N,
            u_min=self.u_min, u_max=self.u_max
        )

    # ----------------------------------------------------------------
    # MPC slow loop
    # ----------------------------------------------------------------

    def _run_mpc(self, x0, sim_time):
        """
        Solve the MPC QP and store the resulting trajectory.

        The actual Python solve runs instantly, but we record wall-clock
        time and enforce a simulated compute budget so that MPC results
        are not used until ``mpc_busy_until``.

        ZMP / DCM integration
        ~~~~~~~~~~~~~~~~~~~~~
        Each solve cycle we:
          1. Query ZMPFlipTrigger to evaluate the DCM / capture-point condition.
          2. If FLIPPING: set the triplet reference to +120° deviation and
             switch to flip-mode Q/R (aggressive triplet, tight pitch).
          3. If SETTLING → NORMAL transition: advance triplet_equilibrium by
             +120° so the state-vector deviations reset to ≈0 for the next
             solve cycle.
          4. Otherwise: use normal mode Q/R with standard reference.
        """
        # ---- Current triplet deviations from equilibrium ----
        trip_dev_L = self._triplet_angle_L - self.triplet_equilibrium
        trip_dev_R = self._triplet_angle_R - self.triplet_equilibrium

        # ---- Query ZMP trigger ----
        flip = self.zmp_trigger.update(
            pitch          = x0[self.IDX_PITCH],
            pitch_rate     = x0[self.IDX_PITCH_RATE],
            triplet_eq_angle = self.triplet_equilibrium,
            trip_dev_L     = trip_dev_L,
            trip_dev_R     = trip_dev_R,
            sim_time       = sim_time,
        )

        phase = flip['phase']

        # ---- Mode switching: rebuild solver when flip mode changes ----
        want_flip_solver = (phase in (FLIP_PHASE_ARMED,
                                      FLIP_PHASE_FLIPPING,
                                      FLIP_PHASE_SETTLING))
        if want_flip_solver and not self._flip_solver_active:
            self._rebuild_solver(self.Q_flip, self.R_flip)
            self._flip_solver_active = True
            self._flip_eq_updated    = False
            # Snapshot where the triplet equilibrium is RIGHT NOW (before any
            # triplet movement) so we can snap to the nearest valid 2WD angle
            # when SETTLING fires, regardless of how far the triplet actually rotated.
            self._equil_before_flip  = self.triplet_equilibrium
            self._flip_abs_target    = (self.triplet_equilibrium
                                        + flip['flip_target'])
        elif not want_flip_solver and self._flip_solver_active:
            self._rebuild_solver(self.Q, self.R)
            self._flip_solver_active = False

        # ---- Advance triplet equilibrium once the new wheel has landed ----
        # On the first SETTLING cycle, shift the equilibrium by ±120°
        # (matching the flip direction) so the state deviations snap back to
        # ≈0 for the normal-mode MPC that follows.
        if phase == FLIP_PHASE_SETTLING and not self._flip_eq_updated:
            flip_dir = self.zmp_trigger.flip_dir
            if flip_dir == 0.0:
                flip_dir = math.copysign(1.0, flip['flip_target']) if flip['flip_target'] != 0.0 else 1.0
            # Snap equilibrium to the NEXT valid 2WD angle in the flip direction.
            # Valid positions are: equil_before_flip + n × 120°.
            # We use direction-biased rounding (floor for backward, ceil for forward)
            # to guarantee n ≠ 0 even on early landings (<60° of rotation done).
            # This means SETTLING continues rotating the remaining distance to the
            # proper 2WD angle rather than freezing in a 4WD (between-wheel) position.
            actual_trip = 0.5 * (self._triplet_angle_L + self._triplet_angle_R)
            step = 2.0 * math.pi / 3.0   # 120°
            raw_n = (actual_trip - self._equil_before_flip) / step
            if flip_dir < 0:
                n = math.floor(raw_n)     # e.g. -0.485 → -1  (backward flip)
            else:
                n = math.ceil(raw_n)      # e.g. +0.485 → +1  (forward flip)
            if n == 0:                    # safety: force at least one step
                n = int(math.copysign(1.0, flip_dir))
            self.triplet_equilibrium = self._equil_before_flip + n * step
            # Re-derive deviations w.r.t. the new equilibrium
            trip_dev_L = self._triplet_angle_L - self.triplet_equilibrium
            trip_dev_R = self._triplet_angle_R - self.triplet_equilibrium
            # Reset position target to current location so the controller does NOT
            # drive the robot back toward the pre-fall position after SETTLING ends.
            self.target_position = x0[self.IDX_FWD_POS]
            self._flip_eq_updated = True
            self._flip_dir_settled = flip_dir   # remember for SETTLING pitch bias

        # ---- Build MPC reference for this solve cycle ----
        self.x_ref[:] = 0.0
        self.x_ref[self.IDX_FWD_POS] = self.target_position
        self.x_ref[self.IDX_FWD_VEL] = 0.0

        if phase == FLIP_PHASE_FLIPPING:
            # Target the +120° triplet deviation (relative to current eq)
            self.x_ref[self.IDX_TRIP_L]   = flip['flip_target']
            self.x_ref[self.IDX_TRIP_R]   = flip['flip_target']
            self.x_ref[self.IDX_TRIP_L_D] = 0.0   # arrive at rest
            self.x_ref[self.IDX_TRIP_R_D] = 0.0
            # Freeze the forward-position reference at the current position
            # so the MPC does not try to move forward during the flip.
            self.x_ref[self.IDX_FWD_POS]  = x0[self.IDX_FWD_POS]
        elif phase == FLIP_PHASE_SETTLING:
            # Hold the new equilibrium (deviations should be near 0 after
            # the equilibrium advance above); allow pitch to settle.
            # Do NOT add a pitch bias here — it leaves a residual lean that
            # immediately re-triggers the flip FSM once SETTLING ends.
            self.x_ref[self.IDX_TRIP_L]   = 0.0
            self.x_ref[self.IDX_TRIP_R]   = 0.0
            self.x_ref[self.IDX_FWD_POS]  = x0[self.IDX_FWD_POS]
        elif phase == FLIP_PHASE_ARMED:
            # Start leaning slightly *into* the fall direction to seed the
            # MPC plan.  flip_dir = +1 → lean forward, -1 → lean backward.
            fd = flip['flip_dir']
            self.x_ref[self.IDX_PITCH] = fd * 0.015 * flip['urgency']

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

        # Clamp each channel (per-actuator limits: triplet 5 Nm, drive 1 Nm)
        tau = np.clip(tau, self.u_min, self.u_max)

        return tau

    # ----------------------------------------------------------------
    # Main update  (called every physics tick from tribot_sim)
    # ----------------------------------------------------------------

    def update(self, measured_pitch, measured_pitch_rate,
               position, yaw_rate, sim_time, dt, ref=None):
        """
        Run one tick of the hybrid MPC+PD controller.

        Same signature as LQRBalanceController.update() for drop-in use.

        Returns
        -------
        left_torque, right_torque : float
            Commanded motor torques (Nm) for left and right drive motors.
        """
        # ---- Velocity estimation (only at control rate — avoids noise
        # amplification at 500 Hz physics rate; matches LQR approach) ----
        # Moved inside the fast-loop gate below.

        # ---- Build full 8-state estimate ----
        # Pitch and forward from sensors; triplet from encoders.
        # Triplet angles are expressed as deviations from the 2WD
        # equilibrium so the linearised model (around 0) stays valid.
        self.x_est[self.IDX_PITCH]      = measured_pitch
        self.x_est[self.IDX_PITCH_RATE] = measured_pitch_rate
        self.x_est[self.IDX_TRIP_L]     = self._triplet_angle_L - self.triplet_equilibrium
        self.x_est[self.IDX_TRIP_R]     = self._triplet_angle_R - self.triplet_equilibrium
        self.x_est[self.IDX_TRIP_L_D]   = self._triplet_rate_L
        self.x_est[self.IDX_TRIP_R_D]   = self._triplet_rate_R
        self.x_est[self.IDX_FWD_POS]    = position
        self.x_est[self.IDX_FWD_VEL]    = self.velocity

        # ---- MPC slow loop ----
        if sim_time >= self.next_mpc_time and sim_time >= self.mpc_busy_until:
            self._run_mpc(self.x_est.copy(), sim_time)
            self.next_mpc_time = sim_time + self.mpc_period

        # ---- Fast PD loop ----
        jitter = (np.random.normal(0, self.cfg.control.control_jitter_std)
                  if self.cfg.imu.add_sensor_noise else 0)

        if sim_time >= self.next_control_time:
            self.next_control_time = sim_time + self.control_period + jitter

            # Velocity estimation at control rate (not every physics tick)
            if self.prev_vel_time < 0:
                # First call: just initialise position reference
                self.prev_position = position
                self.prev_vel_time = sim_time
            else:
                vel_dt = sim_time - self.prev_vel_time
                if vel_dt > 1e-6:
                    raw_vel = (position - self.prev_position) / vel_dt
                    self.velocity += self.vel_filter_alpha * (raw_vel - self.velocity)
                self.prev_position = position
                self.prev_vel_time = sim_time
            self.x_est[self.IDX_FWD_VEL] = self.velocity

            # Interpolate MPC trajectory at current sim_time
            # If MPC is still "computing" (sim_time < mpc_busy_until),
            # we use the previous trajectory — just like real hardware.
            x_ref, u_ff = self._interpolate_trajectory(sim_time)

            # PD tracking step
            tau_4 = self._pd_step(x_ref, u_ff, self.x_est)

            # The tribot_sim expects drive + triplet torques separately.
            # Drive torques are returned from update(); triplet torques are
            # exposed via self.triplet_torque_L / R for tribot_sim to apply.
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
        self._triplet_torque_L = delayed_trip_L
        self._triplet_torque_R = delayed_trip_R

        return left_torque, right_torque
