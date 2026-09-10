"""
Planar (sagittal) plant model of the tribot for model predictive control.

Two contact modes share one state vector so the MPC can be scheduled
across them without changing its structure:

    z = [s, s_dot, theta, theta_dot, lam, lam_dot]
    u = [u_d, u_t]

    s      forward position of the grounded wheel centre (m)
    theta  body pitch, positive = top of body leans FORWARD (rad)
    lam    "leg" angle: direction from the grounded wheel centre to the
           hub axis, positive = hub is FORWARD of the contact (rad)
    u_d    total drive torque, both sides (Nm), positive = forward
    u_t    total triplet hub torque, both sides (Nm), positive = hub
           rotates so that the bottom wheel moves forward

Frame: (f, z) with f = forward, z = up.  `theta` and `lam` are measured
from the upward vertical toward forward (forward tilt positive).  The hub
joint angle `phi` is the simulator's encoder value: positive `phi` rotates
the triplet so that its bottom wheel moves FORWARD relative to the hub,
which is the opposite rotation sense to a forward body tilt.  Hence the
absolute triplet rotation is  psi = phi - theta  (bottom-forward positive)
and the leg angle follows from it (see `leg_angle`).  See `PITCH_SIGN` for
how the simulator's measured pitch maps onto `theta`.  All of this was
checked numerically against MuJoCo by tools/validate_mpc_plant.py.

Modes
-----
SINGLE_CONTACT (2WD, and any intermediate hub angle): a double inverted
pendulum on a rolling wheel.  Link 1 is the triplet "leg" (contact -> hub,
length R), link 2 is the body (hub -> CoG, length l).  The drive torque acts
between the wheel and the body (the hub is transparent to it, see the
free-hub trick in docs/TRIPLET_BALANCING_ROBOT.md), the triplet torque acts
between the hub and the body.

FOUR_WD (both lower wheels of each triplet on the ground): the triplet is
locked to the ground plane, so the leg angle is not a degree of freedom.
The body is a single inverted pendulum about the hub axis, the triplet
motor acts directly as a pitch torque on the body, and the unilateral
ground contacts impose the tipping constraint

    |u_t + (z_w / r) u_d| <= m_total g d

The model is written as an explicit Lagrangian (mass matrix from point-mass
Jacobians, gravity from the potential) and linearised numerically at the
requested operating point, so the same code serves any hub angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import linalg as la


# Sign relating the simulator's measured pitch to `theta` above.
#
# In the MuJoCo model the robot drives toward body -X and the measured pitch
# is the rotation about body +Y, so a positive measured pitch tilts the top
# of the body toward +X, i.e. BACKWARD.  The planar model uses forward-lean
# positive, hence theta = PITCH_SIGN * measured_pitch.  Verified numerically
# by tools/validate_mpc_plant.py.
PITCH_SIGN = -1.0

NX = 6   # state dimension
NU = 2   # input dimension

_PI_3 = math.pi / 3.0
_2PI_3 = 2.0 * math.pi / 3.0


class ContactMode(Enum):
    SINGLE_CONTACT = 'single'
    FOUR_WD = '4wd'


def wrap120(angle: float) -> float:
    """Wrap an angle into [-60 deg, +60 deg) — the triplet's 120 deg symmetry."""
    return (angle + _PI_3) % _2PI_3 - _PI_3


def grounded_wheel_offset(theta: float, phi: float) -> float:
    """
    Signed angle of the lowest wheel of one triplet from straight-down (rad).

    theta: body pitch (forward-lean positive, planar convention)
    phi:   hub joint angle (encoder), bottom-wheel-forward positive.

    Positive result = lowest wheel is forward of the hub.  At the 4WD pose
    (hub joint at 0, body upright) two wheels tie at +-60 deg; at the 2WD
    pose (hub joint at +-60 deg) the result is ~0.
    """
    return wrap120(phi - theta + _PI_3)


def leg_angle(theta: float, phi: float) -> float:
    """Leg angle lam (contact -> hub, forward positive) for one triplet."""
    return -grounded_wheel_offset(theta, phi)


@dataclass
class PlantParams:
    """Physical constants of the planar model (both sides summed)."""
    m_body: float          # kg
    l_cog: float           # m, hub axis -> body CoG
    I_body: float          # kg m^2, body pitch inertia about its CoG
    m_hub: float           # kg, both hubs
    I_hub: float           # kg m^2, both hubs about the hub axis
    m_wheel: float         # kg, ONE wheel
    I_wheel: float         # kg m^2, ONE wheel about its axle
    n_wheels: int          # total number of wheels (belt coupled)
    R: float               # m, triplet circumradius
    r: float               # m, wheel radius
    g: float               # m/s^2
    c_hub: float           # Nm s/rad, hub joint damping, both sides
    d_4wd: float           # m, half distance between grounded wheels in 4WD
    z_w4: float            # m, grounded wheel axle depth below hub in 4WD

    @classmethod
    def from_config(cls, config) -> 'PlantParams':
        p = config.plant
        return cls(
            m_body=p.body_mass,
            l_cog=p.cog_height,
            I_body=p.body_inertia,
            m_hub=2.0 * p.hub_mass,
            I_hub=2.0 * p.hub_inertia,
            m_wheel=p.wheel_mass_each,
            I_wheel=p.wheel_inertia,
            n_wheels=6,
            R=config.robot.triplet_radius,
            r=config.robot.wheel_radius,
            g=abs(config.sim.gravity),
            c_hub=2.0 * config.robot.triplet_joint_damping,
            d_4wd=p.wheelbase_half_4wd,
            z_w4=p.wheel_depth_4wd,
        )

    # Derived quantities -------------------------------------------------

    @property
    def m_total(self) -> float:
        return self.m_body + self.m_hub + self.n_wheels * self.m_wheel

    @property
    def m_spin(self) -> float:
        """Equivalent translational mass of the belt-coupled wheel spin."""
        return self.n_wheels * self.I_wheel / self.r ** 2

    @property
    def hub_height_4wd(self) -> float:
        return self.r + self.R * math.cos(_PI_3)

    @property
    def tipping_torque(self) -> float:
        """Max hub moment the two 4WD ground contacts can react (Nm)."""
        return self.m_total * self.g * self.d_4wd


class PlanarPlant:
    """Nonlinear planar model + linearisation for both contact modes."""

    def __init__(self, params: PlantParams):
        self.p = params

    # ------------------------------------------------------------------
    # Single-contact (2WD-like) Lagrangian model in q = (s, lam, theta)
    # ------------------------------------------------------------------

    def _point_masses(self, lam: float, theta: float):
        """
        Return a list of (mass, position (f,z), jacobian 2x3) for all point
        masses of the single-contact model.  Both sides are summed.
        """
        p = self.p
        R, l = p.R, p.l_cog
        sl, cl = math.sin(lam), math.cos(lam)
        st, ct = math.sin(theta), math.cos(theta)

        masses = []

        # Grounded wheels (one per side): at the wheel centre.
        J_w = np.array([[1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0]])
        masses.append((2.0 * p.m_wheel, np.array([0.0, p.r]), J_w))

        # Hubs: wheel centre + R (sin lam, cos lam)
        p_h = np.array([R * sl, p.r + R * cl])
        J_h = np.array([[1.0, R * cl, 0.0],
                        [0.0, -R * sl, 0.0]])
        masses.append((p.m_hub, p_h, J_h))

        # Free wheels (two per side) at hub + R (sin(lam +- 60), cos(lam +- 60))
        for delta in (+_PI_3, -_PI_3):
            a = lam + delta
            pos = p_h + R * np.array([math.sin(a), math.cos(a)])
            J = J_h + np.array([[0.0, R * math.cos(a), 0.0],
                                [0.0, -R * math.sin(a), 0.0]])
            masses.append((2.0 * p.m_wheel, pos, J))

        # Body CoG: hub + l (sin theta, cos theta)
        p_b = p_h + l * np.array([st, ct])
        J_b = J_h + np.array([[0.0, 0.0, l * ct],
                              [0.0, 0.0, -l * st]])
        masses.append((p.m_body, p_b, J_b))

        return masses

    def mass_matrix_single(self, lam: float, theta: float) -> np.ndarray:
        p = self.p
        M = np.zeros((3, 3))
        for m, _, J in self._point_masses(lam, theta):
            M += m * (J.T @ J)
        M[0, 0] += p.m_spin           # belt-coupled wheel spin
        M[1, 1] += p.I_hub            # hubs rotate with the leg
        M[2, 2] += p.I_body
        return M

    def gravity_grad_single(self, lam: float, theta: float) -> np.ndarray:
        """dV/dq for q = (s, lam, theta)."""
        p = self.p
        grad = np.zeros(3)
        for m, _, J in self._point_masses(lam, theta):
            grad += m * p.g * J[1, :]   # d z_i / d q
        return grad

    def _damping_single(self) -> np.ndarray:
        """Generalised damping matrix D (Q_damp = -D q_dot) for (s, lam, theta)."""
        c = self.p.c_hub
        D = np.zeros((3, 3))
        # The hub joint angle is phi = theta - lam + const, so the joint
        # damping torque -c*phi_dot does work on (theta_dot - lam_dot).
        D[1, 1] = c
        D[1, 2] = -c
        D[2, 1] = -c
        D[2, 2] = c
        return D

    @staticmethod
    def _input_map_single() -> np.ndarray:
        """
        Generalised force per unit input for (s, lam, theta) x (u_d, u_t).

        u_d acts between wheel and body: ground force u_d/r on s and the
        reaction -u_d on the body (forward drive pitches the body back).
        u_t acts on the hub joint phi = theta - lam + const: +u_t on theta,
        -u_t on lam.
        """
        return np.array([[0.0, 0.0],    # filled with 1/r at runtime
                         [0.0, -1.0],
                         [-1.0, 1.0]])

    def linearize_single(self, lam0: float, theta0: float):
        """
        Linearise the single-contact model about (lam0, theta0), zero
        velocity and zero input.

        Returns (A, B, c) with z_dot = A z + B u + c, z = [s, s_dot, theta,
        theta_dot, lam, lam_dot] (positions are absolute, so c carries the
        gravity drift of the linearisation point).
        """
        p = self.p
        M = self.mass_matrix_single(lam0, theta0)
        Minv = la.inv(M)

        # Stiffness: -d^2 V / dq^2 by central differences of the gradient.
        h = 1e-5
        K = np.zeros((3, 3))
        for j, (dl, dt) in enumerate(((0, 0), (h, 0), (0, h))):
            if j == 0:
                continue
            gp = self.gravity_grad_single(lam0 + dl, theta0 + dt)
            gm = self.gravity_grad_single(lam0 - dl, theta0 - dt)
            K[:, j] = -(gp - gm) / (2.0 * h)
        # column 0 (s) is zero: potential does not depend on s.

        D = self._damping_single()
        Bq = self._input_map_single()
        Bq[0, 0] = 1.0 / p.r

        g0 = -self.gravity_grad_single(lam0, theta0)          # drift force
        # Remove the part already represented by K q0 so that
        # Q(q) ~= g0 + K (q - q0) holds around the point.
        q0 = np.array([0.0, lam0, theta0])
        drift = g0 - K @ q0

        acc_q = Minv @ K        # q_ddot from q
        acc_v = -Minv @ D       # q_ddot from q_dot
        acc_u = Minv @ Bq
        acc_c = Minv @ drift

        # Map (s, lam, theta) ordering -> state [s, s_dot, th, th_dot, lam, lam_dot]
        idx = [0, 2, 1]          # state position index -> q index
        A = np.zeros((NX, NX))
        B = np.zeros((NX, NU))
        c = np.zeros(NX)
        for i_state, i_q in enumerate(idx):
            A[2 * i_state, 2 * i_state + 1] = 1.0
            for j_state, j_q in enumerate(idx):
                A[2 * i_state + 1, 2 * j_state] = acc_q[i_q, j_q]
                A[2 * i_state + 1, 2 * j_state + 1] = acc_v[i_q, j_q]
            B[2 * i_state + 1, :] = acc_u[i_q, :]
            c[2 * i_state + 1] = acc_c[i_q]
        return A, B, c

    # ------------------------------------------------------------------
    # 4WD model in q = (s, theta), leg pinned
    # ------------------------------------------------------------------

    def linearize_4wd(self, theta0: float):
        """
        Linearise the 4WD cart-pendulum about theta0 (leg locked).  Same
        state layout as `linearize_single`; the leg rows are zero.
        """
        p = self.p
        m_cart = p.m_hub + p.n_wheels * p.m_wheel + p.m_spin
        ml = p.m_body * p.l_cog
        ct, st = math.cos(theta0), math.sin(theta0)
        M = np.array([[m_cart + p.m_body, ml * ct],
                      [ml * ct, p.m_body * p.l_cog ** 2 + p.I_body]])
        Minv = la.inv(M)

        K = np.array([[0.0, 0.0],
                      [0.0, ml * p.g * ct]])
        D = np.array([[0.0, 0.0],
                      [0.0, p.c_hub]])
        # Leg locked: phi = theta + const, so u_t is a direct body torque.
        Bq = np.array([[1.0 / p.r, 0.0],
                       [-1.0, 1.0]])
        g0 = np.array([0.0, ml * p.g * st])
        drift = g0 - K @ np.array([0.0, theta0])

        acc_q = Minv @ K
        acc_v = -Minv @ D
        acc_u = Minv @ Bq
        acc_c = Minv @ drift

        A = np.zeros((NX, NX))
        B = np.zeros((NX, NU))
        c = np.zeros(NX)
        # (state row of the position, q index): s -> row 0, theta -> row 2
        slots = ((0, 0), (2, 1))
        for row_pos, i_q in slots:
            A[row_pos, row_pos + 1] = 1.0
            for col_pos, j_q in slots:
                A[row_pos + 1, col_pos] = acc_q[i_q, j_q]
                A[row_pos + 1, col_pos + 1] = acc_v[i_q, j_q]
            B[row_pos + 1, :] = acc_u[i_q, :]
            c[row_pos + 1] = acc_c[i_q]
        return A, B, c

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def linearize(self, mode: ContactMode, lam0: float, theta0: float):
        if mode == ContactMode.FOUR_WD:
            return self.linearize_4wd(theta0)
        return self.linearize_single(lam0, theta0)

    def equilibrium_leg_angle(self, theta_ref: float) -> float:
        """
        Leg angle that places the composite CoG above the contact for a
        body lean theta_ref in single-contact mode (small-angle balance of
        first moments).
        """
        p = self.p
        M1 = p.R * (p.m_hub + 1.5 * (4.0 * p.m_wheel) + p.m_body)
        M2 = p.m_body * p.l_cog
        s = -M2 * math.sin(theta_ref) / M1
        return math.asin(max(-1.0, min(1.0, s)))

    def unstable_rate(self, mode: ContactMode, lam0: float = 0.0,
                      theta0: float = 0.0) -> float:
        """Largest real eigenvalue of A (rad/s), the divergence rate."""
        A, _, _ = self.linearize(mode, lam0, theta0)
        ev = np.linalg.eigvals(A)
        return float(max(ev.real))

    def composite_pendulum(self):
        """
        Whole robot (minus the grounded wheels) as one rigid pendulum about
        the ground contact with the leg locked at lam = 0, as used by the
        flip trigger.  Returns (mass, CoG height above contact, inertia
        about the contact).
        """
        p = self.p
        parts = [
            (p.m_body, 0.0, p.r + p.R + p.l_cog, p.I_body),
            (p.m_hub, 0.0, p.r + p.R, p.I_hub),
        ]
        for delta in (+_PI_3, -_PI_3):
            parts.append((2.0 * p.m_wheel, p.R * math.sin(delta),
                          p.r + p.R + p.R * math.cos(delta), 2.0 * p.I_wheel))
        m = sum(q[0] for q in parts)
        L = sum(q[0] * q[2] for q in parts) / m
        I = sum(q[0] * (q[1] ** 2 + q[2] ** 2) + q[3] for q in parts)
        return m, L, I

    def unstable_rate_body(self) -> float:
        """omega_0^2 (1/s^2) of the composite pendulum about the contact."""
        m, L, I = self.composite_pendulum()
        return m * self.p.g * L / I


def discretize(A: np.ndarray, B: np.ndarray, c: np.ndarray, dt: float):
    """Exact zero-order-hold discretisation of z_dot = A z + B u + c."""
    n, m = A.shape[0], B.shape[1]
    Maug = np.zeros((n + m + 1, n + m + 1))
    Maug[:n, :n] = A
    Maug[:n, n:n + m] = B
    Maug[:n, n + m] = c
    E = la.expm(Maug * dt)
    Ad = E[:n, :n]
    Bd = E[:n, n:n + m]
    cd = E[:n, n + m]
    return Ad, Bd, cd
