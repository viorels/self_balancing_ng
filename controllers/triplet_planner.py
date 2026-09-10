"""
Event layer for the MPC controller: hub-angle references, drive-mode
transitions and the emergency flip.

The MPC owns the *symmetric* part of the triplet motion (the leg angle
that balances the robot).  This planner owns everything discrete:

  * per-side hub joint references (unwrapped, continuous);
  * 4WD <-> 2WD transitions, executed as an *asymmetric* rotation (left
    hub backward, right hub forward) so that one front and one rear wheel
    stay on the ground and the support line passes under the hub axis at
    every instant — no lean is needed and the pitch model is undisturbed;
  * the emergency flip: when the divergent component of motion (DCM)
    exceeds what the drive motor can arrest, both hubs rotate 120 deg in
    the fall direction to plant a fresh wheel ahead of the fall
    (docs/DCM_AUTHORITY_FLIP_TRIGGER.md).

References follow minimum-jerk (quintic) profiles, which have zero
velocity and acceleration at both ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot_state import DriveMode

_PI_3 = math.pi / 3.0
_2PI_3 = 2.0 * math.pi / 3.0


def _quintic(t: float, T: float):
    """Minimum-jerk profile s(t), s_dot(t) on [0, T]."""
    if t <= 0.0:
        return 0.0, 0.0
    if t >= T:
        return 1.0, 0.0
    tau = t / T
    s = 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5
    sd = (30.0 * tau ** 2 - 60.0 * tau ** 3 + 30.0 * tau ** 4) / T
    return s, sd


class _Ramp:
    """Quintic ramp of a scalar between two values."""

    def __init__(self):
        self.active = False
        self.t0 = 0.0
        self.T = 1.0
        self.a = 0.0
        self.b = 0.0
        self.value = 0.0
        self.rate = 0.0

    def start(self, t, a, b, T):
        self.t0, self.a, self.b, self.T = t, a, b, max(T, 1e-3)
        self.active = True
        self.value, self.rate = a, 0.0

    def update(self, t):
        if not self.active:
            self.rate = 0.0
            return self.value, self.rate
        s, sd = _quintic(t - self.t0, self.T)
        self.value = self.a + (self.b - self.a) * s
        self.rate = (self.b - self.a) * sd
        if t - self.t0 >= self.T:
            self.active = False
            self.value, self.rate = self.b, 0.0
        return self.value, self.rate

    def set(self, v):
        self.active = False
        self.value, self.rate = v, 0.0


class FlipPhase:
    NORMAL = 0
    ARMED = 1
    FLIPPING = 2
    SETTLING = 3


@dataclass
class PlannerOutput:
    phi_ref_L: float          # unwrapped hub joint reference, left (rad)
    phi_ref_R: float          # right (rad)
    phi_rate_L: float
    phi_rate_R: float
    transitioning: bool       # asymmetric mode transition in progress
    flipping: bool            # flip in progress (hub torque owned by planner)
    settling: bool            # just landed after a flip
    freeze_position: bool     # do not chase the old position reference
    reset_position: bool      # true on the tick the position ref must latch
    phase: int
    dcm: float
    dcm_margin: float
    t_capture: float


class TripletPlanner:
    """Drive-mode transitions and DCM flip trigger."""

    def __init__(self, config, plant):
        m = config.mpc
        self.plant = plant
        self.T_transition = m.transition_time
        self.T_flip = m.flip_time
        self.T_settle = m.flip_settle_time
        self.cooldown = m.flip_cooldown
        self.authority = m.flip_authority
        self.theta_crash = m.flip_theta_crash
        self.min_fall_rate = math.radians(m.flip_min_fall_rate_deg_s)
        self.t_margin = m.flip_time_margin
        self.enabled = m.flip_enabled
        self.theta_trigger = m.flip_theta_trigger
        self.drive_torque_total = 2.0 * config.motor.max_torque

        # Effective pendulum used for the DCM: whole robot about the contact.
        m_pole, self.L, _ = plant.composite_pendulum()
        self.mgl = m_pole * plant.p.g * self.L
        self.omega0 = math.sqrt(max(1e-6, plant.unstable_rate_body()))

        self._ramp_L = _Ramp()
        self._ramp_R = _Ramp()
        self._initialised = False
        self.mode = DriveMode.FOUR_WD
        self.phase = FlipPhase.NORMAL
        self._flip_dir = 0.0
        self._phase_t0 = 0.0
        self._cooldown_until = 0.0
        self._reset_pending = False
        self.dcm = 0.0
        self.dcm_margin = 0.0
        self.t_capture = float('inf')

    # ------------------------------------------------------------------

    def reset(self):
        self._initialised = False
        self.mode = DriveMode.FOUR_WD
        self.phase = FlipPhase.NORMAL
        self._flip_dir = 0.0
        self._cooldown_until = 0.0
        self._reset_pending = False
        self._ramp_L = _Ramp()
        self._ramp_R = _Ramp()

    def initialise(self, phi_L: float, phi_R: float):
        """Latch references from the first encoder reading (snapped to 60 deg)."""
        snap = lambda a: round(a / _PI_3) * _PI_3
        self._ramp_L.set(snap(phi_L))
        self._ramp_R.set(snap(phi_R))
        # Infer the starting mode: a hub joint at an odd multiple of 60 deg
        # has one wheel straight down (2WD), an even multiple has two
        # wheels tied at the bottom (4WD).
        k = round(snap(phi_L) / _PI_3)
        self.mode = DriveMode.TWO_WD if (k % 2 != 0) else DriveMode.FOUR_WD
        self._initialised = True

    @property
    def initialised(self) -> bool:
        return self._initialised

    def request_mode(self, mode: DriveMode, t: float):
        """Start an asymmetric 4WD<->2WD transition."""
        if not self._initialised or mode == self.mode:
            return
        if self.phase in (FlipPhase.FLIPPING,):
            return
        L, R = self._ramp_L.value, self._ramp_R.value
        if mode == DriveMode.TWO_WD:
            self._ramp_L.start(t, L, L - _PI_3, self.T_transition)
            self._ramp_R.start(t, R, R + _PI_3, self.T_transition)
        else:
            self._ramp_L.start(t, L, L + _PI_3, self.T_transition)
            self._ramp_R.start(t, R, R - _PI_3, self.T_transition)
        self.mode = mode

    # ------------------------------------------------------------------

    def _dcm_update(self, theta: float, theta_dot: float):
        """DCM and time-to-crash for the whole-robot pendulum."""
        L, w0 = self.L, self.omega0
        xi = L * math.sin(theta) + L * math.cos(theta) * theta_dot / w0
        tau_eff = self.authority * self.drive_torque_total
        theta_eq = min(tau_eff / self.mgl, self.theta_crash)
        xi_max = L * math.sin(theta_eq)
        xi_crash = L * math.sin(self.theta_crash)
        self.dcm = xi
        self.dcm_margin = abs(xi) - xi_max
        if abs(xi) > 1e-6 and abs(xi) < xi_crash:
            self.t_capture = math.log(xi_crash / abs(xi)) / w0
        elif abs(xi) >= xi_crash:
            self.t_capture = 0.0
        else:
            self.t_capture = float('inf')

    def update(self, t: float, theta: float, theta_dot: float,
               phi_L: float, phi_R: float,
               predicted_peak_theta: float = 0.0) -> PlannerOutput:
        """
        Advance the planner one tick.

        theta, theta_dot:     body pitch (forward-lean positive) and rate.
        phi_L, phi_R:         measured hub joint angles (rad).
        predicted_peak_theta: signed pitch of largest magnitude along the
                              MPC's latest predicted trajectory.  The flip
                              is armed when the MPC itself predicts the
                              pitch running past `flip_theta_trigger`
                              despite full use of the drive torque — a
                              tighter test than the free-fall DCM, which
                              ignores that the wheels can catch the body.
        """
        if not self._initialised:
            self.initialise(phi_L, phi_R)
        elif (self.phase == FlipPhase.NORMAL
              and not (self._ramp_L.active or self._ramp_R.active)):
            # The MPC may have rolled the triplets onto another wheel by
            # itself (e.g. after a hard push).  Once the hubs rest near a
            # different 60 deg multiple, re-latch references and mode.
            snapped_L = round(phi_L / _PI_3) * _PI_3
            snapped_R = round(phi_R / _PI_3) * _PI_3
            if (abs(snapped_L - self._ramp_L.value) > _PI_3 / 2.0
                    or abs(snapped_R - self._ramp_R.value) > _PI_3 / 2.0) \
                    and abs(phi_L - snapped_L) < math.radians(8.0) \
                    and abs(phi_R - snapped_R) < math.radians(8.0):
                self.initialise(phi_L, phi_R)

        reset_position = False
        self._dcm_update(theta, theta_dot)
        two_wd_like = (self.mode == DriveMode.TWO_WD
                       and not (self._ramp_L.active or self._ramp_R.active))
        falling_dir = 0.0
        if abs(predicted_peak_theta) > self.theta_trigger \
                and predicted_peak_theta * theta_dot > 0.0 \
                and abs(theta_dot) > self.min_fall_rate:
            falling_dir = 1.0 if predicted_peak_theta > 0.0 else -1.0
        if abs(theta) > self.theta_trigger and theta * theta_dot > 0.0:
            falling_dir = 1.0 if theta > 0.0 else -1.0

        # --- Flip state machine (2WD only) ---
        if self.phase == FlipPhase.NORMAL:
            if (self.enabled and two_wd_like and t >= self._cooldown_until
                    and falling_dir != 0.0):
                self.phase = FlipPhase.ARMED
                self._flip_dir = falling_dir
                self._phase_t0 = t
            # fall through to check immediate trigger
        if self.phase == FlipPhase.ARMED:
            budget = self.T_flip + self.t_margin
            if falling_dir == 0.0 and t - self._phase_t0 > 0.05:
                self.phase = FlipPhase.NORMAL            # recovered
            elif (self.t_capture <= budget
                  or abs(theta) > 0.6 * self.theta_trigger):
                # Falling forward (dcm > 0): the new wheel must land ahead,
                # which is a NEGATIVE hub rotation (bottom wheel moves back,
                # the front-upper wheel comes down in front).
                d = -self._flip_dir * _2PI_3
                L, R = self._ramp_L.value, self._ramp_R.value
                self._ramp_L.start(t, L, L + d, self.T_flip)
                self._ramp_R.start(t, R, R + d, self.T_flip)
                self.phase = FlipPhase.FLIPPING
                self._phase_t0 = t
        elif self.phase == FlipPhase.FLIPPING:
            if not (self._ramp_L.active or self._ramp_R.active):
                self.phase = FlipPhase.SETTLING
                self._phase_t0 = t
                reset_position = True
        elif self.phase == FlipPhase.SETTLING:
            if t - self._phase_t0 >= self.T_settle:
                self.phase = FlipPhase.NORMAL
                self._cooldown_until = t + self.cooldown

        pL, vL = self._ramp_L.update(t)
        pR, vR = self._ramp_R.update(t)
        transitioning = (self._ramp_L.active or self._ramp_R.active) \
            and self.phase != FlipPhase.FLIPPING
        flipping = self.phase == FlipPhase.FLIPPING
        return PlannerOutput(
            phi_ref_L=pL, phi_ref_R=pR, phi_rate_L=vL, phi_rate_R=vR,
            transitioning=transitioning,
            flipping=flipping,
            settling=self.phase == FlipPhase.SETTLING,
            freeze_position=flipping,
            reset_position=reset_position,
            phase=self.phase,
            dcm=self.dcm, dcm_margin=self.dcm_margin,
            t_capture=self.t_capture,
        )
