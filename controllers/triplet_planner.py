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

from .mpc_plant import wrap120

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


class StepPhase:
    """Stair step manoeuvre (4WD): lean over the front wheel, roll the leg."""
    NONE = 0
    LEAN = 1
    ROLL = 2
    SETTLE = 3


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
    # Stair step manoeuvre
    step_phase: int = StepPhase.NONE
    theta_ref: float | None = None     # pitch reference override (rad)
    lam_ref: float | None = None       # leg angle reference override (rad)
    lam_rate: float = 0.0
    force_single_contact: bool = False # use the leg model even at the tie
    relax_limits: bool = False         # raise pitch limit, drop tipping limit
    keep_tipping: bool = False         # ...but keep the 4WD tipping limit
    limit_drive: bool = False          # cap drive torque (pivot wheel blocked)
    pin_drive: bool = False            # drive torque fixed to the press bias
    hold_for_drop: bool = False        # obstacle ahead we cannot step over: hold
    v_cap: float = float('inf')        # velocity command cap (approach to a step)


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

        # Stair step manoeuvre
        self.step_enabled = m.step_enabled
        self.step_min_h = m.step_min_height
        self.step_max_h = m.step_max_height
        self.step_gap = m.step_trigger_gap
        self.T_step_lean = m.step_lean_time
        self.T_step_lean_extra = m.step_lean_extra
        self.T_step_rest = m.step_rest_time
        self.T_step_settle = m.step_settle_time
        self.front_edge_4wd = plant.p.R * math.sin(_PI_3) + plant.p.r
        self.step_land_margin = m.step_land_margin
        self.step_lean_margin = m.step_lean_margin
        self.step_roll_margin = m.step_roll_margin
        self.step_roll_lead = m.step_roll_lead
        self.step_theta_rate = m.step_theta_rate
        self.step_theta_floor = m.step_theta_floor
        self.step_roll_timeout = m.step_roll_timeout
        self.T_step_roll = m.step_roll_time
        self.step_impact_rate = m.step_impact_rate
        self._th_prev = 0.0
        self._lam_track = -_PI_3
        self._lam_dot_hist = [0.0]
        self.step_down_enabled = m.step_down_enabled
        self.step_max_drop = m.step_max_drop
        self.step_approach_speed = m.step_approach_speed
        self.step_approach_distance = m.step_approach_distance
        self.v_cap = float('inf')
        self.drop_hold = m.drop_hold
        self._step_height = 0.0
        self._lam_pivot0 = -_PI_3
        self._lean_target = 0.0
        self._rest_since = 0.0
        self._stall_time = 0.0
        self._last_t = 0.0
        self._lam_land = _PI_3
        self.step_phase = StepPhase.NONE
        self._step_t0 = 0.0
        self._theta_ramp = _Ramp()
        self._lam_ramp = _Ramp()
        self.hold_for_drop = False

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
        self.step_phase = StepPhase.NONE
        self._theta_ramp = _Ramp()
        self._lam_ramp = _Ramp()
        self.hold_for_drop = False
        self._rest_since = 0.0

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
        if self.phase == FlipPhase.FLIPPING or self.step_phase != StepPhase.NONE:
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

    # ------------------------------------------------------------------
    # Stair step manoeuvre
    # ------------------------------------------------------------------

    def _step_update(self, t, theta, theta_dot, lam, lam_dot, lam_pivot, v_cmd,
                     terrain_dist, terrain_height):
        """
        Advance the stair-step state machine.  Returns a dict of overrides
        for the controller (see PlannerOutput fields).

        The manoeuvre runs only in steady 4WD while driving forward into a
        riser of climbable height:
          LEAN   pitch reference ramps to the lean that puts the composite
                 CoG ahead of the front axle (the rear wheel unloads), see
                 `lean_for_pivot`;
          ROLL   the leg reference sweeps from the measured pivot angle
                 (-60 deg on level ground, more in an oblique stance on a
                 step) to the landing angle for the measured step height;
                 the MPC's single-contact model rolls the cluster over the
                 blocked wheel and the upper wheel lands on the tread;
          SETTLE back in the (now oblique) 4WD stance.
        """
        out = {}
        steady_4wd = (self.mode == DriveMode.FOUR_WD
                      and not (self._ramp_L.active or self._ramp_R.active)
                      and self.phase == FlipPhase.NORMAL)
        driving_fwd = v_cmd > 0.05
        # The riser must still be in front of the front wheel: a height
        # change reported closer than the wheel's leading edge is one the
        # wheel has already climbed (probe under the hub lags the wheel).
        in_front = terrain_dist >= self.front_edge_4wd - 0.05
        riser_close = in_front and (terrain_dist <= self.front_edge_4wd + self.step_gap)
        climbable = self.step_min_h <= terrain_height <= self.step_max_h
        descendable = (self.step_down_enabled
                       and -self.step_max_drop <= terrain_height <= -self.step_min_h)
        climbable = climbable or descendable
        drop_ahead = terrain_height < -self.step_min_h and \
            terrain_dist <= self.front_edge_4wd + 2.0 * self.step_gap

        # `lam_pivot` is the leg angle relative to the FRONT lower wheel
        # (the pivot of the roll), supplied by the controller per side; in
        # an oblique stance on a step the lowest wheel is the rear one and
        # the front wheel is 120 deg on.

        if self.step_phase == StepPhase.NONE:
            # Only start from rest: after a landing the cluster may still be
            # rocking onto its stance (it can over-rotate past the tie and
            # fall back), and a lean begun in that jolt lags its ramp.
            at_rest = abs(lam_dot) < 0.5 and abs(theta_dot) < 0.6
            if not at_rest:
                self._rest_since = t
            can_step = (self.step_enabled and steady_4wd
                        and t - self._rest_since >= self.T_step_rest)
            obstacle_ahead = (abs(terrain_height) > self.step_min_h and in_front
                              and terrain_dist <= self.front_edge_4wd + 2.0 * self.step_gap)
            # Slow down on the approach so the wheel meets the riser gently.
            self.v_cap = float('inf')
            if climbable and terrain_dist <= self.step_approach_distance:
                self.v_cap = self.step_approach_speed
            # Stop in front of anything we cannot step over: a drop when
            # descending is off, any riser in 2WD or mid-transition, and
            # obstacles taller than the manoeuvre handles.
            self.hold_for_drop = driving_fwd and obstacle_ahead and (
                (terrain_height < 0.0 and (self.drop_hold and not descendable))
                or (terrain_height > 0.0 and not (can_step and climbable))
                or (terrain_height < 0.0 and descendable and not can_step))
            if (can_step and driving_fwd and riser_close and climbable):
                self.step_phase = StepPhase.LEAN
                self._step_t0 = t
                self._step_height = terrain_height
                self._lam_pivot0 = lam_pivot
                self._lam_track = lam_pivot
                lean = self.lean_for_pivot(lam_pivot)
                self._lean_target = lean
                self._theta_ramp.start(t, theta, lean, self.T_step_lean)
        elif self.step_phase == StepPhase.LEAN:
            th_ref, _ = self._theta_ramp.update(t)
            # The drive is pinned to a damper on backward base motion so
            # the pivot wheel stays at the riser (or the edge, stepping
            # down) while the body leans.  Left to the MPC, even with a
            # position hold, the base rolls back 10-15 cm to build the
            # lean: climbing, the roll then starts short of the riser;
            # descending, the rear wheel drops off the previous tread.
            out.update(theta_ref=th_ref, relax=True, limit_drive=True,
                       pin_drive=True)
            self._lam_track = self._lam_track + wrap120(lam_pivot - self._lam_track)
            # Early exit only on a real roll-over, not on a contact wobble
            # while the rear wheel unloads.
            cluster_moving = (lam_dot > 0.8
                              and self._lam_track > self._lam_pivot0 + math.radians(8.0))
            # Roll once the body has actually reached its lean (the ramp
            # can end with the body 10-15 deg short after a jolt, and a
            # roll from there stalls), with a timeout as the backstop.
            lean_reached = ((not self._theta_ramp.active)
                            and theta >= self._lean_target - math.radians(4.0))
            timed_out = t - self._step_t0 > self.T_step_lean + self.T_step_lean_extra
            if lean_reached or cluster_moving or timed_out:
                self.step_phase = StepPhase.ROLL
                self._step_t0 = t
                self._lam_land = self.landing_leg_angle(self._step_height)
                # The leg reference sweeps to just short of the next tie so
                # the MPC never brakes the leg before the wheel has landed;
                # the landing itself is detected below.
                lam_end = _PI_3 - math.radians(3.0)
                T = self.T_step_roll * (lam_end - self._lam_track) / (2.0 * _PI_3)
                self._lam_ramp.start(t, self._lam_track, lam_end, T)
                self._stall_time = 0.0
                self._th_prev = th_ref
                self._lam_dot_hist = [lam_dot]
        elif self.step_phase == StepPhase.ROLL:
            lam_ref, lam_rate = self._lam_ramp.update(t)
            # Track the leg angle relative to the pivot, unwrapped.
            self._lam_track = self._lam_track + wrap120(lam_pivot - self._lam_track)
            lam_unwrapped = self._lam_track
            if lam_unwrapped > 0.0:
                # Past the top gravity carries the roll faster than any
                # planned ramp; keep the reference a fixed lead ahead of
                # the measurement so the MPC never brakes the leg (which
                # would pitch the body forward) before the wheel lands.
                lam_ref = lam_unwrapped + self.step_roll_lead
                lam_rate = max(lam_dot, 1.0)
            # Pitch reference follows the leg: keep the composite CoG a
            # small margin ahead of the pivot while the hub is behind it,
            # then settle to a small forward lean.  Rate-limited so the body
            # never needs a violent correction around the landing.
            th_geom = self.lean_for_pivot(min(lam_unwrapped, 0.0),
                                          margin=self.step_roll_margin)
            th_geom = max(th_geom, self.step_theta_floor)
            dth = self.step_theta_rate * (t - self._last_t)
            th_ref = min(max(th_geom, self._th_prev - dth), self._th_prev + dth)
            self._th_prev = th_ref
            # The drive stays available (forward only): against a riser it
            # acts on the body through the blocked wheel; descending it
            # moves the base under the body.  Pinning it here makes the
            # MPC reverse the roll with hub torque instead.
            out.update(theta_ref=th_ref, lam_ref=lam_ref, lam_rate=lam_rate,
                       single=True, relax=True, limit_drive=True)
            # Landing detection, in order of speed:
            #  impact  — the cluster's rotation collapses within one tick
            #            when the wheel hits the tread (hub encoders);
            #  stalled — the leg has passed the top and no longer advances;
            #  reached — geometric landing angle for the measured height.
            past_top = lam_unwrapped > math.radians(12.0)
            # Soft contacts spread the impact over a few ticks: compare with
            # the leg rate ~20 ms ago.
            self._lam_dot_hist.append(lam_dot)
            if len(self._lam_dot_hist) > 5:
                self._lam_dot_hist.pop(0)
            impact = past_top and (self._lam_dot_hist[0] - lam_dot) > self.step_impact_rate
            if past_top and abs(lam_dot) < 0.8:
                self._stall_time += t - self._last_t
            else:
                self._stall_time = 0.0
            reached = lam_unwrapped >= self._lam_land + self.step_land_margin
            stalled = self._stall_time > 0.05
            timed_out = t - self._step_t0 > self.step_roll_timeout
            if impact or reached or stalled or timed_out:
                self.step_phase = StepPhase.SETTLE
                self._step_t0 = t
                self._lam_ramp.set(lam_unwrapped)
        elif self.step_phase == StepPhase.SETTLE:
            # The body may land well forward of upright; allow the lean but
            # keep the tipping limit: a large hub torque in the locked-leg
            # model would roll the cluster over the front wheel again (and
            # off the next edge) instead of pitching the body.
            out['relax'] = True
            out['keep_tipping'] = True
            if t - self._step_t0 >= self.T_step_settle:
                self.step_phase = StepPhase.NONE
        self._last_t = t
        out['freeze'] = self.step_phase != StepPhase.NONE
        return out

    def lean_for_pivot(self, lam_pivot: float, margin: float | None = None) -> float:
        """
        Body lean that places the composite CoG a margin ahead of the front
        wheel when the hub sits at leg angle `lam_pivot` behind it:
            m_b l sin(theta) = -M1 sin(lam_pivot) + margin
        with M1 the first moment of everything carried by the leg.
        """
        p = self.plant.p
        M1 = p.R * (p.m_hub + 1.5 * (4.0 * p.m_wheel) + p.m_body)
        if margin is None:
            margin = self.step_lean_margin
        need = -M1 * math.sin(lam_pivot) + margin
        s = need / (p.m_body * p.l_cog)
        return math.asin(max(-0.95, min(0.95, s)))

    def landing_leg_angle(self, step_height: float) -> float:
        """
        Leg angle (pivot wheel -> hub) at which the upper-front wheel of
        the cluster touches a tread `step_height` above the pivot wheel's
        ground:  R cos(lam) + R cos(lam + 60 deg) = step_height.
        Equals 60 deg on flat ground (the 4WD tie) and shrinks with height.
        """
        R = self.plant.p.R
        target = min(max(step_height, -0.99 * R), 0.99 * R) / R
        lo, hi = 0.0, 2.0 * _PI_3     # cos(l) + cos(l + 60) is monotonic here
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if math.cos(mid) + math.cos(mid + _PI_3) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def update(self, t: float, theta: float, theta_dot: float,
               phi_L: float, phi_R: float,
               predicted_peak_theta: float = 0.0,
               lam: float = 0.0, lam_dot: float = 0.0,
               lam_pivot: float = -_PI_3, v_cmd: float = 0.0,
               terrain_dist: float = float('inf'),
               terrain_height: float = 0.0) -> PlannerOutput:
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
        lam, lam_dot:         measured leg angle and rate (stair stepping).
        v_cmd:                operator velocity command (m/s).
        terrain_dist/height:  terrain probe reading (see RobotState).
        """
        if not self._initialised:
            self.initialise(phi_L, phi_R)
        elif (self.phase == FlipPhase.NORMAL
              and self.step_phase == StepPhase.NONE
              and not (self._ramp_L.active or self._ramp_R.active)):
            # The MPC may have rolled the triplets onto another wheel by
            # itself (e.g. after a hard push, or a stair step).  Once the
            # clusters rest near a different 60 deg multiple of ABSOLUTE
            # rotation (hub joint minus body pitch), re-latch references
            # and mode.  Tolerate the tilt of an oblique stance on a step.
            psi_L, psi_R = phi_L - theta, phi_R - theta
            snapped_L = round(psi_L / _PI_3) * _PI_3
            snapped_R = round(psi_R / _PI_3) * _PI_3
            if (abs(snapped_L - self._ramp_L.value) > _PI_3 / 2.0
                    or abs(snapped_R - self._ramp_R.value) > _PI_3 / 2.0) \
                    and abs(psi_L - snapped_L) < math.radians(20.0) \
                    and abs(psi_R - snapped_R) < math.radians(20.0):
                self.initialise(snapped_L, snapped_R)

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

        step = self._step_update(t, theta, theta_dot, lam, lam_dot, lam_pivot,
                                 v_cmd, terrain_dist, terrain_height)
        return PlannerOutput(
            phi_ref_L=pL, phi_ref_R=pR, phi_rate_L=vL, phi_rate_R=vR,
            transitioning=transitioning,
            flipping=flipping,
            settling=self.phase == FlipPhase.SETTLING,
            freeze_position=flipping or step.get('freeze', False),
            reset_position=reset_position,
            phase=self.phase,
            dcm=self.dcm, dcm_margin=self.dcm_margin,
            t_capture=self.t_capture,
            step_phase=self.step_phase,
            theta_ref=step.get('theta_ref'),
            lam_ref=step.get('lam_ref'),
            lam_rate=step.get('lam_rate', 0.0),
            force_single_contact=step.get('single', False),
            relax_limits=step.get('relax', False),
            keep_tipping=step.get('keep_tipping', False),
            limit_drive=step.get('limit_drive', False),
            pin_drive=step.get('pin_drive', False),
            hold_for_drop=self.hold_for_drop,
            v_cap=self.v_cap,
        )
