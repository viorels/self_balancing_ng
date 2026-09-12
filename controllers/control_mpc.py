"""
Mode-scheduled linear MPC balance controller for the tribot.

Architecture (see docs/plans/DYNAMIC_CONTROLLER_RECOMMENDATION.md):

  planner  (TripletPlanner)  drive-mode transitions, DCM flip trigger,
                             per-side hub joint references
  MPC      (this file)       one constrained QP over the planar model with
                             states [s, s_dot, theta, theta_dot, lam, lam_dot]
                             and inputs [u_d, u_t]; scheduled between the
                             4WD (leg locked) and single-contact models
  yaw PI                     differential drive torque, decoupled
  leg PD                     antisymmetric hub torque tracking the planner's
                             per-side references (mode transitions), or
                             both sides during a flip
  reflex                     torque cut-off above the tilt limit

Actuator allocation is not hand-tuned: the QP decides how much of the
pitch correction comes from the wheels (1 Nm each) and how much from the
hub motors (5 Nm each) through the cost weights and the torque limits,
including the 4WD tipping constraint.

Sign conventions inside the controller follow controllers/mpc_plant.py
(forward-lean positive); the simulator's pitch is converted on entry with
PITCH_SIGN.  Output torques use the simulator's conventions: positive
drive torque = forward, positive hub torque = hub joint angle increases.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg as la

from robot_state import DriveMode

from .base import BalanceControllerBase, StateReference
from .mpc_plant import (
    NX, NU, PITCH_SIGN, ContactMode, PlanarPlant, PlantParams,
    discretize, grounded_wheel_offset, wrap120,
)
from .mpc_qp import MPCQP
from .triplet_planner import TripletPlanner, FlipPhase, StepPhase

_PI_3 = math.pi / 3.0


class MPCBalanceController(BalanceControllerBase):
    """Constrained MIMO MPC over the drive and triplet torques."""

    def __init__(self, config):
        self.cfg = config
        m = config.mpc

        # --- Plant and QP ---
        self.params = PlantParams.from_config(config)
        self.plant = PlanarPlant(self.params)
        self.N = int(m.horizon)
        self.dt = float(m.dt_pred)
        self.Q_single = np.diag(m.q_diag)
        self.Q_4wd = np.diag(m.q_diag).copy()
        self.Q_4wd[4:, :] = 0.0
        self.Q_4wd[:, 4:] = 0.0
        self.R = np.diag(m.r_diag)
        self.Rd = np.diag(m.rd_diag)
        self.qp = MPCQP(self.N, self.Q_single, self.R, self.Rd,
                        theta_max=m.theta_soft_limit,
                        rho1=m.slack_linear, rho2=m.slack_quadratic,
                        max_iter=m.max_iter, eps=m.solver_eps)

        # --- Event layer ---
        self.planner = TripletPlanner(config, self.plant)

        # --- Limits ---
        self.tau_drive = config.motor.max_torque
        self.tau_trip = m.triplet_torque_max
        self.u_d_max = 2.0 * self.tau_drive * (1.0 - m.yaw_reserve)
        self.u_t_max = 2.0 * self.tau_trip * (1.0 - m.leg_reserve)
        self.tip_safety = m.tipping_safety
        self.kappa_tip = self.params.z_w4 / self.params.r
        self.theta_cutoff = m.theta_cutoff
        self.lin_clip = m.linearisation_clip
        self.four_wd_tol = math.radians(m.four_wd_tolerance_deg)

        # --- Yaw PI ---
        self.yaw_kp = config.control.yaw_damping_k
        self.yaw_ki = m.yaw_ki
        self._yaw_int = 0.0
        self.yaw_rate_setpoint = 0.0

        # --- Leg (antisymmetric) PD and flip PD ---
        self.leg_kp = m.leg_kp
        self.leg_kd = m.leg_kd
        self.flip_kp = m.flip_kp
        self.flip_kd = m.flip_kd
        self.I_triplet_side = (config.plant.hub_inertia
                               + 3.0 * config.plant.wheel_mass_each
                               * config.robot.triplet_radius ** 2)

        # --- Timing ---
        self.control_period = 1.0 / config.control.control_rate_hz
        self.next_control_time = 0.0

        # --- Operator references ---
        self._velocity_command = 0.0
        self.target_position = 0.0
        self._was_driving = False
        self._requested_lean = 0.0
        self.target_lean = 0.0
        self.target_pitch = 0.0

        # --- Measurements ---
        self._phi_L = self._phi_R = 0.0
        self._phid_L = self._phid_R = 0.0
        self._sim_time = 0.0
        self._terrain = (float('inf'), 0.0, 0.0)

        # --- Stair step manoeuvre limits ---
        self.step_pitch_limit = m.step_pitch_limit
        self.step_drive_limit = m.step_drive_limit
        self.step_hub_limit = m.step_hub_limit
        self.step_press_torque = m.step_press_torque
        self.step_lean_press = m.step_lean_press
        self.step_lean_damping = m.step_lean_damping
        self.step_lin_clip = m.step_lin_clip
        self.theta_soft_limit = m.theta_soft_limit

        # --- Model cache ---
        self._mode = ContactMode.FOUR_WD
        self._lin_point = (None, None)
        self._Ad = self._Bd = self._cd = None
        self._P_terminal = None

        # --- Outputs / telemetry ---
        self._left_torque = self._right_torque = 0.0
        self._trip_L = self._trip_R = 0.0
        self.u = np.zeros(NU)
        self.z0 = np.zeros(NX)
        self.z_ref0 = np.zeros(NX)
        self.u_ref = np.zeros(NU)
        self.u_traj = None
        self.z_traj = None
        self.solve_ok = False
        self.fallen = False
        self.max_solve_ms = 0.0
        self._plan_index = 0
        self._last_out = None
        self._holding = False

        print("  MPC balance controller initialised")
        print(f"    horizon N={self.N}, dt_pred={self.dt * 1e3:.0f} ms "
              f"({self.N * self.dt:.2f} s), rate={config.control.control_rate_hz} Hz")
        print(f"    Q={m.q_diag}  R={m.r_diag}  Rd={m.rd_diag}")
        print(f"    |u_d|<={self.u_d_max:.2f} Nm  |u_t|<={self.u_t_max:.2f} Nm  "
              f"tipping<={self.tip_safety * self.params.tipping_torque:.2f} Nm")
        print(f"    divergence: 4WD {self.plant.unstable_rate(ContactMode.FOUR_WD):.1f} "
              f"rad/s, 2WD leg {self.plant.unstable_rate(ContactMode.SINGLE_CONTACT):.1f} "
              f"rad/s, DCM omega0 {self.planner.omega0:.2f} rad/s")

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def reset(self):
        self._velocity_command = 0.0
        self.target_position = 0.0
        self._was_driving = False
        self._requested_lean = 0.0
        self.target_lean = self.target_pitch = 0.0
        self.yaw_rate_setpoint = 0.0
        self._yaw_int = 0.0
        self.next_control_time = 0.0
        self._left_torque = self._right_torque = 0.0
        self._trip_L = self._trip_R = 0.0
        self.u = np.zeros(NU)
        self.fallen = False
        self.u_traj = self.z_traj = None
        self._plan_index = 0
        self._lin_point = (None, None)
        self._holding = False
        self._terrain = (float('inf'), 0.0, 0.0)
        self.planner.reset()

    def set_velocity_command(self, velocity):
        self._velocity_command = velocity

    def set_target_position(self, position):
        self.target_position = position

    def set_yaw_rate(self, yaw_rate):
        self.yaw_rate_setpoint = yaw_rate

    def set_lean(self, lean_rad):
        self._requested_lean = lean_rad

    @property
    def requested_lean(self) -> float:
        return self._requested_lean

    def set_triplet_state(self, angle_L, angle_R, rate_L, rate_R):
        self._phi_L, self._phi_R = angle_L, angle_R
        self._phid_L, self._phid_R = rate_L, rate_R

    def set_drive_mode(self, mode: DriveMode):
        self.planner.request_mode(mode, self._sim_time)

    def set_terrain_ahead(self, distance, height, clearance):
        self._terrain = (distance, height, clearance)

    @property
    def plans_triplet_torque(self) -> bool:
        return True

    @property
    def triplet_torque_L(self) -> float:
        return self._trip_L

    @property
    def triplet_torque_R(self) -> float:
        return self._trip_R

    def get_telemetry(self) -> dict:
        out = self._last_out
        d = {
            "u_drive":        float(self.u[0]),
            "u_triplet":      float(self.u[1]),
            "u_drive_ref":    float(self.u_ref[0]),
            "u_triplet_ref":  float(self.u_ref[1]),
            "trip_cmd_L":     float(self._trip_L),
            "trip_cmd_R":     float(self._trip_R),
            "z_s":            float(self.z0[0]),
            "z_sdot":         float(self.z0[1]),
            "z_theta":        float(self.z0[2]),
            "z_thetadot":     float(self.z0[3]),
            "z_lam":          float(self.z0[4]),
            "z_lamdot":       float(self.z0[5]),
            "ref_s":          float(self.z_ref0[0]),
            "ref_theta":      float(self.z_ref0[2]),
            "ref_lam":        float(self.z_ref0[4]),
            "mode_4wd":       float(self._mode == ContactMode.FOUR_WD),
            "solve_ms":       float(self.qp.last_solve_ms),
            "solve_max_ms":   float(self.max_solve_ms),
            "solve_iter":     float(self.qp.last_iter),
            "solve_ok":       float(self.solve_ok),
            "solve_fail_cnt": float(self.qp.fail_count),
            "target_pos":     float(self.target_position),
            "velocity_cmd":   float(self._velocity_command),
            "target_pitch":   float(self.target_pitch),
            "requested_lean": float(self._requested_lean),
            "yaw_int":        float(self._yaw_int),
            "fallen":         float(self.fallen),
        }
        if out is not None:
            d.update({
                "phi_ref_L":   float(out.phi_ref_L),
                "phi_ref_R":   float(out.phi_ref_R),
                "flip_phase":  float(out.phase),
                "dcm":         float(out.dcm),
                "dcm_margin":  float(out.dcm_margin),
                "t_capture":   float(min(out.t_capture, 9.9)),
                "transition":  float(out.transitioning),
                "step_phase":  float(out.step_phase),
                "hold_drop":   float(out.hold_for_drop),
                "planner_2wd": float(self.planner.mode == DriveMode.TWO_WD),
            })
        return d

    def get_flip_diagnostics(self) -> dict:
        out = self._last_out
        if out is None:
            return {}
        return {"flip_phase": out.phase, "dcm": out.dcm,
                "dcm_margin": out.dcm_margin, "t_capture": out.t_capture}

    # ------------------------------------------------------------------
    # Model scheduling
    # ------------------------------------------------------------------

    def _select_mode(self, a_L, a_R, out):
        """Pick the contact model for this tick."""
        if out.force_single_contact:
            return ContactMode.SINGLE_CONTACT
        both_tied = (abs(a_L) > _PI_3 - self.four_wd_tol
                     and abs(a_R) > _PI_3 - self.four_wd_tol)
        steady_4wd = (self.planner.mode == DriveMode.FOUR_WD
                      and not out.transitioning)
        if out.flipping or steady_4wd or both_tied:
            return ContactMode.FOUR_WD
        return ContactMode.SINGLE_CONTACT

    def _update_model(self, mode, lam0, theta0, clip=None):
        """Re-linearise and refresh the terminal cost when needed."""
        clip = self.lin_clip if clip is None else clip
        lam0 = float(np.clip(lam0, -clip, clip))
        theta0 = float(np.clip(theta0, -clip, clip))
        if mode == ContactMode.FOUR_WD:
            lam0 = 0.0
        prev_mode, prev_pt = self._lin_point
        changed = (prev_mode != mode or prev_pt is None
                   or abs(prev_pt[0] - lam0) > math.radians(3.0)
                   or abs(prev_pt[1] - theta0) > math.radians(3.0))
        if not changed:
            return
        A, B, c = self.plant.linearize(mode, lam0, theta0)
        Ad, Bd, cd = discretize(A, B, c, self.dt)
        Q = self.Q_4wd if mode == ContactMode.FOUR_WD else self.Q_single
        P = self._terminal_cost(mode, Ad, Bd, Q)
        self.qp.set_weights(Q, P)
        self._Ad, self._Bd, self._cd = Ad, Bd, cd
        self._P_terminal = P
        self._mode = mode
        self._lin_point = (mode, (lam0, theta0))

    def _terminal_cost(self, mode, Ad, Bd, Q):
        """Infinite-horizon LQR cost-to-go on the controllable states."""
        idx = list(range(NX)) if mode == ContactMode.SINGLE_CONTACT else [0, 1, 2, 3]
        A_r = Ad[np.ix_(idx, idx)]
        B_r = Bd[idx, :]
        Q_r = Q[np.ix_(idx, idx)]
        P = np.zeros((NX, NX))
        try:
            P_r = la.solve_discrete_are(A_r, B_r, Q_r, self.R)
            P[np.ix_(idx, idx)] = P_r
        except (la.LinAlgError, ValueError):
            P[np.ix_(idx, idx)] = Q_r * 10.0
        return P

    def _equilibrium_input(self, z_ref):
        """Least-squares input that holds z_ref stationary in the model."""
        acc_rows = [1, 3, 5] if self._mode == ContactMode.SINGLE_CONTACT else [1, 3]
        Ad, Bd, cd = self._Ad, self._Bd, self._cd
        # Ad z + Bd u + cd = z  on the velocity rows (steady state)
        rhs = (z_ref - Ad @ z_ref - cd)[acc_rows]
        M = Bd[acc_rows, :]
        u_eq, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        return np.clip(u_eq, [-self.u_d_max, -self.u_t_max],
                       [self.u_d_max, self.u_t_max])

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def update(self, measured_pitch, measured_pitch_rate,
               position, forward_velocity, yaw_rate, sim_time, dt,
               ref=None):
        self._sim_time = sim_time
        if ref is None:
            ref = StateReference(pitch=self._requested_lean)
        self.target_lean = ref.pitch
        self.target_pitch = ref.pitch

        theta = PITCH_SIGN * measured_pitch
        theta_dot = PITCH_SIGN * measured_pitch_rate

        # --- Safety reflex: cut torque when clearly fallen ---
        if abs(theta) > self.theta_cutoff:
            self.fallen = True
        if self.fallen:
            self._left_torque = self._right_torque = 0.0
            self._trip_L = self._trip_R = 0.0
            return 0.0, 0.0

        if sim_time < self.next_control_time:
            return self._left_torque, self._right_torque
        self.next_control_time = sim_time + self.control_period
        T = self.control_period

        # --- Leg geometry from the hub encoders ---
        a_L = grounded_wheel_offset(theta, self._phi_L)
        a_R = grounded_wheel_offset(theta, self._phi_R)
        # Plain average: the contact midpoint, which is the physical leg
        # for the MPC during an asymmetric transition, where the two sides
        # are legitimately up to 60 deg apart and start 120 deg apart at
        # the tie (a circular mean reads the first half of a transition as
        # a tie).  Outside a transition, sides more than 90 deg apart are
        # straddling the +-60 deg wrap boundary a few degrees apart (the
        # hubs drift apart mid-roll on a step), and the circular mean is
        # the physical one: a_L = -52, a_R = +58 is a leg at 57, not 3.
        in_transition = self._last_out is not None and self._last_out.transitioning
        if not in_transition and abs(a_L - a_R) > 1.5 * _PI_3:
            lam_raw = -(a_L + 0.5 * wrap120(a_R - a_L))
        else:
            lam_raw = -0.5 * (a_L + a_R)
        # Leg angle to the FRONT lower wheel, the pivot of a stair step,
        # in (-120, 0].  Its wrap boundary is at 0 (pivot wheel straight
        # under the hub), which the cluster crosses mid-roll with the two
        # encoders a hair apart, so the sides are combined with a circular
        # mean: one side at 0.1 deg and the other at 119.9 deg is 0, not
        # 60 (that jump put the planner's leg tracker on the wrong branch).
        af_L = a_L if a_L >= 0.0 else a_L + 2.0 * _PI_3
        af_R = a_R if a_R >= 0.0 else a_R + 2.0 * _PI_3
        lam_pivot = -(af_L + 0.5 * wrap120(af_R - af_L))
        lam_dot_raw = theta_dot - 0.5 * (self._phid_L + self._phid_R)

        # --- Event layer ---
        peak_theta = 0.0
        if self.z_traj is not None:
            k = int(np.argmax(np.abs(self.z_traj[:, 2])))
            peak_theta = float(self.z_traj[k, 2])
        t_dist, t_height, _ = self._terrain
        out = self.planner.update(sim_time, theta, theta_dot,
                                  self._phi_L, self._phi_R,
                                  predicted_peak_theta=peak_theta,
                                  lam=lam_raw, lam_dot=lam_dot_raw,
                                  lam_pivot=lam_pivot,
                                  v_cmd=self._velocity_command,
                                  terrain_dist=t_dist, terrain_height=t_height)
        self._last_out = out
        mode = self._select_mode(a_L, a_R, out)

        # --- State estimate in planar coordinates ---
        if mode == ContactMode.SINGLE_CONTACT:
            lam, lam_dot = lam_raw, lam_dot_raw
            if out.lam_ref is not None:
                # Unwrap the 120-deg periodic leg angle onto the branch of
                # the reference so a full roll over the pivot is continuous.
                lam = out.lam_ref + wrap120(lam_raw - out.lam_ref)
        else:
            lam, lam_dot = 0.0, 0.0
        z0 = np.array([position, forward_velocity, theta, theta_dot, lam, lam_dot])
        self.z0 = z0

        # --- Position / velocity reference (controller owned) ---
        driving = abs(self._velocity_command) > 1e-4
        if out.reset_position:
            self.target_position = position
        if out.freeze_position or out.hold_for_drop:
            # Hold the position where the hold began (latched once).
            v_cmd = 0.0
            if not self._holding:
                self.target_position = position
                self._holding = True
        else:
            if self._holding:
                # Release: restart the reference from wherever we are, the
                # odometry may have drifted while a wheel was blocked.
                self.target_position = position
            self._holding = False
            v_cmd = self._velocity_command
            if v_cmd > 0.0:
                v_cmd = min(v_cmd, out.v_cap)
            if driving:
                self.target_position += v_cmd * T
                self._was_driving = True
            elif self._was_driving:
                self.target_position = position
                self._was_driving = False
        if out.step_phase == StepPhase.ROLL:
            # The pivot wheel is blocked by the riser and spins freely, so
            # the wheel odometry is meaningless: pin the position states.
            z0[0] = self.target_position
            z0[1] = 0.0

        theta_ref = PITCH_SIGN * ref.pitch
        theta_rate_ref = PITCH_SIGN * ref.pitch_rate
        if out.theta_ref is not None:
            theta_ref, theta_rate_ref = out.theta_ref, 0.0
        lam_rate_ref = 0.0
        if out.lam_ref is not None:
            lam_ref, lam_rate_ref = out.lam_ref, out.lam_rate
        elif mode == ContactMode.SINGLE_CONTACT:
            lam_ref = self.plant.equilibrium_leg_angle(theta_ref)
        else:
            lam_ref = 0.0

        # --- Model for this tick ---
        clip = self.step_lin_clip if out.force_single_contact else None
        self._update_model(mode, lam, theta, clip=clip)
        self.qp.theta_max = (self.step_pitch_limit if out.relax_limits
                             else self.theta_soft_limit)

        z_ref = np.zeros((self.N + 1, NX))
        for k in range(self.N + 1):
            z_ref[k] = [self.target_position + v_cmd * k * self.dt, v_cmd,
                        theta_ref, theta_rate_ref,
                        lam_ref + lam_rate_ref * k * self.dt, lam_rate_ref]
        self.z_ref0 = z_ref[0]
        u_ref = self._equilibrium_input(z_ref[0])
        if out.flipping:
            u_ref[1] = 0.0
        self.u_ref = u_ref

        # --- Input bounds ---
        u_min = np.array([-self.u_d_max, -self.u_t_max])
        u_max = np.array([self.u_d_max, self.u_t_max])
        if out.flipping:
            u_min[1] = u_max[1] = 0.0      # hub torque owned by the flip PD
        if out.limit_drive:
            # Forward only, with a bias that keeps the pivot wheel pressed
            # against the riser; never back away from it to help the pitch.
            u_min[0], u_max[0] = 0.0, self.step_drive_limit
            u_ref[0] = self.step_press_torque
            if out.pin_drive:
                # Leaning to climb: the drive only damps backward drift of
                # the base, so the pivot wheel stays at the riser without
                # ever being driven into it (a constant press shoves the
                # base into the riser at the end of the lean and the jolt
                # throws the body past its lean; a free wheel lets the
                # base roll back 10 cm and the roll starts short).
                pin = min(self.step_lean_press,
                          max(0.0, -self.step_lean_damping * forward_velocity))
                u_min[0] = u_max[0] = pin
        if out.force_single_contact:
            # Rolling over the pivot: moderate hub torque so a lagging leg
            # reference cannot yank the body.
            u_min[1], u_max[1] = -self.step_hub_limit, self.step_hub_limit
        if mode == ContactMode.FOUR_WD and not out.flipping and (
                not out.relax_limits or out.keep_tipping):
            tip_max = self.tip_safety * self.params.tipping_torque
            kappa = self.kappa_tip
        else:
            tip_max = np.inf
            kappa = 0.0

        # --- Solve ---
        u_traj, z_traj, ok = self.qp.solve(z0, self._Ad, self._Bd, self._cd,
                                           z_ref, u_ref, u_min, u_max,
                                           kappa=kappa, tip_max=tip_max)
        self.solve_ok = ok
        self.max_solve_ms = max(self.max_solve_ms, self.qp.last_solve_ms)
        if u_traj is None:
            u = np.zeros(NU)
        elif ok:
            u = u_traj[0].copy()
            self.u_traj, self.z_traj = u_traj, z_traj
            self._plan_index = 0
        else:
            # Solver failed: fall back to the previous plan, shifted.
            self._plan_index = min(self._plan_index + 1, self.N - 1)
            u = self.u_traj[self._plan_index].copy() if self.u_traj is not None \
                else np.zeros(NU)
        u[0] = float(np.clip(u[0], -self.u_d_max, self.u_d_max))
        u[1] = float(np.clip(u[1], -self.u_t_max, self.u_t_max))
        self.u = u

        # --- Yaw PI (differential drive torque) ---
        yaw_err = yaw_rate - self.yaw_rate_setpoint
        self._yaw_int += yaw_err * T
        int_lim = self.tau_drive * 0.5 / max(self.yaw_ki, 1e-9)
        self._yaw_int = float(np.clip(self._yaw_int, -int_lim, int_lim))
        yaw_corr = self.yaw_kp * yaw_err + self.yaw_ki * self._yaw_int
        left = 0.5 * u[0] - yaw_corr
        right = 0.5 * u[0] + yaw_corr
        self._left_torque = float(np.clip(left, -self.tau_drive, self.tau_drive))
        self._right_torque = float(np.clip(right, -self.tau_drive, self.tau_drive))

        # --- Hub torques: symmetric from MPC, antisymmetric from the leg PD ---
        if out.flipping:
            ff = self.I_triplet_side * 0.0
            tL = (self.flip_kp * (out.phi_ref_L - self._phi_L)
                  + self.flip_kd * (out.phi_rate_L - self._phid_L) + ff)
            tR = (self.flip_kp * (out.phi_ref_R - self._phi_R)
                  + self.flip_kd * (out.phi_rate_R - self._phid_R) + ff)
        else:
            phi_diff = 0.5 * (self._phi_L - self._phi_R)
            phid_diff = 0.5 * (self._phid_L - self._phid_R)
            ref_diff = 0.5 * (out.phi_ref_L - out.phi_ref_R)
            rate_diff = 0.5 * (out.phi_rate_L - out.phi_rate_R)
            tau_a = (self.leg_kp * (ref_diff - phi_diff)
                     + self.leg_kd * (rate_diff - phid_diff))
            if mode == ContactMode.SINGLE_CONTACT:
                # Each leg carries half the robot; hold it against gravity.
                p = self.params
                lam_L, lam_R = -a_L, -a_R
                tau_a += 0.25 * p.m_total * p.g * p.R * (
                    math.sin(lam_L) - math.sin(lam_R))
            tL = 0.5 * u[1] + tau_a
            tR = 0.5 * u[1] - tau_a
        self._trip_L = float(np.clip(tL, -self.tau_trip, self.tau_trip))
        self._trip_R = float(np.clip(tR, -self.tau_trip, self.tau_trip))

        return self._left_torque, self._right_torque
