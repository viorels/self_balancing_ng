"""
Triplet Lean Controller and Geometry Helpers

PD position controller for a triplet hub joint with gravity compensation
and nonlinear balance assist.  Also contains the sine-theorem geometry
functions for computing triplet ↔ body-lean angles.

Extracted from tribot_sim.py — no behavioural changes.
"""

import math

from robot_state import DriveMode


# ============================================================================
# TRIPLET GEOMETRY — sine-theorem lean compensation
# ============================================================================

def compute_triplet_from_pitch(alpha, h, l=0.12, base_angle=None):
    """
    Compute triplet joint angle for a given body lean α using the
    sine theorem on the CoG–hub–contact triangle.

    Triangle vertices:
        A = CoG (directly above contact point when balanced)
        B = triplet hub
        C = ground contact (wheel), directly below A

    Angles:
        A = α          (body lean from vertical)
        B = π - β      (supplement of triplet-to-body angle)
        C = β - α      (wheel-to-vertical, foot angle)

    Sine rule on side BC = l, opposite angle A = α:
        l / sin α = h / sin(β - α)
        ⟹  β = α + arcsin((h/l) · sin α)

    Valid for |α| < arcsin(l/h).  Clamped to ±π/2 for safety.

    Args:
        alpha:      body lean angle (rad, positive = forward)
        h:          hub-to-CoG distance (m)
        l:          hub-to-wheel contact distance (m), default 0.12
        base_angle: joint reading at equilibrium (e.g. ±π/3 in 2WD).
                    When provided, returns the absolute joint angle
                    (base_angle − β).  When None (legacy), returns
                    the raw geometric offset β.

    Returns:
        If base_angle is None: geometric offset β (rad)
        If base_angle given:   absolute joint angle (rad)
    """
    ratio = h / l
    sin_arg = ratio * math.sin(alpha)
    sin_arg = max(-1.0, min(1.0, sin_arg))  # clamp for safety
    beta = alpha + math.asin(sin_arg)
    if base_angle is not None:
        return base_angle - beta
    return beta


def compute_pitch_from_triplet(beta, h, l=0.12, base_angle=None):
    """
    Inverse of compute_triplet_from_pitch: compute body lean α from
    a triplet joint angle.

    Args:
        beta:       triplet angle (rad).  If base_angle is None this is
                    the raw geometric offset (0 = balanced).  If base_angle
                    is given this is the *absolute* joint reading.
        h:          hub-to-CoG distance (m)
        l:          hub-to-wheel contact distance (m), default 0.12
        base_angle: joint reading at equilibrium (e.g. ±π/3 in 2WD).
                    When provided, the geometric offset is computed as
                    (base_angle − beta) so the caller can pass the raw
                    encoder value directly.

    Returns:
        alpha: body lean angle supported by this triplet position (rad)
    """
    if base_angle is not None:
        beta = base_angle - beta
    alpha = math.atan2(l * math.sin(beta), h + l * math.cos(beta))
    return alpha


# ============================================================================
# TRIPLET LEAN CONTROLLER
# ============================================================================

class TripletController:
    """
    PD position controller for a triplet hub joint with gravity compensation
    and nonlinear balance assist.

    Maintains the triplet at `target_angle` (rad, relative to body) using
    PD control plus a sin()-based gravity feedforward.  At small pitch angles
    the feedforward is near-zero and the PD alone holds the hub; at high pitch
    the feedforward cancels the dominant gravity disturbance.

    Gravity compensation
    --------------------
    When the body pitches by θ, asymmetric ground-contact normal forces
    create a net torque on the hub that pulls the triplet *with* the lean:
        τ_grav ≈ +K · sin(θ + φ - φ₀)
    where K depends on the drive mode (4WD ≈ 2 Nm, 2WD ≈ 0.4 Nm), φ is
    the triplet angle relative to body, and φ₀ is the equilibrium base angle.
    The feedforward adds -K·sin(θ + φ - φ₀) to cancel this disturbance.

    Nonlinear assist (dead-zoned quadratic)
    ------------------------------------------------------
    Beyond ASSIST_DEADZONE, when the robot is *falling* (pitch and pitch_rate
    same sign), a quadratic torque boost activates.  This gives the triplet real
    authority exactly when the drive motors are saturated, without fighting LQR at working angles.

    Output torque is `triplet_cmd` in:
        triplet_total = -motor_torque + triplet_cmd
    and is therefore additive with the drive-motor reaction cancellation.
    """

    def __init__(self, config):
        self.kp = config.triplet.lean_kp
        self.kd = config.triplet.lean_kd
        self.target_angle = config.sim.initial_triplet_angle
        self.base_angle = config.sim.initial_triplet_angle  # equilibrium angle (set by sim loop)

        # Lean compensation geometry
        self.cog_dist_2wd = config.triplet.cog_dist_2wd
        self.lean_scale_4wd = config.triplet.lean_scale_4wd

        # Gravity compensation gains (mode-dependent)
        self.grav_comp_4wd = config.triplet.grav_comp_4wd
        self.grav_comp_2wd = config.triplet.grav_comp_2wd

        # Nonlinear balance assist parameters
        self.assist_gain = config.triplet.assist_gain        # Nm/rad²
        self.assist_deadzone = config.triplet.assist_deadzone  # rad (~8.6°)
        self.assist_max = config.triplet.assist_max           # Nm clamp
        self.assist_tau = config.triplet.assist_tau           # s EMA smoothing
        self.last_assist_force = 0.0  # filtered output (for telemetry and actuation)
        self.drive_mode = DriveMode.FOUR_WD  # assist disabled in 2WD (fights the triplet hold)

        # Telemetry (populated each update, read by sim loop)
        self.last_grav_comp = 0.0

        print(f"  TripletController: Kp={self.kp}, Kd={self.kd}, "
              f"target={math.degrees(self.target_angle):.1f}\u00b0")
        print(f"    Gravity comp: 4WD={self.grav_comp_4wd:.2f} Nm, "
              f"2WD={self.grav_comp_2wd:.2f} Nm")
        print(f"    Nonlinear assist: gain={self.assist_gain}, "
              f"deadzone={math.degrees(self.assist_deadzone):.1f}°, "
              f"max={self.assist_max} Nm, "
              f"tau={self.assist_tau*1000:.0f}ms")

    # ------------------------------------------------------------------

    def update(self, angle, rate, body_pitch, body_pitch_rate=0.0, dt=0.002):
        """
        Compute triplet hub torque with gravity compensation and nonlinear
        balance assist.

        Torque = PD + gravity_feedforward + nonlinear_assist

        The gravity feedforward cancels the dominant disturbance (asymmetric
        ground-contact forces when the body pitches).  The nonlinear assist
        provides emergency authority at extreme pitch.

        Args:
            angle:            triplet joint angle (rad, relative to body)
            rate:             triplet joint angular velocity (rad/s)
            body_pitch:       current body pitch in world frame (rad)
            body_pitch_rate:  current body pitch rate (rad/s)
            dt:               timestep (s) for EMA filter

        Returns:
            torque (Nm) to apply at the triplet hub joint
        """
        error = self.target_angle - angle

        # --- PD term — drives joint toward target_angle, damps velocity ---
        tau_pd = self.kp * error - self.kd * rate

        # --- Gravity compensation feedforward ---
        # The gravity disturbance *pulls* the triplet with the lean:
        #   τ_disturb ≈ +K · sin(body_pitch + angle - base_angle)
        # The feedforward cancels it with the opposite sign.
        # At equilibrium (body_pitch=0, angle=base_angle) the term is zero;
        # no spurious offset in 2WD where base_angle=±60°.
        grav_gain = (self.grav_comp_2wd if self.drive_mode == DriveMode.TWO_WD
                     else self.grav_comp_4wd)
        tau_grav = -grav_gain * math.sin(body_pitch + angle - self.base_angle)
        self.last_grav_comp = tau_grav

        # print(f"  TripletController: target_angle={math.degrees(self.target_angle):.1f}\u00b0, "
        #       f"angle={math.degrees(angle):.1f}\u00b0, body_pitch={math.degrees(body_pitch):.1f}\u00b0, error={math.degrees(error):.1f}\u00b0, "
        #       f"tau_pd={tau_pd:.2f} Nm, tau_grav={tau_grav:.2f} Nm")

        base_force = tau_pd

        # --- Nonlinear assist: dead-zoned quadratic, rate-gated ---
        # Only active in 4WD.  In 2WD the triplet is holding a specific
        # angle for balance; the assist torque would fight that hold.
        if self.drive_mode == DriveMode.TWO_WD:
            self.last_assist_force = 0.0
            return base_force + tau_grav

        # At small pitch the PD alone holds the hub.
        # Beyond the deadzone, when the robot is *falling*, the quadratic
        # boost activates to provide emergency authority.
        #
        # Only fires when:
        #   1. |pitch| exceeds the deadzone (LQR handles small angles fine)
        #   2. The robot is *falling* (pitch and pitch_rate same sign)
        #   3. pitch_rate is above noise floor (0.1 rad/s)
        assist_force_raw = 0.0
        excess = abs(body_pitch) - self.assist_deadzone
        if excess > 0 and abs(body_pitch_rate) > 0.1:
            # Only assist when falling (pitch and pitch_rate same sign)
            if body_pitch * body_pitch_rate > 0:
                # Quadratic nonlinear term
                assist_force_raw = self.assist_gain * excess * excess * math.copysign(1.0, body_pitch)
                assist_force_raw = max(-self.assist_max, min(self.assist_max, assist_force_raw))

        # Asymmetric filter: instant attack, smooth decay.
        # When |raw| >= |filtered|, snap immediately (no delay on initial reaction).
        # When |raw| < |filtered| (gate closed or torque dropping), EMA-decay
        # to prevent on/off chattering at the physics rate.
        if abs(assist_force_raw) >= abs(self.last_assist_force):
            self.last_assist_force = assist_force_raw
        else:
            alpha = min(1.0, dt / self.assist_tau) if self.assist_tau > 0 else 1.0
            self.last_assist_force += alpha * (assist_force_raw - self.last_assist_force)

        return base_force + self.last_assist_force

    def compute_lean_and_update(self, target_pitch, desired_lean, base_angle,
                                 triplet_angle, triplet_rate, body_pitch,
                                 body_pitch_rate=0.0, dt=0.002):
        """
        Compute lean-compensation target and run the PD update in one call.

        Encapsulates the mode-dependent geometry that converts a body-pitch
        target into a triplet-hub target angle, then drives the hub toward it.

        In 2WD the full sine-theorem mapping is used; in 4WD a linear scale
        plus the raw desired-lean offset is applied.

        Args:
            target_pitch:    controller's target body pitch (rad)
            desired_lean:    raw lean offset from remote input (rad)
            base_angle:      current triplet equilibrium / base angle (rad)
            triplet_angle:   measured triplet joint angle (rad, relative to body)
            triplet_rate:    measured triplet joint angular velocity (rad/s)
            body_pitch:      measured body pitch in world frame (rad)
            body_pitch_rate: measured body pitch rate (rad/s)
            dt:              timestep (s)

        Returns:
            torque (Nm) to apply at the triplet hub joint
        """
        if self.drive_mode == DriveMode.TWO_WD:
            joint_target = compute_triplet_from_pitch(
                target_pitch, h=self.cog_dist_2wd, base_angle=base_angle)
        else:
            lean_comp = -target_pitch * self.lean_scale_4wd - desired_lean
            joint_target = base_angle + lean_comp

        self.set_base_angle(base_angle)
        self.set_target(joint_target)

        return self.update(triplet_angle, triplet_rate, body_pitch,
                           body_pitch_rate=body_pitch_rate, dt=dt)

    def set_target(self, angle_rad):
        """Override the target triplet joint angle (rad)."""
        self.target_angle = angle_rad

    def set_base_angle(self, angle_rad):
        """Set the equilibrium angle for gravity compensation (rad)."""
        self.base_angle = angle_rad
