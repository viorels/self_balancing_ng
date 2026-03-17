"""
Lean Transition Trajectory Planner

Computes a smooth [position, velocity, pitch, pitch_rate] reference
trajectory for transitioning between lean angles.  The LQR tracks
this trajectory instead of a step reference, preventing free-fall
overshoot and position excursions.

The trajectory is shaped so that:
  - The body pitches smoothly (no step in pitch reference)
  - The wheels lead the lean (position moves first to catch the CoG)
  - Peak position excursion is bounded (stair-safe)
  - All states arrive at the new equilibrium simultaneously

Uses a minimum-jerk (quintic) time profile:
    s(τ) = 10τ³ − 15τ⁴ + 6τ⁵,   τ = t/T
which has zero velocity and acceleration at both endpoints.
"""

import math


class LeanTrajectory:
    """
    Generates a smooth state reference [x, ẋ, θ, θ̇] for transitioning
    between two lean angles.

    Position reference accounts for the geometric CoG shift:
        Δx = h_cog · (sin(θ_end) − sin(θ_start))

    Duration scales with lean change magnitude:
        T = lean_per_sec_factor · |Δθ| / (π/12)
    clamped to [min_duration, max_duration].

    When no trajectory is active, the planner is transparent —
    update() returns the last-set steady-state reference.
    """

    def __init__(self, h_cog,
                 min_duration=0.3,
                 max_duration=2.0,
                 lean_per_sec_factor=0.5):
        """
        Args:
            h_cog:               CoG height above wheel contact (m)
            min_duration:        minimum transition time (s)
            max_duration:        maximum transition time (s)
            lean_per_sec_factor: seconds of transition per 15° of lean change
        """
        self._h_cog = h_cog
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._lean_per_sec_factor = lean_per_sec_factor

        # Trajectory state
        self._active = False
        self._t0 = 0.0
        self._duration = 0.0

        self._theta_start = 0.0
        self._theta_end = 0.0
        self._x_start = 0.0
        self._dx = 0.0  # total position change

        # Current reference output
        self.ref_position = 0.0
        self.ref_velocity = 0.0
        self.ref_pitch = 0.0
        self.ref_pitch_rate = 0.0

    @property
    def active(self):
        """True while a transition trajectory is being tracked."""
        return self._active

    def start(self, sim_time, current_position, theta_start, theta_end,
              duration=None):
        """
        Begin a new lean transition.

        Args:
            sim_time:         current simulation time (s)
            current_position: current wheel position (m)
            theta_start:      starting lean angle (rad)
            theta_end:        target lean angle (rad)
            duration:         transition time (s).  If None, auto-computed
                              from lean change magnitude.
        """
        self._theta_start = theta_start
        self._theta_end = theta_end
        self._x_start = current_position

        # Geometric position shift: wheels must move to stay under new CoG
        self._dx = self._h_cog * (
            math.sin(theta_end) - math.sin(theta_start))

        if duration is None:
            lean_change = abs(theta_end - theta_start)
            duration = self._lean_per_sec_factor * lean_change / math.radians(15.0)

        self._duration = max(self._min_duration,
                             min(self._max_duration, duration))
        self._t0 = sim_time
        self._active = True

        # Initial reference = current state (smooth start)
        self.ref_position = current_position
        self.ref_velocity = 0.0
        self.ref_pitch = theta_start
        self.ref_pitch_rate = 0.0

    def update(self, sim_time):
        """
        Compute the reference state for the current time.

        Call every control tick.  When the trajectory completes,
        the reference holds at the final state and active becomes False.

        Args:
            sim_time: current simulation time (s)

        Returns:
            (ref_position, ref_velocity, ref_pitch, ref_pitch_rate)
        """
        if not self._active:
            return (self.ref_position, self.ref_velocity,
                    self.ref_pitch, self.ref_pitch_rate)

        elapsed = sim_time - self._t0
        T = self._duration

        if elapsed >= T:
            # Trajectory complete — hold final state
            self._active = False
            self.ref_position = self._x_start + self._dx
            self.ref_velocity = 0.0
            self.ref_pitch = self._theta_end
            self.ref_pitch_rate = 0.0
            return (self.ref_position, self.ref_velocity,
                    self.ref_pitch, self.ref_pitch_rate)

        # Normalised time τ ∈ [0, 1]
        tau = elapsed / T

        # Minimum-jerk profile: s(τ) = 10τ³ − 15τ⁴ + 6τ⁵
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau
        s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5

        # Derivative: ds/dτ = 30τ² − 60τ³ + 30τ⁴
        ds_dtau = 30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4
        ds_dt = ds_dtau / T  # chain rule

        # Position reference (geometric CoG shift)
        d_theta = self._theta_end - self._theta_start
        self.ref_position = self._x_start + self._dx * s
        self.ref_velocity = self._dx * ds_dt

        # Pitch reference (smooth lean transition)
        self.ref_pitch = self._theta_start + d_theta * s
        self.ref_pitch_rate = d_theta * ds_dt

        return (self.ref_position, self.ref_velocity,
                self.ref_pitch, self.ref_pitch_rate)

    def cancel(self):
        """Abort the current trajectory.  Reference freezes at last values."""
        self._active = False
