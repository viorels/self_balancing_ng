"""
Abstract interface for all balance controllers.

Every controller implements `update()` and `get_telemetry()`.
The sim loop never inspects controller internals — it only calls these
two methods plus the setters defined here.

Design notes
------------
* `update()` returns a plain (left_torque, right_torque) tuple for
  backward compatibility with the existing motor-application code in
  TribotBalanceBot.  In a future step, this will return a ControlOutput
  dataclass instead.

* `get_telemetry()` returns a flat dict of controller-specific diagnostic
  signals.  Keys should be short, snake_case.  Values must be float/int.
  The sim loop merges these under a "ctrl/" prefix for PlotJuggler.

* `set_lean()` and `set_triplet_state()` have default no-op implementations
  so controllers that don't support them (PID) don't need stubs.

* Properties `triplet_torque_L/R`, `desired_lean`, `target_pitch` are
  exposed as properties with safe defaults so the sim loop and triplet
  controller can read them without hasattr() checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StateReference:
    """Reference state passed to the controller each tick.

    Built by the robot loop (which knows the drive mode and owns the
    trajectory planner).  The controller is a pure function of
    (measured_state, ref, gains) → torques.
    """
    position: float = 0.0
    velocity: float = 0.0
    pitch: float = 0.0
    pitch_rate: float = 0.0


class BalanceControllerBase(ABC):
    """
    Base class for all balance controllers (PID, LQR, MPC, …).

    Subclasses MUST implement:
        update()          — compute one control tick
        get_telemetry()   — return diagnostic dict

    Subclasses MAY override:
        set_lean()           — accept operator lean bias (default: no-op)
        set_triplet_state()  — accept triplet encoder readings (default: no-op)
        get_flip_diagnostics() — ZMP/DCM flip telemetry (default: empty dict)

    Properties with safe defaults (override in subclass if meaningful):
        triplet_torque_L/R   — planned triplet torques (0.0)
        desired_lean         — LQR-implied lean demand (0.0)
        plans_triplet_torque — whether sim loop should use controller's
                               triplet torques vs the external TripletController
    """

    # ------------------------------------------------------------------
    # Abstract methods — every controller MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def update(self, measured_pitch: float, measured_pitch_rate: float,
               position: float, yaw_rate: float,
               sim_time: float, dt: float,
               ref: StateReference | None = None) -> tuple[float, float]:
        """
        Compute one control step.

        Args:
            measured_pitch:      fused pitch angle (rad)
            measured_pitch_rate: gyro pitch rate (rad/s)
            position:            forward position estimate (m)
            yaw_rate:            body-frame yaw rate (rad/s)
            sim_time:            current simulation time (s)
            dt:                  physics timestep (s)
            ref:                 reference state (position, velocity, pitch,
                                 pitch_rate).  Built by the robot loop.
                                 Controllers that manage their own references
                                 may ignore this (default None).

        Returns:
            (left_torque, right_torque) — commanded motor torques (Nm)
        """
        ...

    @abstractmethod
    def get_telemetry(self) -> dict:
        """
        Return controller-specific diagnostic signals as a flat dict.

        Keys should be short, snake_case.  Values must be float or int.
        The sim loop will merge these under a "ctrl/" prefix.
        """
        ...

    # ------------------------------------------------------------------
    # Setters — shared interface, called by the sim loop
    # ------------------------------------------------------------------

    @abstractmethod
    def set_target_position(self, position: float) -> None:
        """Set the desired forward position (m)."""
        ...

    @abstractmethod
    def set_yaw_rate(self, yaw_rate: float) -> None:
        """Set desired yaw rate (rad/s). 0 = drive straight."""
        ...

    def set_lean(self, lean_rad: float) -> None:
        """Set operator-commanded pitch bias (rad).  Default: no-op."""
        pass

    def set_triplet_state(self, angle_L: float, angle_R: float,
                          rate_L: float, rate_R: float) -> None:
        """Update triplet encoder readings.  Default: no-op."""
        pass

    # ------------------------------------------------------------------
    # Properties — safe defaults, override where meaningful
    # ------------------------------------------------------------------

    @property
    def plans_triplet_torque(self) -> bool:
        """True if this controller produces its own triplet hub torques.

        When True the sim loop uses triplet_torque_L/R directly.
        When False (default) the sim loop runs the external TripletController.
        """
        return False

    @property
    def triplet_torque_L(self) -> float:
        """Planned left triplet hub torque (Nm).  Default 0."""
        return 0.0

    @property
    def triplet_torque_R(self) -> float:
        """Planned right triplet hub torque (Nm).  Default 0."""
        return 0.0

    @property
    def desired_lean(self) -> float:
        """LQR/MPC-implied lean demand (rad) for triplet cooperation.  Default 0."""
        return 0.0

    @property
    def requested_lean(self) -> float:
        """Raw operator lean command (rad) before triplet coordination.  Default 0."""
        return 0.0

    # ------------------------------------------------------------------
    # Optional telemetry extensions
    # ------------------------------------------------------------------

    def get_flip_diagnostics(self) -> dict:
        """Return ZMP/DCM flip diagnostics (MPC only).  Default: empty."""
        return {}
