"""
Canonical data structures shared across all subsystems.

These dataclasses define the contracts between components:
  - RobotState:    sensor-derived state (written by robot, read by controllers)
  - ControlOutput: actuator commands  (written by controllers, read by robot)
  - ControlGoals:  operator intent    (written by input layer, read by controllers)
  - Telemetry:     flat dict for UDP streaming (aggregated by sim loop)

DriveMode is also defined here to avoid circular imports between
tribot_sim.py and the extracted controller/model modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Drive mode (shared by TripletController, TribotBalanceBot, sim loop)
# ---------------------------------------------------------------------------

class DriveMode(Enum):
    """4WD = two wheels/side on ground; 2WD = one wheel/side (active balance)."""
    FOUR_WD = '4wd'
    TWO_WD = '2wd'


# ---------------------------------------------------------------------------
# Robot state — single source of truth for measured / estimated quantities
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    """
    Written by the sensor pipeline (TribotRobot.read_sensors);
    read by controllers.  All angles in radians, distances in metres.
    """
    # Time
    sim_time: float = 0.0
    dt: float = 0.002

    # Body state (IMU-fused)
    pitch: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0

    # Ground-truth (for telemetry — controllers must NOT use)
    true_pitch: float = 0.0
    true_pitch_rate: float = 0.0

    # Odometry (integrated forward velocity)
    position: float = 0.0
    forward_velocity: float = 0.0

    # Triplet joint encoders
    triplet_angle_L: float = 0.0
    triplet_angle_R: float = 0.0
    triplet_rate_L: float = 0.0
    triplet_rate_R: float = 0.0

    # Wheel encoders (one representative per side, belt-coupled)
    wheel_velocity_L: float = 0.0
    wheel_velocity_R: float = 0.0

    # Current drive mode
    drive_mode: DriveMode = DriveMode.FOUR_WD
    triplet_base_angle: float = 0.0


# ---------------------------------------------------------------------------
# Control output — actuator commands returned by any balance controller
# ---------------------------------------------------------------------------

@dataclass
class ControlOutput:
    """
    Produced by controllers; consumed by TribotRobot.apply_control().
    """
    # Drive motor torques (Nm, one per side)
    torque_L: float = 0.0
    torque_R: float = 0.0

    # Triplet hub torques (Nm, one per side)
    triplet_torque_L: float = 0.0
    triplet_torque_R: float = 0.0

    # Informational (for telemetry / triplet PD cooperation)
    target_pitch: float = 0.0
    desired_lean: float = 0.0


# ---------------------------------------------------------------------------
# Control goals — high-level operator commands
# ---------------------------------------------------------------------------

@dataclass
class ControlGoals:
    """
    Written by the input layer (gamepad / autonomy);
    read by controllers.
    """
    target_position: float = 0.0
    yaw_rate: float = 0.0
    lean_offset: float = 0.0                     # intentional lean (rad)
    request_drive_mode: DriveMode | None = None   # None = no change


# ---------------------------------------------------------------------------
# Telemetry aggregator
# ---------------------------------------------------------------------------

@dataclass
class Telemetry:
    """Flat dict aggregated per tick and streamed via UDP."""
    data: dict = field(default_factory=dict)

    def merge(self, prefix: str, d: dict):
        """Merge a subsystem dict with a namespace prefix."""
        for k, v in d.items():
            self.data[f"{prefix}/{k}"] = v
