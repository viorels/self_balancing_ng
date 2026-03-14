"""
InputManager — translates raw gamepad inputs into ControlGoals.

Owns:
  - Axis mapping & scaling (sticks → target distance, yaw rate, lean)
  - Rising-edge detection for mode toggle (LB button)
  - Target-position latching (hold last target when stick is idle)
  - World-frame marker position (for the visual target line)

Does NOT own:
  - The low-level Gamepad reader (injected)
  - Calling controller methods or PyBullet APIs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from robot_state import ControlGoals, DriveMode


@dataclass
class MarkerState:
    """World-frame marker position for the visual target line."""
    x: float = 0.0
    y: float = 0.0


class InputManager:
    """
    Reads a Gamepad each tick and produces a ControlGoals + side-effects
    (mode-toggle flag, marker position).

    Usage::

        inp = InputManager(gamepad, config)
        # each tick:
        goals, toggle, marker = inp.update(robot_position, world_pose_2d)
    """

    def __init__(self, gamepad, config):
        self.gp = gamepad
        self.cfg = config

        # Axis / button indices
        self._speed_axis = config.gamepad.speed_axis
        self._yaw_axis = config.gamepad.yaw_axis
        self._lean_axis = config.gamepad.lean_axis
        self._mode_button = config.gamepad.mode_button

        # Scaling
        self._max_distance = config.gamepad.max_distance
        self._max_yaw_rate = config.gamepad.max_yaw_rate
        self._max_lean = config.gamepad.max_lean

        # Rising-edge state for mode toggle
        self._mode_was_pressed = False

        # Latched target position (1-D, robot forward axis)
        self.target_position: float = 0.0

        # Latched world-frame marker position
        self.marker = MarkerState()

    @property
    def connected(self) -> bool:
        return self.gp.connected

    def update(
        self,
        robot_position: float,
        world_pose_2d: Tuple[float, float, float, float, float],
    ) -> Tuple[ControlGoals, bool, MarkerState]:
        """
        Poll the gamepad and return (goals, mode_toggle_requested, marker).

        Parameters
        ----------
        robot_position : float
            Current 1-D forward position (from RobotState.position).
        world_pose_2d : tuple
            (x, y, yaw, fwd_x, fwd_y) from robot.get_world_pose_2d().

        Returns
        -------
        goals : ControlGoals
            Operator commands for the controller.
        mode_toggle : bool
            True on the *rising edge* of the mode button (LB).
        marker : MarkerState
            World-frame marker position for the debug line.
        """
        self.gp.poll()

        if not self.gp.connected:
            return ControlGoals(target_position=self.target_position), False, self.marker

        # --- Map axes ---
        # Right stick Y → forward distance offset (push up = negative axis)
        forward_offset = -self.gp.axis(self._speed_axis) * self._max_distance
        # Right stick X → yaw rate
        yaw_cmd = self.gp.axis(self._yaw_axis) * self._max_yaw_rate
        # Left stick Y → lean (push up = lean forward = positive pitch bias)
        lean_cmd = self.gp.axis(self._lean_axis) * self._max_lean

        # --- Mode toggle (rising edge) ---
        mode_pressed = self.gp.button(self._mode_button)
        mode_toggle = mode_pressed and not self._mode_was_pressed
        self._mode_was_pressed = mode_pressed

        # --- Target-position latching ---
        rx, ry, _, fwd_x, fwd_y = world_pose_2d

        if abs(forward_offset) > 1e-4:
            # Stick deflected → update target relative to current position
            self.target_position = robot_position + forward_offset
            self.marker.x = rx + forward_offset * fwd_x
            self.marker.y = ry + forward_offset * fwd_y
        elif abs(yaw_cmd) > 1e-4:
            # Turning with no forward → reset target to current (no chase)
            self.target_position = robot_position
            self.marker.x = rx
            self.marker.y = ry
        # else: stick idle → keep latched target_position and marker

        goals = ControlGoals(
            target_position=self.target_position,
            yaw_rate=yaw_cmd,
            pitch_bias=lean_cmd,
            request_drive_mode=None,  # mode toggle handled separately
        )
        return goals, mode_toggle, self.marker
