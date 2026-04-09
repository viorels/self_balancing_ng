"""
Cascaded PID Balance Controller for Self-Balancing Robots

Outer loop:  position → target pitch angle  (low rate)
Inner loop:  pitch error → commanded torque  (high rate)
Yaw damping: yaw rate → differential torque

Inputs:  measured pitch, gyro rate, forward position, yaw rate
Outputs: per-side commanded torques (left, right)
"""

import numpy as np

from .base import BalanceControllerBase


class BalanceController(BalanceControllerBase):
    """
    Cascaded PID controller with sensor-to-actuator delay pipeline.

    Parameters (passed via config dict):
        Inner PID:  PID_KP, PID_KI, PID_KD
        Outer PID:  POS_PID_KP, POS_PID_KI, POS_PID_KD, POS_PID_MAX_PITCH
        Rates:      CONTROL_RATE_HZ, POS_PID_RATE_HZ
        Limits:     MAX_TORQUE
        Realism:    CONTROL_JITTER_STD, SENSOR_TO_ACTUATOR_DELAY_STEPS,
                    ADD_SENSOR_NOISE, YAW_DAMPING_K
    """

    def __init__(self, config):
        self.cfg = config

        # --- Inner PID state (pitch → torque) ---
        self.integral_pitch_error = 0.0

        # --- Outer PID state (position → target pitch) ---
        self.target_pitch = 0.0
        self.target_position = 0.0
        self.prev_position = 0.0
        self.integral_pos_error = 0.0

        # --- Control loop timing ---
        self.control_period = 1.0 / config.control.control_rate_hz
        self.next_control_time = 0.0

        self.pos_control_period = 1.0 / config.pid.pos_rate_hz
        self.next_pos_control_time = 0.0

        # --- Last commanded torque (for logging) ---
        self.control_torque = 0.0
        self._left_torque = 0.0
        self._right_torque = 0.0

        # --- Yaw rate setpoint (for joystick control) ---
        self.yaw_rate_setpoint = 0.0

        # --- Compatibility fields (base class properties read these) ---
        self.state_error = [0.0, 0.0, 0.0, 0.0]

    # ----------------------------------------------------------------
    # BalanceControllerBase interface
    # ----------------------------------------------------------------

    def set_velocity_command(self, velocity):
        """PID doesn't support velocity mode; ignore."""
        pass

    def set_target_position(self, position):
        """Set the desired forward position (m)."""
        self.target_position = position

    def set_yaw_rate(self, yaw_rate):
        """Set desired yaw rate (rad/s). 0 = drive straight."""
        self.yaw_rate_setpoint = yaw_rate

    def get_telemetry(self) -> dict:
        """Return PID-specific diagnostic signals."""
        return {
            "torque_cmd":    float(self.control_torque),
            "target_pitch":  float(self.target_pitch),
            "target_pos":    float(self.target_position),
        }

    def update(self, measured_pitch, measured_pitch_rate,
               position, forward_velocity, yaw_rate, sim_time, dt, ref=None):
        """
        Run one controller tick.

        Args:
            measured_pitch:      fused pitch angle (rad, positive = forward lean)
            measured_pitch_rate: gyro pitch rate (rad/s)
            position:            forward position estimate (m)
            yaw_rate:            body-frame yaw rate (rad/s)
            sim_time:            current simulation time (s)
            dt:                  physics timestep (s)

        Returns:
            (left_torque, right_torque): commanded motor torques (Nm)
                before the motor model.  Returns the delayed values from
                the sensor-to-actuator pipeline.
        """
        # === Outer PID loop: position → target pitch (lower rate) ===
        if sim_time >= self.next_pos_control_time:
            self.next_pos_control_time = sim_time + self.pos_control_period

            pos_error = self.target_position - position
            velocity = (position - self.prev_position) / self.pos_control_period
            self.prev_position = position

            pos_p = self.cfg.pid.pos_kp * pos_error
            pos_d = -self.cfg.pid.pos_kd * velocity
            self.integral_pos_error += pos_error * self.pos_control_period
            self.integral_pos_error = float(np.clip(
                self.integral_pos_error, -1.0, 1.0))
            pos_i = self.cfg.pid.pos_ki * self.integral_pos_error

            self.target_pitch = float(np.clip(
                pos_p + pos_d + pos_i,
                -self.cfg.pid.pos_max_pitch,
                 self.cfg.pid.pos_max_pitch
            ))

        # === Inner PID loop: pitch → torque (at CONTROL_RATE_HZ) ===
        jitter = (np.random.normal(0, self.cfg.control.control_jitter_std)
                  if self.cfg.imu.add_sensor_noise else 0)

        if sim_time >= self.next_control_time:
            self.next_control_time = sim_time + self.control_period + jitter

            pitch_error = self.target_pitch - measured_pitch
            p_term = self.cfg.pid.kp * pitch_error
            d_term = self.cfg.pid.kd * (0.0 - measured_pitch_rate)
            self.integral_pitch_error += pitch_error * self.control_period
            self.integral_pitch_error = float(np.clip(
                self.integral_pitch_error, -0.5, 0.5))
            i_term = self.cfg.pid.ki * self.integral_pitch_error

            commanded_torque = p_term + d_term + i_term
            commanded_torque = float(np.clip(
                commanded_torque,
                -self.cfg.motor.max_torque, self.cfg.motor.max_torque
            ))
            self.control_torque = commanded_torque

            # Yaw damping: oppose yaw rate relative to setpoint
            yaw_correction = self.cfg.control.yaw_damping_k * (yaw_rate - self.yaw_rate_setpoint)

            # === Per-side torques (left −yaw, right +yaw) ===
            # l_triplet is at -Y (robot's left from behind), r_triplet at +Y (right).
            # Positive yaw_correction → more torque on right side → turns right.
            self._left_torque = commanded_torque - yaw_correction
            self._right_torque = commanded_torque + yaw_correction

        return self._left_torque, self._right_torque
