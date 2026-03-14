"""
Realistic MEMS IMU Sensor Model with Complementary Filter

Simulates gyroscope bias drift, accelerometer vibration noise,
quantisation, and fuses them via a complementary filter.

Extracted from tribot_sim.py — no behavioural changes.
"""

import math
import numpy as np


class IMUSensorModel:
    """
    Simulates a MEMS IMU with complementary filter fusion.
    """

    def __init__(self, config):
        self.cfg = config
        self.fused_pitch = 0.0
        self.gyro_bias = 0.0
        self.last_sample_time = 0.0
        self.sample_period = 1.0 / config.imu.sample_rate_hz

        accel_range_mps2 = config.imu.accel_range_g * 9.81
        self.accel_lsb = (2 * accel_range_mps2) / (2 ** config.imu.quantization_bits)
        gyro_range_rps = math.radians(config.imu.gyro_range_dps)
        self.gyro_lsb = (2 * gyro_range_rps) / (2 ** config.imu.quantization_bits)

    def _quantize(self, value, lsb):
        return round(value / lsb) * lsb

    def read(self, true_pitch, true_pitch_rate, sim_time, dt):
        if not self.cfg.imu.add_sensor_noise:
            return true_pitch, true_pitch_rate

        # Gyroscope
        self.gyro_bias += np.random.normal(0, self.cfg.imu.gyro_drift_rate * dt)
        gyro_reading = true_pitch_rate + self.gyro_bias
        gyro_reading += np.random.normal(0, self.cfg.imu.gyro_noise_std)
        gyro_reading = self._quantize(gyro_reading, self.gyro_lsb)

        # Accelerometer
        accel_pitch = true_pitch
        accel_pitch += np.random.normal(0, self.cfg.imu.angle_noise_std)
        vibration = np.random.normal(0, self.cfg.imu.accel_vib_noise_std)
        accel_pitch += math.atan2(vibration, 9.81)
        accel_pitch = self._quantize(accel_pitch, self.accel_lsb)

        # Complementary filter
        alpha = self.cfg.imu.comp_filter_alpha
        gyro_angle = self.fused_pitch + gyro_reading * dt
        self.fused_pitch = (1.0 - alpha) * gyro_angle + alpha * accel_pitch

        return self.fused_pitch, gyro_reading
