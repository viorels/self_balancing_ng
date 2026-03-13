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
        self.sample_period = 1.0 / config['IMU_SAMPLE_RATE_HZ']

        accel_range_mps2 = config['IMU_ACCEL_RANGE_G'] * 9.81
        self.accel_lsb = (2 * accel_range_mps2) / (2 ** config['IMU_QUANTIZATION_BITS'])
        gyro_range_rps = math.radians(config['IMU_GYRO_RANGE_DPS'])
        self.gyro_lsb = (2 * gyro_range_rps) / (2 ** config['IMU_QUANTIZATION_BITS'])

    def _quantize(self, value, lsb):
        return round(value / lsb) * lsb

    def read(self, true_pitch, true_pitch_rate, sim_time, dt):
        if not self.cfg['ADD_SENSOR_NOISE']:
            return true_pitch, true_pitch_rate

        # Gyroscope
        self.gyro_bias += np.random.normal(0, self.cfg['IMU_GYRO_DRIFT_RATE'] * dt)
        gyro_reading = true_pitch_rate + self.gyro_bias
        gyro_reading += np.random.normal(0, self.cfg['IMU_GYRO_NOISE_STD'])
        gyro_reading = self._quantize(gyro_reading, self.gyro_lsb)

        # Accelerometer
        accel_pitch = true_pitch
        accel_pitch += np.random.normal(0, self.cfg['IMU_ANGLE_NOISE_STD'])
        vibration = np.random.normal(0, self.cfg['IMU_ACCEL_VIB_NOISE_STD'])
        accel_pitch += math.atan2(vibration, 9.81)
        accel_pitch = self._quantize(accel_pitch, self.accel_lsb)

        # Complementary filter
        alpha = self.cfg['COMP_FILTER_ALPHA']
        gyro_angle = self.fused_pitch + gyro_reading * dt
        self.fused_pitch = (1.0 - alpha) * gyro_angle + alpha * accel_pitch

        return self.fused_pitch, gyro_reading
