"""
Realistic Brushless DC Motor Model

Simulates electrical dynamics, back-EMF, cogging, deadband, and noise
for a single motor.  Two instances are used (one per side).

Extracted from tribot_sim.py — no behavioural changes.
"""

import math
import numpy as np


class BrushlessMotorModel:
    """
    Simulates a brushless DC motor with:
    - First-order lag (electrical time constant)
    - Back-EMF (torque drops with speed)
    - Cogging torque
    - Deadband
    - Torque noise
    """

    def __init__(self, config):
        self.cfg = config
        self.actual_torque = 0.0
        self.tau = config['MOTOR_TAU']

    def update(self, commanded_torque, wheel_velocity, dt):
        """
        Compute actual motor torque given commanded torque and wheel speed.
        """
        # 1. First-order lag
        if self.tau > 0:
            alpha = min(1.0, dt / self.tau)
            self.actual_torque += (commanded_torque - self.actual_torque) * alpha
        else:
            self.actual_torque = commanded_torque

        torque = self.actual_torque

        # 2. Back-EMF
        back_emf_loss = self.cfg['MOTOR_BACK_EMF_K'] * abs(wheel_velocity)
        max_available = max(0.0, self.cfg['MAX_TORQUE'] - back_emf_loss)
        torque = np.clip(torque, -max_available, max_available)

        # 3. Cogging torque
        cogging = self.cfg['MOTOR_COGGING_AMPLITUDE'] * math.sin(
            self.cfg['MOTOR_COGGING_POLES'] * wheel_velocity * dt * 100
        )
        torque += cogging

        # 4. Deadband
        if abs(torque) < self.cfg['MOTOR_DEADBAND']:
            torque = 0.0

        # 5. Torque noise
        torque += np.random.normal(0, self.cfg['MOTOR_TORQUE_NOISE_STD'])

        # Final clamp
        torque = np.clip(torque, -self.cfg['MAX_TORQUE'], self.cfg['MAX_TORQUE'])

        return float(torque)
