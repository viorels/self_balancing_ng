#!/usr/bin/env python3
"""
Self-Balancing Robot Prototype using PyBullet and PID Control

Realistic simulation including motor dynamics, sensor fusion, control latency,
and mechanical imperfections to approximate real-world balancing difficulty.

REQUIREMENTS:
    - PyBullet: pip install pybullet
    - NumPy: pip install numpy

USAGE:
    python3 bullet_sim.py
"""

import time
import math
import numpy as np
import pybullet as p
import pybullet_data


# ============================================================================
# CONFIGURATION - All parameters are easily tunable here
# ============================================================================

CONFIG = {
    # Simulation parameters
    'GRAVITY': -9.81,
    'TIMESTEP': 1.0 / 500.0,  # 500 Hz physics
    'SIM_DURATION': 60.0,
    'GROUND_FRICTION': 0.8,

    # Robot physical parameters
    'ROBOT_MASS': 1.1,
    'BODY_HEIGHT': 0.15,
    'BODY_WIDTH': 0.08,
    'BODY_DEPTH': 0.08,
    'WHEEL_DIAMETER': 0.095,
    'WHEEL_MASS': 0.15,
    'WHEEL_FRICTION': 1.0,
    'AXLE_WIDTH': 0.10,

    # Center of mass offset from geometric center (meters, body-local)
    # Real robots are never perfectly symmetric
    'COM_OFFSET': [0.002, 0.001, 0.005],

    # Initial conditions
    'INITIAL_PITCH': -0.05,  # rad (~2.9 degrees)
    'INITIAL_HEIGHT': 0.08,

    # PID Controller gains
    'PID_KP': 2.0,
    'PID_KD': 0.1,
    'PID_KI': 0.5,

    # Motor/actuator limits (realistic small brushless)
    'MAX_TORQUE': 0.5,  # Nm (stall torque)

    # === REALISM PARAMETERS ===

    # Motor model
    'MOTOR_TAU': 0.003,          # Electrical time constant (s), 1st-order lag
    'MOTOR_BACK_EMF_K': 0.005,   # Back-EMF constant: torque_loss = K * omega (Nm per rad/s)
    'MOTOR_COGGING_AMPLITUDE': 0.005,  # Nm, cogging torque amplitude
    'MOTOR_COGGING_POLES': 14,    # Number of magnetic poles (cogging frequency)
    'MOTOR_DEADBAND': 0.01,       # Nm, minimum torque the motor can produce
    'MOTOR_TORQUE_NOISE_STD': 0.005, # Nm, random torque variation

    # Control loop
    'CONTROL_RATE_HZ': 200,       # PID update rate (Hz) — typical for Arduino/ESP32
    'CONTROL_JITTER_STD': 0.0005, # Timing jitter standard deviation (s)
    'SENSOR_TO_ACTUATOR_DELAY_STEPS': 1,  # Extra timestep delay (sample → actuate pipeline)

    # IMU sensor model
    'ADD_SENSOR_NOISE': True,
    'IMU_ANGLE_NOISE_STD': 0.003,   # rad — complementary filter output noise
    'IMU_GYRO_NOISE_STD': 0.01,     # rad/s — gyro noise after filtering
    'IMU_GYRO_DRIFT_RATE': 0.001,   # rad/s per second — slow gyro bias drift
    'IMU_ACCEL_VIB_NOISE_STD': 0.15, # m/s² — vibration-induced accel noise
    'IMU_SAMPLE_RATE_HZ': 500,       # IMU sample rate
    'IMU_QUANTIZATION_BITS': 16,     # ADC resolution
    'IMU_ACCEL_RANGE_G': 2,          # ±2g accelerometer range
    'IMU_GYRO_RANGE_DPS': 500,       # ±500 deg/s gyro range

    # Complementary filter (simulates real sensor fusion)
    'COMP_FILTER_ALPHA': 0.02,       # Weight of accelerometer (0=pure gyro, 1=pure accel)

    # Mechanical imperfections
    'WHEEL_IMBALANCE_TORQUE': 0.002,  # Nm, periodic torque from wheel imbalance
}


# ============================================================================
# REALISTIC MOTOR MODEL
# ============================================================================

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
        self.actual_torque = 0.0  # Current output torque (after lag)
        self.tau = config['MOTOR_TAU']

    def update(self, commanded_torque, wheel_velocity, dt):
        """
        Compute actual motor torque given commanded torque and wheel speed.

        Args:
            commanded_torque: Desired torque from controller (Nm)
            wheel_velocity: Current wheel angular velocity (rad/s)
            dt: Time step (s)

        Returns:
            Actual torque applied to the wheel (Nm)
        """
        # 1. First-order lag (motor electrical dynamics)
        #    tau * d(torque)/dt + torque = command
        #    Discrete: torque += (command - torque) * dt / tau
        if self.tau > 0:
            alpha = min(1.0, dt / self.tau)
            self.actual_torque += (commanded_torque - self.actual_torque) * alpha
        else:
            self.actual_torque = commanded_torque

        torque = self.actual_torque

        # 2. Back-EMF: torque capability decreases with speed
        #    Available torque = stall_torque - K * |omega|
        back_emf_loss = self.cfg['MOTOR_BACK_EMF_K'] * abs(wheel_velocity)
        max_available = max(0.0, self.cfg['MAX_TORQUE'] - back_emf_loss)
        torque = np.clip(torque, -max_available, max_available)

        # 3. Cogging torque (periodic resistance from magnets)
        cogging = self.cfg['MOTOR_COGGING_AMPLITUDE'] * math.sin(
            self.cfg['MOTOR_COGGING_POLES'] * wheel_velocity * dt * 100  # approximate rotor angle
        )
        torque += cogging

        # 4. Deadband: motor can't produce very small torques
        if abs(torque) < self.cfg['MOTOR_DEADBAND']:
            torque = 0.0

        # 5. Torque noise (electrical noise, commutation ripple)
        torque += np.random.normal(0, self.cfg['MOTOR_TORQUE_NOISE_STD'])

        # Final clamp
        torque = np.clip(torque, -self.cfg['MAX_TORQUE'], self.cfg['MAX_TORQUE'])

        return float(torque)


# ============================================================================
# REALISTIC IMU SENSOR MODEL
# ============================================================================

class IMUSensorModel:
    """
    Simulates a MEMS IMU (e.g. MPU6050) with:
    - Accelerometer-based angle (noisy, vibration-sensitive, no drift)
    - Gyroscope-based angle (low noise, but drifts)
    - Complementary filter fusion
    - Quantization
    - Sample rate limiting
    """

    def __init__(self, config):
        self.cfg = config
        self.fused_pitch = 0.0  # Complementary filter output
        self.gyro_bias = 0.0    # Slowly drifting gyro bias
        self.last_sample_time = 0.0
        self.sample_period = 1.0 / config['IMU_SAMPLE_RATE_HZ']

        # Quantization parameters
        accel_range_mps2 = config['IMU_ACCEL_RANGE_G'] * 9.81
        self.accel_lsb = (2 * accel_range_mps2) / (2 ** config['IMU_QUANTIZATION_BITS'])
        gyro_range_rps = math.radians(config['IMU_GYRO_RANGE_DPS'])
        self.gyro_lsb = (2 * gyro_range_rps) / (2 ** config['IMU_QUANTIZATION_BITS'])

    def _quantize(self, value, lsb):
        """Simulate ADC quantization."""
        return round(value / lsb) * lsb

    def read(self, true_pitch, true_pitch_rate, sim_time, dt):
        """
        Simulate an IMU reading with realistic noise and fusion.

        Args:
            true_pitch: Actual pitch angle (rad) from physics
            true_pitch_rate: Actual pitch rate (rad/s) from physics
            sim_time: Current simulation time
            dt: Physics timestep

        Returns:
            (measured_pitch, measured_pitch_rate)
        """
        if not self.cfg['ADD_SENSOR_NOISE']:
            return true_pitch, true_pitch_rate

        # === Gyroscope model ===
        # Gyro drift: bias wanders slowly (random walk)
        self.gyro_bias += np.random.normal(0, self.cfg['IMU_GYRO_DRIFT_RATE'] * dt)

        # Gyro reading = true rate + bias + noise
        gyro_reading = true_pitch_rate + self.gyro_bias
        gyro_reading += np.random.normal(0, self.cfg['IMU_GYRO_NOISE_STD'])
        gyro_reading = self._quantize(gyro_reading, self.gyro_lsb)

        # === Accelerometer model ===
        # Accel measures gravity direction, giving absolute pitch
        # But it's corrupted by vibration and linear acceleration
        accel_pitch = true_pitch
        accel_pitch += np.random.normal(0, self.cfg['IMU_ANGLE_NOISE_STD'])
        # Vibration noise (from motors, wheel impacts)
        vibration = np.random.normal(0, self.cfg['IMU_ACCEL_VIB_NOISE_STD'])
        # Convert vibration acceleration to angle error: approx atan(a_noise / g)
        accel_pitch += math.atan2(vibration, 9.81)
        # Quantize the underlying accelerometer values
        accel_pitch = self._quantize(accel_pitch, self.accel_lsb)

        # === Complementary filter ===
        # Fuse: trust gyro for fast changes, accel for absolute reference
        alpha = self.cfg['COMP_FILTER_ALPHA']
        gyro_angle = self.fused_pitch + gyro_reading * dt
        self.fused_pitch = (1.0 - alpha) * gyro_angle + alpha * accel_pitch

        return self.fused_pitch, gyro_reading


# ============================================================================
# ROBOT CLASS
# ============================================================================

class BalanceBot:
    """A 2-wheel self-balancing robot with realistic motor and sensor models."""

    def __init__(self, physics_client_id, config):
        self.pc = physics_client_id
        self.cfg = config

        self.body_id = None
        self.wheel_ids = []
        self._create_robot()

        # PID controller state
        self.prev_pitch_error = 0.0
        self.integral_pitch_error = 0.0

        # Realistic motor models (one per wheel)
        self.motors = [BrushlessMotorModel(config), BrushlessMotorModel(config)]

        # Realistic IMU sensor model
        self.imu = IMUSensorModel(config)

        # Control loop timing
        self.control_period = 1.0 / config['CONTROL_RATE_HZ']
        self.time_since_last_control = 0.0
        self.next_control_time = 0.0

        # Sensor-to-actuator delay buffer
        delay_steps = config['SENSOR_TO_ACTUATOR_DELAY_STEPS']
        self.torque_delay_buffer = [0.0] * (delay_steps + 1)

        # Output state for logging
        self.pitch_angle = 0.0
        self.pitch_rate = 0.0
        self.control_torque = 0.0
        self.actual_torques = [0.0, 0.0]

    def _create_robot(self):
        """Create the robot body and wheels using createMultiBody."""
        body_mass = self.cfg['ROBOT_MASS'] - 2 * self.cfg['WHEEL_MASS']
        wheel_radius = self.cfg['WHEEL_DIAMETER'] / 2
        wheel_width = 0.02
        wheel_spacing = self.cfg['AXLE_WIDTH'] / 2 + 0.01
        body_z = wheel_radius + self.cfg['BODY_HEIGHT'] / 2

        # Collision and visual shapes
        body_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.cfg['BODY_WIDTH']/2, self.cfg['BODY_DEPTH']/2, self.cfg['BODY_HEIGHT']/2]
        )
        body_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.cfg['BODY_WIDTH']/2, self.cfg['BODY_DEPTH']/2, self.cfg['BODY_HEIGHT']/2],
            rgbaColor=[0.2, 0.6, 1.0, 1.0]
        )

        wheel_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_width)
        wheel_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_width,
                                           rgbaColor=[0.1, 0.1, 0.1, 1.0])

        self.body_id = p.createMultiBody(
            baseMass=body_mass,
            baseCollisionShapeIndex=body_shape,
            baseVisualShapeIndex=body_visual,
            basePosition=[0, 0, body_z],
            baseOrientation=p.getQuaternionFromEuler([0, self.cfg['INITIAL_PITCH'], 0]),
            # Center of mass offset for asymmetry
            baseInertialFramePosition=self.cfg['COM_OFFSET'],
            baseInertialFrameOrientation=[0, 0, 0, 1],

            linkMasses=[self.cfg['WHEEL_MASS'], self.cfg['WHEEL_MASS']],
            linkCollisionShapeIndices=[wheel_shape, wheel_shape],
            linkVisualShapeIndices=[wheel_visual, wheel_visual],
            linkPositions=[
                [0, -wheel_spacing, -self.cfg['BODY_HEIGHT']/2],
                [0,  wheel_spacing, -self.cfg['BODY_HEIGHT']/2]
            ],
            linkOrientations=[
                p.getQuaternionFromEuler([math.pi/2, 0, 0]),
                p.getQuaternionFromEuler([math.pi/2, 0, 0])
            ],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 0, 1], [0, 0, 1]]
        )

        self.wheel_ids = [0, 1]

        # Disable default joint motors
        for wid in self.wheel_ids:
            p.setJointMotorControl2(self.body_id, wid, p.VELOCITY_CONTROL,
                                    targetVelocity=0.0, force=0.0)

        # Dynamics
        p.changeDynamics(self.body_id, -1, lateralFriction=self.cfg['GROUND_FRICTION'],
                         linearDamping=0.0, angularDamping=0.05)
        for wid in self.wheel_ids:
            p.changeDynamics(self.body_id, wid,
                             lateralFriction=self.cfg['WHEEL_FRICTION'],
                             spinningFriction=0.01, rollingFriction=0.001,
                             linearDamping=0.0, angularDamping=0.0)

    def _get_true_state(self):
        """Read true pitch from physics (body-frame, yaw-invariant)."""
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id)

        rot = p.getMatrixFromQuaternion(orn)
        body_up_z = rot[8]
        body_fwd_z = rot[6]
        pitch = math.atan2(-body_fwd_z, body_up_z)

        pitch_rate = rot[1] * ang_vel[0] + rot[4] * ang_vel[1] + rot[7] * ang_vel[2]

        return pitch, pitch_rate

    def update(self, sim_time, dt):
        """
        Update sensor reading, control loop, and motor output.
        Called every physics timestep, but PID only runs at CONTROL_RATE_HZ.
        """
        # --- Read true state and pass through IMU model ---
        true_pitch, true_pitch_rate = self._get_true_state()
        measured_pitch, measured_pitch_rate = self.imu.read(
            true_pitch, true_pitch_rate, sim_time, dt
        )
        self.pitch_angle = measured_pitch
        self.pitch_rate = measured_pitch_rate

        # --- Control loop runs at limited rate with jitter ---
        jitter = np.random.normal(0, self.cfg['CONTROL_JITTER_STD']) if self.cfg['ADD_SENSOR_NOISE'] else 0
        if sim_time >= self.next_control_time:
            self.next_control_time = sim_time + self.control_period + jitter

            # PID on measured (noisy, delayed) state
            pitch_error = 0.0 - measured_pitch
            p_term = self.cfg['PID_KP'] * pitch_error
            d_term = self.cfg['PID_KD'] * (0.0 - measured_pitch_rate)
            self.integral_pitch_error += pitch_error * self.control_period
            self.integral_pitch_error = np.clip(self.integral_pitch_error, -0.5, 0.5)
            i_term = self.cfg['PID_KI'] * self.integral_pitch_error

            commanded_torque = p_term + d_term + i_term
            commanded_torque = float(np.clip(commanded_torque,
                                             -self.cfg['MAX_TORQUE'], self.cfg['MAX_TORQUE']))

            # Push into delay buffer (simulates sample-compute-actuate pipeline)
            self.torque_delay_buffer.append(commanded_torque)
            self.control_torque = commanded_torque

        # Pop delayed torque command
        if len(self.torque_delay_buffer) > self.cfg['SENSOR_TO_ACTUATOR_DELAY_STEPS'] + 1:
            delayed_torque = self.torque_delay_buffer.pop(0)
        else:
            delayed_torque = self.torque_delay_buffer[0]

        # --- Apply through motor model (per wheel) ---
        for i, wid in enumerate(self.wheel_ids):
            wheel_vel = p.getJointState(self.body_id, wid)[1]

            # Add wheel imbalance (periodic disturbance)
            wheel_pos = p.getJointState(self.body_id, wid)[0]
            imbalance = self.cfg['WHEEL_IMBALANCE_TORQUE'] * math.sin(wheel_pos)

            actual_torque = self.motors[i].update(delayed_torque, wheel_vel, dt)
            actual_torque += imbalance

            self.actual_torques[i] = actual_torque

            p.setJointMotorControl2(
                self.body_id, wid,
                controlMode=p.TORQUE_CONTROL,
                force=actual_torque
            )

    def get_debug_state(self):
        """Return comprehensive debug info."""
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id)
        euler = p.getEulerFromQuaternion(orn)

        wheel_states = []
        for wid in self.wheel_ids:
            js = p.getJointState(self.body_id, wid)
            wheel_states.append({'pos': js[0], 'vel': js[1], 'torque': js[3]})

        return {
            'pos': pos,
            'euler_deg': (math.degrees(euler[0]), math.degrees(euler[1]), math.degrees(euler[2])),
            'ang_vel_deg': (math.degrees(ang_vel[0]), math.degrees(ang_vel[1]), math.degrees(ang_vel[2])),
            'wheels': wheel_states,
        }

    def check_fallen(self):
        """Check if robot has fallen over (abs pitch > 45 degrees)."""
        true_pitch, _ = self._get_true_state()
        return abs(true_pitch) > math.radians(45)


# ============================================================================
# SIMULATION MAIN LOOP
# ============================================================================

def run_simulation():
    """Run the self-balancing robot simulation."""

    print("=" * 70)
    print("Self-Balancing Robot — Realistic Simulation")
    print("=" * 70)

    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, CONFIG['GRAVITY'])
    p.setPhysicsEngineParameter(fixedTimeStep=CONFIG['TIMESTEP'], numSubSteps=1)

    ground_id = p.loadURDF("plane.urdf")
    p.changeDynamics(ground_id, -1, lateralFriction=CONFIG['GROUND_FRICTION'])

    robot = BalanceBot(physics_client, CONFIG)

    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=90, cameraPitch=-30,
                                 cameraTargetPosition=[0, 0, 0.1])

    print(f"\nRobot: {CONFIG['ROBOT_MASS']}kg, body {CONFIG['BODY_HEIGHT']}m tall, "
          f"wheels ø{CONFIG['WHEEL_DIAMETER']}m")
    print(f"PID: Kp={CONFIG['PID_KP']}, Ki={CONFIG['PID_KI']}, Kd={CONFIG['PID_KD']}")
    print(f"Motor: τ={CONFIG['MOTOR_TAU']*1000:.0f}ms lag, "
          f"back-EMF K={CONFIG['MOTOR_BACK_EMF_K']}, "
          f"cogging={CONFIG['MOTOR_COGGING_AMPLITUDE']}Nm, "
          f"deadband={CONFIG['MOTOR_DEADBAND']}Nm")
    print(f"IMU: complementary filter α={CONFIG['COMP_FILTER_ALPHA']}, "
          f"gyro drift={CONFIG['IMU_GYRO_DRIFT_RATE']} rad/s²")
    print(f"Control: {CONFIG['CONTROL_RATE_HZ']}Hz, "
          f"{CONFIG['SENSOR_TO_ACTUATOR_DELAY_STEPS']} step pipeline delay")
    print(f"Initial pitch: {math.degrees(CONFIG['INITIAL_PITCH']):.1f}°")
    print("-" * 70)

    sim_time = 0.0
    log_interval = 0.1
    last_log_time = 0.0

    while sim_time < CONFIG['SIM_DURATION']:
        robot.update(sim_time, CONFIG['TIMESTEP'])
        p.stepSimulation()
        sim_time += CONFIG['TIMESTEP']

        if robot.check_fallen():
            print(f"\n[{sim_time:.2f}s] Robot fell over!")
            break

        if sim_time - last_log_time >= log_interval:
            dbg = robot.get_debug_state()
            rx, ry, rz = dbg['euler_deg']
            wx, wy, wz = dbg['ang_vel_deg']
            w0, w1 = dbg['wheels']
            at0, at1 = robot.actual_torques
            print(f"[{sim_time:5.2f}s] "
                  f"Euler(r={rx:6.1f} p={ry:6.1f} y={rz:6.1f})° | "
                  f"AngVel(x={wx:6.1f} y={wy:6.1f} z={wz:6.1f})°/s | "
                  f"Whl({w0['vel']:6.1f},{w1['vel']:6.1f})rad/s | "
                  f"Cmd:{robot.control_torque:6.3f} Act:{at0:6.3f},{at1:6.3f}")
            last_log_time = sim_time

        time.sleep(CONFIG['TIMESTEP'])

    print("-" * 70)
    final_pitch_deg = math.degrees(robot.pitch_angle)
    print(f"\nSimulation complete! Final pitch: {final_pitch_deg:.2f}°")
    if abs(final_pitch_deg) < 10:
        print("✓ Robot balanced!")
    else:
        print("✗ Robot fell.")

    print("\nClose the PyBullet window to exit.")
    while p.isConnected(physics_client):
        time.sleep(0.01)
    p.disconnect()


if __name__ == "__main__":
    run_simulation()
