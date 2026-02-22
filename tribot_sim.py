#!/usr/bin/env python3
"""
Tribot Self-Balancing Robot Simulation using PyBullet and PID Control

Uses a URDF robot model with triplet (tri-wheel) clusters on each side.
Each side has 3 small drive wheels connected by virtual belts to a single
motor. The triplet assemblies spin freely, acting as large effective wheels.

Based on bullet_sim.py — same realistic motor model, IMU sensor model,
and cascaded PID control structure.

REQUIREMENTS:
    - PyBullet: pip install pybullet
    - NumPy: pip install numpy
    - URDF model: tribot_description/urdf/tribot.urdf
    - STL meshes: tribot_description/meshes/*.stl

USAGE:
    python3 tribot_sim.py
"""

import time
import math
import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import pybullet as p
import pybullet_data


# ============================================================================
# CONFIGURATION - All parameters are easily tunable here
# ============================================================================

CONFIG = {
    # Simulation parameters
    'GRAVITY': -9.81,
    'TIMESTEP': 1.0 / 500.0,      # 500 Hz physics
    'SIM_DURATION': 60.0,
    'GROUND_FRICTION': 1.0,

    # URDF model path (relative to this script)
    'URDF_PATH': 'tribot_description/urdf/tribot.urdf',

    # Robot geometry (must match the URDF/STL)
    'WHEEL_RADIUS': 0.058,         # Small drive wheel radius (m) — measured from STL AABB
    'TRIPLET_RADIUS': 0.12,        # Circumradius of the wheel triangle (m)

    # Initial conditions
    'INITIAL_PITCH': -0.03,        # rad (~1.7°) — slight initial tilt
    'INITIAL_HEIGHT': 0.12,        # m — c_body origin above ground (bottom wheel at ~ground level)

    # Inner PID gains (pitch → motor torque)
    'PID_KP': 15.0,
    'PID_KD': 0.8,
    'PID_KI': 3.0,

    # Outer PID gains (position → target pitch angle)
    'POS_PID_KP': 0.15,
    'POS_PID_KD': 0.03,
    'POS_PID_KI': 0.01,
    'POS_PID_MAX_PITCH': 0.15,     # rad (~8.6°) max lean angle from outer loop
    'POS_PID_RATE_HZ': 50,

    # Motor / actuator limits
    'MAX_TORQUE': 0.5,             # Nm (stall torque per motor, one motor per side)

    # === REALISM PARAMETERS ===

    # Motor model
    'MOTOR_TAU': 0.003,            # Electrical time constant (s)
    'MOTOR_BACK_EMF_K': 0.005,     # Back-EMF constant (Nm per rad/s)
    'MOTOR_COGGING_AMPLITUDE': 0.005,  # Nm
    'MOTOR_COGGING_POLES': 14,
    'MOTOR_DEADBAND': 0.01,        # Nm
    'MOTOR_TORQUE_NOISE_STD': 0.005,   # Nm

    # Control loop
    'CONTROL_RATE_HZ': 200,        # PID update rate (Hz)
    'CONTROL_JITTER_STD': 0.0005,  # Timing jitter std dev (s)
    'SENSOR_TO_ACTUATOR_DELAY_STEPS': 1,

    # IMU sensor model
    'ADD_SENSOR_NOISE': True,
    'IMU_ANGLE_NOISE_STD': 0.003,
    'IMU_GYRO_NOISE_STD': 0.01,
    'IMU_GYRO_DRIFT_RATE': 0.001,
    'IMU_ACCEL_VIB_NOISE_STD': 0.15,
    'IMU_SAMPLE_RATE_HZ': 500,
    'IMU_QUANTIZATION_BITS': 16,
    'IMU_ACCEL_RANGE_G': 2,
    'IMU_GYRO_RANGE_DPS': 500,

    # Complementary filter
    'COMP_FILTER_ALPHA': 0.02,

    # Mechanical imperfections
    'WHEEL_IMBALANCE_TORQUE': 0.002,   # Nm, periodic torque from wheel imbalance

    # Yaw damping gain (differential torque to oppose yaw rotation)
    'YAW_DAMPING_K': 0.05,

    # Wheel contact properties
    'WHEEL_FRICTION': 1.2,
    'TRIPLET_FRICTION': 0.3,           # Low friction on triplet hubs (shouldn't contact ground much)

    # Virtual belt stiffness (gear constraint max force)
    'BELT_MAX_FORCE': 100.0,
}


# ============================================================================
# REALISTIC MOTOR MODEL (same as bullet_sim.py)
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


# ============================================================================
# REALISTIC IMU SENSOR MODEL (same as bullet_sim.py)
# ============================================================================

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


# ============================================================================
# URDF HELPERS
# ============================================================================

def preprocess_urdf(urdf_path):
    """
    Preprocess the tribot URDF for PyBullet:
    - Replace package:// paths with absolute filesystem paths
    - Remove the zero-mass base_link and its joint so that c_body
      becomes the root link (floating base with proper mass/inertia)
    Returns the path to a temporary URDF file.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Resolve package://tribot_description/ → absolute path
    pkg_dir = os.path.abspath(os.path.join(os.path.dirname(urdf_path), '..'))
    for mesh_elem in root.iter('mesh'):
        fn = mesh_elem.get('filename', '')
        if fn.startswith('package://tribot_description/'):
            mesh_elem.set('filename', fn.replace(
                'package://tribot_description/', pkg_dir + '/'))

    # Remove the zero-mass base_link and its connecting joint entirely.
    # This makes c_body the root link so PyBullet gives it proper mass.
    for link in root.findall('link'):
        if link.get('name') == 'base_link':
            root.remove(link)
    for joint in root.findall('joint'):
        if joint.get('name') == 'base_link_to_c_body':
            root.remove(joint)

    # Write the preprocessed URDF to a temp file
    fd, temp_path = tempfile.mkstemp(suffix='.urdf', prefix='tribot_')
    with os.fdopen(fd, 'w') as f:
        tree.write(f, xml_declaration=True, encoding='unicode')

    return temp_path


# ============================================================================
# TRIBOT ROBOT CLASS
# ============================================================================

class TribotBalanceBot:
    """
    A self-balancing robot with triplet wheel clusters, loaded from URDF.

    Each side has a triplet assembly (3 small wheels in a triangle) connected
    to the body by a freely-rotating hub. All 3 wheels on each side are
    coupled by a virtual belt (same angular velocity) driven by a single motor.
    """

    def __init__(self, physics_client_id, config):
        self.pc = physics_client_id
        self.cfg = config

        self.body_id = None

        # Joint indices (populated after loading)
        self.joint_map = {}               # name → index
        self.l_triplet_joint = -1
        self.r_triplet_joint = -1
        self.l_wheel_joints = []          # [idx, idx, idx]
        self.r_wheel_joints = []
        self.belt_constraints = []        # gear constraint IDs

        # Load and configure the robot
        self._load_robot()
        self._discover_joints()
        self._configure_dynamics()
        self._setup_belt_constraints()

        # Inner PID controller state (pitch → torque)
        self.prev_pitch_error = 0.0
        self.integral_pitch_error = 0.0

        # Outer PID controller state (position → target pitch)
        self.target_pitch = 0.0
        self.position = 0.0
        self.prev_position = 0.0
        self.integral_pos_error = 0.0
        self.target_position = 0.0
        self.pos_control_period = 1.0 / config['POS_PID_RATE_HZ']
        self.next_pos_control_time = 0.0

        # Two motors (one per side)
        self.motors = [BrushlessMotorModel(config), BrushlessMotorModel(config)]

        # IMU sensor model
        self.imu = IMUSensorModel(config)

        # Control loop timing
        self.control_period = 1.0 / config['CONTROL_RATE_HZ']
        self.next_control_time = 0.0

        # Sensor-to-actuator delay buffer
        delay_steps = config['SENSOR_TO_ACTUATOR_DELAY_STEPS']
        self.torque_delay_buffer = [(0.0, 0.0)] * (delay_steps + 1)

        # Estimated wheel radius (may be overridden from AABB after loading)
        self.wheel_radius = config['WHEEL_RADIUS']

        # Output state for logging
        self.pitch_angle = 0.0
        self.pitch_rate = 0.0
        self.control_torque = 0.0
        self.actual_torques = [0.0, 0.0]

    # ----------------------------------------------------------------
    # Robot setup
    # ----------------------------------------------------------------

    def _load_robot(self):
        """Load the URDF with preprocessed paths."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, self.cfg['URDF_PATH'])

        temp_urdf = preprocess_urdf(urdf_path)
        try:
            self.body_id = p.loadURDF(
                temp_urdf,
                basePosition=[0, 0, self.cfg['INITIAL_HEIGHT']],
                baseOrientation=p.getQuaternionFromEuler(
                    [0, self.cfg['INITIAL_PITCH'], 0]
                ),
                useFixedBase=False,
                # Do NOT use URDF_USE_INERTIA_FROM_FILE — the CAD-exported
                # inertia values are in wrong units.  Let PyBullet compute
                # inertias from the collision meshes + URDF masses instead.
            )
        finally:
            os.unlink(temp_urdf)

    def _discover_joints(self):
        """Build a name→index map and identify joint groups."""
        num_joints = p.getNumJoints(self.body_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.body_id, i)
            name = info[1].decode('utf-8')
            self.joint_map[name] = i

        # Triplet hub joints (free-spinning)
        self.l_triplet_joint = self.joint_map['c_body_to_l_triplet']
        self.r_triplet_joint = self.joint_map['c_body_to_r_triplet']

        # Drive wheel joints (3 per side, belt-coupled)
        self.l_wheel_joints = [
            self.joint_map['l_triplet_to_l_wheel_1'],
            self.joint_map['l_triplet_to_l_wheel_2'],
            self.joint_map['l_triplet_to_l_wheel_3'],
        ]
        self.r_wheel_joints = [
            self.joint_map['r_triplet_to_r_wheel_1'],
            self.joint_map['r_triplet_to_r_wheel_2'],
            self.joint_map['r_triplet_to_r_wheel_3'],
        ]

        print(f"  Joints discovered: {num_joints} total")
        print(f"    L triplet: joint {self.l_triplet_joint}")
        print(f"    R triplet: joint {self.r_triplet_joint}")
        print(f"    L wheels:  joints {self.l_wheel_joints}")
        print(f"    R wheels:  joints {self.r_wheel_joints}")

    def _configure_dynamics(self):
        """Set friction, damping for all links."""
        num_joints = p.getNumJoints(self.body_id)

        # Disable all default joint motors (we'll apply torques explicitly)
        for i in range(num_joints):
            p.setJointMotorControl2(
                self.body_id, i, p.VELOCITY_CONTROL,
                targetVelocity=0.0, force=0.0
            )

        # Print computed inertias for debugging
        for i in range(-1, num_joints):
            dyn = p.getDynamicsInfo(self.body_id, i)
            if i == -1:
                name = 'c_body(base)'
            else:
                name = p.getJointInfo(self.body_id, i)[12].decode()
            print(f"    {name}: mass={dyn[0]:.4f} inertia={tuple(round(v,6) for v in dyn[2])}")

        # Body base link: moderate friction, slight angular damping
        p.changeDynamics(self.body_id, -1,
                         lateralFriction=0.4,
                         linearDamping=0.0,
                         angularDamping=0.05)

        # Triplet hubs: low friction (shouldn't touch ground much)
        for tj in [self.l_triplet_joint, self.r_triplet_joint]:
            p.changeDynamics(self.body_id, tj,
                             lateralFriction=self.cfg['TRIPLET_FRICTION'],
                             linearDamping=0.0,
                             angularDamping=0.0)

        # Drive wheels: high friction for traction
        for wj in self.l_wheel_joints + self.r_wheel_joints:
            p.changeDynamics(self.body_id, wj,
                             lateralFriction=self.cfg['WHEEL_FRICTION'],
                             spinningFriction=0.01,
                             rollingFriction=0.001,
                             linearDamping=0.0,
                             angularDamping=0.0)

    def _setup_belt_constraints(self):
        """
        Create gear constraints to simulate virtual belts.
        All 3 wheels on each side rotate at the same angular velocity
        (relative to their triplet) — enforced via 1:1 gear constraints.
        """
        for side_wheels in [self.l_wheel_joints, self.r_wheel_joints]:
            master = side_wheels[0]
            for slave in side_wheels[1:]:
                c = p.createConstraint(
                    self.body_id, master,
                    self.body_id, slave,
                    jointType=p.JOINT_GEAR,
                    jointAxis=[0, 1, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0]
                )
                # gearRatio=-1 → same direction rotation (child = -ratio * parent,
                # and the constraint eq is ratio*q_parent + q_child = 0)
                p.changeConstraint(c, gearRatio=-1,
                                   maxForce=self.cfg['BELT_MAX_FORCE'])
                self.belt_constraints.append(c)

        print(f"  Belt constraints: {len(self.belt_constraints)} "
              f"gear joints created")

    # ----------------------------------------------------------------
    # State estimation
    # ----------------------------------------------------------------

    def _get_true_state(self):
        """
        Read true pitch from physics (body-frame, yaw-invariant).
        Pitch = rotation around body Y axis (the wheel axle direction).
        """
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id)

        # Rotation matrix columns = body axes in world frame
        rot = p.getMatrixFromQuaternion(orn)
        # body Z in world Z (rot[8] = R[2,2])
        body_up_z = rot[8]
        # body X in world Z (rot[6] = R[2,0])
        body_fwd_z = rot[6]
        pitch = math.atan2(-body_fwd_z, body_up_z)

        # Pitch rate = angular velocity projected onto body Y axis
        # body Y in world = (rot[1], rot[4], rot[7])
        pitch_rate = rot[1] * ang_vel[0] + rot[4] * ang_vel[1] + rot[7] * ang_vel[2]

        return pitch, pitch_rate

    def _estimate_position(self):
        """
        Estimate linear position from wheel encoders.
        position = (wheel_angle_relative + triplet_angle) * wheel_radius
        averaged over both sides.
        """
        r = self.wheel_radius

        # Left side: use first wheel (all are belt-coupled, same angle)
        l_wheel_pos = p.getJointState(self.body_id, self.l_wheel_joints[0])[0]
        l_triplet_pos = p.getJointState(self.body_id, self.l_triplet_joint)[0]
        l_pos = (l_wheel_pos + l_triplet_pos) * r

        # Right side
        r_wheel_pos = p.getJointState(self.body_id, self.r_wheel_joints[0])[0]
        r_triplet_pos = p.getJointState(self.body_id, self.r_triplet_joint)[0]
        r_pos = (r_wheel_pos + r_triplet_pos) * r

        return (l_pos + r_pos) / 2.0

    # ----------------------------------------------------------------
    # Control update
    # ----------------------------------------------------------------

    def update(self, sim_time, dt):
        """
        Update sensor reading, PID control, and motor output.
        Called every physics timestep; PID only runs at CONTROL_RATE_HZ.
        """
        # --- Read true state and pass through IMU model ---
        true_pitch, true_pitch_rate = self._get_true_state()
        measured_pitch, measured_pitch_rate = self.imu.read(
            true_pitch, true_pitch_rate, sim_time, dt
        )
        self.pitch_angle = measured_pitch
        self.pitch_rate = measured_pitch_rate

        # --- Position estimation from wheel encoders ---
        self.position = self._estimate_position()

        # --- Outer PID loop: position → target pitch (lower rate) ---
        if sim_time >= self.next_pos_control_time:
            self.next_pos_control_time = sim_time + self.pos_control_period

            pos_error = self.position - self.target_position
            velocity = (self.position - self.prev_position) / self.pos_control_period
            self.prev_position = self.position

            pos_p = self.cfg['POS_PID_KP'] * pos_error
            pos_d = self.cfg['POS_PID_KD'] * velocity
            self.integral_pos_error += pos_error * self.pos_control_period
            self.integral_pos_error = np.clip(self.integral_pos_error, -1.0, 1.0)
            pos_i = self.cfg['POS_PID_KI'] * self.integral_pos_error

            self.target_pitch = float(np.clip(
                pos_p + pos_d + pos_i,
                -self.cfg['POS_PID_MAX_PITCH'],
                 self.cfg['POS_PID_MAX_PITCH']
            ))

        # --- Inner control loop (pitch → torque) at limited rate ---
        jitter = (np.random.normal(0, self.cfg['CONTROL_JITTER_STD'])
                  if self.cfg['ADD_SENSOR_NOISE'] else 0)
        if sim_time >= self.next_control_time:
            self.next_control_time = sim_time + self.control_period + jitter

            pitch_error = self.target_pitch - measured_pitch
            p_term = self.cfg['PID_KP'] * pitch_error
            d_term = self.cfg['PID_KD'] * (0.0 - measured_pitch_rate)
            self.integral_pitch_error += pitch_error * self.control_period
            self.integral_pitch_error = np.clip(self.integral_pitch_error, -0.5, 0.5)
            i_term = self.cfg['PID_KI'] * self.integral_pitch_error

            commanded_torque = p_term + d_term + i_term
            commanded_torque = float(np.clip(
                commanded_torque,
                -self.cfg['MAX_TORQUE'], self.cfg['MAX_TORQUE']
            ))
            self.control_torque = commanded_torque

            # --- Yaw damping: oppose yaw rate with differential torque ---
            _, ang_vel = p.getBaseVelocity(self.body_id)
            _, orn = p.getBasePositionAndOrientation(self.body_id)
            rot = p.getMatrixFromQuaternion(orn)
            # Yaw rate = angular velocity projected onto body Z axis
            yaw_rate = rot[2] * ang_vel[0] + rot[5] * ang_vel[1] + rot[8] * ang_vel[2]
            yaw_correction = self.cfg['YAW_DAMPING_K'] * yaw_rate

            # Push into delay buffer
            self.torque_delay_buffer.append((commanded_torque, yaw_correction))

        # Pop delayed torque command
        if len(self.torque_delay_buffer) > self.cfg['SENSOR_TO_ACTUATOR_DELAY_STEPS'] + 1:
            delayed_torque, delayed_yaw = self.torque_delay_buffer.pop(0)
        else:
            delayed_torque, delayed_yaw = self.torque_delay_buffer[0]

        # --- Apply motor torque through motor models ---
        # Left motor (+yaw_correction), Right motor (-yaw_correction)
        side_configs = [
            (0, self.l_wheel_joints, +1.0),  # left motor, left wheels, yaw sign
            (1, self.r_wheel_joints, -1.0),  # right motor, right wheels, yaw sign
        ]

        for motor_idx, wheel_joints, yaw_sign in side_configs:
            # Representative wheel velocity (belt-coupled, all same)
            wheel_vel = p.getJointState(self.body_id, wheel_joints[0])[1]

            # Motor produces total torque for this side
            motor_torque = self.motors[motor_idx].update(
                delayed_torque + yaw_sign * delayed_yaw,
                wheel_vel, dt
            )
            self.actual_torques[motor_idx] = motor_torque

            # Distribute torque equally among 3 belt-coupled wheels
            torque_per_wheel = motor_torque / 3.0

            for wj in wheel_joints:
                # Add wheel imbalance (per-wheel periodic disturbance)
                wheel_pos = p.getJointState(self.body_id, wj)[0]
                imbalance = self.cfg['WHEEL_IMBALANCE_TORQUE'] * math.sin(wheel_pos)

                p.setJointMotorControl2(
                    self.body_id, wj,
                    controlMode=p.TORQUE_CONTROL,
                    force=torque_per_wheel + imbalance
                )

    # ----------------------------------------------------------------
    # Debug / status
    # ----------------------------------------------------------------

    def get_debug_state(self):
        """Return comprehensive debug info."""
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id)
        euler = p.getEulerFromQuaternion(orn)

        # Triplet angles
        lt_state = p.getJointState(self.body_id, self.l_triplet_joint)
        rt_state = p.getJointState(self.body_id, self.r_triplet_joint)

        # Representative wheel velocities (one per side, belt-coupled)
        lw_vel = p.getJointState(self.body_id, self.l_wheel_joints[0])[1]
        rw_vel = p.getJointState(self.body_id, self.r_wheel_joints[0])[1]

        return {
            'pos': pos,
            'euler_deg': tuple(math.degrees(e) for e in euler),
            'ang_vel_deg': tuple(math.degrees(v) for v in ang_vel),
            'triplet_vel': (lt_state[1], rt_state[1]),
            'wheel_vel': (lw_vel, rw_vel),
        }

    def check_fallen(self):
        """Check if robot has fallen over (|pitch| > 45°)."""
        true_pitch, _ = self._get_true_state()
        return abs(true_pitch) > math.radians(45)


# ============================================================================
# SIMULATION MAIN LOOP
# ============================================================================

def run_simulation():
    """Run the tribot self-balancing simulation."""

    print("=" * 70)
    print("Tribot Self-Balancing Robot — URDF-based Simulation")
    print("=" * 70)

    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, CONFIG['GRAVITY'])
    p.setPhysicsEngineParameter(fixedTimeStep=CONFIG['TIMESTEP'], numSubSteps=1)

    # Ground plane
    ground_id = p.loadURDF("plane.urdf")
    p.changeDynamics(ground_id, -1, lateralFriction=CONFIG['GROUND_FRICTION'])

    # Create robot
    print("\nLoading tribot URDF...")
    robot = TribotBalanceBot(physics_client, CONFIG)

    # Camera
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.15]
    )

    # Print configuration summary
    total_mass = sum(p.getDynamicsInfo(robot.body_id, i)[0]
                     for i in range(-1, p.getNumJoints(robot.body_id)))
    print(f"\nRobot total mass: {total_mass:.3f} kg")
    print(f"Inner PID (pitch→torque): Kp={CONFIG['PID_KP']}, "
          f"Ki={CONFIG['PID_KI']}, Kd={CONFIG['PID_KD']}")
    print(f"Outer PID (pos→pitch):   Kp={CONFIG['POS_PID_KP']}, "
          f"Ki={CONFIG['POS_PID_KI']}, Kd={CONFIG['POS_PID_KD']}, "
          f"max_pitch={math.degrees(CONFIG['POS_PID_MAX_PITCH']):.1f}°")
    print(f"Motor: τ={CONFIG['MOTOR_TAU']*1000:.0f}ms lag, "
          f"back-EMF K={CONFIG['MOTOR_BACK_EMF_K']}, "
          f"deadband={CONFIG['MOTOR_DEADBAND']}Nm")
    print(f"IMU: complementary filter α={CONFIG['COMP_FILTER_ALPHA']}, "
          f"gyro drift={CONFIG['IMU_GYRO_DRIFT_RATE']} rad/s²")
    print(f"Control: {CONFIG['CONTROL_RATE_HZ']}Hz, "
          f"{CONFIG['SENSOR_TO_ACTUATOR_DELAY_STEPS']} step pipeline delay")
    print(f"Initial pitch: {math.degrees(CONFIG['INITIAL_PITCH']):.1f}°  "
          f"height: {CONFIG['INITIAL_HEIGHT']:.3f}m")
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
            tv = dbg['triplet_vel']
            wv = dbg['wheel_vel']
            at0, at1 = robot.actual_torques
            print(
                f"[{sim_time:5.2f}s] "
                f"Euler(r={rx:6.1f} p={ry:6.1f} y={rz:6.1f})° | "
                f"Pos:{robot.position:6.3f}m "
                f"TgtPitch:{math.degrees(robot.target_pitch):5.2f}° | "
                f"Triplet({tv[0]:5.1f},{tv[1]:5.1f}) "
                f"Whl({wv[0]:5.1f},{wv[1]:5.1f})rad/s | "
                f"Act:{at0:6.3f},{at1:6.3f}"
            )
            last_log_time = sim_time

        time.sleep(CONFIG['TIMESTEP'])

    # Final report
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
