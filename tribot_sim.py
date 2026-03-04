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

from control_pid import BalanceController
from control_lqr import LQRBalanceController
from control_mpc_hybrid import MPCHybridController
from gamepad import Gamepad
from plotjuggler_udp import PlotJugglerStreamer
from terrain import create_terrain


# ============================================================================
# CONFIGURATION - All parameters are easily tunable here
# ============================================================================

CONFIG = {
    # Simulation parameters
    'GRAVITY': -9.81,
    'TIMESTEP': 1.0 / 500.0,      # 500 Hz physics (required for 2WD triplet stability)
    'SIM_DURATION': 60.0,
    'GROUND_FRICTION': 1.0,

    # Terrain: 'flat', 'heightfield', or 'box_stairs'
    'TERRAIN': 'box_stairs',

    # URDF model path (relative to this script)
    'URDF_PATH': 'tribot_description/urdf/tribot.urdf',

    # Robot geometry (must match the URDF/STL)
    'WHEEL_RADIUS': 0.058,         # Small drive wheel radius (m) — measured from STL AABB
    'TRIPLET_RADIUS': 0.12,        # Circumradius of the wheel triangle (m)

    # Initial conditions
    'INITIAL_PITCH': -0.03,        # rad (~1.7°) — slight initial tilt
    'INITIAL_HEIGHT': 0.118,       # m — c_body origin above ground (4WD: triplet_Z_offset + wheel_R = 0.060125 + 0.058)
    'INITIAL_TRIPLET_ANGLE': 0.0,  # rad (0°) — 4WD mode: two wheels down per side (flat triangle base on ground)

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
    'MAX_TORQUE': 1.0,             # Nm (stall torque per motor, one motor per side)

    # === REALISM PARAMETERS ===

    # Motor model
    'MOTOR_TAU': 0.003,            # Electrical time constant (s)
    'MOTOR_BACK_EMF_K': 0.005,     # Back-EMF constant (Nm per rad/s)
    'MOTOR_COGGING_AMPLITUDE': 0.005,  # Nm
    'MOTOR_COGGING_POLES': 14,
    'MOTOR_DEADBAND': 0.01,        # Nm
    'MOTOR_TORQUE_NOISE_STD': 0.005,   # Nm

    # Control loop
    'CONTROL_RATE_HZ': 200,        # PD tracking loop rate (Hz) — independent of physics
    'CONTROL_JITTER_STD': 0.0005,  # Timing jitter std dev (s)
    'SENSOR_TO_ACTUATOR_DELAY_STEPS': 0,

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
    'YAW_DAMPING_K': 0.5,

    # Wheel contact properties
    'WHEEL_FRICTION': 1.2,
    'TRIPLET_FRICTION': 0.3,           # Low friction on triplet hubs (shouldn't contact ground much)

    # Virtual belt stiffness (gear constraint max force)
    'BELT_MAX_FORCE': 100.0,

    # Triplet hub joint damping (simulates motor back-EMF / bearing friction)
    'TRIPLET_JOINT_DAMPING': 0.05,     # Nm·s/rad

    # === CONTROLLER SELECTION ===
    # 'lqr', 'pid', or 'mpc'
    'CONTROLLER': 'lqr',

    # === MPC HYBRID PARAMETERS ===
    'MPC_RATE_HZ': 30,                 # MPC solve rate (Hz) — realistic for ESP32-S3
    'MPC_HORIZON': 10,                 # Prediction horizon N (DARE terminal cost handles the rest)
    'MPC_SIMULATED_SOLVE_MS': 20.0,    # Artificial delay per solve (ms) — realistic for ESP32-S3 SIMD
    # Q weights: [pitch, pitch_rate, tripL, tripR, tripL_rate, tripR_rate, fwd_pos, fwd_vel]
    'MPC_Q_DIAG': [50.0, 5.0, 40.0, 40.0, 5.0, 5.0, 12.0, 5.0],
    # R weights: [tau_tripL, tau_tripR, tau_driveL, tau_driveR]
    'MPC_R_DIAG': [1.0, 1.0, 8.0, 8.0],
    'MPC_Q_TERMINAL_SCALE': 3.0,
    'MPC_TRIPLET_TORQUE_MAX': 5.0,     # Nm — larger triplet motor for 2WD balance
    'MPC_TRIPLET_INERTIA': 0.00238,    # kg·m² (0.5 * 0.33 * 0.12²)
    # PD tracking gains: [trip_L, trip_R, drive_L, drive_R]
    'MPC_PD_KP': [10.0, 10.0, 3.0, 3.0],
    'MPC_PD_KD': [1.0, 1.0, 0.3, 0.3],
    'MPC_PITCH_PD_CROSS_DRIVE': 8.0,
    'MPC_PITCH_RATE_PD_CROSS_DRIVE': 0.5,

    # === ZMP / DCM TRIPLET FLIP TRIGGER (physics-based) ===
    # Flip timing
    'ZMP_T_FLIP_NOMINAL': 0.18,    # s  — observed 120° rotation time (physics min ~63 ms)
    'ZMP_T_FLIP_MARGIN':  0.05,    # s  — extra margin for motor lag, belt compliance
    'ZMP_T_SETTLE':       0.40,    # s  — post-flip settling window
    'ZMP_TRIP_TOL':       0.15,    # rad — "arrived at new angle" tolerance (~8.6°)
    # MPC cost reshaping during flip
    'ZMP_FLIP_Q_TRIP':  120.0,
    'ZMP_FLIP_Q_PITCH': 120.0,
    'ZMP_FLIP_R_TRIP':    0.05,
    # Control-authority model — fraction η of max drive torque the controller
    # can muster during a fall.  Lower = more conservative (fires earlier).
    # 0.0 = free-fall (old behaviour), 1.0 = full authority (fires very late).
    # Empirical: MPC delivers ~50-70 % during impact, but 20 % is conservative
    # because motor lag & battery sag eat into the usable authority.
    'ZMP_CTRL_AUTHORITY': 0.3,
    # Mechanical crash limit (rad).  Beyond this angle, recovery is impossible
    # regardless of torque.  45° is a good default for an inverted pendulum.
    'ZMP_THETA_CRASH': 0.785,       # rad (≈45°)
    # Stair-step height (m).  If 0, flat-ground assumptions are used.
    # Non-zero reduces the required triplet rotation and landing ω₀.
    'ZMP_STAIR_HEIGHT': 0.1,
    # Secondary pitch-rate gate — filters out slow balance sway.
    'ZMP_MIN_FALL_RATE_DEG_S': 15.0,
    # Post-flip cooldown (s) — block re-arming after a flip completes.
    'ZMP_FLIP_COOLDOWN': 0.8,
    # Early-landing exit from FLIPPING
    'ZMP_PITCH_RECOVER_THRESHOLD': 0.12,   # rad (~7°)
    'ZMP_FLIP_MIN_ROTATION':       0.698,  # rad (40°)

    # === LQR PARAMETERS ===
    # Linearised plant physical constants (derived from URDF via PyBullet)
    'LQR_BODY_MASS': 2.7167,       # kg — c_body (from URDF mesh + mass)
    'LQR_WHEEL_MASS': 0.6698,      # kg — 2 triplets + 6 wheels
    'LQR_COG_HEIGHT': 0.247,       # m  — c_body CoG z above wheel axis
    'LQR_BODY_INERTIA': 0.056436,  # kg·m² — c_body Iyy (PyBullet-computed)

    # Q diagonal: [position, velocity, pitch, pitch_rate]
    'LQR_Q_DIAG': [12.0, 4.0, 55.0, 4.0],
    # R: torque cost (scalar) — higher = less aggressive, more robust to
    # unmodeled motor dynamics (lag, deadband, back-EMF)
    'LQR_R': 2.0,

    # === GAIN-SCHEDULED LQR (aggressive mode while far from target) ===
    # When |pos_error| > threshold, switch to aggressive Q/R for fast tracking.
    # Hysteresis band prevents chattering around the boundary.
    'LQR_AGGRESSIVE_Q_DIAG': [40.0, 8.0, 35.0, 3.0],   # lean harder, chase faster
    'LQR_AGGRESSIVE_R': 1.0,                              # allow more torque
    'LQR_SWITCH_THRESHOLD': 0.20,     # m — switch to aggressive when |error| > this
    'LQR_SWITCH_HYSTERESIS': 0.05,    # m — switch back when |error| < threshold - hyst

    # === TRIPLET LEAN PD CONTROLLER ===
    # Holds the triplet hub at INITIAL_TRIPLET_ANGLE using torque control.
    # Gravity compensation is computed analytically; for the symmetric
    # equilateral-triangle triplet it is near-zero but included for accuracy.
    'TRIPLET_LEAN_KP': 8.0,     # Nm/rad  — proportional gain
    'TRIPLET_LEAN_KD': 0.4,     # Nm·s/rad — derivative (damping) gain

    # === GAMEPAD ===
    'GAMEPAD_DEVICE': '/dev/input/js0',
    'GAMEPAD_DEADZONE': 0.08,
    'GAMEPAD_SPEED_AXIS': 4,        # Right stick Y
    'GAMEPAD_YAW_AXIS': 3,          # Right stick X
    'GAMEPAD_LEAN_AXIS': 1,         # Left stick Y  (push up = lean forward)
    'GAMEPAD_MAX_DISTANCE': 1.0,    # m max target distance in front of robot
    'GAMEPAD_MAX_YAW_RATE': 2.0,    # rad/s max yaw rate
    'GAMEPAD_MAX_LEAN': math.radians(30), # max intentional lean angle
    'TARGET_MARKER_HEIGHT': 0.3,    # m height of the visual target marker
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
# TRIPLET LEAN CONTROLLER
# ============================================================================

class TripletController:
    """
    PD position controller for a triplet hub joint.

    Maintains the triplet at `target_angle` (rad, relative to body) using
    PD control plus gravity compensation for the body mass above the hub.

    Gravity compensation
    --------------------
    The body's CoG is at height `l_cog` above the triplet hub axis.
    When the body is pitched at angle θ (measured_pitch, world-frame),
    the body weight creates a torque about the hub Y-axis:

        τ_grav = m_body · g · l_cog · sin(θ)

    This term is independent of the triplet joint angle (which only
    changes where the wheels sit relative to the body, not where the
    body CoG is in world frame).

    Output torque is `triplet_cmd` in:
        triplet_total = -motor_torque + triplet_cmd
    and is therefore additive with the drive-motor reaction cancellation.
    """

    def __init__(self, config):
        self.kp = config.get('TRIPLET_LEAN_KP', 8.0)
        self.kd = config.get('TRIPLET_LEAN_KD', 0.4)
        self.target_angle = config.get('INITIAL_TRIPLET_ANGLE', 0.0)
        self.g = abs(config.get('GRAVITY', 9.81))

        # Body parameters for gravity compensation
        self._m_body = config['LQR_BODY_MASS']     # kg  (2.7167)
        self._l_cog  = config['LQR_COG_HEIGHT']    # m   (0.247)

        tau_max_grav = self._m_body * self.g * self._l_cog
        print(f"  TripletController: Kp={self.kp}, Kd={self.kd}, "
              f"target={math.degrees(self.target_angle):.1f}\u00b0")
        print(f"    Body grav-comp: m={self._m_body:.3f}kg, "
              f"l_cog={self._l_cog:.3f}m → "
              f"τ_max={tau_max_grav:.3f} Nm (at 90°)")

    # ------------------------------------------------------------------

    def update(self, angle, rate, body_pitch):
        """
        Compute triplet hub torque.

        Args:
            angle:       triplet joint angle (rad, relative to body)
            rate:        triplet joint angular velocity (rad/s)
            body_pitch:  current body pitch in world frame (rad)

        Returns:
            torque (Nm) to apply at the triplet hub joint
        """
        # PD term — drives joint toward target_angle, damps velocity
        tau_pd = self.kp * (self.target_angle - angle) - self.kd * rate

        # Gravity compensation for body mass above the hub.
        # Body CoG is at l_cog along body Z.  When the body is pitched
        # at angle body_pitch, its weight creates a torque about the hub:
        # divided by 2 because the triplet shares the load with the other side.
        tau_grav = self._m_body * self.g * self._l_cog * math.sin(body_pitch) / 2

        return tau_pd + tau_grav

    def set_target(self, angle_rad):
        """Override the target triplet joint angle (rad)."""
        self.target_angle = angle_rad


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
        self._set_initial_pose()
        self._configure_dynamics()
        self._setup_belt_constraints()

        # Balance controller
        ctrl_type = config.get('CONTROLLER', 'lqr').lower()
        if ctrl_type == 'mpc':
            self.controller = MPCHybridController(config)
        elif ctrl_type == 'lqr':
            self.controller = LQRBalanceController(config)
        else:
            self.controller = BalanceController(config)

        # Two motors (one per side)
        self.motors = [BrushlessMotorModel(config), BrushlessMotorModel(config)]

        # Triplet lean PD controller (holds hub at INITIAL_TRIPLET_ANGLE)
        self.triplet_ctrl = TripletController(config)

        # IMU sensor model
        self.imu = IMUSensorModel(config)

        # Estimated wheel radius (may be overridden from AABB after loading)
        self.wheel_radius = config['WHEEL_RADIUS']

        # Current state for logging
        # NOTE: position is accumulated by integrating forward velocity
        # projected onto the robot's current heading, so it stays valid
        # after yaw rotations (unlike a raw world-X projection).
        self.position = 0.0
        self.pitch_angle = 0.0
        self.pitch_rate = 0.0
        self.actual_torques = [0.0, 0.0]

    # ----------------------------------------------------------------
    # Robot setup
    # ----------------------------------------------------------------

    def _set_initial_pose(self):
        """Set initial triplet angles (0° = 4WD, 60° = 2WD)."""
        trip_angle = self.cfg.get('INITIAL_TRIPLET_ANGLE', 0.0)
        if abs(trip_angle) > 1e-6:
            p.resetJointState(self.body_id, self.l_triplet_joint, trip_angle, 0.0)
            p.resetJointState(self.body_id, self.r_triplet_joint, trip_angle, 0.0)
        mode = '2WD' if abs(trip_angle - math.pi / 3) < 0.05 else ('4WD' if abs(trip_angle) < 0.05 else 'Lean')
        print(f"  Initial triplet angle: {math.degrees(trip_angle):.1f}° ({mode} mode)")

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

        # Triplet hubs: low friction + joint damping (simulates motor back-EMF)
        trip_damping = self.cfg.get('TRIPLET_JOINT_DAMPING', 0.05)
        for tj in [self.l_triplet_joint, self.r_triplet_joint]:
            p.changeDynamics(self.body_id, tj,
                             lateralFriction=self.cfg['TRIPLET_FRICTION'],
                             linearDamping=0.0,
                             angularDamping=0.0,
                             jointDamping=trip_damping)

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
        NOT USED — position is now integrated in update().
        Left here for reference only.
        """
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return -pos[0]

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

        # --- Yaw rate (body-frame) for yaw damping ---
        # Negated so that positive yaw_rate = turning right (from behind)
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id)
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        rot = p.getMatrixFromQuaternion(orn)
        yaw_rate = -(rot[2] * ang_vel[0] + rot[5] * ang_vel[1] + rot[8] * ang_vel[2])

        # --- Forward odometry: integrate velocity projected onto heading ---
        # Robot forward is body -X.  Project world linear velocity onto that
        # axis so position accumulates correctly after any yaw rotation.
        body_fwd_x = -rot[0]   # body -X in world X
        body_fwd_y = -rot[3]   # body -X in world Y
        fwd_vel = lin_vel[0] * body_fwd_x + lin_vel[1] * body_fwd_y
        self.position += fwd_vel * dt

        # --- Triplet encoders → controller (for MPC) ---
        lt_state = p.getJointState(self.body_id, self.l_triplet_joint)
        rt_state = p.getJointState(self.body_id, self.r_triplet_joint)
        if hasattr(self.controller, 'set_triplet_state'):
            self.controller.set_triplet_state(
                lt_state[0], rt_state[0],   # angles
                lt_state[1], rt_state[1],   # rates
            )

        # --- Controller → per-side commanded torques ---
        left_cmd, right_cmd = self.controller.update(
            measured_pitch, measured_pitch_rate,
            self.position, yaw_rate, sim_time, dt
        )

        # Triplet hub commands:
        #   MPC plans its own triplet torques (flip/2WD logic) → use those.
        #   LQR / PID have no triplet plan → use the lean PD controller to
        #   hold the hub at INITIAL_TRIPLET_ANGLE (keeps wheels on the ground
        #   in the configured 4WD/2WD geometry regardless of body pitch).
        if hasattr(self.controller, 'triplet_torque_L'):
            triplet_cmd_L = self.controller.triplet_torque_L
            triplet_cmd_R = self.controller.triplet_torque_R
        else:
            triplet_cmd_L = self.triplet_ctrl.update(
                lt_state[0], lt_state[1], self.pitch_angle)
            triplet_cmd_R = self.triplet_ctrl.update(
                rt_state[0], rt_state[1], self.pitch_angle)

        # --- Apply motor torque through motor models ---
        # The motor stator is mounted on the BODY, driving the wheel shaft
        # through the free-spinning triplet hub bearing.  In the URDF chain
        # (body → triplet → wheel), we must apply the same motor torque to:
        #   1. Wheel joints  (+τ on wheels, −τ reaction on triplet)
        #   2. Triplet joint  (+τ on triplet, −τ reaction on body)
        # Net: wheels +τ, triplet 0 (free), body −τ (motor reaction).
        #
        # Left motor (+yaw_correction), Right motor (−yaw_correction)
        side_configs = [
            (0, self.l_wheel_joints, self.l_triplet_joint, left_cmd, triplet_cmd_L),
            (1, self.r_wheel_joints, self.r_triplet_joint, right_cmd, triplet_cmd_R),
        ]

        for motor_idx, wheel_joints, triplet_joint, cmd_torque, triplet_cmd in side_configs:
            # Representative wheel velocity (belt-coupled, all same)
            wheel_vel = p.getJointState(self.body_id, wheel_joints[0])[1]

            # Motor produces total torque for this side
            motor_torque = self.motors[motor_idx].update(
                cmd_torque, wheel_vel, dt
            )
            self.actual_torques[motor_idx] = motor_torque

            # --- Triplet hub joint ---
            # Two torque components act on the triplet joint:
            #  1. Drive motor reaction: -motor_torque transfers wheel-joint
            #     reaction to the body (keeps the triplet hub free-spinning).
            #  2. Triplet motor command: +triplet_cmd actively rotates the
            #     triplet relative to the body (MPC-planned, 0 for PID/LQR).
            triplet_total = -motor_torque + triplet_cmd
            p.setJointMotorControl2(
                self.body_id, triplet_joint,
                controlMode=p.TORQUE_CONTROL,
                force=triplet_total
            )

            # --- Wheel joints: distribute torque among 3 belt-coupled wheels ---
            # Negate because URDF wheel axis is +Y, whereas bullet_sim's
            # effective axis is -Y; same PID sign needs opposite joint torque.
            torque_per_wheel = -motor_torque / 3.0

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
            'triplet_ang': (lt_state[0], rt_state[0]),
            'triplet_vel': (lt_state[1], rt_state[1]),
            'wheel_vel': (lw_vel, rw_vel),
        }

    def get_world_pose_2d(self):
        """
        Return (x, y, yaw, fwd_x, fwd_y) in world frame.
        Robot forward is body -X (URDF convention).
        """
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        rot = p.getMatrixFromQuaternion(orn)
        # Body -X in world = (-rot[0], -rot[3], -rot[6])
        fwd_x = -rot[0]
        fwd_y = -rot[3]
        yaw = math.atan2(fwd_y, fwd_x)
        return pos[0], pos[1], yaw, fwd_x, fwd_y

    def check_fallen(self):
        """Check if robot has fallen over (|pitch| > 80°)."""
        true_pitch, _ = self._get_true_state()
        return abs(true_pitch) > math.radians(80)


# ============================================================================
# SIMULATION MAIN LOOP
# ============================================================================

def run_simulation():
    """Run the tribot self-balancing simulation."""

    print("=" * 70)
    print("Tribot Self-Balancing Robot — URDF-based Simulation")
    print("=" * 70)

    physics_client = p.connect(p.GUI, options="--width=1920 --height=1080 --maximized")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, CONFIG['GRAVITY'])
    p.setPhysicsEngineParameter(fixedTimeStep=CONFIG['TIMESTEP'], numSubSteps=1)

    # Terrain
    ground_ids = create_terrain(CONFIG)

    # Create robot
    print("\nLoading tribot URDF...")
    robot = TribotBalanceBot(physics_client, CONFIG)

    # Camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2, cameraYaw=0, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.15]
    )

    # Print configuration summary
    total_mass = sum(p.getDynamicsInfo(robot.body_id, i)[0]
                     for i in range(-1, p.getNumJoints(robot.body_id)))
    ctrl_type = CONFIG.get('CONTROLLER', 'lqr').upper()
    print(f"\nRobot total mass: {total_mass:.3f} kg")
    print(f"Controller: {ctrl_type}")
    if ctrl_type == 'PID':
        print(f"  Inner PID (pitch\u2192torque): Kp={CONFIG['PID_KP']}, "
              f"Ki={CONFIG['PID_KI']}, Kd={CONFIG['PID_KD']}")
        print(f"  Outer PID (pos\u2192pitch):   Kp={CONFIG['POS_PID_KP']}, "
              f"Ki={CONFIG['POS_PID_KI']}, Kd={CONFIG['POS_PID_KD']}, "
              f"max_pitch={math.degrees(CONFIG['POS_PID_MAX_PITCH']):.1f}\u00b0")
    elif ctrl_type == 'MPC':
        print(f"  MPC rate: {CONFIG['MPC_RATE_HZ']}Hz, N={CONFIG['MPC_HORIZON']}, "
              f"sim_solve={CONFIG['MPC_SIMULATED_SOLVE_MS']}ms")
        print(f"  MPC Q_diag={CONFIG['MPC_Q_DIAG']}")
        print(f"  MPC R_diag={CONFIG['MPC_R_DIAG']}")
        print(f"  Plant: m_body={CONFIG['LQR_BODY_MASS']}kg, "
              f"m_wheel={CONFIG['LQR_WHEEL_MASS']}kg, "
              f"l_cog={CONFIG['LQR_COG_HEIGHT']}m, "
              f"I_body={CONFIG['LQR_BODY_INERTIA']}kg\u00b7m\u00b2")
    else:
        print(f"  Q_diag={CONFIG['LQR_Q_DIAG']}, R={CONFIG['LQR_R']}")
        print(f"  Plant: m_body={CONFIG['LQR_BODY_MASS']}kg, "
              f"m_wheel={CONFIG['LQR_WHEEL_MASS']}kg, "
              f"l_cog={CONFIG['LQR_COG_HEIGHT']}m, "
              f"I_body={CONFIG['LQR_BODY_INERTIA']}kg\u00b7m\u00b2")
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

    # Gamepad
    gp = Gamepad(CONFIG['GAMEPAD_DEVICE'], deadzone=CONFIG['GAMEPAD_DEADZONE'])
    if gp.connected:
        print(f"Gamepad: right stick Y (axis {CONFIG['GAMEPAD_SPEED_AXIS']}) = distance, "
              f"X (axis {CONFIG['GAMEPAD_YAW_AXIS']}) = yaw")

    # PlotJuggler real-time streaming
    pj = PlotJugglerStreamer()   # UDP → 127.0.0.1:9870
    print("PlotJuggler UDP streamer active on 127.0.0.1:9870")

    # Visual target marker (vertical debug line)
    marker_id = -1
    marker_color = [0.0, 1.0, 0.0]   # green
    marker_h = CONFIG['TARGET_MARKER_HEIGHT']

    # Target position (1-D, robot forward axis). Latched when stick is idle.
    target_pos = 0.0
    # World-frame marker position (latched alongside target_pos)
    marker_world = [0.0, 0.0]

    sim_time = 0.0
    log_interval = 0.1
    last_log_time = 0.0

    while sim_time < CONFIG['SIM_DURATION']:
        # --- Gamepad input ---
        gp.poll()
        if gp.connected:
            # Right stick Y → forward distance offset (push up = negative axis = in front)
            forward_offset = -gp.axis(CONFIG['GAMEPAD_SPEED_AXIS']) * CONFIG['GAMEPAD_MAX_DISTANCE']

            # Right stick X → yaw rate command
            yaw_cmd = gp.axis(CONFIG['GAMEPAD_YAW_AXIS']) * CONFIG['GAMEPAD_MAX_YAW_RATE']
            robot.controller.set_yaw_rate(yaw_cmd)

            # Left stick Y → intentional lean command
            # Push up (negative axis) = lean forward (positive pitch offset).
            # set_lean() adjusts the pitch reference so LQR sees
            # (measured_pitch - requested_lean) and does not fight the lean.
            if hasattr(robot.controller, 'set_lean'):
                lean_cmd = gp.axis(CONFIG['GAMEPAD_LEAN_AXIS']) * CONFIG['GAMEPAD_MAX_LEAN']
                robot.controller.set_lean(lean_cmd)
                robot.triplet_ctrl.set_target(-lean_cmd)

            # Update target while stick is actively deflected;
            # when released, the last target stays fixed in world.
            # While turning (yaw active, forward idle), reset target to
            # current position so the robot doesn't chase a stale target.
            if abs(forward_offset) > 1e-4:
                target_pos = robot.position + forward_offset
                # Compute world-frame marker position
                rx, ry, _, fwd_x, fwd_y = robot.get_world_pose_2d()
                marker_world = [rx + forward_offset * fwd_x,
                                ry + forward_offset * fwd_y]
            elif abs(yaw_cmd) > 1e-4:
                target_pos = robot.position
                rx, ry, _, _, _ = robot.get_world_pose_2d()
                marker_world = [rx, ry]

            robot.controller.set_target_position(target_pos)

            # --- Update visual marker ---
            pt_from = [marker_world[0], marker_world[1], 0.0]
            pt_to   = [marker_world[0], marker_world[1], marker_h]
            if marker_id >= 0:
                marker_id = p.addUserDebugLine(
                    pt_from, pt_to, marker_color, lineWidth=3,
                    replaceItemUniqueId=marker_id)
            else:
                marker_id = p.addUserDebugLine(
                    pt_from, pt_to, marker_color, lineWidth=3)

        robot.update(sim_time, CONFIG['TIMESTEP'])
        p.stepSimulation()
        sim_time += CONFIG['TIMESTEP']

        # --- Stream signals to PlotJuggler ---
        ctrl = robot.controller
        true_pitch, true_pitch_rate = robot._get_true_state()
        _flip_diag = ctrl.get_flip_diagnostics() if hasattr(ctrl, 'get_flip_diagnostics') else {}
        pj.send({
            "timestamp": sim_time,
            # State signals
            "pos_err": float(ctrl.state_error[0]),
            "vel_est": float(ctrl.state_error[1]),
            "pitch_meas": float(ctrl.state_error[2]),
            "pitch_rate_meas": float(ctrl.state_error[3]),
            "true_pitch": true_pitch,
            "true_pitch_rate": true_pitch_rate,
            "measured_pitch": float(robot.pitch_angle),
            # Torque signals
            "torque_cmd": float(ctrl.control_torque),
            "torque_L_actual": float(robot.actual_torques[0]),
            "torque_R_actual": float(robot.actual_torques[1]),
            # Per-state LQR contributions (K_i * x_i)
            "K_pos": float(ctrl.K_contributions[0]),
            "K_vel": float(ctrl.K_contributions[1]),
            "K_pitch": float(ctrl.K_contributions[2]),
            "K_pitch_rate": float(ctrl.K_contributions[3]),
            # Gain-scheduled LQR mode (1=aggressive, 0=normal)
            "lqr_aggressive": float(getattr(ctrl, 'aggressive_active', False)),
            # Targets
            "target_pos": float(ctrl.target_position),
            "target_lean": float(getattr(ctrl, 'target_lean', 0.0)),
            "position": float(robot.position),
            # MPC diagnostics (only meaningful when controller is MPC)
            **({
                "mpc_solve_count": int(ctrl.mpc_solve_count),
                "mpc_last_wall_ms": float(ctrl.mpc_last_wall_ms),
                "mpc_max_wall_ms": float(ctrl.mpc_max_wall_ms),
                "mpc_target_pitch": float(ctrl.target_pitch),
                "mpc_ff_drive_L": float(ctrl.K_contributions[0]),
                "mpc_ff_drive_R": float(ctrl.K_contributions[1]),
                "mpc_pd_drive_L": float(ctrl.K_contributions[2]),
                "mpc_pd_drive_R": float(ctrl.K_contributions[3]),
                "mpc_triplet_torque_L": float(ctrl.triplet_torque_L),
                "mpc_triplet_torque_R": float(ctrl.triplet_torque_R),
                "mpc_triplet_angle_L": float(ctrl._triplet_angle_L),
                "mpc_triplet_angle_R": float(ctrl._triplet_angle_R),
                "mpc_triplet_dev_L": float(ctrl.x_est[ctrl.IDX_TRIP_L]),
                "mpc_triplet_dev_R": float(ctrl.x_est[ctrl.IDX_TRIP_R]),
            } if hasattr(ctrl, 'mpc_solve_count') else {}),
            # ZMP / DCM flip diagnostics (only when MPC controller is active)
            **({
                "zmp_phase":       int(_flip_diag['zmp/phase']),
                "zmp_dcm":         float(_flip_diag['zmp/dcm']),
                "zmp_dcm_max":     float(_flip_diag['zmp/dcm_max']),
                "zmp_dcm_trigger": float(_flip_diag['zmp/dcm_trigger']),
                "zmp_t_capture":   float(_flip_diag['zmp/t_capture']),
                "zmp_urgency":     float(_flip_diag['zmp/urgency']),
                "zmp_eq_deg":      float(_flip_diag['zmp/eq_angle_deg']),
            } if _flip_diag else {}),
        })

        if robot.check_fallen():
            print(f"\n[{sim_time:.2f}s] Robot fell over!")
            break

        if sim_time - last_log_time >= log_interval:
            dbg = robot.get_debug_state()
            rx, ry, rz = dbg['euler_deg']
            tv = dbg['triplet_vel']
            wv = dbg['wheel_vel']
            ta = dbg.get('triplet_ang', (0.0, 0.0))
            at0, at1 = robot.actual_torques
            print(
                f"[{sim_time:5.2f}s] "
                # f"Euler(r={rx:6.1f} p={ry:6.1f} y={rz:6.1f})° | "
                f"Euler(p={ry:6.1f})° | "
                # f"Pos:{robot.position:6.3f}m "
                f"TgtPitch:{math.degrees(robot.controller.target_pitch):5.2f}° | "
                f"TripAng({math.degrees(ta[0]):5.1f},{math.degrees(ta[1]):5.1f})° "
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

    pj.close()
    print("\nClose the PyBullet window to exit.")
    while p.isConnected(physics_client):
        time.sleep(0.01)
    p.disconnect()


if __name__ == "__main__":
    run_simulation()
