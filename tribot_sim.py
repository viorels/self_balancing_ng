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
from enum import Enum
import numpy as np
import pybullet as p
import pybullet_data

from robot_state import DriveMode, RobotState, ControlOutput, ControlGoals, Telemetry

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
    'TERRAIN': 'flat',

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
    'LQR_AGGRESSIVE_R': 1.0,    # should be half of LQR_R or less for a noticeable effect
    'LQR_SWITCH_THRESHOLD': 0.20,     # m — switch to aggressive when |error| > this
    'LQR_SWITCH_HYSTERESIS': 0.05,    # m — switch back when |error| < threshold - hyst

    # === TRIPLET LEAN PID CONTROLLER ===
    # Holds the triplet hub at INITIAL_TRIPLET_ANGLE using torque control.
    'TRIPLET_LEAN_KP': 8.0,     # Nm/rad  — proportional gain
    'TRIPLET_LEAN_KD': 0.4,     # Nm·s/rad — derivative (damping) gain
    # Gravity compensation feedforward (per side):
    #   τ_ff = K · sin(body_pitch + triplet_angle)
    # 4WD: K ≈ (m_total/2)·g·z_contact = 1.70·9.81·0.118 ≈ 1.97 Nm
    #   Two grounded wheels create asymmetric normal forces when body pitches;
    #   the net torque about the hub grows with sin(θ).  "Bilateral support"
    #   cancels the cos(θ) component, leaving the sin(θ) term.
    # 2WD: K ≈ m_trip·g·R = 0.335·9.81·0.12 ≈ 0.39 Nm
    #   Single grounded wheel, restoring torque from contact offset.
    'TRIPLET_GRAV_COMP_4WD': 2.0,  # Nm — gravity comp gain (per-side, 4WD)
    'TRIPLET_GRAV_COMP_2WD': 2.0,  # Nm — gravity comp gain (per-side, 2WD)

    # Lean compensation geometry (sine theorem):
    #   β = α + arcsin((h/l) sin α)  where
    #     α = body lean angle (rad, from vertical)
    #     h = distance from triplet hub to body CoG (m)
    #     l = triplet foot length (hub to wheel contact, m)
    # Valid for |α| < arcsin(l/h) ≈ 29°.
    # 4WD uses a simpler linear scale (small lean corrections).
    'TRIPLET_4WD_LEAN_SCALE': 1.0 / 1.7,  # ≈ 0.59 (linear approx, 4WD only)
    'TRIPLET_2WD_COG_DIST': 0.19,        # m — h: hub-to-CoG (whole robot, lower than c_body alone)
    'TRIPLET_2WD_FOOT_LENGTH': 0.12,      # m — l: hub-to-wheel distance

    # Nonlinear triplet balance assist (dead-zoned quadratic, rate-gated)
    'TRIPLET_ASSIST_GAIN': 4.0,       # Nm/rad² — quadratic gain beyond deadzone
    'TRIPLET_ASSIST_DEADZONE': 0.15,  # rad (~8.6°) — no assist below this pitch
    'TRIPLET_ASSIST_MAX': 3.0,        # Nm — clamp (triplet motor limit is 5 Nm)
    'TRIPLET_ASSIST_TAU': 0.1,        # s — EMA time constant for assist smoothing (~25 Hz cutoff)

    # === GAMEPAD ===
    'GAMEPAD_DEVICE': '/dev/input/js0',
    'GAMEPAD_DEADZONE': 0.08,
    'GAMEPAD_SPEED_AXIS': 4,        # Right stick Y
    'GAMEPAD_YAW_AXIS': 3,          # Right stick X
    'GAMEPAD_LEAN_AXIS': 1,         # Left stick Y  (push up = lean forward)
    'GAMEPAD_MAX_DISTANCE': 1.0,    # m max target distance in front of robot
    'GAMEPAD_MAX_YAW_RATE': 2.0,    # rad/s max yaw rate
    'GAMEPAD_MAX_LEAN': math.radians(30), # max intentional lean angle
    'GAMEPAD_2WD_BUTTON': 4,         # LB (left bumper) on F710 (XInput)
    'TARGET_MARKER_HEIGHT': 0.3,    # m height of the visual target marker

    # === MODE SWITCH (4WD ↔ 2WD) ===
    'TRIPLET_2WD_ANGLE': math.pi / 3,  # 60° target for 2WD mode
}


# ============================================================================
# Extracted modules (Step 2 — same classes, moved to own files)
# ============================================================================

from motor_model import BrushlessMotorModel              # noqa: E402
from imu_model import IMUSensorModel                     # noqa: E402
from triplet_controller import (                         # noqa: E402
    TripletController,
    compute_triplet_from_pitch,
    compute_pitch_from_triplet,
)


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

        # Triplet lean PD controllers (one per side, holds hub at target angle)
        self.triplet_ctrl_L = TripletController(config)
        self.triplet_ctrl_R = TripletController(config)

        # Drive mode: '4wd' (two wheels/side) or '2wd' (one wheel/side)
        self.drive_mode = DriveMode.FOUR_WD
        self.triplet_base_angle = config.get('INITIAL_TRIPLET_ANGLE', 0.0)

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

        self.lean_offset_ema = 0.0

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
        #
        #   When LQR is active, it exposes `desired_lean` — the pitch angle
        #   it implicitly wants for position tracking.  Feed this to the
        #   triplet PD so it cooperates with (rather than fights) the lean.
        if self.controller.plans_triplet_torque:
            triplet_cmd_L = self.controller.triplet_torque_L
            triplet_cmd_R = self.controller.triplet_torque_R
        else:
            # Compute lean compensation on top of the mode-dependent base angle.
            # In 4WD the base is 0°; lean_comp uses a linear scale.
            # In 2WD the base is ±60°; lean_comp uses the exact sine-theorem
            # formula via compute_triplet_from_pitch().
            lean_comp = 0.0
            lean_offset = self.controller.desired_lean
            alpha = self.controller.target_pitch   # body lean from vertical
            if self.drive_mode == DriveMode.TWO_WD:
                beta = compute_triplet_from_pitch(alpha, h=self.cfg.get('TRIPLET_2WD_COG_DIST'))
                lean_comp = -beta # + lean_offset/2 # feed in some of the desired lean as an offset
            else:
                lean_scale = self.cfg.get('TRIPLET_4WD_LEAN_SCALE', 1.0 / 1.7)
                lean_comp = -alpha * lean_scale - lean_offset

            self.triplet_ctrl_L.set_base_angle(self.triplet_base_angle)
            self.triplet_ctrl_R.set_base_angle(self.triplet_base_angle)
            self.triplet_ctrl_L.set_target(self.triplet_base_angle + lean_comp)
            self.triplet_ctrl_R.set_target(self.triplet_base_angle + lean_comp)

            triplet_cmd_L = self.triplet_ctrl_L.update(
                lt_state[0], lt_state[1], self.pitch_angle,
                body_pitch_rate=self.pitch_rate, dt=dt)
            triplet_cmd_R = self.triplet_ctrl_R.update(
                rt_state[0], rt_state[1], self.pitch_angle,
                body_pitch_rate=self.pitch_rate, dt=dt)

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

    def toggle_drive_mode(self):
        """Toggle between 4WD and 2WD drive modes.

        In 4WD (triplet angle ≈ 0°), two wheels per side touch the ground.
        In 2WD (triplet angle ≈ ±60°), one wheel per side — active balance.

        Due to triplet 3-fold symmetry, both +60° and −60° are valid 2WD
        configurations.  We pick whichever is closer to the current triplet
        angle so the transition is always a short (~35°) rotation rather
        than a violent 120° flip.
        """
        angle_2wd = self.cfg.get('TRIPLET_2WD_ANGLE', math.pi / 3)
        if self.drive_mode == DriveMode.FOUR_WD:
            self.drive_mode = DriveMode.TWO_WD
            # Pick sign of 60° closest to current triplet angle
            cur_angle = p.getJointState(self.body_id, self.l_triplet_joint)[0]
            if cur_angle >= 0:
                self.triplet_base_angle = -angle_2wd
            else:
                self.triplet_base_angle = angle_2wd
            print(f"  [MODE] 4WD → 2WD  (triplet target "
                  f"{math.degrees(self.triplet_base_angle):+.0f}°)")
        else:
            self.drive_mode = DriveMode.FOUR_WD
            self.triplet_base_angle = 0.0
            print(f"  [MODE] 2WD → 4WD  (triplet target 0°)")
        self.triplet_ctrl_L.drive_mode = self.drive_mode
        self.triplet_ctrl_R.drive_mode = self.drive_mode

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

    # LT trigger state for rising-edge detection (4WD ↔ 2WD toggle)
    lt_was_pressed = False

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
            lean_cmd = gp.axis(CONFIG['GAMEPAD_LEAN_AXIS']) * CONFIG['GAMEPAD_MAX_LEAN']
            robot.controller.set_lean(lean_cmd)

            # LB (left bumper) → toggle 4WD ↔ 2WD on rising edge
            lb_pressed = gp.button(CONFIG['GAMEPAD_2WD_BUTTON'])
            if lb_pressed and not lt_was_pressed:
                robot.toggle_drive_mode()
            lt_was_pressed = lb_pressed

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
        ctrl_telem = ctrl.get_telemetry()
        pj.send({
            "timestamp": sim_time,
            # True state (not from controller — for reference only)
            "true_pitch": true_pitch,
            "true_pitch_rate": true_pitch_rate,
            # Actuator outputs
            "torque_L_actual": float(robot.actual_torques[0]),
            "torque_R_actual": float(robot.actual_torques[1]),
            # Triplet controller breakdown (for tuning visibility)
            "triplet_assist_L": float(robot.triplet_ctrl_L.last_assist_force),
            "triplet_assist_R": float(robot.triplet_ctrl_R.last_assist_force),
            "triplet_grav_comp_L": float(robot.triplet_ctrl_L.last_grav_comp),
            "triplet_grav_comp_R": float(robot.triplet_ctrl_R.last_grav_comp),
            # Drive mode (0=4WD, 1=2WD)
            "drive_mode": float(robot.drive_mode == DriveMode.TWO_WD),
            # Robot state
            "position": float(robot.position),
            # All controller-specific signals (prefixed by controller type)
            **{f"ctrl/{k}": v for k, v in ctrl_telem.items()},
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
