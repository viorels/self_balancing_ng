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
import pybullet as p
import pybullet_data

from robot_state import DriveMode, RobotState, ControlOutput, ControlGoals, Telemetry
from config import load_config

from controllers.control_pid import BalanceController
from controllers.control_lqr import LQRBalanceController
from controllers.control_mpc_hybrid import MPCHybridController
from input.gamepad import Gamepad
from input.input_manager import InputManager
from plotjuggler_udp import PlotJugglerStreamer
from terrain import create_terrain
from motor_model import BrushlessMotorModel
from imu_model import IMUSensorModel
from triplet_controller import (
    TripletController,
    compute_triplet_from_pitch,
    compute_pitch_from_triplet,
)


# ============================================================================
# CONFIGURATION — typed dataclass, dict-like backward compat via shim
# ============================================================================

CONFIG = load_config()


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

        # Sensor state (populated by read_sensors() on each tick)
        self.state = RobotState(drive_mode=self.drive_mode,
                                triplet_base_angle=self.triplet_base_angle)

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
    # Sensor pipeline
    # ----------------------------------------------------------------

    def read_sensors(self, sim_time, dt):
        """
        Read all sensors and return a populated RobotState.

        This method is the single source of truth for measured / estimated
        quantities. Controllers and the telemetry loop consume the returned
        RobotState rather than reaching into PyBullet directly.
        """
        # --- Ground truth from physics ---
        true_pitch, true_pitch_rate = self._get_true_state()

        # --- IMU-fused measurements ---
        measured_pitch, measured_pitch_rate = self.imu.read(
            true_pitch, true_pitch_rate, sim_time, dt
        )
        # Keep legacy attributes in sync (used by triplet PD, logging)
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

        # --- Triplet encoders ---
        lt_state = p.getJointState(self.body_id, self.l_triplet_joint)
        rt_state = p.getJointState(self.body_id, self.r_triplet_joint)

        # --- Wheel velocities (one representative per side, belt-coupled) ---
        wheel_vel_L = p.getJointState(self.body_id, self.l_wheel_joints[0])[1]
        wheel_vel_R = p.getJointState(self.body_id, self.r_wheel_joints[0])[1]

        state = RobotState(
            sim_time=sim_time,
            dt=dt,
            pitch=measured_pitch,
            pitch_rate=measured_pitch_rate,
            yaw_rate=yaw_rate,
            true_pitch=true_pitch,
            true_pitch_rate=true_pitch_rate,
            position=self.position,
            forward_velocity=fwd_vel,
            triplet_angle_L=lt_state[0],
            triplet_angle_R=rt_state[0],
            triplet_rate_L=lt_state[1],
            triplet_rate_R=rt_state[1],
            wheel_velocity_L=wheel_vel_L,
            wheel_velocity_R=wheel_vel_R,
            drive_mode=self.drive_mode,
            triplet_base_angle=self.triplet_base_angle,
        )
        self.state = state
        return state

    # ----------------------------------------------------------------
    # Control update
    # ----------------------------------------------------------------

    def update(self, sim_time, dt):
        """
        Run one control + actuation cycle.

        Reads sensors via read_sensors(), feeds the controller, computes
        triplet commands, and applies motor torques.
        Called every physics timestep; controller only runs at CONTROL_RATE_HZ.
        """
        s = self.read_sensors(sim_time, dt)

        # --- Feed triplet state to controller (for MPC) ---
        self.controller.set_triplet_state(
            s.triplet_angle_L, s.triplet_angle_R,
            s.triplet_rate_L, s.triplet_rate_R,
        )

        # --- Controller → per-side commanded torques ---
        left_cmd, right_cmd = self.controller.update(
            s.pitch, s.pitch_rate,
            s.position, s.yaw_rate, sim_time, dt
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
                s.triplet_angle_L, s.triplet_rate_L, s.pitch,
                body_pitch_rate=s.pitch_rate, dt=dt)
            triplet_cmd_R = self.triplet_ctrl_R.update(
                s.triplet_angle_R, s.triplet_rate_R, s.pitch,
                body_pitch_rate=s.pitch_rate, dt=dt)

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
            (0, self.l_wheel_joints, self.l_triplet_joint, left_cmd, triplet_cmd_L, s.wheel_velocity_L),
            (1, self.r_wheel_joints, self.r_triplet_joint, right_cmd, triplet_cmd_R, s.wheel_velocity_R),
        ]

        for motor_idx, wheel_joints, triplet_joint, cmd_torque, triplet_cmd, wheel_vel in side_configs:

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

    # Gamepad + InputManager
    gp = Gamepad(CONFIG['GAMEPAD_DEVICE'], deadzone=CONFIG['GAMEPAD_DEADZONE'])
    inp = InputManager(gp, CONFIG)
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

    sim_time = 0.0
    log_interval = 0.1
    last_log_time = 0.0

    while sim_time < CONFIG['SIM_DURATION']:
        # --- Input ---
        goals, mode_toggle, marker = inp.update(
            robot.position, robot.get_world_pose_2d()
        )
        robot.controller.set_target_position(goals.target_position)
        robot.controller.set_yaw_rate(goals.yaw_rate)
        robot.controller.set_lean(goals.pitch_bias)
        if mode_toggle:
            robot.toggle_drive_mode()

        # --- Update visual marker ---
        if inp.connected:
            pt_from = [marker.x, marker.y, 0.0]
            pt_to   = [marker.x, marker.y, marker_h]
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
        s = robot.state
        ctrl_telem = ctrl.get_telemetry()
        pj.send({
            "timestamp": sim_time,
            # True state (not from controller — for reference only)
            "true_pitch": s.true_pitch,
            "true_pitch_rate": s.true_pitch_rate,
            # Actuator outputs
            "torque_L_actual": float(robot.actual_torques[0]),
            "torque_R_actual": float(robot.actual_torques[1]),
            # Triplet controller breakdown (for tuning visibility)
            "triplet_assist_L": float(robot.triplet_ctrl_L.last_assist_force),
            "triplet_assist_R": float(robot.triplet_ctrl_R.last_assist_force),
            "triplet_grav_comp_L": float(robot.triplet_ctrl_L.last_grav_comp),
            "triplet_grav_comp_R": float(robot.triplet_ctrl_R.last_grav_comp),
            # Drive mode (0=4WD, 1=2WD)
            "drive_mode": float(s.drive_mode == DriveMode.TWO_WD),
            # Robot state from sensor pipeline
            "position": s.position,
            "forward_velocity": s.forward_velocity,
            "pitch": s.pitch,
            "pitch_rate": s.pitch_rate,
            "yaw_rate": s.yaw_rate,
            "triplet_angle_L": s.triplet_angle_L,
            "triplet_angle_R": s.triplet_angle_R,
            "wheel_velocity_L": s.wheel_velocity_L,
            "wheel_velocity_R": s.wheel_velocity_R,
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
