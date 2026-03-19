"""
TribotBalanceBot — URDF loading, sensors, actuators, belt constraints.

This module owns the physical robot representation: loading the URDF,
discovering and configuring joints, reading sensors into a RobotState,
and applying ControlOutput torques to the physics engine.

No control logic lives here — only the robot's interface to PyBullet.
"""

import math
import os
import tempfile
import xml.etree.ElementTree as ET

import pybullet as p

from robot_state import DriveMode, RobotState
from motor_model import BrushlessMotorModel
from imu_model import IMUSensorModel
from triplet_controller import (
    TripletController,
    compute_triplet_from_pitch,
)
from controllers.base import StateReference
from controllers.control_pid import BalanceController
from controllers.control_lqr import LQRBalanceController
from controllers.control_mpc_hybrid import MPCHybridController
from controllers.lean_trajectory import LeanTrajectory


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
        ctrl_type = config.sim.controller.lower()
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
        self.triplet_base_angle = config.sim.initial_triplet_angle

        # Lean transition trajectory planner (2WD only).
        # Generates smooth state references so the LQR tracks a feasible
        # path instead of a step when the operator commands a lean change.
        self._lean_traj = LeanTrajectory(
            h_cog=config.triplet.cog_dist_2wd,
            min_duration=0.2,
            max_duration=1.0,
            lean_per_sec_factor=0.25,
        )
        self._prev_requested_lean = 0.0

        # IMU sensor model
        self.imu = IMUSensorModel(config)

        # Estimated wheel radius (may be overridden from AABB after loading)
        self.wheel_radius = config.robot.wheel_radius

        # Current state for logging
        # NOTE: position is accumulated by integrating forward velocity
        # projected onto the robot's current heading, so it stays valid
        # after yaw rotations (unlike a raw world-X projection).
        self.position = 0.0
        self.pitch_angle = 0.0
        self.pitch_rate = 0.0
        self.actual_torques = [0.0, 0.0]

        # Sensor state (populated by read_sensors() on each tick)
        self.state = RobotState(drive_mode=self.drive_mode,
                                triplet_base_angle=self.triplet_base_angle)

    # ----------------------------------------------------------------
    # Robot setup
    # ----------------------------------------------------------------

    def _set_initial_pose(self):
        """Set initial triplet angles (0° = 4WD, 60° = 2WD)."""
        trip_angle = self.cfg.sim.initial_triplet_angle
        if abs(trip_angle) > 1e-6:
            p.resetJointState(self.body_id, self.l_triplet_joint, trip_angle, 0.0)
            p.resetJointState(self.body_id, self.r_triplet_joint, trip_angle, 0.0)
        mode = '2WD' if abs(trip_angle - math.pi / 3) < 0.05 else ('4WD' if abs(trip_angle) < 0.05 else 'Lean')
        print(f"  Initial triplet angle: {math.degrees(trip_angle):.1f}° ({mode} mode)")

    def reset(self):
        """Reset the robot to its initial upright pose with zero velocities.

        Intended for AI-driven experiments: call between trials to start fresh
        without restarting the simulation process.
        """
        # --- Restore base pose ---
        init_pos = [0, 0, self.cfg.sim.initial_height]
        init_orn = p.getQuaternionFromEuler([0, self.cfg.sim.initial_pitch, 0])
        p.resetBasePositionAndOrientation(self.body_id, init_pos, init_orn)
        p.resetBaseVelocity(self.body_id,
                            linearVelocity=[0, 0, 0],
                            angularVelocity=[0, 0, 0])

        # --- Restore all joint states ---
        num_joints = p.getNumJoints(self.body_id)
        trip_angle = self.cfg.sim.initial_triplet_angle
        for i in range(num_joints):
            if i in (self.l_triplet_joint, self.r_triplet_joint):
                p.resetJointState(self.body_id, i, trip_angle, 0.0)
            else:
                p.resetJointState(self.body_id, i, 0.0, 0.0)

        # --- Drain gear-constraint solver warmstart before any physics steps.
        #
        # PyBullet's JOINT_GEAR constraints keep internal Lagrange-multiplier
        # state (warmstart) across ticks.  After resetJointState the positions
        # and velocities are zero, but the solver's internal force estimate is
        # still whatever it was on the last tick before reset.  On the very
        # first step after reset those stale forces are applied in full,
        # launching the robot skyward.
        #
        # Fix: temporarily disable all wheel actuators (force=0) and step the
        # simulation ~20 times so the solver converges to zero forces before
        # we hand control back to the balance controller.
        all_wheel_joints = self.l_wheel_joints + self.r_wheel_joints
        for ji in all_wheel_joints:
            p.setJointMotorControl2(
                self.body_id, ji,
                p.VELOCITY_CONTROL,
                targetVelocity=0, force=0)
        for _ in range(20):
            p.stepSimulation()
        # Re-enable default friction damping on wheel joints.
        for ji in all_wheel_joints:
            p.setJointMotorControl2(
                self.body_id, ji,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=self.cfg.robot.belt_max_force)

        # --- Reset motor first-order lag models ---
        self.motors = [
            type(self.motors[0])(self.cfg),
            type(self.motors[1])(self.cfg),
        ]

        # --- Reset IMU ---
        self.imu = type(self.imu)(self.cfg)

        # --- Reset software state ---
        self.position = 0.0
        self.pitch_angle = 0.0
        self.pitch_rate = 0.0
        self.actual_torques = [0.0, 0.0]
        self.drive_mode = DriveMode.FOUR_WD
        self.triplet_base_angle = self.cfg.sim.initial_triplet_angle
        self.state = RobotState(drive_mode=self.drive_mode,
                                triplet_base_angle=self.triplet_base_angle)
        self.triplet_ctrl_L.drive_mode = self.drive_mode
        self.triplet_ctrl_R.drive_mode = self.drive_mode

        # --- Reset controller integrators (best-effort) ---
        ctrl = self.controller
        for attr in ('_integral', '_pos_integral', 'x_hat',
                     '_prev_error', '_prev_pos_error'):
            if hasattr(ctrl, attr):
                import numpy as np
                val = getattr(ctrl, attr)
                if hasattr(val, 'shape'):
                    setattr(ctrl, attr, np.zeros_like(val))
                else:
                    setattr(ctrl, attr, 0.0)
        ctrl.set_target_position(0.0)
        ctrl.set_yaw_rate(0.0)
        ctrl.set_lean(0.0)

        print("[reset] Robot pose and state restored to initial conditions.")

    def _load_robot(self):
        """Load the URDF with preprocessed paths."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, self.cfg.sim.urdf_path)

        temp_urdf = preprocess_urdf(urdf_path)
        try:
            self.body_id = p.loadURDF(
                temp_urdf,
                basePosition=[0, 0, self.cfg.sim.initial_height],
                baseOrientation=p.getQuaternionFromEuler(
                    [0, self.cfg.sim.initial_pitch, 0]
                ),
                useFixedBase=False,
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
        trip_damping = self.cfg.robot.triplet_joint_damping
        for tj in [self.l_triplet_joint, self.r_triplet_joint]:
            p.changeDynamics(self.body_id, tj,
                             lateralFriction=self.cfg.robot.triplet_friction,
                             linearDamping=0.0,
                             angularDamping=0.0,
                             jointDamping=trip_damping)

        # Drive wheels: high friction for traction
        for wj in self.l_wheel_joints + self.r_wheel_joints:
            p.changeDynamics(self.body_id, wj,
                             lateralFriction=self.cfg.robot.wheel_friction,
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
                p.changeConstraint(c, gearRatio=-1,
                                   maxForce=self.cfg.robot.belt_max_force)
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

        rot = p.getMatrixFromQuaternion(orn)
        body_up_z = rot[8]
        body_fwd_z = rot[6]
        pitch = math.atan2(-body_fwd_z, body_up_z)

        # Pitch rate = angular velocity projected onto body Y axis
        pitch_rate = rot[1] * ang_vel[0] + rot[4] * ang_vel[1] + rot[7] * ang_vel[2]

        return pitch, pitch_rate

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
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id)
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        rot = p.getMatrixFromQuaternion(orn)
        yaw_rate = -(rot[2] * ang_vel[0] + rot[5] * ang_vel[1] + rot[8] * ang_vel[2])

        # --- Forward odometry: integrate velocity projected onto heading ---
        body_fwd_x = -rot[0]
        body_fwd_y = -rot[3]
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
    # State reference
    # ----------------------------------------------------------------

    def _build_state_reference(self, sim_time):
        """Build the StateReference for the current tick.

        In 2WD, when the operator commands a lean change, a minimum-jerk
        trajectory provides smooth [pos, vel, pitch, pitch_rate] references
        so the LQR tracks a feasible path.  In 4WD (or when no trajectory
        is active) the reference is simply the operator's commanded lean
        at the current target position.
        """
        requested = self.controller.requested_lean

        # Detect lean change → start trajectory (2WD only)
        if (self.drive_mode == DriveMode.TWO_WD
                and abs(requested - self._prev_requested_lean) > math.radians(1.0)):
            self._lean_traj.start(
                sim_time=sim_time,
                current_position=self.controller.target_position,
                theta_start=self.controller.target_lean,
                theta_end=requested,
            )
        self._prev_requested_lean = requested

        # Active trajectory → use its smooth reference
        if self._lean_traj.active:
            ref_pos, ref_vel, ref_pitch, ref_prate = \
                self._lean_traj.update(sim_time)
            return StateReference(
                position=ref_pos,
                velocity=ref_vel,
                pitch=ref_pitch,
                pitch_rate=ref_prate,
            )

        # Steady state: track operator commands directly
        return StateReference(
            position=self.controller.target_position,
            pitch=requested,
        )

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

        if not self.controller.plans_triplet_torque:
            self.triplet_ctrl_L.set_base_angle(self.triplet_base_angle)
            self.triplet_ctrl_R.set_base_angle(self.triplet_base_angle)

        # --- Build state reference ---
        # The robot loop owns the trajectory planner and knows the drive
        # mode, so it builds the StateReference that the controller
        # tracks.  The controller is a pure function of (state, ref, K).
        ref = self._build_state_reference(sim_time)

        # --- Controller → per-side commanded torques ---
        left_cmd, right_cmd = self.controller.update(
            s.pitch, s.pitch_rate,
            s.position, s.yaw_rate, sim_time, dt, ref=ref
        )

        # Triplet hub commands
        if self.controller.plans_triplet_torque:
            triplet_cmd_L = self.controller.triplet_torque_L
            triplet_cmd_R = self.controller.triplet_torque_R
        else:
            # Triplet target = ref.pitch (trajectory pitch during
            # transitions, requested lean otherwise).  Works for
            # both 2WD and 4WD — the ref already encodes the mode.
            triplet_cmd_L = self.triplet_ctrl_L.compute_lean_and_update(
                ref.pitch, self.controller.desired_lean,
                self.triplet_base_angle,
                s.triplet_angle_L, s.triplet_rate_L, s.pitch,
                body_pitch_rate=s.pitch_rate, dt=dt)
            triplet_cmd_R = self.triplet_ctrl_R.compute_lean_and_update(
                ref.pitch, self.controller.desired_lean,
                self.triplet_base_angle,
                s.triplet_angle_R, s.triplet_rate_R, s.pitch,
                body_pitch_rate=s.pitch_rate, dt=dt)

        # --- Apply motor torque through motor models ---
        side_configs = [
            (0, self.l_wheel_joints, self.l_triplet_joint, left_cmd, triplet_cmd_L, s.wheel_velocity_L),
            (1, self.r_wheel_joints, self.r_triplet_joint, right_cmd, triplet_cmd_R, s.wheel_velocity_R),
        ]

        for motor_idx, wheel_joints, triplet_joint, cmd_torque, triplet_cmd, wheel_vel in side_configs:

            motor_torque = self.motors[motor_idx].update(
                cmd_torque, wheel_vel, dt
            )
            self.actual_torques[motor_idx] = motor_torque

            triplet_total = -motor_torque + triplet_cmd
            p.setJointMotorControl2(
                self.body_id, triplet_joint,
                controlMode=p.TORQUE_CONTROL,
                force=triplet_total
            )

            torque_per_wheel = -motor_torque / 3.0

            for wj in wheel_joints:
                wheel_pos = p.getJointState(self.body_id, wj)[0]
                imbalance = self.cfg.robot.wheel_imbalance_torque * math.sin(wheel_pos)

                p.setJointMotorControl2(
                    self.body_id, wj,
                    controlMode=p.TORQUE_CONTROL,
                    force=torque_per_wheel + imbalance
                )

    # ----------------------------------------------------------------
    # Telemetry helpers
    # ----------------------------------------------------------------

    def get_telemetry(self):
        """Return robot-level telemetry for the current tick."""
        s = self.state
        return {
            "true_pitch": s.true_pitch,
            "true_pitch_rate": s.true_pitch_rate,
            "torque_L_actual": float(self.actual_torques[0]),
            "torque_R_actual": float(self.actual_torques[1]),
            "triplet_assist_L": float(self.triplet_ctrl_L.last_assist_force),
            "triplet_assist_R": float(self.triplet_ctrl_R.last_assist_force),
            "triplet_grav_comp_L": float(self.triplet_ctrl_L.last_grav_comp),
            "triplet_grav_comp_R": float(self.triplet_ctrl_R.last_grav_comp),
            "drive_mode": float(s.drive_mode == DriveMode.TWO_WD),
            "position": s.position,
            "forward_velocity": s.forward_velocity,
            "pitch": s.pitch,
            "pitch_rate": s.pitch_rate,
            "yaw_rate": s.yaw_rate,
            "triplet_angle_L": s.triplet_angle_L,
            "triplet_angle_R": s.triplet_angle_R,
            "wheel_velocity_L": s.wheel_velocity_L,
            "wheel_velocity_R": s.wheel_velocity_R,
        }

    # ----------------------------------------------------------------
    # Debug / status
    # ----------------------------------------------------------------

    def get_debug_state(self):
        """Return comprehensive debug info."""
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id)
        euler = p.getEulerFromQuaternion(orn)

        lt_state = p.getJointState(self.body_id, self.l_triplet_joint)
        rt_state = p.getJointState(self.body_id, self.r_triplet_joint)

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
        fwd_x = -rot[0]
        fwd_y = -rot[3]
        yaw = math.atan2(fwd_y, fwd_x)
        return pos[0], pos[1], yaw, fwd_x, fwd_y

    def set_drive_mode(self, mode: DriveMode):
        """Set a specific drive mode; no-op if already in that mode."""
        if mode != self.drive_mode:
            self.toggle_drive_mode()

    def toggle_drive_mode(self):
        """Toggle between 4WD and 2WD drive modes."""
        angle_2wd = self.cfg.robot.triplet_2wd_angle
        if self.drive_mode == DriveMode.FOUR_WD:
            self.drive_mode = DriveMode.TWO_WD
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
        if self.drive_mode == DriveMode.FOUR_WD:
            self._lean_traj.cancel()

    def check_fallen(self):
        """Check if robot has fallen over (|pitch| > 80°)."""
        true_pitch, _ = self._get_true_state()
        return abs(true_pitch) > math.radians(80)
