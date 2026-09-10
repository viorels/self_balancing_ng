"""
TribotBalanceBot — MuJoCo robot interface: sensors, actuators, state.

This module owns the physical robot representation: discovering joints
and actuators in the compiled MuJoCo model, reading sensors into a
RobotState, and applying ControlOutput torques via data.ctrl.

No control logic lives here — only the robot's interface to MuJoCo.
"""

import math

import mujoco
import numpy as np

from robot_state import DriveMode, RobotState
from models.motor_model import BrushlessMotorModel
from models.imu_model import IMUSensorModel
from controllers.triplet_controller import (
    TripletController,
    compute_triplet_from_pitch,
)
from controllers.base import StateReference
from controllers.control_pid import BalanceController
from controllers.control_lqr import LQRBalanceController
from controllers.control_mpc import MPCBalanceController
from controllers.lean_trajectory import LeanTrajectory


# ============================================================================
# QUATERNION HELPERS
# ============================================================================

def _euler_to_mj_quat(roll, pitch, yaw):
    """Convert Euler angles (XYZ extrinsic) to MuJoCo quaternion [w, x, y, z]."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,   # w
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
    ]


def _mj_quat_to_euler(quat_wxyz):
    """Convert MuJoCo quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
    w, x, y, z = quat_wxyz
    # Roll (X-axis)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    # Pitch (Y-axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # Yaw (Z-axis)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


# ============================================================================
# TRIBOT ROBOT CLASS
# ============================================================================

class TribotBalanceBot:
    """
    A self-balancing robot with triplet wheel clusters.

    Each side has a triplet assembly (3 small wheels in a triangle) connected
    to the body by a freely-rotating hub. All 3 wheels on each side are
    coupled by equality constraints (same angular velocity) driven by a
    single motor.

    Receives pre-loaded MuJoCo model and data from tribot_sim.py.
    """

    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.cfg = config

        # Joint and actuator index maps (populated by _discover)
        self.joint_map = {}
        self.actuator_map = {}

        # Joint indices
        self.l_triplet_jnt = -1
        self.r_triplet_jnt = -1
        self.l_wheel_jnts = []
        self.r_wheel_jnts = []

        # qpos / qvel address caches
        self._qpos_addr = {}   # joint_id → qpos index
        self._qvel_addr = {}   # joint_id → qvel (dof) index

        # Actuator indices
        self.act_l_triplet = -1
        self.act_r_triplet = -1
        self.act_l_wheels = []
        self.act_r_wheels = []

        # Body id for c_body
        self.body_id = -1

        # Discover joints/actuators and cache addresses
        self._discover()

        # Store initial pose for reset()
        self._initial_qpos = self.data.qpos.copy()

        # Balance controller
        ctrl_type = config.sim.controller.lower()
        if ctrl_type == 'mpc':
            self.controller = MPCBalanceController(config)
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
        self._lean_traj = LeanTrajectory(
            h_cog=config.triplet.cog_dist_2wd,
            min_duration=0.2,
            max_duration=1.0,
            lean_per_sec_factor=0.25,
        )
        self._prev_requested_lean = 0.0

        # IMU sensor model
        self.imu = IMUSensorModel(config)

        # Wheel radius and velocity filter for odometry
        self.wheel_radius = config.robot.wheel_radius
        self._fwd_vel_filtered = 0.0

        # Current state for logging
        self.position = 0.0
        self.pitch_angle = 0.0
        self.pitch_rate = 0.0
        self.actual_torques = [0.0, 0.0]

        # Sensor state (populated by read_sensors() on each tick)
        self.state = RobotState(drive_mode=self.drive_mode,
                                triplet_base_angle=self.triplet_base_angle)

        # Set initial triplet angles and print summary
        self._set_initial_pose()

    # ----------------------------------------------------------------
    # Robot setup
    # ----------------------------------------------------------------

    def _discover(self):
        """Build name→id maps for joints, actuators, and body; cache addresses."""
        # Joints
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_map[name] = i
                self._qpos_addr[i] = self.model.jnt_qposadr[i]
                self._qvel_addr[i] = self.model.jnt_dofadr[i]

        self.l_triplet_jnt = self.joint_map['c_body_to_l_triplet']
        self.r_triplet_jnt = self.joint_map['c_body_to_r_triplet']

        self.l_wheel_jnts = [
            self.joint_map['l_triplet_to_l_wheel_1'],
            self.joint_map['l_triplet_to_l_wheel_2'],
            self.joint_map['l_triplet_to_l_wheel_3'],
        ]
        self.r_wheel_jnts = [
            self.joint_map['r_triplet_to_r_wheel_1'],
            self.joint_map['r_triplet_to_r_wheel_2'],
            self.joint_map['r_triplet_to_r_wheel_3'],
        ]

        # Actuators
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuator_map[name] = i

        self.act_l_triplet = self.actuator_map['motor_l_triplet']
        self.act_r_triplet = self.actuator_map['motor_r_triplet']
        self.act_l_wheels = [
            self.actuator_map['motor_l_wheel_1'],
            self.actuator_map['motor_l_wheel_2'],
            self.actuator_map['motor_l_wheel_3'],
        ]
        self.act_r_wheels = [
            self.actuator_map['motor_r_wheel_1'],
            self.actuator_map['motor_r_wheel_2'],
            self.actuator_map['motor_r_wheel_3'],
        ]

        # Body id
        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'c_body')

        num_joints = self.model.njnt
        print(f"  Joints discovered: {num_joints} total")
        print(f"    L triplet: joint {self.l_triplet_jnt}")
        print(f"    R triplet: joint {self.r_triplet_jnt}")
        print(f"    L wheels:  joints {self.l_wheel_jnts}")
        print(f"    R wheels:  joints {self.r_wheel_jnts}")

        # Print body masses for debugging
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            mass = self.model.body_mass[i]
            if mass > 0:
                print(f"    {name}: mass={mass:.4f}")

        print(f"  Belt constraints: {self.model.neq} equality constraints")

    def _set_initial_pose(self):
        """Set initial triplet angles (0 = 4WD, 60 = 2WD)."""
        trip_angle = self.cfg.sim.initial_triplet_angle
        if abs(trip_angle) > 1e-6:
            self.data.qpos[self._qpos_addr[self.l_triplet_jnt]] = trip_angle
            self.data.qpos[self._qpos_addr[self.r_triplet_jnt]] = trip_angle
            mujoco.mj_forward(self.model, self.data)
            # Update stored initial qpos
            self._initial_qpos = self.data.qpos.copy()

        mode = ('2WD' if abs(trip_angle - math.pi / 3) < 0.05
                else ('4WD' if abs(trip_angle) < 0.05 else 'Lean'))
        print(f"  Initial triplet angle: {math.degrees(trip_angle):.1f} ({mode} mode)")

    def reset(self):
        """Reset the robot to its initial upright pose with zero velocities.

        Intended for AI-driven experiments: call between trials to start fresh
        without restarting the simulation process.
        """
        # Restore initial qpos (includes base pose + triplet angles)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # Forward kinematics to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        # Reset motor first-order lag models
        self.motors = [
            type(self.motors[0])(self.cfg),
            type(self.motors[1])(self.cfg),
        ]

        # Reset IMU
        self.imu = type(self.imu)(self.cfg)

        # Reset software state
        self.position = 0.0
        self._fwd_vel_filtered = 0.0
        self.pitch_angle = 0.0
        self.pitch_rate = 0.0
        self.actual_torques = [0.0, 0.0]
        self.drive_mode = DriveMode.FOUR_WD
        self.triplet_base_angle = self.cfg.sim.initial_triplet_angle
        self.state = RobotState(drive_mode=self.drive_mode,
                                triplet_base_angle=self.triplet_base_angle)
        self.triplet_ctrl_L.drive_mode = self.drive_mode
        self.triplet_ctrl_R.drive_mode = self.drive_mode

        # Reset controller
        ctrl = self.controller
        if hasattr(ctrl, 'reset'):
            ctrl.reset()
        else:
            for attr in ('_integral', '_pos_integral', 'x_hat',
                         '_prev_error', '_prev_pos_error'):
                if hasattr(ctrl, attr):
                    val = getattr(ctrl, attr)
                    if hasattr(val, 'shape'):
                        setattr(ctrl, attr, np.zeros_like(val))
                    else:
                        setattr(ctrl, attr, 0.0)
            ctrl.set_target_position(0.0)
            ctrl.set_yaw_rate(0.0)
            ctrl.set_lean(0.0)

        # Reset lean trajectory planner
        self._lean_traj.cancel()
        self._prev_requested_lean = 0.0

        print("[reset] Robot pose and state restored to initial conditions.")

    # ----------------------------------------------------------------
    # State estimation
    # ----------------------------------------------------------------

    def _get_rot_and_vel(self):
        """Return (rot_3x3, lin_vel_world, ang_vel_body) for c_body."""
        rot = self.data.xmat[self.body_id].reshape(3, 3)
        # Use CoM velocity for odometry (matches PyBullet's getBaseVelocity).
        # qvel[0:3] is the body-frame origin velocity, but the LQR's position
        # estimate was tuned with CoM velocity.  The difference matters during
        # pitching: v_CoM = v_origin + ω×r_CoG, which shifts ~0.25 m/s at
        # pitch_rate≈1 rad/s (h_CoG=0.247 m) and causes position drift.
        lin_vel = self.data.cvel[self.body_id][3:6]  # CoM velocity, world frame
        ang_vel = self.data.qvel[3:6]   # body frame (MuJoCo convention)
        return rot, lin_vel, ang_vel

    def _get_true_state(self):
        """
        Read true pitch from physics (body-frame, yaw-invariant).
        Pitch = rotation around body Y axis (the wheel axle direction).

        MuJoCo xmat is row-major 3x3:
          rot[2,2] = world-Z component of body-Z (body_up_z)
          rot[2,0] = world-Z component of body-X (body_fwd_z)
        """
        rot, _, ang_vel = self._get_rot_and_vel()

        body_up_z = rot[2, 2]
        body_fwd_z = rot[2, 0]
        pitch = math.atan2(-body_fwd_z, body_up_z)

        # Pitch rate = body-frame Y angular velocity (MuJoCo qvel is body frame)
        pitch_rate = float(ang_vel[1])

        return pitch, pitch_rate

    # ----------------------------------------------------------------
    # Sensor pipeline
    # ----------------------------------------------------------------

    def read_sensors(self, sim_time, dt):
        """
        Read all sensors and return a populated RobotState.

        This method is the single source of truth for measured / estimated
        quantities. Controllers and the telemetry loop consume the returned
        RobotState rather than reaching into MuJoCo directly.
        """
        # --- Ground truth from physics ---
        true_pitch, true_pitch_rate = self._get_true_state()

        # --- IMU-fused measurements ---
        measured_pitch, measured_pitch_rate = self.imu.read(
            true_pitch, true_pitch_rate, sim_time, dt
        )
        self.pitch_angle = measured_pitch
        self.pitch_rate = measured_pitch_rate

        # --- Yaw rate (body-frame Z angular velocity) ---
        _, _, ang_vel = self._get_rot_and_vel()
        yaw_rate = -float(ang_vel[2])

        # --- Triplet encoders ---
        lt_angle = self.data.qpos[self._qpos_addr[self.l_triplet_jnt]]
        lt_rate = self.data.qvel[self._qvel_addr[self.l_triplet_jnt]]
        rt_angle = self.data.qpos[self._qpos_addr[self.r_triplet_jnt]]
        rt_rate = self.data.qvel[self._qvel_addr[self.r_triplet_jnt]]

        # --- Wheel velocities (one representative per side, belt-coupled) ---
        wheel_vel_L = self.data.qvel[self._qvel_addr[self.l_wheel_jnts[0]]]
        wheel_vel_R = self.data.qvel[self._qvel_addr[self.r_wheel_jnts[0]]]

        # --- Forward odometry from wheel encoders ---
        # The wheel encoder measures spin relative to the hub.  Ground
        # travel follows the wheel's ABSOLUTE spin, which also includes
        # the hub rotation and the body pitch rate (all about +Y):
        #     omega_abs = wheel_joint_rate + hub_joint_rate + pitch_rate
        # In steady 4WD the hub is ground-locked (hub rate = -pitch rate)
        # so this reduces to the raw encoder; it matters during mode
        # transitions and flips, where the hub rotates by up to 120 deg.
        # A low-pass filter then strips the high-frequency balance
        # component (~5-10 Hz) while preserving the actual translation.
        omega_abs = 0.5 * (wheel_vel_L + wheel_vel_R + lt_rate + rt_rate) \
            + measured_pitch_rate
        v_wheel_raw = -self.wheel_radius * omega_abs
        alpha = min(1.0, dt * 20.0)  # ~50ms time constant
        self._fwd_vel_filtered += alpha * (v_wheel_raw - self._fwd_vel_filtered)
        fwd_vel = self._fwd_vel_filtered
        self.position += fwd_vel * dt

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
            triplet_angle_L=float(lt_angle),
            triplet_angle_R=float(rt_angle),
            triplet_rate_L=float(lt_rate),
            triplet_rate_R=float(rt_rate),
            wheel_velocity_L=float(wheel_vel_L),
            wheel_velocity_R=float(wheel_vel_R),
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

        # Detect lean change -> start trajectory (2WD only)
        if (self.drive_mode == DriveMode.TWO_WD
                and abs(requested - self._prev_requested_lean) > math.radians(1.0)):
            self._lean_traj.start(
                sim_time=sim_time,
                current_position=self.controller.target_position,
                theta_start=self.controller.target_lean,
                theta_end=requested,
            )
        self._prev_requested_lean = requested

        # Active trajectory -> use its smooth lean reference
        if self._lean_traj.active:
            _, _, ref_pitch, ref_prate = self._lean_traj.update(sim_time)
            return StateReference(
                pitch=ref_pitch,
                pitch_rate=ref_prate,
            )

        # Steady state: track operator lean directly
        return StateReference(
            pitch=requested,
        )

    # ----------------------------------------------------------------
    # Control update
    # ----------------------------------------------------------------

    def update(self, sim_time, dt):
        """
        Run one control + actuation cycle.

        Reads sensors via read_sensors(), feeds the controller, computes
        triplet commands, and applies motor torques via data.ctrl.
        Called every physics timestep; controller only runs at CONTROL_RATE_HZ.
        """
        s = self.read_sensors(sim_time, dt)

        # --- Feed triplet state to controller (used by the MPC) ---
        self.controller.set_triplet_state(
            s.triplet_angle_L, s.triplet_angle_R,
            s.triplet_rate_L, s.triplet_rate_R,
        )

        if not self.controller.plans_triplet_torque:
            self.triplet_ctrl_L.set_base_angle(self.triplet_base_angle)
            self.triplet_ctrl_R.set_base_angle(self.triplet_base_angle)

        # --- Build state reference ---
        ref = self._build_state_reference(sim_time)

        # --- Controller -> per-side commanded torques ---
        left_cmd, right_cmd = self.controller.update(
            s.pitch, s.pitch_rate,
            s.position, s.forward_velocity, s.yaw_rate, sim_time, dt, ref=ref
        )

        # Triplet hub commands
        if self.controller.plans_triplet_torque:
            triplet_cmd_L = self.controller.triplet_torque_L
            triplet_cmd_R = self.controller.triplet_torque_R
        else:
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
            (0, self.l_wheel_jnts, self.act_l_wheels, self.act_l_triplet,
             left_cmd, triplet_cmd_L, s.wheel_velocity_L),
            (1, self.r_wheel_jnts, self.act_r_wheels, self.act_r_triplet,
             right_cmd, triplet_cmd_R, s.wheel_velocity_R),
        ]

        for (motor_idx, wheel_jnts, wheel_acts, triplet_act,
             cmd_torque, triplet_cmd, wheel_vel) in side_configs:

            motor_torque = self.motors[motor_idx].update(
                cmd_torque, wheel_vel, dt
            )
            self.actual_torques[motor_idx] = motor_torque

            # Triplet hub: reaction-cancelled torque
            triplet_total = -motor_torque + triplet_cmd
            self.data.ctrl[triplet_act] = triplet_total

            # Drive wheels: distribute motor torque + per-wheel imbalance
            torque_per_wheel = -motor_torque / 3.0
            for wj, wa in zip(wheel_jnts, wheel_acts):
                wheel_pos = self.data.qpos[self._qpos_addr[wj]]
                imbalance = (self.cfg.robot.wheel_imbalance_torque
                             * math.sin(wheel_pos))
                self.data.ctrl[wa] = torque_per_wheel + imbalance

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
        pos = self.data.qpos[0:3]
        quat_wxyz = self.data.qpos[3:7]
        euler = _mj_quat_to_euler(quat_wxyz)

        ang_vel = self.data.qvel[3:6]

        lt_angle = self.data.qpos[self._qpos_addr[self.l_triplet_jnt]]
        rt_angle = self.data.qpos[self._qpos_addr[self.r_triplet_jnt]]
        lt_rate = self.data.qvel[self._qvel_addr[self.l_triplet_jnt]]
        rt_rate = self.data.qvel[self._qvel_addr[self.r_triplet_jnt]]

        lw_vel = self.data.qvel[self._qvel_addr[self.l_wheel_jnts[0]]]
        rw_vel = self.data.qvel[self._qvel_addr[self.r_wheel_jnts[0]]]

        return {
            'pos': tuple(pos),
            'euler_deg': tuple(math.degrees(e) for e in euler),
            'ang_vel_deg': tuple(math.degrees(v) for v in ang_vel),
            'triplet_ang': (float(lt_angle), float(rt_angle)),
            'triplet_vel': (float(lt_rate), float(rt_rate)),
            'wheel_vel': (float(lw_vel), float(rw_vel)),
        }

    def get_world_pose_2d(self):
        """
        Return (x, y, yaw, fwd_x, fwd_y) in world frame.
        Robot forward is body -X (URDF convention).
        """
        pos = self.data.qpos[0:3]
        rot = self.data.xmat[self.body_id].reshape(3, 3)
        fwd_x = -rot[0, 0]
        fwd_y = -rot[1, 0]
        yaw = math.atan2(fwd_y, fwd_x)
        return float(pos[0]), float(pos[1]), yaw, fwd_x, fwd_y

    def set_drive_mode(self, mode: DriveMode):
        """Set a specific drive mode; no-op if already in that mode."""
        if mode != self.drive_mode:
            self.toggle_drive_mode()

    def toggle_drive_mode(self):
        """Toggle between 4WD and 2WD drive modes."""
        angle_2wd = self.cfg.robot.triplet_2wd_angle
        if self.drive_mode == DriveMode.FOUR_WD:
            self.drive_mode = DriveMode.TWO_WD
            cur_angle = self.data.qpos[self._qpos_addr[self.l_triplet_jnt]]
            if cur_angle >= 0:
                self.triplet_base_angle = -angle_2wd
            else:
                self.triplet_base_angle = angle_2wd
            print(f"  [MODE] 4WD -> 2WD  (triplet target "
                  f"{math.degrees(self.triplet_base_angle):+.0f})")
        else:
            self.drive_mode = DriveMode.FOUR_WD
            self.triplet_base_angle = 0.0
            print(f"  [MODE] 2WD -> 4WD  (triplet target 0)")
        self.triplet_ctrl_L.drive_mode = self.drive_mode
        self.triplet_ctrl_R.drive_mode = self.drive_mode
        self.controller.set_drive_mode(self.drive_mode)
        if self.drive_mode == DriveMode.FOUR_WD:
            self._lean_traj.cancel()

    def check_fallen(self):
        """Check if robot has fallen over (|pitch| > 80)."""
        true_pitch, _ = self._get_true_state()
        return abs(true_pitch) > math.radians(80)
