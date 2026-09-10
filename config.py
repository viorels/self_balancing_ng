"""
Typed configuration for the tribot simulation.

Replaces the flat CONFIG dict with namespaced dataclasses.
All consumer code uses typed attribute access::

    from config import load_config
    CONFIG = load_config()

    CONFIG.sim.gravity         # → -9.81
    CONFIG.lqr.r               # → 2.0
    CONFIG.motor.max_torque    # → 1.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


# ============================================================================
# Sub-configs (one per subsystem)
# ============================================================================

@dataclass
class SimConfig:
    """Simulation-level parameters and initial conditions."""
    gravity: float = -9.81
    timestep: float = 1.0 / 500.0       # 500 Hz physics
    sim_duration: float = 600.0
    controller: str = 'lqr'             # 'lqr', 'pid', or 'mpc'
    mjcf_path: str = 'tribot_description/mjcf/tribot.xml'
    initial_pitch: float = -0.03        # rad (~1.7°)
    initial_height: float = 0.118       # m
    initial_triplet_angle: float = 0.0  # rad (0° = 4WD)


@dataclass
class RobotConfig:
    """Robot geometry, contact properties, and mechanical parameters."""
    wheel_radius: float = 0.058         # m — small drive wheel
    triplet_radius: float = 0.12        # m — circumradius of wheel triangle
    wheel_friction: float = 0.7
    triplet_friction: float = 0.3
    belt_max_force: float = 100.0       # N — gear constraint max force
    triplet_joint_damping: float = 0.05 # Nm·s/rad
    wheel_imbalance_torque: float = 0.002  # Nm
    triplet_2wd_angle: float = math.pi / 3  # 60° target for 2WD mode


@dataclass
class MotorConfig:
    """Brushless motor electrical / mechanical model."""
    max_torque: float = 1.0             # Nm stall torque per side
    tau: float = 0.003                  # s — electrical time constant
    back_emf_k: float = 0.005          # Nm per rad/s
    cogging_amplitude: float = 0.005   # Nm
    cogging_poles: int = 14
    deadband: float = 0.01             # Nm
    torque_noise_std: float = 0.005    # Nm


@dataclass
class IMUConfig:
    """IMU sensor model + complementary filter."""
    add_sensor_noise: bool = True
    angle_noise_std: float = 0.003
    gyro_noise_std: float = 0.01
    gyro_drift_rate: float = 0.001     # rad/s²
    accel_vib_noise_std: float = 0.15
    sample_rate_hz: int = 500
    quantization_bits: int = 16
    accel_range_g: int = 2
    gyro_range_dps: int = 500
    comp_filter_alpha: float = 0.02


@dataclass
class ControlConfig:
    """Shared control-loop timing and parameters."""
    control_rate_hz: int = 200          # PD tracking loop rate
    control_jitter_std: float = 0.0005  # s — timing jitter σ
    yaw_damping_k: float = 0.5


@dataclass
class PlantConfig:
    """Linearised plant physical constants (shared by LQR and MPC)."""
    body_mass: float = 2.7167          # kg
    wheel_mass: float = 0.6698         # kg — 2 triplets + 6 wheels
    cog_height: float = 0.247          # m — CoG above wheel axis
    body_inertia: float = 0.056436     # kg·m² — Iyy


@dataclass
class PIDConfig:
    """PID controller gains (inner pitch + outer position loops)."""
    kp: float = 15.0
    kd: float = 0.8
    ki: float = 3.0
    pos_kp: float = 0.15
    pos_kd: float = 0.03
    pos_ki: float = 0.01
    pos_max_pitch: float = 0.15        # rad (~8.6°)
    pos_rate_hz: int = 50


@dataclass
class LQRConfig:
    """LQR gain tuning and gain-scheduling parameters."""
    q_diag: List[float] = field(
        default_factory=lambda: [20.0, 12.0, 45.0, 6.0])
    r: float = 2.0
    aggressive_q_diag: List[float] = field(
        default_factory=lambda: [40.0, 16.0, 35.0, 5.0])
    aggressive_r: float = 1.5
    switch_threshold: float = 0.20     # m
    switch_hysteresis: float = 0.05    # m


@dataclass
class MPCConfig:
    """MPC solver parameters + ZMP/DCM flip trigger."""
    # MPC solver
    rate_hz: int = 30
    horizon: int = 10
    simulated_solve_ms: float = 20.0
    q_diag: List[float] = field(
        default_factory=lambda: [50.0, 5.0, 40.0, 40.0, 5.0, 5.0, 12.0, 5.0])
    r_diag: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 8.0, 8.0])
    q_terminal_scale: float = 3.0
    triplet_torque_max: float = 5.0    # Nm
    triplet_inertia: float = 0.00238   # kg·m²
    # PD tracking gains: [trip_L, trip_R, drive_L, drive_R]
    pd_kp: List[float] = field(
        default_factory=lambda: [10.0, 10.0, 3.0, 3.0])
    pd_kd: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 0.3, 0.3])
    pitch_pd_cross_drive: float = 8.0
    pitch_rate_pd_cross_drive: float = 0.5
    # ZMP / DCM flip trigger
    zmp_t_flip_nominal: float = 0.18   # s
    zmp_t_flip_margin: float = 0.05    # s
    zmp_t_settle: float = 0.40         # s
    zmp_trip_tol: float = 0.15         # rad
    zmp_flip_q_trip: float = 120.0
    zmp_flip_q_pitch: float = 120.0
    zmp_flip_r_trip: float = 0.05
    zmp_ctrl_authority: float = 0.3
    zmp_theta_crash: float = 0.785     # rad (≈45°)
    zmp_stair_height: float = 0.1      # m
    zmp_min_fall_rate_deg_s: float = 15.0
    zmp_flip_cooldown: float = 0.8     # s
    zmp_pitch_recover_threshold: float = 0.12  # rad
    zmp_flip_min_rotation: float = 0.698       # rad (40°)


@dataclass
class TripletConfig:
    """Triplet lean PD controller + nonlinear balance assist."""
    lean_kp: float = 8.0               # Nm/rad
    lean_kd: float = 0.4               # Nm·s/rad
    grav_comp_4wd: float = 2.0         # Nm
    grav_comp_2wd: float = 2.0         # Nm
    lean_scale_4wd: float = 1.0 / 1.7  # ≈ 0.59
    cog_dist_2wd: float = 0.19         # m — hub-to-CoG
    foot_length_2wd: float = 0.12      # m — hub-to-wheel
    assist_gain: float = 4.0           # Nm/rad²
    assist_deadzone: float = 0.15      # rad (~8.6°)
    assist_max: float = 3.0            # Nm
    assist_tau: float = 0.1            # s — EMA time constant


@dataclass
class GamepadConfig:
    """Gamepad axis/button mapping and scaling."""
    device: str = '/dev/input/js0'
    deadzone: float = 0.08
    speed_axis: int = 4                # Right stick Y
    yaw_axis: int = 3                  # Right stick X
    lean_axis: int = 1                 # Left stick Y
    max_speed: float = 1.0             # m/s
    max_yaw_rate: float = 2.0          # rad/s
    max_lean: float = math.radians(30)
    mode_button: int = 4               # LB on F710 (XInput)
    target_marker_height: float = 0.3  # m


@dataclass
class TerrainConfig:
    """Terrain type and stair geometry."""
    terrain_type: str = 'flat'         # 'flat', 'heightfield', or 'box_stairs'
    ground_friction: float = 0.7
    stair_num_steps: int = 2
    stair_step_depth: float = 0.20     # m
    stair_step_height: List[float] = field(
        default_factory=lambda: [0.1, 0.15])
    stair_width: float = 0.6           # m
    stair_start_x: float = 0.5         # m


# ============================================================================
# Root configuration
# ============================================================================

@dataclass
class Config:
    """Root configuration — contains all sub-configs.

    Access via typed attributes::

        config.sim.gravity
        config.lqr.r
        config.motor.max_torque
    """
    sim: SimConfig = field(default_factory=SimConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    lqr: LQRConfig = field(default_factory=LQRConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    triplet: TripletConfig = field(default_factory=TripletConfig)
    gamepad: GamepadConfig = field(default_factory=GamepadConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)


# ============================================================================
# Factory
# ============================================================================

def load_config() -> Config:
    """Create a Config with all defaults.

    Usage::

        cfg = load_config()
        cfg.sim.gravity       # → -9.81
        cfg.lqr.r             # → 2.0
    """
    return Config()
