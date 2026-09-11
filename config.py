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
    controller: str = 'mpc'             # 'mpc' (default), 'lqr', or 'pid'
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
    """Plant physical constants (shared by LQR and MPC), from the MJCF."""
    body_mass: float = 2.7167          # kg
    wheel_mass: float = 0.6698         # kg — 2 triplets + 6 wheels (LQR)
    cog_height: float = 0.247          # m — CoG above hub axis
    body_inertia: float = 0.056436     # kg·m² — Iyy
    # Per-part constants used by the planar MPC model
    hub_mass: float = 0.253511         # kg — one triplet hub
    hub_inertia: float = 0.002097      # kg·m² — one hub, Iyy about hub axis
    wheel_mass_each: float = 0.027132  # kg — one wheel
    wheel_inertia: float = 0.000068    # kg·m² — one wheel, Iyy about axle
    wheelbase_half_4wd: float = 0.10414  # m — grounded wheel x offset in 4WD
    wheel_depth_4wd: float = 0.060125    # m — grounded wheel axle below hub


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
    """Mode-scheduled MPC (controllers/control_mpc.py) + event layer."""
    # --- Prediction model / horizon ---
    horizon: int = 20                  # steps
    dt_pred: float = 0.02              # s per step (0.4 s lookahead)
    # State weights [s, s_dot, theta, theta_dot, lam, lam_dot]
    q_diag: List[float] = field(
        default_factory=lambda: [20.0, 4.0, 150.0, 2.0, 60.0, 0.4])
    # Input weights [u_d, u_t] and input-rate weights
    r_diag: List[float] = field(default_factory=lambda: [1.0, 0.3])
    rd_diag: List[float] = field(default_factory=lambda: [2.0, 0.5])
    # Soft pitch limit and slack penalties
    theta_soft_limit: float = 0.35     # rad (~20°)
    slack_linear: float = 500.0
    slack_quadratic: float = 5000.0
    # Solver
    max_iter: int = 400
    solver_eps: float = 1e-4
    # --- Limits and scheduling ---
    triplet_torque_max: float = 5.0    # Nm per side
    yaw_reserve: float = 0.15          # drive torque fraction kept for yaw
    leg_reserve: float = 0.3           # hub torque fraction kept for the leg PD
    tipping_safety: float = 0.7        # fraction of the 4WD tipping moment
    theta_cutoff: float = 0.9          # rad, torque cut-off (fallen)
    linearisation_clip: float = 0.45   # rad, clamp on the linearisation point
    four_wd_tolerance_deg: float = 4.0 # both wheels within this of the tie
    # --- Yaw PI (kp comes from control.yaw_damping_k) ---
    yaw_ki: float = 3.0
    # --- Antisymmetric leg PD (mode transitions) ---
    leg_kp: float = 12.0               # Nm/rad
    leg_kd: float = 0.6                # Nm·s/rad
    transition_time: float = 0.8       # s for a 4WD<->2WD transition
    # --- Emergency flip (DCM trigger, see docs/DCM_AUTHORITY_FLIP_TRIGGER.md) ---
    flip_enabled: bool = True
    flip_time: float = 0.25            # s for the 120° rotation
    flip_time_margin: float = 0.05     # s
    flip_settle_time: float = 0.4      # s
    flip_cooldown: float = 0.8         # s
    flip_authority: float = 0.3        # fraction of drive torque assumed usable (DCM telemetry)
    flip_theta_crash: float = 0.785    # rad (~45°)
    flip_theta_trigger: float = 0.7    # rad (~40°): predicted/measured pitch that arms a flip
    flip_min_fall_rate_deg_s: float = 15.0
    flip_kp: float = 25.0              # Nm/rad
    flip_kd: float = 0.8               # Nm·s/rad
    # --- Stair step manoeuvre (4WD): lean over the blocked front wheel, then
    #     roll the cluster so the upper wheel lands on the tread ---
    step_enabled: bool = True
    step_min_height: float = 0.03      # m — smaller bumps are just driven over
    step_max_height: float = 0.09      # m — taller obstacles: stop instead
    step_trigger_gap: float = 0.04     # m — front wheel edge to riser distance
    step_approach_speed: float = 0.2   # m/s — velocity cap when a riser is near
    step_approach_distance: float = 0.45  # m — from the hub, where the cap applies
    step_lean_margin: float = -0.05    # kg·m — CoG first-moment margin vs the pivot (negative: just behind)
    step_roll_margin: float = 0.02     # kg·m — CoG margin ahead of the pivot while rolling
    step_roll_lead: float = 0.5        # rad — leg reference lead past the top of the roll
    step_theta_rate: float = 1.2       # rad/s — max rate of the pitch reference while rolling
    step_theta_floor: float = 0.05     # rad — pitch reference floor while rolling
    step_lean_time: float = 0.7        # s
    drop_hold: bool = True             # stop at a drop-off instead of driving over it
    step_down_enabled: bool = True     # roll down a drop with the same manoeuvre
    step_max_drop: float = 0.12        # m — largest drop the step-down attempts
    step_roll_time: float = 0.7        # s for a full 120° roll (scaled by the actual sweep)
    step_roll_timeout: float = 1.5     # s — give up on a roll after this
    step_impact_rate: float = 1.5      # rad/s — leg-rate drop in one tick that marks the landing
    step_land_margin: float = 0.09     # rad (~5°) past the geometric landing angle
    step_settle_time: float = 0.4      # s
    step_pitch_limit: float = 0.8      # rad — soft pitch limit while stepping
    step_drive_limit: float = 0.6      # Nm — drive torque cap while stepping
    step_press_torque: float = 0.0     # Nm — drive bias while leaning (0: pivot wheel rolls freely)
    step_hub_limit: float = 5.0        # Nm — total hub torque cap while rolling
    step_lin_clip: float = 1.35        # rad — linearisation clip while rolling


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
    terrain_type: str = 'box_stairs'         # 'flat', 'heightfield', or 'box_stairs'
    ground_friction: float = 0.7
    stair_num_steps: int = 2
    stair_step_depth: float = 0.20     # m
    stair_step_height: List[float] = field(
        default_factory=lambda: [0.05, 0.05])
    stair_width: float = 0.6           # m
    stair_start_x: float = 0.5         # m
    # Forward-looking terrain probe (ToF model, see TribotBalanceBot._probe_terrain)
    probe_rate_hz: int = 50
    probe_step_threshold: float = 0.02  # m — ground height change that counts


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
