"""
Typed configuration for the tribot simulation.

Replaces the flat CONFIG dict with namespaced dataclasses.
Backward-compatible: Config supports dict-like access (CONFIG['KEY'])
so all existing ``config['KEY']`` / ``self.cfg['KEY']`` code works
without changes.  New code should prefer the typed attributes
(e.g. ``config.sim.gravity``).

Usage::

    from config import load_config
    CONFIG = load_config()

    # Old style (still works everywhere):
    CONFIG['GRAVITY']          # → -9.81
    CONFIG.get('LQR_R', 2.0)  # → 2.0

    # New typed style:
    CONFIG.sim.gravity         # → -9.81
    CONFIG.lqr.r               # → 2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, List


# ============================================================================
# Sub-configs (one per subsystem)
# ============================================================================

@dataclass
class SimConfig:
    """Simulation-level parameters and initial conditions."""
    gravity: float = -9.81
    timestep: float = 1.0 / 500.0       # 500 Hz physics
    sim_duration: float = 60.0
    controller: str = 'lqr'             # 'lqr', 'pid', or 'mpc'
    urdf_path: str = 'tribot_description/urdf/tribot.urdf'
    initial_pitch: float = -0.03        # rad (~1.7°)
    initial_height: float = 0.118       # m
    initial_triplet_angle: float = 0.0  # rad (0° = 4WD)

    _FLAT_KEYS: ClassVar[dict] = {
        'GRAVITY': 'gravity',
        'TIMESTEP': 'timestep',
        'SIM_DURATION': 'sim_duration',
        'CONTROLLER': 'controller',
        'URDF_PATH': 'urdf_path',
        'INITIAL_PITCH': 'initial_pitch',
        'INITIAL_HEIGHT': 'initial_height',
        'INITIAL_TRIPLET_ANGLE': 'initial_triplet_angle',
    }


@dataclass
class RobotConfig:
    """Robot geometry, contact properties, and mechanical parameters."""
    wheel_radius: float = 0.058         # m — small drive wheel
    triplet_radius: float = 0.12        # m — circumradius of wheel triangle
    wheel_friction: float = 1.2
    triplet_friction: float = 0.3
    belt_max_force: float = 100.0       # N — gear constraint max force
    triplet_joint_damping: float = 0.05 # Nm·s/rad
    wheel_imbalance_torque: float = 0.002  # Nm
    triplet_2wd_angle: float = math.pi / 3  # 60° target for 2WD mode

    _FLAT_KEYS: ClassVar[dict] = {
        'WHEEL_RADIUS': 'wheel_radius',
        'TRIPLET_RADIUS': 'triplet_radius',
        'WHEEL_FRICTION': 'wheel_friction',
        'TRIPLET_FRICTION': 'triplet_friction',
        'BELT_MAX_FORCE': 'belt_max_force',
        'TRIPLET_JOINT_DAMPING': 'triplet_joint_damping',
        'WHEEL_IMBALANCE_TORQUE': 'wheel_imbalance_torque',
        'TRIPLET_2WD_ANGLE': 'triplet_2wd_angle',
    }


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

    _FLAT_KEYS: ClassVar[dict] = {
        'MAX_TORQUE': 'max_torque',
        'MOTOR_TAU': 'tau',
        'MOTOR_BACK_EMF_K': 'back_emf_k',
        'MOTOR_COGGING_AMPLITUDE': 'cogging_amplitude',
        'MOTOR_COGGING_POLES': 'cogging_poles',
        'MOTOR_DEADBAND': 'deadband',
        'MOTOR_TORQUE_NOISE_STD': 'torque_noise_std',
    }


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

    _FLAT_KEYS: ClassVar[dict] = {
        'ADD_SENSOR_NOISE': 'add_sensor_noise',
        'IMU_ANGLE_NOISE_STD': 'angle_noise_std',
        'IMU_GYRO_NOISE_STD': 'gyro_noise_std',
        'IMU_GYRO_DRIFT_RATE': 'gyro_drift_rate',
        'IMU_ACCEL_VIB_NOISE_STD': 'accel_vib_noise_std',
        'IMU_SAMPLE_RATE_HZ': 'sample_rate_hz',
        'IMU_QUANTIZATION_BITS': 'quantization_bits',
        'IMU_ACCEL_RANGE_G': 'accel_range_g',
        'IMU_GYRO_RANGE_DPS': 'gyro_range_dps',
        'COMP_FILTER_ALPHA': 'comp_filter_alpha',
    }


@dataclass
class ControlConfig:
    """Shared control-loop timing and parameters."""
    control_rate_hz: int = 200          # PD tracking loop rate
    control_jitter_std: float = 0.0005  # s — timing jitter σ
    sensor_to_actuator_delay_steps: int = 0
    yaw_damping_k: float = 0.5

    _FLAT_KEYS: ClassVar[dict] = {
        'CONTROL_RATE_HZ': 'control_rate_hz',
        'CONTROL_JITTER_STD': 'control_jitter_std',
        'SENSOR_TO_ACTUATOR_DELAY_STEPS': 'sensor_to_actuator_delay_steps',
        'YAW_DAMPING_K': 'yaw_damping_k',
    }


@dataclass
class PlantConfig:
    """Linearised plant physical constants (shared by LQR and MPC)."""
    body_mass: float = 2.7167          # kg
    wheel_mass: float = 0.6698         # kg — 2 triplets + 6 wheels
    cog_height: float = 0.247          # m — CoG above wheel axis
    body_inertia: float = 0.056436     # kg·m² — Iyy

    _FLAT_KEYS: ClassVar[dict] = {
        'LQR_BODY_MASS': 'body_mass',
        'LQR_WHEEL_MASS': 'wheel_mass',
        'LQR_COG_HEIGHT': 'cog_height',
        'LQR_BODY_INERTIA': 'body_inertia',
    }


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

    _FLAT_KEYS: ClassVar[dict] = {
        'PID_KP': 'kp',
        'PID_KD': 'kd',
        'PID_KI': 'ki',
        'POS_PID_KP': 'pos_kp',
        'POS_PID_KD': 'pos_kd',
        'POS_PID_KI': 'pos_ki',
        'POS_PID_MAX_PITCH': 'pos_max_pitch',
        'POS_PID_RATE_HZ': 'pos_rate_hz',
    }


@dataclass
class LQRConfig:
    """LQR gain tuning and gain-scheduling parameters."""
    q_diag: List[float] = field(
        default_factory=lambda: [12.0, 4.0, 55.0, 4.0])
    r: float = 2.0
    aggressive_q_diag: List[float] = field(
        default_factory=lambda: [40.0, 8.0, 35.0, 3.0])
    aggressive_r: float = 1.0
    switch_threshold: float = 0.20     # m
    switch_hysteresis: float = 0.05    # m

    _FLAT_KEYS: ClassVar[dict] = {
        'LQR_Q_DIAG': 'q_diag',
        'LQR_R': 'r',
        'LQR_AGGRESSIVE_Q_DIAG': 'aggressive_q_diag',
        'LQR_AGGRESSIVE_R': 'aggressive_r',
        'LQR_SWITCH_THRESHOLD': 'switch_threshold',
        'LQR_SWITCH_HYSTERESIS': 'switch_hysteresis',
    }


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

    _FLAT_KEYS: ClassVar[dict] = {
        'MPC_RATE_HZ': 'rate_hz',
        'MPC_HORIZON': 'horizon',
        'MPC_SIMULATED_SOLVE_MS': 'simulated_solve_ms',
        'MPC_Q_DIAG': 'q_diag',
        'MPC_R_DIAG': 'r_diag',
        'MPC_Q_TERMINAL_SCALE': 'q_terminal_scale',
        'MPC_TRIPLET_TORQUE_MAX': 'triplet_torque_max',
        'MPC_TRIPLET_INERTIA': 'triplet_inertia',
        'MPC_PD_KP': 'pd_kp',
        'MPC_PD_KD': 'pd_kd',
        'MPC_PITCH_PD_CROSS_DRIVE': 'pitch_pd_cross_drive',
        'MPC_PITCH_RATE_PD_CROSS_DRIVE': 'pitch_rate_pd_cross_drive',
        'ZMP_T_FLIP_NOMINAL': 'zmp_t_flip_nominal',
        'ZMP_T_FLIP_MARGIN': 'zmp_t_flip_margin',
        'ZMP_T_SETTLE': 'zmp_t_settle',
        'ZMP_TRIP_TOL': 'zmp_trip_tol',
        'ZMP_FLIP_Q_TRIP': 'zmp_flip_q_trip',
        'ZMP_FLIP_Q_PITCH': 'zmp_flip_q_pitch',
        'ZMP_FLIP_R_TRIP': 'zmp_flip_r_trip',
        'ZMP_CTRL_AUTHORITY': 'zmp_ctrl_authority',
        'ZMP_THETA_CRASH': 'zmp_theta_crash',
        'ZMP_STAIR_HEIGHT': 'zmp_stair_height',
        'ZMP_MIN_FALL_RATE_DEG_S': 'zmp_min_fall_rate_deg_s',
        'ZMP_FLIP_COOLDOWN': 'zmp_flip_cooldown',
        'ZMP_PITCH_RECOVER_THRESHOLD': 'zmp_pitch_recover_threshold',
        'ZMP_FLIP_MIN_ROTATION': 'zmp_flip_min_rotation',
    }


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

    _FLAT_KEYS: ClassVar[dict] = {
        'TRIPLET_LEAN_KP': 'lean_kp',
        'TRIPLET_LEAN_KD': 'lean_kd',
        'TRIPLET_GRAV_COMP_4WD': 'grav_comp_4wd',
        'TRIPLET_GRAV_COMP_2WD': 'grav_comp_2wd',
        'TRIPLET_4WD_LEAN_SCALE': 'lean_scale_4wd',
        'TRIPLET_2WD_COG_DIST': 'cog_dist_2wd',
        'TRIPLET_2WD_FOOT_LENGTH': 'foot_length_2wd',
        'TRIPLET_ASSIST_GAIN': 'assist_gain',
        'TRIPLET_ASSIST_DEADZONE': 'assist_deadzone',
        'TRIPLET_ASSIST_MAX': 'assist_max',
        'TRIPLET_ASSIST_TAU': 'assist_tau',
    }


@dataclass
class GamepadConfig:
    """Gamepad axis/button mapping and scaling."""
    device: str = '/dev/input/js0'
    deadzone: float = 0.08
    speed_axis: int = 4                # Right stick Y
    yaw_axis: int = 3                  # Right stick X
    lean_axis: int = 1                 # Left stick Y
    max_distance: float = 1.0          # m
    max_yaw_rate: float = 2.0          # rad/s
    max_lean: float = math.radians(30)
    mode_button: int = 4               # LB on F710 (XInput)
    target_marker_height: float = 0.3  # m

    _FLAT_KEYS: ClassVar[dict] = {
        'GAMEPAD_DEVICE': 'device',
        'GAMEPAD_DEADZONE': 'deadzone',
        'GAMEPAD_SPEED_AXIS': 'speed_axis',
        'GAMEPAD_YAW_AXIS': 'yaw_axis',
        'GAMEPAD_LEAN_AXIS': 'lean_axis',
        'GAMEPAD_MAX_DISTANCE': 'max_distance',
        'GAMEPAD_MAX_YAW_RATE': 'max_yaw_rate',
        'GAMEPAD_MAX_LEAN': 'max_lean',
        'GAMEPAD_2WD_BUTTON': 'mode_button',
        'TARGET_MARKER_HEIGHT': 'target_marker_height',
    }


@dataclass
class TerrainConfig:
    """Terrain type and stair geometry."""
    terrain_type: str = 'flat'         # 'flat', 'heightfield', or 'box_stairs'
    ground_friction: float = 1.0
    stair_num_steps: int = 2
    stair_step_depth: float = 0.20     # m
    stair_step_height: List[float] = field(
        default_factory=lambda: [0.1, 0.15])
    stair_width: float = 0.6           # m
    stair_start_x: float = 0.5         # m

    _FLAT_KEYS: ClassVar[dict] = {
        'TERRAIN': 'terrain_type',
        'GROUND_FRICTION': 'ground_friction',
        'STAIR_NUM_STEPS': 'stair_num_steps',
        'STAIR_STEP_DEPTH': 'stair_step_depth',
        'STAIR_STEP_HEIGHT': 'stair_step_height',
        'STAIR_WIDTH': 'stair_width',
        'STAIR_START_X': 'stair_start_x',
    }


# ============================================================================
# Root configuration
# ============================================================================

_SENTINEL = object()


@dataclass
class Config:
    """Root configuration — contains all sub-configs.

    Supports dict-like access for backward compatibility::

        CONFIG['GRAVITY']          # shim → CONFIG.sim.gravity
        CONFIG.get('LQR_R', 2.0)  # shim → CONFIG.lqr.r

    All existing ``config['KEY']`` / ``self.cfg['KEY']`` / ``self.cfg.get(...)``
    calls work without changes.
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

    # Pre-built flat dict for backward compat (populated in __post_init__)
    _flat: dict = field(default_factory=dict, repr=False, init=False,
                        compare=False)

    # Ordered list of sub-config attribute names (for iteration)
    _SUB_NAMES: ClassVar[list] = [
        'sim', 'robot', 'motor', 'imu', 'control', 'plant',
        'pid', 'lqr', 'mpc', 'triplet', 'gamepad', 'terrain',
    ]

    def __post_init__(self):
        self._rebuild_flat()

    def _rebuild_flat(self):
        """(Re-)build the flat dict from all sub-config fields."""
        self._flat.clear()
        for name in self._SUB_NAMES:
            sub = getattr(self, name)
            for flat_key, field_name in sub._FLAT_KEYS.items():
                self._flat[flat_key] = getattr(sub, field_name)

    # --- dict-like interface (backward-compat shim) ----------------------

    def __getitem__(self, key: str):
        return self._flat[key]

    def get(self, key: str, default=_SENTINEL):
        if default is _SENTINEL:
            return self._flat.get(key)
        return self._flat.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._flat

    def __iter__(self):
        return iter(self._flat)

    def keys(self):
        return self._flat.keys()

    def values(self):
        return self._flat.values()

    def items(self):
        return self._flat.items()

    def to_flat_dict(self) -> dict:
        """Return a plain dict copy (e.g. for serialisation)."""
        return dict(self._flat)


# ============================================================================
# Factory
# ============================================================================

def load_config(**overrides) -> Config:
    """Create a Config with defaults, applying any flat-key overrides.

    Usage::

        cfg = load_config()                    # all defaults
        cfg = load_config(GRAVITY=-10.0)       # override one key
        cfg = load_config(**{'LQR_R': 3.0})    # override via dict

    Returns a Config that behaves like a dict (backward compat) AND
    provides typed attribute access (new style).
    """
    cfg = Config()
    if overrides:
        # Build reverse map: flat_key → (sub_config_attr_name, field_name)
        reverse: dict[str, tuple[str, str]] = {}
        for sub_name in Config._SUB_NAMES:
            sub = getattr(cfg, sub_name)
            for flat_key, field_name in sub._FLAT_KEYS.items():
                reverse[flat_key] = (sub_name, field_name)
        for key, value in overrides.items():
            if key not in reverse:
                raise KeyError(f"Unknown config key: {key!r}")
            sub_name, field_name = reverse[key]
            setattr(getattr(cfg, sub_name), field_name, value)
        cfg._rebuild_flat()
    return cfg
