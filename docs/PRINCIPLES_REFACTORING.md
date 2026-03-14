# Design Principles & Refactoring Plan

## Problem Statement

`tribot_sim.py` has grown into a ~900-line monolith mixing robot modelling,
sensor simulation, motor physics, controller wiring, gamepad input, telemetry
streaming, and the main loop.  Shared state is scattered across classes via
ad-hoc attribute access and `hasattr()` branching.

---

## Design Principles

### 1. One canonical state struct, owned by the robot

All sensors write into a single `RobotState` dataclass.  All controllers
read from it.  No other code mutates it.

### 2. Controllers are pure functions of state → commands

Every controller receives `RobotState` + `ControlGoals` and returns a
`ControlOutput`.  Controllers never import PyBullet, never call simulation
APIs, and never hold references to the robot.

### 3. Uniform controller interface — no `hasattr()` polymorphism

All controllers implement `BalanceControllerBase`:

```python
class BalanceControllerBase(ABC):
    def update(self, state: RobotState, goals: ControlGoals) -> ControlOutput: ...
    def get_telemetry(self) -> dict: ...
```

Optional signals default to zero/None inside `ControlOutput`, not behind
runtime attribute checks.

### 4. Telemetry is pull-based

Each subsystem exposes `get_telemetry() → dict`.  The sim loop collects,
prefixes, merges, and streams.  No reaching into class internals for
diagnostics.

### 5. Config is namespaced and typed

Each subsystem receives its own config dataclass (`SimConfig`, `MotorConfig`,
`LQRConfig`, …).  The flat `CONFIG` dict is replaced incrementally via a
`load_config()` shim.

### 6. One file, one responsibility

If a class needs its own test fixture, it belongs in its own module.

### 7. The sim loop is thin

`tribot_sim.py` only orchestrates:  
input → sensors → controller → actuators → physics step → telemetry.  
No control logic, no mode switching, no gain calculations.

---

## Data Structures

| Struct | Purpose | Written by | Read by |
|--------|---------|-----------|---------|
| `RobotState` | Measured/estimated state (pitch, rates, odometry, triplet angles) | `TribotRobot.read_sensors()` | Controllers, telemetry |
| `ControlOutput` | Drive + triplet torques, targets | Controllers | `TribotRobot.apply_control()` |
| `ControlGoals` | Operator commands (target position, yaw, lean, mode) | `InputManager` | Controllers |
| `Telemetry` | Flat `dict` aggregated per tick for UDP streaming | Sim loop (merges from subsystems) | `PlotJugglerStreamer` |

---

## Target File Structure

```
self_balancing_ng/
├── tribot_sim.py              # Thin sim loop (~150 lines)
├── robot_state.py             # RobotState, ControlOutput, ControlGoals, DriveMode, Telemetry
├── robot.py                   # TribotRobot: URDF, sensors, actuators, belt constraints
├── config.py                  # Namespaced config dataclasses + load_config()
├── motor_model.py             # BrushlessMotorModel
├── imu_model.py               # IMUSensorModel
├── triplet_controller.py      # TripletController + geometry helpers
├── controllers/
│   ├── __init__.py            # create_controller() factory
│   ├── base.py                # BalanceControllerBase (ABC)
│   ├── control_pid.py         # PIDBalanceController
│   ├── control_lqr.py         # LQRBalanceController
│   └── control_mpc_hybrid.py  # MPCHybridController
├── input/
│   ├── __init__.py
│   ├── gamepad.py             # Gamepad (low-level)
│   └── input_manager.py       # InputManager: gamepad → ControlGoals
├── telemetry/
│   ├── __init__.py
│   └── plotjuggler_udp.py     # PlotJugglerStreamer
├── terrain.py                 # create_terrain()
├── docs/
│   ├── TRIPLET_BALANCING_ROBOT.md
│   └── PRINCIPLES_REFACTORING.md   # ← this file
└── tribot_description/
    └── urdf/tribot.urdf
```

---

## Responsibility Matrix

| Component | Reads | Writes | Uses PyBullet? |
|-----------|-------|--------|----------------|
| `tribot_sim.py` | everything | sim_time, physics steps, telemetry | Yes |
| `TribotRobot` | `ControlOutput` | `RobotState`, PyBullet torques | Yes |
| `BalanceControllerBase` | `RobotState`, `ControlGoals` | `ControlOutput` | **No** |
| `TripletController` | `RobotState` (pitch, triplet angles) | triplet torques in `ControlOutput` | **No** |
| `InputManager` | Gamepad axes/buttons, `RobotState.position` | `ControlGoals` | No |
| `BrushlessMotorModel` | commanded torque, wheel velocity | actual torque | No |
| `IMUSensorModel` | true pitch/rate | fused pitch/rate | No |
| `PlotJugglerStreamer` | `Telemetry.data` | UDP socket | No |

---

## Migration Steps

Each step is a **single commit** that must pass `python3 tribot_sim.py` without
regression.

| Step | Description | Risk |
|------|-------------|------|
| 1 | Create `robot_state.py` with all data structs. Import everywhere, don't use yet. | Zero |
| 2 | Extract `motor_model.py`, `imu_model.py`, `triplet_controller.py`. Re-import in `tribot_sim.py`. | Low |
| 3 | Create `controllers/base.py`. Make `LQRBalanceController` implement it. Replace `hasattr()` with `get_telemetry()`. | Medium |
| 4 | Create `TribotRobot.read_sensors()` that populates `RobotState`. Move sensor code out of the update loop. | Medium |
| 5 | Refactor `TribotRobot.update()` → `apply_control(ControlOutput)`. | Medium |
| 6 | Extract `InputManager` from the gamepad block in the main loop. | Low |
| 7 | Replace flat `CONFIG` dict with typed `Config` dataclass + `load_config()` shim. | Low (tedious) |
| 8 | Slim `tribot_sim.py` to the thin orchestration loop. | Final cleanup |