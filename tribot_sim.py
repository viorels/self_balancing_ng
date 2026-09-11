#!/usr/bin/env python3
"""
Tribot Self-Balancing Robot Simulation — thin orchestration loop (MuJoCo).

Wires together: config, robot, input, controller, physics, telemetry.
No control logic, no sensor code, no motor physics lives here.

USAGE:
    python3 tribot_sim.py
"""

import math
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from config import load_config
from robot import TribotBalanceBot, _euler_to_mj_quat
from robot_state import DriveMode
from input.gamepad import Gamepad
from input.input_manager import InputManager
from plotjuggler_udp import PlotJugglerStreamer
from terrain import get_terrain_xml, post_load_terrain
from mcp.sim_bridge import SimBridge


CONFIG = load_config()


# ============================================================================
# SCENE BUILDER
# ============================================================================

def _build_scene_xml(mjcf_path, config):
    """Read base MJCF, inject terrain fragments, return complete XML string."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Resolve meshdir to absolute path (needed when loading via from_xml_string)
    compiler = root.find('compiler')
    if compiler is not None:
        meshdir = compiler.get('meshdir', '')
        if meshdir and not os.path.isabs(meshdir):
            abs_meshdir = os.path.normpath(
                os.path.join(os.path.dirname(mjcf_path), meshdir))
            compiler.set('meshdir', abs_meshdir + '/')

    terrain = get_terrain_xml(config)
    worldbody = root.find('worldbody')

    # Inject terrain asset (heightfield declaration)
    if terrain.get('asset'):
        asset = root.find('asset')
        if asset is None:
            asset = ET.SubElement(root, 'asset')
        for line in terrain['asset'].strip().split('\n'):
            line = line.strip()
            if line:
                asset.append(ET.fromstring(line))

    # Inject terrain worldbody geoms at start of worldbody
    wb_xml = terrain.get('worldbody', '')
    insert_idx = 0
    for line in wb_xml.strip().split('\n'):
        line = line.strip()
        if line:
            worldbody.insert(insert_idx, ET.fromstring(line))
            insert_idx += 1

    return ET.tostring(root, encoding='unicode')


# ============================================================================
# KEYBOARD CALLBACK
# ============================================================================

_reset_requested = False


def _key_callback(keycode):
    global _reset_requested
    # MuJoCo viewer key codes: 'r' = 82 (uppercase) or check for glfw code
    if keycode == ord('r') or keycode == ord('R'):
        _reset_requested = True


# ============================================================================
# VISUAL MARKER
# ============================================================================

def _draw_marker(viewer, x, y, h, color):
    """Draw a vertical line marker at (x, y) using the viewer scene."""
    if viewer is None:
        return
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array([*color, 1.0], dtype=np.float32),
    )
    mujoco.mjv_connector(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        0.003,  # width
        np.array([x, y, 0.0]),
        np.array([x, y, h]),
    )
    scn.ngeom += 1


# ============================================================================
# SIMULATION MAIN LOOP
# ============================================================================

def _print_config_summary(model, robot, config):
    """Print a one-time startup banner with key parameters."""
    total_mass = sum(model.body_mass)
    ctrl_type = config.sim.controller.upper()
    print(f"\nRobot total mass: {total_mass:.3f} kg")
    print(f"Controller: {ctrl_type}")
    if ctrl_type == 'PID':
        print(f"  Inner PID (pitch->torque): Kp={config.pid.kp}, "
              f"Ki={config.pid.ki}, Kd={config.pid.kd}")
        print(f"  Outer PID (pos->pitch):   Kp={config.pid.pos_kp}, "
              f"Ki={config.pid.pos_ki}, Kd={config.pid.pos_kd}, "
              f"max_pitch={math.degrees(config.pid.pos_max_pitch):.1f}")
    elif ctrl_type == 'MPC':
        print(f"  MPC: N={config.mpc.horizon} x {config.mpc.dt_pred * 1e3:.0f} ms, "
              f"solved at {config.control.control_rate_hz} Hz (OSQP)")
        print(f"  MPC Q_diag={config.mpc.q_diag}")
        print(f"  MPC R_diag={config.mpc.r_diag}  Rd_diag={config.mpc.rd_diag}")
        print(f"  Flip trigger: {'on' if config.mpc.flip_enabled else 'off'}, "
              f"T_flip={config.mpc.flip_time * 1e3:.0f} ms, "
              f"transition={config.mpc.transition_time:.2f} s")
        print(f"  Plant: m_body={config.plant.body_mass}kg, "
              f"l_cog={config.plant.cog_height}m, "
              f"I_body={config.plant.body_inertia}kg*m^2")
    else:
        print(f"  Q_diag={config.lqr.q_diag}, R={config.lqr.r}")
        print(f"  Plant: m_body={config.plant.body_mass}kg, "
              f"m_wheel={config.plant.wheel_mass}kg, "
              f"l_cog={config.plant.cog_height}m, "
              f"I_body={config.plant.body_inertia}kg*m^2")
    print(f"Motor: tau={config.motor.tau*1000:.0f}ms lag, "
          f"back-EMF K={config.motor.back_emf_k}, "
          f"deadband={config.motor.deadband}Nm")
    print(f"IMU: complementary filter alpha={config.imu.comp_filter_alpha}, "
          f"gyro drift={config.imu.gyro_drift_rate} rad/s^2")
    print(f"Control: {config.control.control_rate_hz}Hz")
    print(f"Initial pitch: {math.degrees(config.sim.initial_pitch):.1f}  "
          f"height: {config.sim.initial_height:.3f}m")
    print("-" * 70)


def run_simulation():
    """Run the tribot self-balancing simulation."""
    global _reset_requested

    print("=" * 70)
    print("Tribot Self-Balancing Robot — MuJoCo Simulation")
    print("=" * 70)

    # --- Build scene XML (MJCF + terrain) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mjcf_path = os.path.join(script_dir, CONFIG.sim.mjcf_path)
    xml_string = _build_scene_xml(mjcf_path, CONFIG)

    # --- Load model and create data ---
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # Apply initial pitch via quaternion
    quat = _euler_to_mj_quat(0, CONFIG.sim.initial_pitch, 0)
    data.qpos[3:7] = quat

    # Forward kinematics before robot init
    mujoco.mj_forward(model, data)

    # Fill runtime terrain data (heightfield)
    post_load_terrain(model, CONFIG)

    # --- Robot ---
    print("\nLoading tribot MJCF...")
    robot = TribotBalanceBot(model, data, CONFIG)

    _print_config_summary(model, robot, CONFIG)

    # --- Viewer ---
    viewer = mujoco.viewer.launch_passive(
        model, data, key_callback=_key_callback)
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.lookat[:] = [0, 0, 0.15]
    viewer.opt.geomgroup[3] = 1   # terrain geoms live in group 3 (ray probes)

    # --- Input ---
    gp = Gamepad(CONFIG.gamepad.device, deadzone=CONFIG.gamepad.deadzone)
    inp = InputManager(gp, CONFIG)
    if gp.connected:
        print(f"Gamepad: right stick Y (axis {CONFIG.gamepad.speed_axis}) = distance, "
              f"X (axis {CONFIG.gamepad.yaw_axis}) = yaw")

    # --- Telemetry ---
    pj = PlotJugglerStreamer()
    print("PlotJuggler UDP streamer active on 127.0.0.1:9870")

    # --- AI bridge ---
    bridge = SimBridge(CONFIG)
    bridge.start()

    # --- Visual target marker ---
    marker_color = [0.0, 1.0, 0.0]
    marker_h = CONFIG.gamepad.target_marker_height

    # Persistent bridge overrides
    _bridge_vel_cmd = None
    _bridge_yaw = None
    _bridge_lean = None
    _bridge_ticks_left = 0

    sim_time = 0.0
    wall_start = time.monotonic()
    log_interval = 0.1
    last_log_time = 0.0

    # ==================================================================
    # Main loop: input -> sensors -> controller -> actuators -> step -> telem
    # ==================================================================
    while sim_time < CONFIG.sim.sim_duration and viewer.is_running():
        # --- Drain bridge command queue ---
        for cmd in bridge.pop_commands():
            ctype = cmd.get("type")
            if ctype == "drive":
                _bridge_vel_cmd = cmd.get("vel", 0.0)
                _bridge_yaw = cmd["yaw"]
                _bridge_ticks_left = cmd.get("ticks", 500)
            elif ctype == "lean":
                _bridge_lean = cmd["lean_rad"]
                _bridge_ticks_left = 5000
            elif ctype == "target_position":
                robot.controller.set_target_position(cmd["position"])
            elif ctype == "set_drive_mode":
                target = DriveMode.TWO_WD if cmd["mode"] == "2wd" else DriveMode.FOUR_WD
                robot.set_drive_mode(target)
            elif ctype == "reset":
                robot.reset()
                _bridge_vel_cmd = _bridge_yaw = _bridge_lean = None
                _bridge_ticks_left = 0
                sim_time = 0.0
                last_log_time = 0.0
                wall_start = time.monotonic()

        # --- Keyboard shortcuts ---
        if _reset_requested:
            _reset_requested = False
            print("[key] R pressed — resetting robot")
            robot.reset()
            _bridge_vel_cmd = _bridge_yaw = _bridge_lean = None
            _bridge_ticks_left = 0
            sim_time = 0.0
            last_log_time = 0.0
            wall_start = time.monotonic()

        # --- Input (gamepad / autonomy) ---
        goals, mode_toggle, marker = inp.update(
            robot.position, robot.get_world_pose_2d()
        )
        robot.controller.set_velocity_command(goals.velocity_command)
        robot.controller.set_yaw_rate(goals.yaw_rate)
        robot.controller.set_lean(goals.pitch_bias)
        if mode_toggle:
            robot.toggle_drive_mode()

        # --- Apply bridge overrides (take priority over gamepad) ---
        if _bridge_ticks_left > 0:
            if _bridge_vel_cmd is not None:
                robot.controller.set_velocity_command(_bridge_vel_cmd)
            if _bridge_yaw is not None:
                robot.controller.set_yaw_rate(_bridge_yaw)
            if _bridge_lean is not None:
                robot.controller.set_lean(_bridge_lean)
            _bridge_ticks_left -= 1

        # --- Update visual marker at controller's target position ---
        viewer.user_scn.ngeom = 0  # clear previous frame's markers
        if inp.connected:
            rx, ry, _, fwd_x, fwd_y = robot.get_world_pose_2d()
            dist = robot.controller.target_position - robot.position
            marker.x = rx + dist * fwd_x
            marker.y = ry + dist * fwd_y
            _draw_marker(viewer, marker.x, marker.y, marker_h, marker_color)

        # --- Sensors -> controller -> actuators ---
        robot.update(sim_time, CONFIG.sim.timestep)

        # --- Physics step ---
        mujoco.mj_step(model, data)
        sim_time += CONFIG.sim.timestep

        # --- Sync viewer ---
        viewer.sync()

        # --- Telemetry: pull from robot + controller, merge, stream ---
        ctrl_telem = robot.controller.get_telemetry()
        robot_telem = robot.get_telemetry()
        telem = {
            "timestamp": sim_time,
            **robot_telem,
            **{f"ctrl/{k}": v for k, v in ctrl_telem.items()},
        }
        pj.send(telem)
        bridge.push_state(telem)

        # --- Periodic console log ---
        if sim_time - last_log_time >= log_interval:
            dbg = robot.get_debug_state()
            _, ry, _ = dbg['euler_deg']
            wv = dbg['wheel_vel']
            ta = dbg.get('triplet_ang', (0.0, 0.0))
            at0, at1 = robot.actual_torques
            print(
                f"[{sim_time:5.2f}s] "
                f"Euler(p={ry:6.1f}) | "
                f"TgtPitch:{math.degrees(robot.controller.target_pitch):5.2f} | "
                f"TripAng({math.degrees(ta[0]):5.1f},{math.degrees(ta[1]):5.1f}) "
                f"Whl({wv[0]:5.1f},{wv[1]:5.1f})rad/s | "
                f"Act:{at0:6.3f},{at1:6.3f}"
            )
            last_log_time = sim_time

        # Realtime pacing: sleep only the remaining time until next step
        wall_target = wall_start + sim_time
        wall_remaining = wall_target - time.monotonic()
        if wall_remaining > 0:
            time.sleep(wall_remaining)

    # --- Shutdown ---
    print("-" * 70)
    final_pitch_deg = math.degrees(robot.pitch_angle)
    print(f"\nSimulation complete! Final pitch: {final_pitch_deg:.2f}")
    if abs(final_pitch_deg) < 10:
        print("Robot balanced!")
    else:
        print("Robot fell.")

    bridge.stop()
    pj.close()
    viewer.close()
    print("Simulation ended.")


if __name__ == "__main__":
    run_simulation()
