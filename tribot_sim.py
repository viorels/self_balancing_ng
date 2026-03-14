#!/usr/bin/env python3
"""
Tribot Self-Balancing Robot Simulation — thin orchestration loop.

Wires together: config, robot, input, controller, physics, telemetry.
No control logic, no sensor code, no motor physics lives here.

USAGE:
    python3 tribot_sim.py
"""

import time
import math

import pybullet as p
import pybullet_data

from config import load_config
from robot import TribotBalanceBot
from robot_state import DriveMode
from input.gamepad import Gamepad
from input.input_manager import InputManager
from plotjuggler_udp import PlotJugglerStreamer
from terrain import create_terrain


CONFIG = load_config()


# ============================================================================
# SIMULATION MAIN LOOP
# ============================================================================

def _print_config_summary(robot, config):
    """Print a one-time startup banner with key parameters."""
    total_mass = sum(p.getDynamicsInfo(robot.body_id, i)[0]
                     for i in range(-1, p.getNumJoints(robot.body_id)))
    ctrl_type = config.sim.controller.upper()
    print(f"\nRobot total mass: {total_mass:.3f} kg")
    print(f"Controller: {ctrl_type}")
    if ctrl_type == 'PID':
        print(f"  Inner PID (pitch→torque): Kp={config.pid.kp}, "
              f"Ki={config.pid.ki}, Kd={config.pid.kd}")
        print(f"  Outer PID (pos→pitch):   Kp={config.pid.pos_kp}, "
              f"Ki={config.pid.pos_ki}, Kd={config.pid.pos_kd}, "
              f"max_pitch={math.degrees(config.pid.pos_max_pitch):.1f}°")
    elif ctrl_type == 'MPC':
        print(f"  MPC rate: {config.mpc.rate_hz}Hz, N={config.mpc.horizon}, "
              f"sim_solve={config.mpc.simulated_solve_ms}ms")
        print(f"  MPC Q_diag={config.mpc.q_diag}")
        print(f"  MPC R_diag={config.mpc.r_diag}")
        print(f"  Plant: m_body={config.plant.body_mass}kg, "
              f"m_wheel={config.plant.wheel_mass}kg, "
              f"l_cog={config.plant.cog_height}m, "
              f"I_body={config.plant.body_inertia}kg·m²")
    else:
        print(f"  Q_diag={config.lqr.q_diag}, R={config.lqr.r}")
        print(f"  Plant: m_body={config.plant.body_mass}kg, "
              f"m_wheel={config.plant.wheel_mass}kg, "
              f"l_cog={config.plant.cog_height}m, "
              f"I_body={config.plant.body_inertia}kg·m²")
    print(f"Motor: τ={config.motor.tau*1000:.0f}ms lag, "
          f"back-EMF K={config.motor.back_emf_k}, "
          f"deadband={config.motor.deadband}Nm")
    print(f"IMU: complementary filter α={config.imu.comp_filter_alpha}, "
          f"gyro drift={config.imu.gyro_drift_rate} rad/s²")
    print(f"Control: {config.control.control_rate_hz}Hz, "
          f"{config.control.sensor_to_actuator_delay_steps} step pipeline delay")
    print(f"Initial pitch: {math.degrees(config.sim.initial_pitch):.1f}°  "
          f"height: {config.sim.initial_height:.3f}m")
    print("-" * 70)


def run_simulation():
    """Run the tribot self-balancing simulation."""

    print("=" * 70)
    print("Tribot Self-Balancing Robot — URDF-based Simulation")
    print("=" * 70)

    # --- Physics engine ---
    physics_client = p.connect(p.GUI, options="--width=1920 --height=1080 --maximized")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, CONFIG.sim.gravity)
    p.setPhysicsEngineParameter(fixedTimeStep=CONFIG.sim.timestep, numSubSteps=1)

    # --- Environment + robot ---
    create_terrain(CONFIG)
    print("\nLoading tribot URDF...")
    robot = TribotBalanceBot(physics_client, CONFIG)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.2, cameraYaw=0, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.15]
    )
    _print_config_summary(robot, CONFIG)

    # --- Input ---
    gp = Gamepad(CONFIG.gamepad.device, deadzone=CONFIG.gamepad.deadzone)
    inp = InputManager(gp, CONFIG)
    if gp.connected:
        print(f"Gamepad: right stick Y (axis {CONFIG.gamepad.speed_axis}) = distance, "
              f"X (axis {CONFIG.gamepad.yaw_axis}) = yaw")

    # --- Telemetry ---
    pj = PlotJugglerStreamer()
    print("PlotJuggler UDP streamer active on 127.0.0.1:9870")

    # --- Visual target marker ---
    marker_id = -1
    marker_color = [0.0, 1.0, 0.0]
    marker_h = CONFIG.gamepad.target_marker_height

    sim_time = 0.0
    log_interval = 0.1
    last_log_time = 0.0

    # ==================================================================
    # Main loop: input → sensors → controller → actuators → step → telem
    # ==================================================================
    while sim_time < CONFIG.sim.sim_duration:
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

        # --- Sensors → controller → actuators ---
        robot.update(sim_time, CONFIG.sim.timestep)

        # --- Physics step ---
        p.stepSimulation()
        sim_time += CONFIG.sim.timestep

        # --- Telemetry: pull from robot + controller, merge, stream ---
        ctrl_telem = robot.controller.get_telemetry()
        robot_telem = robot.get_telemetry()
        pj.send({
            "timestamp": sim_time,
            **robot_telem,
            **{f"ctrl/{k}": v for k, v in ctrl_telem.items()},
        })

        # --- Fall detection ---
        if robot.check_fallen():
            print(f"\n[{sim_time:.2f}s] Robot fell over!")
            break

        # --- Periodic console log ---
        if sim_time - last_log_time >= log_interval:
            dbg = robot.get_debug_state()
            _, ry, _ = dbg['euler_deg']
            wv = dbg['wheel_vel']
            ta = dbg.get('triplet_ang', (0.0, 0.0))
            at0, at1 = robot.actual_torques
            print(
                f"[{sim_time:5.2f}s] "
                f"Euler(p={ry:6.1f})° | "
                f"TgtPitch:{math.degrees(robot.controller.target_pitch):5.2f}° | "
                f"TripAng({math.degrees(ta[0]):5.1f},{math.degrees(ta[1]):5.1f})° "
                f"Whl({wv[0]:5.1f},{wv[1]:5.1f})rad/s | "
                f"Act:{at0:6.3f},{at1:6.3f}"
            )
            last_log_time = sim_time

        time.sleep(CONFIG.sim.timestep)

    # --- Shutdown ---
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
