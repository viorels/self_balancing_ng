#!/usr/bin/env python3
"""
Headless scenario runner for the tribot simulation (no viewer, no pacing).

Builds the same scene as tribot_sim.py and drives TribotBalanceBot through
scripted events (velocity commands, mode toggles, pushes), then prints a
summary: whether the robot stayed up, peak pitch, position error, torque
saturation, and MPC solve times.

Usage:
    .venv/bin/python tools/run_headless.py [scenario ...] [--controller mpc|lqr]
                                           [--duration S] [--noise 0|1] [--seed N]
Scenarios: balance4, balance2, drive4, drive2, transition, push4, push2,
           flip, stairs, all   (stairs: --mode 4wd|2wd, --speed; box stairs)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config                          # noqa: E402
from robot import TribotBalanceBot, _euler_to_mj_quat   # noqa: E402
from robot_state import DriveMode                       # noqa: E402
from terrain import post_load_terrain                   # noqa: E402
from tribot_sim import _build_scene_xml                 # noqa: E402


class Quiet:
    def __enter__(self):
        self._out = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *exc):
        sys.stdout.close()
        sys.stdout = self._out


def make_robot(cfg, triplet=0.0, pitch=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml = _build_scene_xml(os.path.join(root, cfg.sim.mjcf_path), cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    cfg.sim.initial_triplet_angle = triplet
    if abs(triplet) > 1e-6:
        cfg.sim.initial_height = 0.178
    data.qpos[2] = cfg.sim.initial_height
    data.qpos[3:7] = _euler_to_mj_quat(
        0, cfg.sim.initial_pitch if pitch is None else pitch, 0)
    mujoco.mj_forward(model, data)
    post_load_terrain(model, cfg)
    with Quiet():
        robot = TribotBalanceBot(model, data, cfg)
    return model, data, robot


def push(robot, impulse_ns, duration=0.05):
    """Schedule a forward push (N·s) on the body CoG as an xfrc force."""
    force = impulse_ns / duration
    return {'t_end': None, 'force': force, 'duration': duration}


def run(cfg, scenario, duration, verbose=False, push_ns=3.0, speed=0.4,
        mode='2wd', log_dt=0.1):
    events = []          # (time, callable)
    triplet0 = 0.0
    if scenario in ('balance2', 'drive2', 'push2', 'flip') or \
            (scenario == 'stairs' and mode == '2wd'):
        triplet0 = math.pi / 3
    # Flat-ground scenarios must not depend on the configured terrain.
    cfg.terrain.terrain_type = 'box_stairs' if scenario == 'stairs' else 'flat'
    model, data, robot = make_robot(cfg, triplet=triplet0)
    ctrl = robot.controller
    if triplet0 != 0.0:
        robot.drive_mode = DriveMode.TWO_WD
        robot.triplet_base_angle = triplet0
        robot.triplet_ctrl_L.drive_mode = robot.drive_mode
        robot.triplet_ctrl_R.drive_mode = robot.drive_mode

    push_force = 0.0
    push_until = -1.0

    def do_push(impulse, dur=0.05):
        def f(t):
            nonlocal push_force, push_until
            push_force = impulse / dur
            push_until = t + dur
        return f

    if scenario in ('drive4', 'drive2'):
        events += [(1.0, lambda t: ctrl.set_velocity_command(0.5)),
                   (3.0, lambda t: ctrl.set_velocity_command(0.0)),
                   (4.0, lambda t: ctrl.set_yaw_rate(1.0)),
                   (5.0, lambda t: ctrl.set_yaw_rate(0.0))]
    elif scenario == 'transition':
        events += [(1.0, lambda t: robot.toggle_drive_mode()),
                   (3.5, lambda t: ctrl.set_velocity_command(0.3)),
                   (5.0, lambda t: ctrl.set_velocity_command(0.0)),
                   (6.0, lambda t: robot.toggle_drive_mode())]
    elif scenario == 'push4':
        events += [(1.0, do_push(1.5)), (3.0, do_push(-1.5)), (5.0, do_push(2.5))]
    elif scenario == 'push2':
        events += [(1.0, do_push(0.8)), (3.0, do_push(-0.8)), (5.0, do_push(1.5))]
    elif scenario == 'flip':
        events += [(1.5, do_push(push_ns, 0.05))]
    elif scenario == 'stairs':
        events += [(1.0, lambda t: ctrl.set_velocity_command(speed))]

    dt = cfg.sim.timestep
    n_steps = int(duration / dt)
    body = robot.body_id
    fwd_world = np.array([-1.0, 0.0, 0.0])

    peak_pitch = 0.0
    sat_ticks = 0
    ticks = 0
    fell = False
    log = []
    min_x = 0.0
    max_z = 0.0
    pos_err_after = []
    t_wall0 = time.perf_counter()
    yaw_rates = []
    for i in range(n_steps):
        t = i * dt
        while events and events[0][0] <= t:
            events.pop(0)[1](t)
        data.xfrc_applied[body, :] = 0.0
        if t < push_until:
            data.xfrc_applied[body, :3] = fwd_world * push_force
        robot.update(t, dt)
        mujoco.mj_step(model, data)
        ticks += 1

        s = robot.state
        peak_pitch = max(peak_pitch, abs(s.true_pitch))
        min_x = min(min_x, float(data.qpos[0]))
        max_z = max(max_z, float(data.qpos[2]))
        if max(abs(v) for v in robot.actual_torques) > 0.98 * cfg.motor.max_torque:
            sat_ticks += 1
        if t > 1.5:
            pos_err_after.append(ctrl.target_position - s.position)
        if scenario.startswith('drive') and 4.2 < t < 5.0:
            yaw_rates.append(s.yaw_rate)
        if robot.check_fallen():
            fell = True
            break
        if verbose and i % max(1, int(round(log_dt / dt))) == 0:
            tel = ctrl.get_telemetry()
            log.append((t, math.degrees(s.true_pitch), s.position,
                        tel.get('u_drive', 0.0), tel.get('u_triplet', 0.0),
                        tel.get('mode_4wd', -1), tel.get('step_phase', -1),
                        math.degrees(s.triplet_angle_L),
                        math.degrees(s.triplet_angle_R),
                        float(data.qpos[0]), float(data.qpos[2]),
                        s.forward_velocity, tel.get('planner_2wd', -1),
                        math.degrees(tel.get('z_lam', 0.0)),
                        math.degrees(tel.get('ref_lam', 0.0)),
                        tel.get('solve_ok', -1)))
    wall = time.perf_counter() - t_wall0
    tel = ctrl.get_telemetry()
    result = {
        'scenario': scenario,
        'fell': fell,
        't_end': ticks * dt,
        'peak_pitch_deg': math.degrees(peak_pitch),
        'sat_frac': sat_ticks / max(ticks, 1),
        'pos_err_rms': float(np.sqrt(np.mean(np.square(pos_err_after)))) if pos_err_after else 0.0,
        'final_pos_err': float(pos_err_after[-1]) if pos_err_after else 0.0,
        'solve_max_ms': tel.get('solve_max_ms', 0.0),
        'solve_fail': tel.get('solve_fail_cnt', 0.0),
        'yaw_rate_mean': float(np.mean(yaw_rates)) if yaw_rates else 0.0,
        'realtime_x': (ticks * dt) / wall,
        'flip_phase_end': tel.get('flip_phase', -1),
        'trip_L_deg': math.degrees(robot.state.triplet_angle_L),
        'trip_R_deg': math.degrees(robot.state.triplet_angle_R),
        'min_x': min_x,
        'max_z': max_z,
    }
    if verbose:
        print(f"    {'t':>5s} {'pitch':>7s} {'pos':>7s} {'u_d':>6s} {'u_t':>6s} "
              f"{'4wd':>4s} {'step':>4s} {'phiL':>7s} {'phiR':>7s} "
              f"{'x':>7s} {'z':>6s} {'v':>6s} {'p2':>3s} {'lam':>6s} {'lamr':>6s} "
              f"{'ok':>2s}")
        for row in log:
            print(f"    {row[0]:5.2f} {row[1]:7.2f} {row[2]:7.3f} {row[3]:6.2f} "
                  f"{row[4]:6.2f} {row[5]:4.0f} {row[6]:4.0f} {row[7]:7.1f} {row[8]:7.1f} "
                  f"{row[9]:7.3f} {row[10]:6.3f} {row[11]:6.2f} {row[12]:3.0f} "
                  f"{row[13]:6.1f} {row[14]:6.1f} {row[15]:2.0f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scenarios', nargs='*', default=['all'])
    ap.add_argument('--controller', default='mpc')
    ap.add_argument('--duration', type=float, default=8.0)
    ap.add_argument('--noise', type=int, default=1)
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--push', type=float, default=3.0,
                    help='impulse (N*s) for the flip scenario')
    ap.add_argument('--flip', type=int, default=1, help='enable the flip trigger')
    ap.add_argument('--speed', type=float, default=0.4,
                    help='velocity command (m/s) for the stairs scenario')
    ap.add_argument('--mode', default='2wd', help='start mode for stairs: 2wd|4wd')
    ap.add_argument('--log-dt', type=float, default=0.1, help='verbose log period (s)')
    ap.add_argument('--set', action='append', default=[],
                    help='override config, e.g. --set mpc.step_lean_margin=-0.05')
    ap.add_argument('--seed', type=int, default=None,
                    help='seed the sensor-noise RNG (reproducible runs)')
    args = ap.parse_args()

    names = args.scenarios
    if names == ['all']:
        names = ['balance4', 'balance2', 'drive4', 'drive2', 'transition',
                 'push4', 'push2', 'flip', 'stairs']
    ok_all = True
    for name in names:
        if args.seed is not None:
            np.random.seed(args.seed)
        cfg = load_config()
        cfg.sim.controller = args.controller
        cfg.imu.add_sensor_noise = bool(args.noise)
        cfg.mpc.flip_enabled = bool(args.flip)
        for item in args.set:
            path, val = item.split('=', 1)
            obj = cfg
            *parents, leaf = path.split('.')
            for part in parents:
                obj = getattr(obj, part)
            cur = getattr(obj, leaf)
            setattr(obj, leaf, type(cur)(val) if not isinstance(cur, bool)
                    else val.lower() in ('1', 'true', 'yes'))
        duration = max(args.duration, 14.0) if name == 'stairs' else args.duration
        res = run(cfg, name, duration, verbose=args.verbose,
                  push_ns=args.push, speed=args.speed, mode=args.mode,
                  log_dt=args.log_dt)
        ok = not res['fell']
        ok_all &= ok
        extra = ''
        if name == 'stairs':
            extra = f"min_x={res['min_x']:+.2f}m max_z={res['max_z']:.3f}m "
        print(f"[{'OK ' if ok else 'FELL'}] {name:10s} t={res['t_end']:.2f}s {extra}"
              f"peak_pitch={res['peak_pitch_deg']:.1f}deg sat={res['sat_frac']*100:.0f}% "
              f"pos_err_rms={res['pos_err_rms']:.3f}m final={res['final_pos_err']:+.3f}m "
              f"yaw={res['yaw_rate_mean']:+.2f} solve_max={res['solve_max_ms']:.2f}ms "
              f"fails={res['solve_fail']:.0f} flip={res['flip_phase_end']:.0f} "
              f"trip=({res['trip_L_deg']:.0f},{res['trip_R_deg']:.0f}) "
              f"rt={res['realtime_x']:.1f}x")
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
