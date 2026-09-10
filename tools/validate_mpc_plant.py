#!/usr/bin/env python3
"""
Validate the planar MPC plant model against MuJoCo (headless).

Checks, in order:
  1. Sign conventions: which way the simulator's pitch and hub joint angle
     tilt the body / move the wheels relative to the driving direction.
  2. Instantaneous accelerations for unit inputs and small displacements in
     4WD and 2WD, compared with the linearised model.

Run from the repository root:
    .venv/bin/python tools/validate_mpc_plant.py
"""

import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config                       # noqa: E402
from controllers.mpc_plant import (                   # noqa: E402
    ContactMode, PITCH_SIGN, PlanarPlant, PlantParams, leg_angle,
)
from robot import TribotBalanceBot, _euler_to_mj_quat  # noqa: E402
from terrain import post_load_terrain                 # noqa: E402
from tribot_sim import _build_scene_xml               # noqa: E402


class _Quiet:
    """Silence the robot's constructor chatter."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, *exc):
        sys.stdout.close()
        sys.stdout = self._stdout
        return False


def load(config, pitch=0.0, triplet=0.0):
    with _Quiet():
        return _load(config, pitch, triplet)


def _load(config, pitch=0.0, triplet=0.0):
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml = _build_scene_xml(os.path.join(script_dir, config.sim.mjcf_path), config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    config.sim.initial_triplet_angle = triplet
    data.qpos[3:7] = _euler_to_mj_quat(0, pitch, 0)
    mujoco.mj_forward(model, data)
    post_load_terrain(model, config)
    robot = TribotBalanceBot(model, data, config)
    return model, data, robot


def settle(model, data, robot, steps=250):
    """Let the contacts settle while pinning every DOF except vertical."""
    qpos0 = data.qpos.copy()
    keep = [2]  # z of the free joint
    for _ in range(steps):
        mujoco.mj_step(model, data)
        for i in range(model.nq):
            if i not in keep and i != 2:
                pass
        # Reset everything but z (qpos index 2) and its velocity (qvel 2)
        z, vz = data.qpos[2], data.qvel[2]
        data.qpos[:] = qpos0
        data.qvel[:] = 0.0
        data.qpos[2] = z
        data.qvel[2] = vz
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def apply(robot, u_d, u_t):
    """Apply total torques with the same convention as TribotBalanceBot.update."""
    data = robot.data
    data.ctrl[:] = 0.0
    for wheel_acts, trip_act in ((robot.act_l_wheels, robot.act_l_triplet),
                                 (robot.act_r_wheels, robot.act_r_triplet)):
        motor = u_d / 2.0
        data.ctrl[trip_act] = -motor + u_t / 2.0
        for wa in wheel_acts:
            data.ctrl[wa] = -motor / 3.0


def measure(model, data, robot, u_d, u_t, dt_window=0.01):
    """Return (hub_fwd_ddot, theta_ddot, lam_ddot) from a short torque window."""
    apply(robot, u_d, u_t)
    def state():
        _, prate = robot._get_true_state()
        phid = 0.5 * (data.qvel[robot._qvel_addr[robot.l_triplet_jnt]]
                      + data.qvel[robot._qvel_addr[robot.r_triplet_jnt]])
        theta_d = PITCH_SIGN * prate
        # Hub (body origin) forward velocity; forward = world -X at yaw 0.
        hub_fwd = -float(data.qvel[0])
        # lam = theta - phi + const  ->  lam_dot = theta_dot - phi_dot
        lam_d = theta_d - phid
        return np.array([hub_fwd, theta_d, lam_d])
    v0 = state()
    n = int(round(dt_window / model.opt.timestep))
    for _ in range(n):
        mujoco.mj_step(model, data)
    v1 = state()
    return (v1 - v0) / dt_window


def model_accel(plant, mode, lam0, theta0, u_d, u_t, dlam=0.0, dtheta=0.0):
    """Model (hub_fwd_ddot, theta_ddot, lam_ddot) at the given state/input."""
    A, B, c = plant.linearize(mode, lam0, theta0)
    z = np.zeros(6)
    z[2] = theta0 + dtheta
    z[4] = lam0 + dlam
    zd = A @ z + B @ np.array([u_d, u_t]) + c
    hub_ddot = zd[1]
    if mode == ContactMode.SINGLE_CONTACT:
        hub_ddot += plant.p.R * math.cos(lam0 + dlam) * zd[5]
    return np.array([hub_ddot, zd[3], zd[5]])


def main():
    cfg = load_config()
    cfg.imu.add_sensor_noise = False
    cfg.motor.torque_noise_std = 0.0
    plant = PlanarPlant(PlantParams.from_config(cfg))

    print("=== 1. Sign conventions ===")
    model, data, robot = load(cfg, pitch=0.2, triplet=0.0)
    mujoco.mj_forward(model, data)
    body = robot.body_id
    cog_world = data.xipos[body]
    base = data.xpos[body]
    fwd = np.array([-data.xmat[body].reshape(3, 3)[0, 0],
                    -data.xmat[body].reshape(3, 3)[1, 0], 0.0])
    lean_fwd = float(np.dot(cog_world - base, fwd))
    print(f"  measured pitch +0.2 -> CoG forward offset {lean_fwd:+.4f} m "
          f"({'forward lean' if lean_fwd > 0 else 'BACKWARD lean'})")
    print(f"  => theta = PITCH_SIGN * pitch with PITCH_SIGN = {PITCH_SIGN:+.0f} "
          f"is {'consistent' if (lean_fwd > 0) == (PITCH_SIGN > 0) else 'WRONG'}")

    model, data, robot = load(cfg, pitch=0.0, triplet=0.3)
    mujoco.mj_forward(model, data)
    # lowest wheel of the left triplet after +0.3 rad hub rotation
    names = ['l_wheel_1', 'l_wheel_2', 'l_wheel_3']
    zs = [data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)][2]
          for n in names]
    k = int(np.argmin(zs))
    wpos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names[k])]
    hub = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'l_triplet')]
    fwd = np.array([-1.0, 0.0, 0.0])
    off = float(np.dot(wpos - hub, fwd))
    print(f"  hub joint +0.3 -> lowest wheel ({names[k]}) is {off:+.4f} m "
          f"{'ahead of' if off > 0 else 'behind'} the hub")
    lam = leg_angle(0.0, 0.3)
    print(f"  model: leg_angle(0, 0.3) = {math.degrees(lam):+.1f} deg "
          f"(positive = hub ahead of contact) -> "
          f"{'consistent' if (lam > 0) == (off < 0) else 'WRONG'}")

    print("\n=== 2. 4WD accelerations (hub joint 0) ===")
    tests = [("u_d = +0.5 Nm", 0.5, 0.0, 0.0),
             ("u_t = +0.5 Nm", 0.0, 0.5, 0.0),
             ("theta0 = +0.05 rad, no input", 0.0, 0.0, 0.05)]
    for label, u_d, u_t, th0 in tests:
        # Keep the triplet locked to the ground: hub joint = -pitch.
        model, data, robot = load(cfg, pitch=PITCH_SIGN * th0,
                                  triplet=-PITCH_SIGN * th0)
        settle(model, data, robot)
        sim = measure(model, data, robot, u_d, u_t)
        mod = model_accel(plant, ContactMode.FOUR_WD, 0.0, 0.0, u_d, u_t, dtheta=th0)
        print(f"  {label:32s} sim: hub''={sim[0]:+7.3f} th''={sim[1]:+7.3f} | "
              f"model: hub''={mod[0]:+7.3f} th''={mod[1]:+7.3f}")

    print("\n=== 3. 2WD accelerations (hub joint +60 deg) ===")
    phi0 = math.pi / 3
    tests = [("u_d = +0.5 Nm", 0.5, 0.0, 0.0, 0.0),
             ("u_t = +0.5 Nm", 0.0, 0.5, 0.0, 0.0),
             ("theta0 = +0.05, no input", 0.0, 0.0, 0.05, 0.0),
             ("phi = +60deg+0.05, no input", 0.0, 0.0, 0.0, 0.05)]
    for label, u_d, u_t, th0, dphi in tests:
        model, data, robot = load(cfg, pitch=PITCH_SIGN * th0, triplet=phi0 + dphi)
        # raise to 2WD ground height so the contact settles quickly
        data.qpos[2] = plant.p.r + plant.p.R + 0.001
        settle(model, data, robot)
        sim = measure(model, data, robot, u_d, u_t)
        lam = leg_angle(th0, phi0 + dphi)
        mod = model_accel(plant, ContactMode.SINGLE_CONTACT, 0.0, 0.0,
                          u_d, u_t, dlam=lam, dtheta=th0)
        print(f"  {label:32s} sim: hub''={sim[0]:+7.3f} th''={sim[1]:+7.3f} "
              f"lam''={sim[2]:+7.3f} | model: hub''={mod[0]:+7.3f} "
              f"th''={mod[1]:+7.3f} lam''={mod[2]:+7.3f}")

    print("\n=== 4. Divergence rates ===")
    print(f"  4WD  : {plant.unstable_rate(ContactMode.FOUR_WD):.2f} rad/s")
    print(f"  2WD  : {plant.unstable_rate(ContactMode.SINGLE_CONTACT):.2f} rad/s")
    print(f"  tipping torque (4WD): {plant.p.tipping_torque:.2f} Nm")


if __name__ == '__main__':
    main()
