# Robot Project Context

This repository contains simulation and control code for a self-balancing triplet robot.

## Robot Structure

The robot's physical structure and design is described in:
- `docs/TRIPLET_BALANCING_ROBOT.md` — human-readable description of the robot's structure, components, and design decisions
- `tribot_description/mjcf/tribot.xml` — contains the definitions of robot's links, joints, sensors, and inertial properties

When answering questions or generating code, always assume familiarity with this robot's structure. Refer to the URDF and the markdown description for joint names, link names, sensor placements, and physical parameters.

## Key conventions
- The robot is simulated using MuJoCo
- Main control strategy is LQR (`control_lqr.py`)
- The main simulation entry point is `tribot_sim.py`
