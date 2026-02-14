#!/usr/bin/env python3
"""
Self-Balancing Robot Prototype using PyBullet and PID Control

This script simulates a 2-wheel differential-drive self-balancing robot (inverted pendulum)
using PyBullet physics simulation. The robot is stabilized using a PID controller that
regulates the pitch angle and pitch rate.

REQUIREMENTS:
    - PyBullet: pip install pybullet
    - NumPy: pip install numpy

USAGE:
    python3 balance_bot.py

The simulation will:
    1. Create a 2-wheel self-balancing robot
    2. Give it a small initial tilt
    3. Attempt to balance using PID control
    4. Run for ~10 seconds (configurable)
    5. Display pitch angle, control torque, and other metrics

Parameters can be easily tuned by modifying the CONFIG dictionary at the top of the file.
"""

import time
import math
import numpy as np
import pybullet as p
import pybullet_data


# ============================================================================
# CONFIGURATION - All parameters are easily tunable here
# ============================================================================

CONFIG = {
    # Simulation parameters
    'GRAVITY': -9.81,
    'TIMESTEP': 1.0 / 500.0,  # 500 Hz simulation frequency
    'SIM_DURATION': 60.0,  # seconds
    'GROUND_FRICTION': 0.8,
    
    # Robot physical parameters
    'ROBOT_MASS': 1.1,  # kg
    'BODY_HEIGHT': 0.15,  # m
    'BODY_WIDTH': 0.08,  # m
    'BODY_DEPTH': 0.08,  # m
    'WHEEL_DIAMETER': 0.095,  # m
    'WHEEL_MASS': 0.15,  # kg each
    'WHEEL_FRICTION': 1.0,
    'AXLE_WIDTH': 0.10,  # distance between wheels
    
    # Initial conditions
    'INITIAL_PITCH': 0.0,  # rad (~5.7 degrees)
    'INITIAL_HEIGHT': 0.08,  # m above ground
    
    # PID Controller gains
    # These are the main tuning parameters for balance control
    'PID_KP': 15.0,   # proportional gain (pitch angle error)
    'PID_KD': 1.0,    # derivative gain (pitch rate)
    'PID_KI': 0.5,    # integral gain (accumulated pitch error)
    
    # Motor/actuator limits
    'MAX_TORQUE': 2.0,  # Nm (motor saturation limit)
    
    # Sensor simulation
    'IMU_ANGLE_NOISE_STD': 0.01,  # rad, standard deviation of angle noise
    'IMU_GYRO_NOISE_STD': 0.05,   # rad/s, standard deviation of gyro noise
    'ADD_SENSOR_NOISE': True,  # Enable/disable noise
}


# ============================================================================
# ROBOT CLASS - Encapsulates robot creation and control
# ============================================================================

class BalanceBot:
    """A 2-wheel self-balancing robot with PID control."""
    
    def __init__(self, physics_client_id, config):
        """
        Create the robot in PyBullet.
        
        Args:
            physics_client_id: PyBullet physics client ID
            config: Configuration dictionary
        """
        self.pc = physics_client_id
        self.cfg = config
        
        # Create robot body and wheels
        self.body_id = None
        self.wheel_ids = []
        self._create_robot()
        
        # PID controller state
        self.prev_pitch_error = 0.0
        self.integral_pitch_error = 0.0
        
        # Sensor data (with optional noise)
        self.pitch_angle = 0.0
        self.pitch_rate = 0.0
        self.control_torque = 0.0
        
    def _create_robot(self):
        """Create the robot body and wheels in PyBullet using a single createMultiBody call."""
        
        # Calculate dimensions
        body_mass = self.cfg['ROBOT_MASS'] - 2 * self.cfg['WHEEL_MASS']
        wheel_radius = self.cfg['WHEEL_DIAMETER'] / 2
        wheel_width = 0.02  # 2cm wide wheels
        wheel_spacing = self.cfg['AXLE_WIDTH'] / 2 + 0.01
        body_z = wheel_radius + self.cfg['BODY_HEIGHT'] / 2
        
        # Create collision shapes
        body_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                self.cfg['BODY_WIDTH'] / 2,
                self.cfg['BODY_DEPTH'] / 2,
                self.cfg['BODY_HEIGHT'] / 2
            ]
        )
        body_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                self.cfg['BODY_WIDTH'] / 2,
                self.cfg['BODY_DEPTH'] / 2,
                self.cfg['BODY_HEIGHT'] / 2
            ],
            rgbaColor=[0.2, 0.6, 1.0, 1.0]  # blue body
        )
        
        wheel_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=wheel_radius,
            height=wheel_width
        )
        wheel_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=wheel_radius,
            length=wheel_width,
            rgbaColor=[0.1, 0.1, 0.1, 1.0]  # dark wheels
        )
        
        # Create robot as multi-body with wheels as revolute-joint links
        self.body_id = p.createMultiBody(
            baseMass=body_mass,
            baseCollisionShapeIndex=body_shape,
            baseVisualShapeIndex=body_visual,
            basePosition=[0, 0, body_z],
            # Initial pitch is rotation around Y axis (the wheel axle)
            baseOrientation=p.getQuaternionFromEuler([0, self.cfg['INITIAL_PITCH'], 0]),

            linkMasses=[self.cfg['WHEEL_MASS'], self.cfg['WHEEL_MASS']],
            linkCollisionShapeIndices=[wheel_shape, wheel_shape],
            linkVisualShapeIndices=[wheel_visual, wheel_visual],

            # Wheels are left/right of the body along the Y axis
            linkPositions=[
                [0, -wheel_spacing, -self.cfg['BODY_HEIGHT'] / 2],
                [0,  wheel_spacing, -self.cfg['BODY_HEIGHT'] / 2]
            ],

            # PyBullet cylinders are Z-aligned by default; rotate so cylinder axis aligns with wheel axle (Y)
            linkOrientations=[
                p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
                p.getQuaternionFromEuler([math.pi / 2, 0, 0])
            ],

            linkInertialFramePositions=[[0,0,0],[0,0,0]],
            linkInertialFrameOrientations=[
                [0,0,0,1],
                [0,0,0,1]
            ],

            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],

            # Joint axis is in the child (wheel) local frame.
            # The wheel cylinder is rotated 90° around X, so local-Z maps to world-Y (the axle).
            # Use [0,0,1] so wheels spin around their cylinder axis (world-Y axle).
            linkJointAxis=[[0, 0, 1], [0, 0, 1]]
        )
        
        # Store wheel link indices (0 and 1 for the two wheels)
        self.wheel_ids = [0, 1]

        # Disable the default joint motors so we can apply pure torque control
        for wheel_joint in self.wheel_ids:
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id,
                jointIndex=wheel_joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=0.0,
            )
        
        # Set friction and damping
        p.changeDynamics(self.body_id, -1, lateralFriction=self.cfg['GROUND_FRICTION'],
                        linearDamping=0.0, angularDamping=0.0)
        
        for wheel_link in self.wheel_ids:
            p.changeDynamics(self.body_id, wheel_link, lateralFriction=self.cfg['WHEEL_FRICTION'],
                            linearDamping=0.0, angularDamping=0.0)
    
    def get_state(self, *, noise: bool = True):
        """
        Read robot state from PyBullet (simulated IMU).
        
        Returns:
            pitch_angle: Body pitch in radians
            pitch_rate: Body pitch angular velocity in rad/s
        """
        # Get body orientation and angular velocity
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id)
        
        # Convert quaternion to Euler angles
        euler = p.getEulerFromQuaternion(orn)
        # PyBullet Euler angles are (roll=X, pitch=Y, yaw=Z). For a 2-wheel bot that moves along X,
        # the balancing tilt is about the Y axis.
        pitch = euler[1]
        pitch_rate = ang_vel[1]
        
        # Add sensor noise if enabled
        if noise and self.cfg['ADD_SENSOR_NOISE']:
            pitch += np.random.normal(0, self.cfg['IMU_ANGLE_NOISE_STD'])
            pitch_rate += np.random.normal(0, self.cfg['IMU_GYRO_NOISE_STD'])
        
        return pitch, pitch_rate
    
    def update_control(self):
        """
        Update the PID controller and apply forces to wheels.
        
        The controller uses pitch angle and pitch rate as feedback to maintain balance.
        We apply linear force to the wheels which creates motion and balancing torque.
        """
        pitch, pitch_rate = self.get_state(noise=True)
        self.pitch_angle = pitch
        self.pitch_rate = pitch_rate
        
        # PID control law
        # Goal: pitch angle -> 0, pitch rate -> 0
        pitch_error = 0.0 - pitch  # positive error -> positive force forward
        
        # Proportional term
        p_term = self.cfg['PID_KP'] * pitch_error
        
        # Derivative term (damping)
        d_term = self.cfg['PID_KD'] * (0.0 - pitch_rate)
        
        # Integral term (long-term bias correction)
        self.integral_pitch_error += pitch_error * self.cfg['TIMESTEP']
        # Clamp integral to prevent windup
        self.integral_pitch_error = np.clip(self.integral_pitch_error, -0.5, 0.5)
        i_term = self.cfg['PID_KI'] * self.integral_pitch_error
        
        # Total control torque (about the wheel axle, Y)
        torque = p_term + d_term + i_term
        torque = float(np.clip(torque, -self.cfg['MAX_TORQUE'], self.cfg['MAX_TORQUE']))
        self.control_torque = torque

        # Apply the same torque to both wheels (differential drive forward/back)
        for wheel_joint in self.wheel_ids:
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id,
                jointIndex=wheel_joint,
                controlMode=p.TORQUE_CONTROL,
                force=torque,
            )

    def get_debug_state(self):
        """
        Return comprehensive debug info for all axes.
        Helps verify that axes are set up correctly.
        """
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id)
        euler = p.getEulerFromQuaternion(orn)

        # Get wheel joint states
        wheel_states = []
        for wid in self.wheel_ids:
            js = p.getJointState(self.body_id, wid)
            wheel_states.append({
                'pos': js[0],     # joint position (angle in rad)
                'vel': js[1],     # joint velocity (rad/s)
                'torque': js[3],  # applied torque
            })

        return {
            'pos': pos,
            'euler_deg': (math.degrees(euler[0]), math.degrees(euler[1]), math.degrees(euler[2])),
            'ang_vel_deg': (math.degrees(ang_vel[0]), math.degrees(ang_vel[1]), math.degrees(ang_vel[2])),
            'wheels': wheel_states,
        }

    def check_fallen(self):
        """Check if robot has fallen over (abs pitch > 60 degrees)."""
        pitch, _ = self.get_state(noise=False)
        return abs(pitch) > math.radians(60)


# ============================================================================
# SIMULATION SETUP AND MAIN LOOP
# ============================================================================

def run_simulation():
    """Run the self-balancing robot simulation."""
    
    print("=" * 70)
    print("Self-Balancing Robot Simulation - PyBullet + PID Control")
    print("=" * 70)
    
    # Connect to PyBullet (GUI mode for visualization)
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Setup simulation
    p.setGravity(0, 0, CONFIG['GRAVITY'])
    p.setPhysicsEngineParameter(fixedTimeStep=CONFIG['TIMESTEP'], numSubSteps=1)
    
    # Load ground plane
    ground_id = p.loadURDF("plane.urdf")
    p.changeDynamics(ground_id, -1, lateralFriction=CONFIG['GROUND_FRICTION'])
    
    # Create robot
    robot = BalanceBot(physics_client, CONFIG)
    
    # Set camera distance to 2m from robot
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.1])
    
    # Print configuration
    print(f"\nRobot Configuration:")
    print(f"  Mass: {CONFIG['ROBOT_MASS']} kg")
    print(f"  Body: {CONFIG['BODY_WIDTH']}m x {CONFIG['BODY_DEPTH']}m x {CONFIG['BODY_HEIGHT']}m")
    print(f"  Wheels: {CONFIG['WHEEL_DIAMETER']}m diameter, {CONFIG['AXLE_WIDTH']}m apart")
    print(f"  Initial pitch: {math.degrees(CONFIG['INITIAL_PITCH']):.1f}°")
    
    print(f"\nControl Configuration:")
    print(f"  PID Gains: Kp={CONFIG['PID_KP']}, Ki={CONFIG['PID_KI']}, Kd={CONFIG['PID_KD']}")
    print(f"  Max torque: {CONFIG['MAX_TORQUE']} Nm")
    print(f"  Simulation rate: {1/CONFIG['TIMESTEP']:.0f} Hz")
    
    print(f"\nStarting simulation for {CONFIG['SIM_DURATION']} seconds...")
    print("-" * 70)
    
    # Simulation loop
    sim_time = 0.0
    log_interval = 0.1  # Print status every 0.1 seconds
    last_log_time = 0.0
    
    while sim_time < CONFIG['SIM_DURATION']:
        # Update robot control
        robot.update_control()
        
        # Step simulation
        p.stepSimulation()
        sim_time += CONFIG['TIMESTEP']
        
        # Check if fallen
        if robot.check_fallen():
            print(f"\n[{sim_time:.2f}s] Robot fell over!")
            break
        
        # Log status periodically
        if sim_time - last_log_time >= log_interval:
            dbg = robot.get_debug_state()
            rx, ry, rz = dbg['euler_deg']
            wx, wy, wz = dbg['ang_vel_deg']
            w0 = dbg['wheels'][0]
            w1 = dbg['wheels'][1]
            print(f"[{sim_time:5.2f}s] "
                  f"Euler(r={rx:6.1f} p={ry:6.1f} y={rz:6.1f})° | "
                  f"AngVel(x={wx:6.1f} y={wy:6.1f} z={wz:6.1f})°/s | "
                  f"Whl({w0['vel']:6.1f},{w1['vel']:6.1f})rad/s | "
                  f"Trq: {robot.control_torque:6.3f}")
            last_log_time = sim_time
        
        # Small sleep to prevent GUI from freezing
        time.sleep(CONFIG['TIMESTEP'])
    
    print("-" * 70)
    final_pitch_deg = math.degrees(robot.pitch_angle)
    print(f"\nSimulation complete!")
    print(f"Final pitch: {final_pitch_deg:.2f}°")
    
    if abs(final_pitch_deg) < 10:
        print("✓ Robot successfully balanced!")
    else:
        print("✗ Robot did not achieve stable balance.")
    
    print("\nClose the PyBullet window to exit.")
    
    # Keep simulation running until user closes the window
    while p.isConnected(physics_client):
        time.sleep(0.01)
    
    p.disconnect()


# ============================================================================
# TUNING GUIDE
# ============================================================================
"""
If the robot is not balancing well, try adjusting these parameters:

PID_KP (Proportional Gain):
    - Increase: Robot responds more aggressively to tilt
    - Decrease: Robot responds more slowly
    - Too high: Oscillation and instability
    - Start: 5-12 N

PID_KD (Derivative Gain):
    - Increase: Damps oscillations, slows response
    - Decrease: Less damping, more oscillatory
    - Too high: Sluggish response
    - Start: 2-4 N/(rad/s)

PID_KI (Integral Gain):
    - Increase: Corrects steady-state errors, reduces bias
    - Decrease: Less correction of bias
    - Too high: Integral windup, slow instability
    - Start: 0.05-0.2 N

MAX_TORQUE:
    - Increase: Motor can exert more force
    - Decrease: Simulate weaker motor
    - Too low: Motor cannot balance robot

WHEEL_FRICTION:
    - Increase: Better traction, can apply force more effectively
    - Decrease: Wheels slip more, reduced control authority

If the robot oscillates:
    - Increase KD (more damping)
    - Decrease KP (less aggressive response)

If the robot drifts and doesn't recover:
    - Increase KI (correct steady-state bias)
    - Decrease KP (may be oscillating)

If the robot responds too slowly:
    - Increase KP
    - Decrease KI (can cause instability)
"""


if __name__ == "__main__":
    run_simulation()
