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
    'SIM_DURATION': 10.0,  # seconds
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
    'INITIAL_PITCH': -0.01,  # rad (~5.7 degrees)
    'INITIAL_HEIGHT': 0.08,  # m above ground
    
    # PID Controller gains
    # These are the main tuning parameters for balance control
    'PID_KP': 8.0,   # proportional gain (pitch angle error)
    'PID_KD': 3.0,   # derivative gain (pitch rate)
    'PID_KI': 0.1,   # integral gain (accumulated pitch error)
    
    # Motor/actuator limits
    'MAX_TORQUE': 1.5,  # Nm (motor saturation limit)
    
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
        """Create the robot body and wheels in PyBullet using hinged wheels."""
        
        # Calculate body mass (remaining mass after wheels)
        body_mass = self.cfg['ROBOT_MASS'] - 2 * self.cfg['WHEEL_MASS']
        wheel_radius = self.cfg['WHEEL_DIAMETER'] / 2
        
        # Create body collision shape (box)
        body_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                self.cfg['BODY_WIDTH'] / 2,
                self.cfg['BODY_DEPTH'] / 2,
                self.cfg['BODY_HEIGHT'] / 2
            ]
        )
        
        # Create the robot body
        # Position body so its bottom aligns with wheel center height
        body_z = wheel_radius + self.cfg['BODY_HEIGHT'] / 2
        self.body_id = p.createMultiBody(
            baseMass=body_mass,
            baseCollisionShapeIndex=body_shape,
            basePosition=[0, 0, body_z],
            baseOrientation=[0, 0, 0, 1]
        )
        
        # Create wheel collision shape (single cylinder)
        # Use a narrow wheel width to avoid intersecting with body
        wheel_width = 0.02  # 2cm wide wheels
        wheel_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=wheel_radius,
            height=wheel_width
        )
        
        # Create left and right wheels as separate bodies
        # Position wheels relative to robot body center
        # Body center is at body_z, so wheels should be at body_z - BODY_HEIGHT/2 (bottom of body)
        wheel_positions = [
            [-self.cfg['AXLE_WIDTH'] / 2, 0, body_z - self.cfg['BODY_HEIGHT'] / 2],
            [self.cfg['AXLE_WIDTH'] / 2, 0, body_z - self.cfg['BODY_HEIGHT'] / 2]
        ]
        
        self.wheel_ids = []
        for pos in wheel_positions:
            wheel_id = p.createMultiBody(
                baseMass=self.cfg['WHEEL_MASS'],
                baseCollisionShapeIndex=wheel_shape,
                basePosition=pos,
                baseOrientation=p.getQuaternionFromEuler([0, math.pi/2, 0])
            )
            self.wheel_ids.append(wheel_id)
        
        # Create hinged connections (allow rotation around Y-axis only)
        for i, wheel_id in enumerate(self.wheel_ids):
            p.createConstraint(
                parentBodyUniqueId=self.body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=wheel_id,
                childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 0],
                parentFramePosition=[wheel_positions[i][0], 0, -self.cfg['BODY_HEIGHT'] / 2],
                childFramePosition=[0, 0, 0]
            )
        
        # Set friction and damping
        p.changeDynamics(self.body_id, -1, lateralFriction=self.cfg['GROUND_FRICTION'],
                        linearDamping=0.0, angularDamping=0.0)
        
        for wheel_id in self.wheel_ids:
            p.changeDynamics(wheel_id, -1, lateralFriction=self.cfg['WHEEL_FRICTION'],
                            linearDamping=0.0, angularDamping=0.0)
        
        # Apply initial tilt (pitch)
        initial_orn = p.getQuaternionFromEuler([self.cfg['INITIAL_PITCH'], 0, 0])
        p.resetBasePositionAndOrientation(self.body_id, [0, 0, body_z], initial_orn)
    
    def get_state(self):
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
        pitch = euler[0]  # pitch is X-axis rotation (forward/backward tilt)
        pitch_rate = ang_vel[0]  # angular velocity around X-axis
        
        # Add sensor noise if enabled
        if self.cfg['ADD_SENSOR_NOISE']:
            pitch += np.random.normal(0, self.cfg['IMU_ANGLE_NOISE_STD'])
            pitch_rate += np.random.normal(0, self.cfg['IMU_GYRO_NOISE_STD'])
        
        return pitch, pitch_rate
    
    def update_control(self):
        """
        Update the PID controller and apply forces to wheels.
        
        The controller uses pitch angle and pitch rate as feedback to maintain balance.
        We apply linear force to the wheels which creates motion and balancing torque.
        """
        pitch, pitch_rate = self.get_state()
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
        
        # Total control force (in Y direction)
        force = p_term + d_term + i_term
        
        # Saturate force to motor limits (convert torque limit to force)
        # F = tau / r
        max_force = self.cfg['MAX_TORQUE'] / (self.cfg['WHEEL_DIAMETER'] / 2)
        force = np.clip(force, -max_force, max_force)
        self.control_torque = force * (self.cfg['WHEEL_DIAMETER'] / 2)
        
        # Apply force to both wheels to create forward/backward motion
        for wheel_id in self.wheel_ids:
            p.applyExternalForce(
                objectUniqueId=wheel_id,
                linkIndex=-1,
                forceObj=[0, force, 0],  # force in wheel's local Y direction
                posObj=[0, 0, 0],  # apply at wheel center
                flags=p.LINK_FRAME
            )
    
    def check_fallen(self):
        """Check if robot has fallen over (abs pitch > 60 degrees)."""
        pitch, _ = self.get_state()
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
            pitch_deg = math.degrees(robot.pitch_angle)
            print(f"[{sim_time:6.2f}s] Pitch: {pitch_deg:7.2f}° | "
                  f"Rate: {math.degrees(robot.pitch_rate):7.2f}°/s | "
                  f"Torque: {robot.control_torque:6.3f} Nm")
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
