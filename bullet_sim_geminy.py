"""
PyBullet simulation of a self-balancing robot using a PID controller.

Instructions:
- Make sure you have PyBullet installed: pip install pybullet
- Run the script: python bullet_sim.py
"""

import pybullet as p
import pybullet_data
import time
import math

# --- Simulation Parameters ---
simulation_running = True
time_step = 1.0 / 500.0  # 500 Hz
gravity = -9.81

# --- Robot Parameters ---
robot_mass = 1.1  # kg
body_height = 0.15  # meters
wheel_diameter = 0.095  # meters
wheel_radius = wheel_diameter / 2
wheel_thickness = 0.02 # meters
initial_tilt_angle = 0.1 # radians (small initial tilt)

# Derived parameters
body_mass_part = robot_mass * 0.9
wheel_mass_part = (robot_mass * 0.1) / 2
body_half_extents = [0.05, 0.05, body_height / 2]

# Center of mass at mid-height of the body
com_z_offset = body_height / 2

# --- PID Controller Parameters ---
# These gains are a starting point and may need tuning.
# Kp: Proportional gain - reacts to the current error (angle).
# Ki: Integral gain - accumulates past errors to correct steady-state error.
# Kd: Derivative gain - dampens the response by reacting to the rate of change of the error (angular velocity).
pid_gains = {
    'kp': 35.0,
    'ki': 25.0,
    'kd': 15.0
}

# --- Motor/Torque Parameters ---
torque_saturation = 2.0  # Nm, represents motor limits

# --- Ground Parameters ---
ground_friction = 0.9

# --- Sensor Noise (Optional) ---
imu_noise_angle = 0.001  # radians, small gaussian noise
imu_noise_gyro = 0.001   # rad/s, small gaussian noise


class PIDController:
    """A simple PID controller."""
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self._integral = 0
        self._previous_error = 0

    def update(self, measured_value, dt):
        """Calculate PID output."""
        error = self.setpoint - measured_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self._integral += error * dt
        i_term = self.ki * self._integral
        
        # Derivative term
        derivative = (error - self._previous_error) / dt
        d_term = self.kd * derivative
        
        # Update previous error
        self._previous_error = error
        
        # Total output
        output = p_term + i_term + d_term
        return output

def setup_simulation():
    """Initializes PyBullet simulation environment."""
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, gravity)
    p.setTimeStep(time_step)
    
    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=ground_friction)
    
    return physics_client

def create_robot():
    """Creates the self-balancing robot in the simulation."""
    # 1. Create collision shapes
    body_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_half_extents)
    wheel_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_thickness)

    # 2. Define visual shapes (optional, but good for visualization)
    body_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=body_half_extents)
    wheel_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_thickness, rgbaColor=[0.8, 0.8, 0.8, 1])

    # 3. Assemble the robot using createMultiBody
    # Initial position and orientation (tilted slightly)
    initial_pos = [0, 0, wheel_radius + com_z_offset]
    initial_orn = p.getQuaternionFromEuler([initial_tilt_angle, 0, 0])

    robot_id = p.createMultiBody(
        baseMass=body_mass_part,
        baseCollisionShapeIndex=body_shape,
        baseVisualShapeIndex=body_visual,
        basePosition=initial_pos,
        baseOrientation=initial_orn,
        # Links (wheels)
        linkMasses=[wheel_mass_part, wheel_mass_part],
        linkCollisionShapeIndices=[wheel_shape, wheel_shape],
        linkVisualShapeIndices=[wheel_visual, wheel_visual],
        # Position of links relative to the parent (body)
        linkPositions=[[0, 0.1, -com_z_offset], [0, -0.1, -com_z_offset]],
        # Orientation of links
        linkOrientations=[p.getQuaternionFromEuler([0, 1.5707, 0]), p.getQuaternionFromEuler([0, 1.5707, 0])],
        # Inertial frame
        linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
        linkInertialFrameOrientations=[[0,0,0,1], [0,0,0,1]],
        # Link parent indices (both wheels are attached to the base)
        linkParentIndices=[0, 0],
        # Joint types (revolute for wheels)
        linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
        # Joint axes
        linkJointAxis=[[1, 0, 0], [1, 0, 0]]
    )
    
    # Disable default motor control to apply our own torque
    p.setJointMotorControlArray(
        bodyIndex=robot_id,
        jointIndices=[0, 1],
        controlMode=p.VELOCITY_CONTROL,
        forces=[0, 0]
    )

    return robot_id

def main():
    """Main simulation loop."""
    # Setup
    physics_client = setup_simulation()
    robot_id = create_robot()
    pid = PIDController(pid_gains['kp'], pid_gains['ki'], pid_gains['kd'])

    print("Starting simulation...")

    try:
        while True:
            # 1. Get Sensor Data (IMU)
            _, base_orn = p.getBasePositionAndOrientation(robot_id)
            _, base_ang_vel = p.getBaseVelocity(robot_id)
            
            # Convert quaternion to Euler angles to get pitch
            pitch_angle = p.getEulerFromQuaternion(base_orn)[0]
            pitch_rate = base_ang_vel[0] # Angular velocity around x-axis

            # 2. Controller Logic
            # The control signal is a torque. We use pitch angle as the error.
            control_signal = pid.update(pitch_angle, time_step)
            
            # 3. Apply Torque
            # Saturate the torque to mimic motor limits
            torque = max(-torque_saturation, min(torque_saturation, control_signal))

            # Apply the same torque to both wheels
            p.setJointMotorControlArray(
                bodyIndex=robot_id,
                jointIndices=[0, 1], # Left and Right wheel joints
                controlMode=p.TORQUE_CONTROL,
                forces=[-torque, -torque] # Negative torque to move forward when leaning forward
            )

            # 4. Step Simulation
            p.stepSimulation()
            time.sleep(time_step)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        p.disconnect()
        print("Simulation disconnected.")

if __name__ == "__main__":
    main()
