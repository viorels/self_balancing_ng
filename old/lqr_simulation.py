import pygame
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import cont2discrete, place_poles

# --- Constants ---
# Physics
M = 0.5  # Mass of the wheel (kg)
m = 0.2  # Mass of the pendulum (kg)
L = 0.3  # Length of the pendulum (m)
g = 9.81 # Gravity (m/s^2)

# Simulation
FPS = 60
DT = 1.0 / FPS

# Screen
WIDTH, HEIGHT = 1000, 600
WHEEL_RADIUS = 30
PENDULUM_LENGTH_PIXELS = 200

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Control
TARGET_STEP = 0.5 # How much to move the target with each key press


# --- State-Space Model ---
# x_dot = Ax + Bu
# State vector: [x, x_dot, theta, theta_dot]
# x: position of the wheel
# theta: angle of the pendulum from vertical

# Linearized model around theta = 0
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -(m * g) / M, 0],
    [0, 0, 0, 1],
    [0, 0, (M + m) * g / (M * L), 0]
])

B = np.array([
    [0],
    [1 / M],
    [0],
    [-1 / (M * L)]
])

# --- LQR Controller ---
# Penalizes state deviation and control effort
Q = np.diag([1.0, 1.0, 10.0, 1.0]) # Penalize angle deviation more
R = np.array([[0.1]])             # Penalize control effort

# Solve Continuous Algebraic Riccati Equation (CARE)
P = solve_continuous_are(A, B, Q, R)

# Calculate LQR gain
K_lqr = np.linalg.inv(R) @ B.T @ P

# --- Deadbeat Controller ---
# Discretize the system
A_d, B_d, _, _, _ = cont2discrete((A, B, np.zeros((1,4)), np.zeros((1,1))), DT, method='zoh')

# Place poles for a "deadbeat-like" fast response
# True deadbeat with all poles at 0 is not possible for this system with a single input
poles = np.array([0.8, 0.81, 0.82, 0.83])

# Calculate Deadbeat gain
K_deadbeat = place_poles(A_d, B_d, poles).gain_matrix

# --- Simulation Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Inverted Pendulum Simulation")
font = pygame.font.SysFont("monospace", 16)
clock = pygame.time.Clock()

# --- Initial State ---
# [x, x_dot, theta, theta_dot]
# Start with a small perturbation
state = np.array([0.0, 0.0, 0.1, 0.0])
target_x = 0.0

# --- Main Loop ---
running = True
controller_index = 0
controllers = [("LQR", K_lqr), ("Deadbeat", K_deadbeat)]

while running:
    clock.tick(FPS)

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                controller_index = (controller_index + 1) % len(controllers)
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_LEFT:
                target_x -= TARGET_STEP
            if event.key == pygame.K_RIGHT:
                target_x += TARGET_STEP

    # --- Control ---
    controller_name, K = controllers[controller_index]

    # Define the target state [target_x, 0, 0, 0]
    target_state = np.array([target_x, 0.0, 0.0, 0.0])
    
    # Calculate the error from the target state
    error = state - target_state

    # The control law is u = -K * error
    u = -K @ error
    u = np.clip(u, -20.0, 20.0) # Clamp the control input

    # --- Physics Update (Euler Integration) ---
    # The physical system is continuous, regardless of the controller type.
    # The controller provides a force `u`, and we simulate the effect of that
    # force on the continuous system.
    state_dot = A @ state + (B @ np.array([[u]])).flatten()
    state = state + state_dot * DT

    # --- Drawing ---
    screen.fill(WHITE)

    # Ground
    pygame.draw.line(screen, BLACK, (0, HEIGHT - 150), (WIDTH, HEIGHT - 150), 2)
    
    # Target marker
    target_pixel_x = int(WIDTH / 2 + target_x * 500)
    pygame.draw.polygon(screen, GREEN, [
        (target_pixel_x, HEIGHT - 150),
        (target_pixel_x - 10, HEIGHT - 160),
        (target_pixel_x + 10, HEIGHT - 160)
    ])


    # Wheel
    wheel_x = int(WIDTH / 2 + state[0] * 500) # Scale position for display
    wheel_y = int(HEIGHT - 150 - WHEEL_RADIUS)
    pygame.draw.circle(screen, BLUE, (wheel_x, wheel_y), WHEEL_RADIUS)

    # Pendulum
    pendulum_angle = state[2]
    pendulum_x1 = wheel_x
    pendulum_y1 = wheel_y
    pendulum_x2 = int(pendulum_x1 + PENDULUM_LENGTH_PIXELS * np.sin(pendulum_angle))
    pendulum_y2 = int(pendulum_y1 - PENDULUM_LENGTH_PIXELS * np.cos(pendulum_angle))
    pygame.draw.line(screen, RED, (pendulum_x1, pendulum_y1), (pendulum_x2, pendulum_y2), 4)

    # --- Text Display ---
    controller_text = font.render(f"Controller: {controller_name} (Press SPACE to switch)", True, BLACK)
    target_text = font.render(f"Target x: {target_x:.2f} (Use LEFT/RIGHT keys)", True, BLACK)
    state_text_x = font.render(f"x: {state[0]:.2f}", True, BLACK)
    state_text_x_dot = font.render(f"x_dot: {state[1]:.2f}", True, BLACK)
    state_text_theta = font.render(f"theta: {np.rad2deg(state[2]):.2f} deg", True, BLACK)
    state_text_theta_dot = font.render(f"theta_dot: {state[3]:.2f}", True, BLACK)

    screen.blit(controller_text, (10, 10))
    screen.blit(target_text, (10, 30))
    screen.blit(state_text_x, (10, 60))
    screen.blit(state_text_x_dot, (10, 80))
    screen.blit(state_text_theta, (10, 100))
    screen.blit(state_text_theta_dot, (10, 120))

    pygame.display.flip()

pygame.quit()
