# Dynamic Controller Recommendation for the Tribot

## What makes this robot unusual

From `docs/TRIPLET_BALANCING_ROBOT.md` and `tribot_description/mjcf/tribot.xml`:

- Inverted pendulum on two 3-wheel clusters, with a second pitch actuator (hub rotation) that moves the ground contact point instead of applying a reaction torque.
- Hybrid dynamics: 4WD (statically stable), 2WD (unstable), continuous transitions between them, and impulsive 120° flips.
- Drive torque is tiny (1 Nm per side) against a 3.4 kg body with CoG 0.25 to 0.38 m above the pivot. Input saturation is the dominant constraint.
- Two pitch actuators with very different authority (1 Nm wheels, 5 Nm hubs) and bandwidth must share control.

## Recommended controller

**Mode-scheduled linear MPC (constrained, MIMO) for balance and locomotion, with offline-optimized trajectories for flips and mode transitions, on top of a cascaded torque/current loop.**

### Architecture, outer to inner

1. **Planner / event layer** (10 to 50 Hz)
   Mode selection (4WD/2WD), stair and obstacle handling, fall detection via capture point (DCM). Triggers flips from a precomputed trajectory library.

2. **MPC** (100 to 250 Hz)
   - Sagittal model with states `[x, ẋ, θ, θ̇, φ, φ̇]` and inputs `[τ_drive, τ_triplet]`.
   - Linearized about the current hub angle φ and scheduled per contact mode (4WD vs 2WD gravity coupling and pendulum length).
   - Horizon 0.3 to 0.5 s (20 to 30 steps).
   - Hard box constraints on both torques. Soft constraints on pitch and on the CoG staying inside the 4WD support polygon.
   - Cost weights encode actuator allocation: the triplet does quasi-static lean, the wheels do dynamic correction. No hand-tuned blending factor is needed.
   - Solve with OSQP (prototype) or acados/HPIPM (generated C). Warm-start every step.

3. **Yaw loop** (same rate as MPC)
   PI on wheel speed difference, added as a differential term to the two drive torques. Decoupled from the MPC.

4. **Inner loop** (1 kHz, on a microcontroller)
   Torque commands to FOC drivers. EKF fusing IMU gyro/accel with wheel and hub encoders for pitch, pitch rate, and forward velocity. Safety reflexes (flip trigger, torque cutoff on tilt limit) live here so they never depend on the MPC solver finishing.

### Why not plain LQR, PID, or RL as the primary

- LQR and PID ignore saturation. With 1 Nm of drive torque they either bang-bang or must be detuned until sluggish.
- MPC handles the two-actuator allocation, saturation, and the mode-scheduled model in one formulation.
- RL (PPO in MuJoCo with domain randomization) is the strongest alternative for stairs and flips, where impacts make model-based control brittle. It is a good second phase once the MPC baseline works, but it is harder to debug and certify.

### Sim-to-real prerequisites

- System-identify motor torque constant, friction, belt losses, and back-EMF on the real actuators.
- Match the MuJoCo friction and contact model to measured behaviour.
- Verify the IMU noise model against the real part.

## Hardware required

### Compute (two-tier, recommended)

| Tier | Role | Candidates |
|------|------|------------|
| Real-time MCU | 1 kHz loop, EKF, reflexes, motor bus | Teensy 4.1, STM32H743/H7-class (Cortex-M7, 480 to 600 MHz, FPU) |
| Linux SBC | MPC, planner, logging | Raspberry Pi 5 with PREEMPT_RT kernel, or Jetson Orin Nano if RL or vision is planned |

The MPC alone can run on the M7 if code-generated with acados or CVXGEN-style tooling (roughly 1 to 3 ms per solve for this problem size). Link SBC and MCU by SPI, UART, or CAN-FD.

### Sensors

- **IMU**: 6-axis at 500 Hz to 1 kHz with low gyro noise. ICM-42688-P or BMI088 (budget), ADIS16470 (low drift). No magnetometer needed.
- **Hub encoders**: must be **absolute**, so the contact mode is known at power-on. AS5047P, AS5048A, or MT6835.
- **Wheel/motor encoders**: 1000+ counts per rev equivalent (magnetic on-axis or the driver's own encoder).
- **Optional ToF** for stair-edge detection: VL53L5CX or similar.

### Actuation (torque control is mandatory, since the MPC outputs torque)

- **Drive**: 2 BLDC motors with FOC drivers, at least 1 Nm continuous at the wheel after the belt. Drivers: moteus r4.11 or c1, ODrive S1/Micro, or SimpleFOC on a Cortex-M board. Current loop 10 to 20 kHz, torque command at 1 kHz.
- **Triplet**: 2 geared BLDC (planetary 6:1 to 9:1, MIT-cheetah style) with FOC drivers, 5 Nm peak, fast enough for a ~230 ms 120° flip.
- **Bus**: CAN-FD between drivers and MCU.

### Power

6S LiPo, sized for the flip current peak (both triplet motors at 5 Nm simultaneously). Add bus capacitance to absorb regen during braking.

## Verification path (if implemented)

1. Prototype the MPC in Python with OSQP against `tribot_sim.py` in MuJoCo: balance in 2WD, in 4WD, and through a transition without more than brief saturation.
2. Measure solve time. Port to acados-generated C if it exceeds 2 ms.
3. Push tests: recover from a step disturbance larger than the LQR baseline handles, in both modes.
4. Time the flip trajectory in sim before trying it on hardware.

## Implementation status

Implemented in this repository as the default controller. See
`docs/MPC_CONTROLLER.md` for the model, QP, event layer, validation
against MuJoCo, and scenario results. Modules: `controllers/control_mpc.py`,
`controllers/mpc_plant.py`, `controllers/mpc_qp.py`,
`controllers/triplet_planner.py`; tools: `tools/validate_mpc_plant.py`,
`tools/run_headless.py`. The EKF, slope handling, and embedded code
generation from this plan are not yet done.
