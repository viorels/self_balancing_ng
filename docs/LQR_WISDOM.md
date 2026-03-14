# LQR Wisdom — Lessons & Derivations

Hard-won knowledge about the LQR controller behavior in the tribot sim.

---

## 1. Extracting the LQR's Desired Lean Angle

### Problem

The LQR state vector is `x = [pos_error, velocity, pitch, pitch_rate]`.
When the robot has a position error, the LQR implicitly wants a temporary
lean (pitch offset) to create forward/backward acceleration.  But it never
explicitly outputs this desired lean — it only outputs a torque `u = -K·x`.

If a separate triplet PD controller holds the hub at 0° (4WD), it fights
the lean the LQR needs, since the lean causes the body to tilt while the
triplet stays world-vertical.  The two controllers cancel each other out.

### Derivation

At the desired lean equilibrium, the pitch rate is zero and the LQR
output should be zero (the lean is doing all the work via gravity):

$$0 = K_0 \cdot e_{pos} + K_1 \cdot v + K_2 \cdot \theta_{desired} + K_3 \cdot \underbrace{\dot\theta}_{=0}$$

Solving for the desired lean angle:

$$\boxed{\theta_{desired} = -\frac{K_0 \cdot e_{pos} + K_1 \cdot v}{K_2}}$$

This is the pitch angle the LQR needs to achieve to drive position error
toward zero.  It changes every control tick as pos_error and velocity evolve.

### Implementation

In `LQRBalanceController.update()`:

```python
K = self.K[0]
if abs(K[2]) > 1e-9:
    self.desired_lean = -(K[0] * x[0] + K[1] * x[1]) / K[2]
else:
    self.desired_lean = 0.0
```

### Feeding it to the Triplet PD

The triplet PD target should be offset by `-desired_lean` so the hub
rotates to keep wheels on the ground during the lean:

```python
if hasattr(self.controller, 'desired_lean'):
    lean_offset = self.controller.desired_lean
    self.triplet_ctrl.set_target(INITIAL_TRIPLET_ANGLE - lean_offset)
```

When LQR tilts the body 5° forward, the triplet rotates 5° backward
(body-relative) to maintain ground contact — cooperation instead of fight.

---

