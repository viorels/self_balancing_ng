"""
Derive equations of motion for tribot using Lagrangian mechanics.

Two models are derived:

1. SIMPLE MODEL (4-state, 2-input) — grounded triplet approximation
   State:  x = [position, velocity, pitch, pitch_rate]
   Input:  u = [τ_wheels, τ_triplet]
   Assumes: triplet wheels are grounded with sufficient friction so the
   triplet angle φ ≈ 0.  The triplet motor applies direct torque to
   the body.  Standard inverted-pendulum-on-cart with two torque inputs.

2. FULL MODEL (6-state, 2-input) — includes triplet dynamics
   State:  x = [θ, φ, x_pos, θ̇, φ̇, ẋ]
   Input:  u = [τ_wheels, τ_triplet]
   The triplet can rotate and shift the ground contact; useful when
   friction is insufficient or for modelling 2WD flip dynamics.
"""
import numpy as np
import sympy as sp

# ============================================================================
# PART 1: Simple 4-state 2-input model (grounded triplet)
# ============================================================================

print("=" * 70)
print("PART 1: Simple model — inverted pendulum + grounded triplet torque")
print("=" * 70)

# Symbols
m_b, m_w, l, I_b, r_w, g_sym = sp.symbols('m_b m_w l I_b r g', positive=True)
tau_w, tau_t = sp.symbols('tau_w tau_t')

# Derived quantities (same as control_lqr.py build_state_space)
I_eff = I_b + m_b * l**2            # body inertia about pivot (parallel axis)
M_tot = m_b + m_w                    # total translational mass

# Mass matrix (from Lagrangian, sign convention matching control_lqr.py)
#   [M_tot   -m_b*l] [ẍ ]   [     0     ] [x]   [1/r   0] [τ_w]
#   [-m_b*l   I_eff] [θ̈ ] = [m_b*g*l    ] [θ] + [ 1   -1] [τ_t]
#
# Generalized forces:
#   τ_w:  Q_x = +τ_w/r  (wheel ground force)
#          Q_θ = +τ_w    (motor stator reaction on body)
#   τ_t:  Q_x = 0        (no horizontal force from triplet motor)
#          Q_θ = -τ_t    (body gets -τ_t reaction because positive joint
#                         torque pushes child/triplet in +Y, body in -Y)

M_mat = sp.Matrix([[M_tot, -m_b*l],
                    [-m_b*l, I_eff]])

B_gf = sp.Matrix([[1/r_w, 0],
                   [ 1,   -1]])

G_vec = sp.Matrix([[0],
                    [m_b * g_sym * l]])

det = M_mat.det()
M_inv = M_mat.inv()

print("\nMass matrix M:")
sp.pprint(M_mat)
print(f"\ndet(M) = {sp.simplify(det)}")

print("\nM⁻¹:")
sp.pprint(sp.simplify(M_inv))

print("\nGeneralised force matrix B_gf (gen forces per [τ_w, τ_t]):")
sp.pprint(B_gf)

# State-space B = M⁻¹ @ B_gf
B_ss = sp.simplify(M_inv * B_gf)
print("\nState-space B = M⁻¹ B_gf (rows: ẍ, θ̈; cols: τ_w, τ_t):")
sp.pprint(B_ss)

# Gravity contribution: M⁻¹ @ G_vec (only θ column)
G_ss = sp.simplify(M_inv * G_vec)
print("\nGravity: M⁻¹ G (rows: ẍ, θ̈):")
sp.pprint(G_ss)

# ----- Numerical evaluation -----
params_simple = {
    m_b: 2.7167,       # body mass (from URDF)
    m_w: 0.6698,       # wheel + triplet mass (2 triplets + 6 wheels)
    l:   0.247,        # body CoG above wheel axis
    I_b: 0.056436,     # body Iyy (PyBullet-computed)
    r_w: 0.058,        # wheel radius (measured from STL)
    g_sym: 9.81,
}

print("\n--- Numerical values (simple model) ---")
print("Parameters:", {str(k): v for k, v in params_simple.items()})

B_num = np.array(B_ss.subs(params_simple).tolist(), dtype=float)
G_num = np.array(G_ss.subs(params_simple).tolist(), dtype=float).flatten()
det_num = float(det.subs(params_simple))

print(f"\ndet(M) = {det_num:.6f}")
print(f"\nB (acceleration per unit input):")
print(f"  ẍ from τ_w:  b1_w = {B_num[0,0]:.6f}")
print(f"  ẍ from τ_t:  b1_t = {B_num[0,1]:.6f}")
print(f"  θ̈ from τ_w:  b3_w = {B_num[1,0]:.6f}")
print(f"  θ̈ from τ_t:  b3_t = {B_num[1,1]:.6f}")
print(f"\nGravity coupling:")
print(f"  ẍ from θ:   a13 = {G_num[0]:.6f}")
print(f"  θ̈ from θ:   a33 = {G_num[1]:.6f}")

# Full 4×4 A matrix and 4×2 B matrix (state = [pos, vel, pitch, pitch_rate])
A_full = np.array([
    [0, 1, 0, 0],
    [0, 0, G_num[0], 0],
    [0, 0, 0, 1],
    [0, 0, G_num[1], 0],
])
B_full = np.array([
    [0,          0         ],
    [B_num[0,0], B_num[0,1]],
    [0,          0         ],
    [B_num[1,0], B_num[1,1]],
])
print(f"\nA (4×4):\n{np.array2string(A_full, precision=6, suppress_small=True)}")
print(f"\nB (4×2):\n{np.array2string(B_full, precision=6, suppress_small=True)}")

# Controllability check
C_mat = np.hstack([np.linalg.matrix_power(A_full, i) @ B_full for i in range(4)])
rank = np.linalg.matrix_rank(C_mat)
print(f"\nControllability rank: {rank} / 4  {'✓ CONTROLLABLE' if rank == 4 else '✗ NOT CONTROLLABLE'}")


# ============================================================================
# PART 2: Full 6-state model (triplet angle as state)
# ============================================================================

print("\n\n" + "=" * 70)
print("PART 2: Full model — body tilt θ, triplet angle φ, cart position x")
print("=" * 70)

t_sym = sp.Symbol('t')

# Generalized coordinates as functions of time
theta = sp.Function('theta')(t_sym)   # body tilt from vertical
phi = sp.Function('phi')(t_sym)       # triplet angle relative to body
x = sp.Function('x')(t_sym)           # axle horizontal position

# Time derivatives
dtheta = theta.diff(t_sym)
dphi = phi.diff(t_sym)
dx = x.diff(t_sym)

# Parameters (from URDF)
M = sp.Symbol('M')          # body mass
m_t = sp.Symbol('m_t')      # total triplet mass (both sides)
m_w2 = sp.Symbol('m_w')     # ground wheel mass
I_b2 = sp.Symbol('I_b')     # body inertia about CoM
I_t = sp.Symbol('I_t')      # triplet inertia about axle
I_w = sp.Symbol('I_w')      # wheel spin inertia
L = sp.Symbol('L')          # body CoM height above axle
R = sp.Symbol('R')          # triplet radius
r = sp.Symbol('r')          # wheel radius
g = sp.Symbol('g')          # gravity

z_a = R + r  # axle height (constant for linearised model)

# Body CoM position
x_body = x + L * sp.sin(theta)
z_body = z_a + L * sp.cos(theta)
dx_body = x_body.diff(t_sym)
dz_body = z_body.diff(t_sym)

# Triplet CoM (at axle)
dx_triplet = dx

# Ground contact wheel position
x_wheel = x - R * sp.sin(phi + theta)
dx_wheel = x_wheel.diff(t_sym)

# Kinetic energy
T = (sp.Rational(1, 2) * M * (dx_body**2 + dz_body**2) +
     sp.Rational(1, 2) * I_b2 * dtheta**2 +
     sp.Rational(1, 2) * m_t * dx_triplet**2 +
     sp.Rational(1, 2) * I_t * (dtheta + dphi)**2 +
     sp.Rational(1, 2) * m_w2 * dx_wheel**2 +
     sp.Rational(1, 2) * I_w * ((dx_wheel / r)**2))
T = sp.expand(T)

# Potential energy
V = M * g * (z_a + L * sp.cos(theta)) + m_t * g * z_a

# Lagrangian
Lag = sp.expand(T - V)

# Generalized forces
tau_t2 = sp.Symbol('tau_t')
tau_w2 = sp.Symbol('tau_w')

Q_theta = -tau_t2 - (tau_w2 / r) * R * sp.cos(phi + theta)
Q_phi = tau_t2 - (tau_w2 / r) * R * sp.cos(phi + theta)
Q_x = tau_w2 / r

# Euler-Lagrange equations
q_list = [theta, phi, x]
Q_list = [Q_theta, Q_phi, Q_x]

EL_eqs = []
for i, qi in enumerate(q_list):
    dqi = qi.diff(t_sym)
    dL_dqi = sp.diff(Lag, qi)
    dL_ddqi = sp.diff(Lag, dqi)
    dt_dL_ddqi = dL_ddqi.diff(t_sym)
    eq = sp.Eq(dt_dL_ddqi - dL_dqi, Q_list[i])
    EL_eqs.append(eq)
    print(f"\nEL equation for {qi}:")
    print(sp.simplify(eq))

# Linearize
subs_linear = {
    sp.sin(theta): theta, sp.cos(theta): 1,
    sp.sin(phi): phi, sp.cos(phi): 1,
    sp.sin(phi + theta): phi + theta, sp.cos(phi + theta): 1,
}

print("\n--- Linearised equations ---")
for i, eq in enumerate(EL_eqs):
    eq_lin = sp.expand(eq.subs(subs_linear))
    print(f"\nLinearized EL equation {i}:")
    print(sp.simplify(eq_lin))

# Numerical parameters for the full model
params_full = {
    M: 2.717, m_t: 0.507, m_w2: 0.054,
    I_b2: 0.167, I_t: 0.940, I_w: 0.025,
    L: 0.247, R: 0.12, r: 0.058, g: 9.81,
}
print("\nFull-model parameters:")
for k, v in params_full.items():
    print(f"  {k} = {v}")