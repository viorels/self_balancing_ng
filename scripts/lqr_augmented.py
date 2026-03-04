"""
Augmented LQR analysis script — standalone.

Computes and prints the A, B, K matrices for the tribot's 4-state 2-input
model (wheel + triplet torque), using the same parameters as tribot_sim.py.

Usage:
    python scripts/lqr_augmented.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_lqr_aug import build_augmented_state_space
from control_lqr import compute_lqr_gain


def main():
    # Parameters matching tribot_sim.py CONFIG
    config = {
        'LQR_BODY_MASS': 2.7167,
        'LQR_WHEEL_MASS': 0.6698,
        'LQR_COG_HEIGHT': 0.247,
        'LQR_BODY_INERTIA': 0.056436,
        'WHEEL_RADIUS': 0.058,
        'GRAVITY': -9.81,
    }

    A, B = build_augmented_state_space(config)

    print("=" * 60)
    print("Augmented LQR: 4-state, 2-input plant")
    print("  State: x = [position, velocity, pitch, pitch_rate]")
    print("  Input: u = [τ_wheels,  τ_triplet]")
    print("=" * 60)

    print(f"\nA (4×4):\n{np.array2string(A, precision=6, suppress_small=True)}")
    print(f"\nB (4×2):\n{np.array2string(B, precision=6, suppress_small=True)}")

    # Controllability
    n = 4
    C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    rank = np.linalg.matrix_rank(C)
    print(f"\nControllability rank: {rank} / {n}"
          f"  {'✓' if rank == n else '✗'}")

    # Open-loop poles
    ols = np.linalg.eigvals(A)
    print(f"Open-loop poles: {[f'{e.real:.4f}' for e in ols]}")

    # ----- Sweep R_triplet to show the effect -----
    print("\n" + "-" * 60)
    print("Effect of R_triplet on gain distribution")
    print(f"  Fixed Q = diag([12, 4, 55, 4]),  R_wheels = 4.0")
    print("-" * 60)
    print(f"{'R_trip':>8s}  {'K_w_pitch':>10s}  {'K_t_pitch':>10s}  "
          f"{'|K_w|':>8s}  {'|K_t|':>8s}  {'CL poles':>30s}")

    Q_diag = [12.0, 4.0, 55.0, 4.0]
    Q = np.diag(Q_diag)
    R_w = 4.0

    for R_t in [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 10.0]:
        R = np.diag([R_w, R_t])
        K = compute_lqr_gain(A, B, Q, R)
        eigs = np.linalg.eigvals(A - B @ K)
        eig_str = ', '.join(f'{e.real:.2f}' for e in sorted(eigs, key=lambda e: e.real))
        print(f"{R_t:8.1f}  {K[0, 2]:10.4f}  {K[1, 2]:10.4f}  "
              f"{np.linalg.norm(K[0]):8.4f}  {np.linalg.norm(K[1]):8.4f}  "
              f"[{eig_str}]")

    # ----- Detailed gain for chosen R -----
    print("\n" + "=" * 60)
    R_chosen = np.diag([4.0, 0.5])
    K = compute_lqr_gain(A, B, Q, R_chosen)
    print(f"Chosen R = diag([4.0, 0.5])")
    print(f"\nK (2×4):")
    print(f"  Wheels:  [{', '.join(f'{k:8.4f}' for k in K[0])}]")
    print(f"  Triplet: [{', '.join(f'{k:8.4f}' for k in K[1])}]")
    print(f"\nInterpretation:")
    print(f"  For 1° pitch error ({np.radians(1):.4f} rad):")
    print(f"    Wheel  torque contribution: {K[0, 2] * np.radians(1):.4f} Nm")
    print(f"    Triplet torque contribution: {K[1, 2] * np.radians(1):.4f} Nm")
    print(f"  For 10cm position error:")
    print(f"    Wheel  torque contribution: {K[0, 0] * 0.1:.4f} Nm")
    print(f"    Triplet torque contribution: {K[1, 0] * 0.1:.4f} Nm")

    eigs = np.linalg.eigvals(A - B @ K)
    print(f"\nClosed-loop eigenvalues: {[f'{e:.4f}' for e in eigs]}")
    print(f"All stable: {all(e.real < 0 for e in eigs)}")


if __name__ == '__main__':
    main()