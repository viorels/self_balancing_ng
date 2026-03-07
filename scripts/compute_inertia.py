#!/usr/bin/env python3
"""
Compute inertia tensor from an STL mesh assuming uniform density.

Handles multi-body STL exports (CAD assemblies exported as a single STL
that internally contains several separate closed shells). Treating such a
mesh as a single body causes signed-volume cancellation errors; this script
splits the mesh into its constituent watertight bodies, propagates density
uniformly across their combined volume, then aggregates their inertia tensors
via the parallel-axis theorem to a common reference point.

Usage: python compute_inertia.py <path_to_stl> <mass_kg>
"""

import sys
import numpy as np
import trimesh


def _parallel_axis(I_com: np.ndarray, mass: float, r: np.ndarray) -> np.ndarray:
    """Shift inertia tensor from CoM to a point displaced by r = point - com."""
    r_sq = np.dot(r, r)
    return I_com + mass * (r_sq * np.eye(3) - np.outer(r, r))


def compute_inertia(stl_path: str, mass_kg: float):
    mesh = trimesh.load(stl_path)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a valid mesh from {stl_path}")

    print(f"STL file   : {stl_path}")
    print(f"Mass       : {mass_kg:.6f} kg")

    # ------------------------------------------------------------------
    # Multi-body detection
    # ------------------------------------------------------------------
    # A valid single closed solid has Euler number = 2.  CAD assemblies
    # exported as one STL often contain N separate shells → Euler = 2*N.
    bodies = mesh.split()
    n_bodies = len(bodies)
    euler = mesh.euler_number

    if euler != 2 or n_bodies > 1:
        print(f"\nWARNING: mesh is a multi-body export "
              f"(Euler={euler}, {n_bodies} separate shells).")
        print("  Treating each body as one uniform-density part and summing.")
        print("  (Single-mesh volume integration gives wrong results here.)\n")

    # Use only watertight bodies for the calculation; report any that aren't.
    good_bodies = [b for b in bodies if b.is_watertight and b.volume > 0]
    bad_bodies  = [b for b in bodies if not b.is_watertight or b.volume <= 0]
    if bad_bodies:
        print(f"  Skipping {len(bad_bodies)} non-watertight / zero-volume bodies.")

    total_volume = sum(b.volume for b in good_bodies)
    density = mass_kg / total_volume

    print(f"Bodies (watertight): {len(good_bodies)}")
    print(f"Total volume : {total_volume:.6f} m³  (assuming mesh units are metres)")
    print(f"Density      : {density:.4f} kg/m³")

    for i, b in enumerate(good_bodies):
        b.density = density

    # ------------------------------------------------------------------
    # Composite CoM (mass-weighted average of per-body CoMs)
    # ------------------------------------------------------------------
    composite_com = np.zeros(3)
    for b in good_bodies:
        composite_com += b.mass_properties['center_mass'] * b.mass_properties['mass']
    composite_com /= mass_kg

    print(f"CoM (link frame): x={composite_com[0]:.6f}  "
          f"y={composite_com[1]:.6f}  z={composite_com[2]:.6f}")

    # ------------------------------------------------------------------
    # Aggregate inertia at composite CoM via parallel-axis theorem
    # ------------------------------------------------------------------
    I_com_total = np.zeros((3, 3))
    for b in good_bodies:
        props  = b.mass_properties
        r_body = props['center_mass'] - composite_com   # body CoM relative to composite CoM
        I_com_total += _parallel_axis(props['inertia'], props['mass'], r_body)

    print()
    print("Inertia tensor at composite CoM (kg·m²):")
    print(f"  ixx={I_com_total[0,0]:.6f}  ixy={I_com_total[0,1]:.6f}  ixz={I_com_total[0,2]:.6f}")
    print(f"               iyy={I_com_total[1,1]:.6f}  iyz={I_com_total[1,2]:.6f}")
    print(f"                              izz={I_com_total[2,2]:.6f}")
    print()

    # ------------------------------------------------------------------
    # Shift to link frame origin (for URDF with <origin xyz="0 0 0">)
    # ------------------------------------------------------------------
    r_to_origin = np.zeros(3) - composite_com   # origin - com
    I_origin = _parallel_axis(I_com_total, mass_kg, r_to_origin)

    print("Inertia tensor at link origin (kg·m²)  [for URDF with <origin xyz='0 0 0'>]:")
    print(f"  ixx={I_origin[0,0]:.6f}  ixy={I_origin[0,1]:.6f}  ixz={I_origin[0,2]:.6f}")
    print(f"               iyy={I_origin[1,1]:.6f}  iyz={I_origin[1,2]:.6f}")
    print(f"                              izz={I_origin[2,2]:.6f}")
    print()
    print("URDF snippet (at composite CoM — preferred):")
    print(f'  <origin xyz="{composite_com[0]:.6f} {composite_com[1]:.6f} {composite_com[2]:.6f}" rpy="0 0 0"/>')
    print(f'  <inertia ixx="{I_com_total[0,0]:.6f}" ixy="{I_com_total[0,1]:.6f}" '
          f'ixz="{I_com_total[0,2]:.6f}" iyy="{I_com_total[1,1]:.6f}" '
          f'iyz="{I_com_total[1,2]:.6f}" izz="{I_com_total[2,2]:.6f}"/>')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <path_to_stl> <mass_kg>")
        sys.exit(1)

    stl_path = sys.argv[1]
    mass_kg = float(sys.argv[2])
    compute_inertia(stl_path, mass_kg)
