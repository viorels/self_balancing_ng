"""
Terrain generators for PyBullet simulation.

Provides swappable terrain factories selected by CONFIG['TERRAIN'].

Supported values:
    'flat'        — default PyBullet plane (plane.urdf)
    'heightfield' — procedural heightfield with stair-step regions
    'box_stairs'  — sharp-edged stairs built from box primitives

Usage:
    from terrain import create_terrain
    ground_ids = create_terrain(config)   # returns list of body IDs
"""

import numpy as np
import pybullet as p
import pybullet_data


# ============================================================================
# PUBLIC API
# ============================================================================

def create_terrain(config):
    """Create the terrain specified by config['TERRAIN'].

    Returns a list of PyBullet body IDs (ground plane + any obstacles).
    """
    kind = config.get('TERRAIN', 'flat')

    if kind == 'flat':
        return _create_flat(config)
    elif kind == 'heightfield':
        return _create_heightfield(config)
    elif kind == 'box_stairs':
        return _create_box_stairs(config)
    else:
        raise ValueError(f"Unknown terrain type: '{kind}'. "
                         "Choose from: flat, heightfield, box_stairs")


# ============================================================================
# FLAT PLANE
# ============================================================================

def _create_flat(config):
    """Plain flat ground (PyBullet built-in plane.urdf)."""
    ground_id = p.loadURDF("plane.urdf")
    p.changeDynamics(ground_id, -1,
                     lateralFriction=config['GROUND_FRICTION'])
    return [ground_id]


# ============================================================================
# HEIGHTFIELD TERRAIN
# ============================================================================

def _create_heightfield(config):
    """Procedural heightfield with ascending/descending stair-step regions.

    Layout (centred on the origin):
      - flat zone around spawn
      - ascending stairs in +X
      - descending stairs in -X
    """
    rows = cols = 256
    height_data = np.zeros(rows * cols, dtype=np.float32)

    mesh_scale_x = 0.05   # metres per cell in X
    mesh_scale_y = 0.05   # metres per cell in Y

    centre = cols // 2
    flat_half = 20         # cells of flat ground around spawn

    step_width_cells = 16  # cells per stair tread
    step_height = 0.015    # height increment per step (metres)

    for col in range(cols):
        for row in range(rows):
            idx = row * cols + col
            offset = col - centre

            if abs(offset) <= flat_half:
                height_data[idx] = 0.0
            elif offset > flat_half:
                stair_index = (offset - flat_half) // step_width_cells
                height_data[idx] = (stair_index + 1) * step_height
            else:
                stair_index = (abs(offset) - flat_half) // step_width_cells
                height_data[idx] = (stair_index + 1) * step_height

    # Shift so the minimum is at 0
    min_h = float(np.min(height_data))
    height_data -= min_h

    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[mesh_scale_x, mesh_scale_y, 1.0],
        heightfieldTextureScaling=1,
        numHeightfieldRows=rows,
        numHeightfieldColumns=cols,
        heightfieldData=height_data.tolist(),
    )
    terrain_body = p.createMultiBody(0, terrain_shape)

    # Position so the flat centre zone sits at Z = 0
    max_h = float(np.max(height_data))
    p.resetBasePositionAndOrientation(
        terrain_body, [0, 0, max_h / 2], [0, 0, 0, 1])

    p.changeVisualShape(terrain_body, -1, rgbaColor=[0.75, 0.75, 0.75, 1])
    p.changeDynamics(terrain_body, -1,
                     lateralFriction=config['GROUND_FRICTION'])
    return [terrain_body]


# ============================================================================
# BOX STAIRS
# ============================================================================

def _create_box_stairs(config):
    """Sharp-edged staircases made from box primitives on a flat plane.

    Creates a flat ground plane plus two staircases:
      - ascending in +X
      - descending in -X (i.e. ascending when approached from the -X side)

    Tunable via CONFIG keys (all optional, sensible defaults provided):
        STAIR_NUM_STEPS   — number of steps per staircase  (default 8)
        STAIR_STEP_DEPTH  — tread depth in metres           (default 0.15)
        STAIR_STEP_HEIGHT — riser height in metres           (default 0.02)
        STAIR_WIDTH       — width of the staircase           (default 0.6)
        STAIR_START_X     — distance from origin to 1st step (default 0.5)
    """
    # Ground plane (base floor)
    ground_id = p.loadURDF("plane.urdf")
    p.changeDynamics(ground_id, -1,
                     lateralFriction=config['GROUND_FRICTION'])

    body_ids = [ground_id]

    num_steps   = config.get('STAIR_NUM_STEPS', 2)
    step_depth  = config.get('STAIR_STEP_DEPTH', 0.20)
    step_height = config.get('STAIR_STEP_HEIGHT', [0.1, 0.15])
    step_width  = config.get('STAIR_WIDTH', 0.6)
    start_x     = config.get('STAIR_START_X', 0.5)
    friction    = config['GROUND_FRICTION']

    step_color_a = [0.55, 0.55, 0.60, 1.0]
    step_color_b = [0.65, 0.65, 0.70, 1.0]

    def _make_staircase(x_origin, x_sign, step_height):
        """Place one staircase. x_sign = +1 for +X, -1 for -X."""
        ids = []
        for i in range(num_steps):
            # Each step is a box whose height = cumulative height up to this step
            h = step_height * (i + 1)
            half = [step_depth / 2, step_width / 2, h / 2]

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                      rgbaColor=step_color_a if i % 2 == 0
                                      else step_color_b)

            cx = x_origin + x_sign * (i * step_depth + step_depth / 2)
            cz = h / 2

            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[cx, 0, cz])
            p.changeDynamics(body, -1, lateralFriction=friction)
            ids.append(body)
        return ids

    # Ascending staircase in +X
    body_ids += _make_staircase(start_x, +1, step_height=step_height[0])
    # Ascending staircase in -X (descending when going from origin)
    body_ids += _make_staircase(-start_x, -1, step_height=step_height[1])

    return body_ids
