"""
Terrain generators for MuJoCo simulation.

Provides swappable terrain factories selected by config.terrain.terrain_type.

Supported values:
    'flat'        — infinite ground plane
    'heightfield' — procedural heightfield with stair-step regions
    'box_stairs'  — sharp-edged stairs built from box primitives

Usage:
    from terrain import get_terrain_xml, post_load_terrain

    # Before model compilation: get XML fragments to inject
    terrain = get_terrain_xml(config)
    # terrain['asset']     — XML string for <asset> section (or None)
    # terrain['worldbody'] — XML string for <worldbody> section

    # After model compilation: fill runtime data (heightfield only)
    post_load_terrain(model, config)
"""

import numpy as np


# ============================================================================
# PUBLIC API
# ============================================================================

def get_terrain_xml(config):
    """Return MJCF XML fragments for the configured terrain type.

    Returns a dict with:
        'asset'     — XML string to insert inside <asset>, or None
        'worldbody' — XML string to insert inside <worldbody>
    """
    kind = config.terrain.terrain_type

    if kind == 'flat':
        return _flat_xml(config)
    elif kind == 'heightfield':
        return _heightfield_xml(config)
    elif kind == 'box_stairs':
        return _box_stairs_xml(config)
    else:
        raise ValueError(f"Unknown terrain type: '{kind}'. "
                         "Choose from: flat, heightfield, box_stairs")


def post_load_terrain(model, config):
    """Fill runtime terrain data after model compilation.

    Currently only needed for heightfield terrain (fills model.hfield_data).
    Call this after mujoco.MjModel.from_xml_string() but before stepping.
    """
    if config.terrain.terrain_type == 'heightfield':
        _fill_heightfield_data(model, config)


# ============================================================================
# FLAT PLANE
# ============================================================================

def _flat_xml(config):
    friction = config.terrain.ground_friction
    return {
        'asset': None,
        'worldbody': (
            f'<geom name="floor" type="plane" size="10 10 0.1" '
            f'friction="{friction} 0.005 0.001" '
            f'rgba="0.8 0.8 0.8 1"/>'
        ),
    }


# ============================================================================
# HEIGHTFIELD TERRAIN
# ============================================================================

def _heightfield_xml(config):
    """Return XML fragments declaring a heightfield.

    The actual height data is filled by post_load_terrain() after compilation.
    """
    rows = cols = 256
    friction = config.terrain.ground_friction
    # size = [x_half_extent, y_half_extent, z_max, z_min_clip]
    x_half = cols * 0.05 / 2  # 256 * 0.05 / 2 = 6.4
    y_half = rows * 0.05 / 2

    return {
        'asset': (
            f'<hfield name="terrain" nrow="{rows}" ncol="{cols}" '
            f'size="{x_half} {y_half} 0.3 0.001"/>'
        ),
        'worldbody': (
            f'<geom name="floor" type="hfield" hfield="terrain" '
            f'friction="{friction} 0.005 0.001" '
            f'rgba="0.75 0.75 0.75 1"/>'
        ),
    }


def _fill_heightfield_data(model, config):
    """Generate and fill heightfield data into the compiled model."""
    import mujoco

    rows = cols = 256
    height_data = np.zeros(rows * cols, dtype=np.float32)

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

    # Normalize to [0, 1] range — MuJoCo hfield_data expects values in [0, 1]
    # which are then scaled by the z component of the hfield size attribute.
    max_h = float(np.max(height_data))
    if max_h > 0:
        height_data /= max_h

    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain')
    model.hfield_data[:] = height_data


# ============================================================================
# BOX STAIRS
# ============================================================================

def _box_stairs_xml(config):
    """Generate box stair XML geoms on a flat ground plane."""
    friction = config.terrain.ground_friction
    num_steps = config.terrain.stair_num_steps
    step_depth = config.terrain.stair_step_depth
    step_height = config.terrain.stair_step_height
    step_width = config.terrain.stair_width
    start_x = config.terrain.stair_start_x

    step_color_a = "0.55 0.55 0.60 1"
    step_color_b = "0.65 0.65 0.70 1"

    parts = [
        f'<geom name="floor" type="plane" size="10 10 0.1" '
        f'friction="{friction} 0.005 0.001" rgba="0.8 0.8 0.8 1"/>'
    ]

    def _make_staircase(prefix, x_origin, x_sign, sh):
        for i in range(num_steps):
            h = sh * (i + 1)
            half_x = step_depth / 2
            half_y = step_width / 2
            half_z = h / 2
            cx = x_origin + x_sign * (i * step_depth + step_depth / 2)
            cz = h / 2
            color = step_color_a if i % 2 == 0 else step_color_b
            parts.append(
                f'<geom name="stair_{prefix}_{i}" type="box" '
                f'size="{half_x} {half_y} {half_z}" '
                f'pos="{cx} 0 {cz}" '
                f'friction="{friction} 0.005 0.001" '
                f'rgba="{color}"/>'
            )

    _make_staircase("pos", start_x, +1, step_height[0])
    _make_staircase("neg", -start_x, -1, step_height[1])

    return {
        'asset': None,
        'worldbody': '\n    '.join(parts),
    }
