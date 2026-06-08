"""Scene backdrop injected via SceneCfg.spec_fn: a real room mesh + a table.

The room is an actual modeled mesh (an Objaverse living room; see ``room.py``),
not a skybox/panorama — a panorama can't be a room you place a robot inside.
It's visual-only geometry, so it renders in the native viewer, the camera
observations, and the Viser browser viewer alike. The only collider is the table
top at ``z = 0``, so the task/reward math is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

TABLE_HEIGHT: float = 0.4  # table top at z=0, room floor at z=-TABLE_HEIGHT
_TABLE_CENTER = (0.3, 0.0)
_TABLE_HALF = (0.45, 0.4)
_TOP_THICK = 0.02
_LEG = 0.03

# Unit quad (2x2, +z normal) with UVs — for the wood table-top surface. Viser
# only textures mesh geoms, so the visible top must be a mesh, not a box.
_QUAD_OBJ = "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"


def _quad_obj_path() -> Path:
    path = Path.home() / ".cache" / "pick_place_challenge" / "table_quad.obj"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_QUAD_OBJ)
    return path


def _add_material(spec, name, rgba, reflectance=0.0):
    mat = spec.add_material(name=name, reflectance=reflectance)
    mat.rgba = rgba
    return mat


def _add_room(spec: mujoco.MjSpec) -> None:
    """Add the Objaverse room mesh (visual only), floor at table-leg height."""
    from pick_place_challenge.room import room_assets

    obj, tex, _ = room_assets()
    mesh = spec.add_mesh(name="room_mesh", file=str(obj))
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    t = spec.add_texture(name="room_tex", type=mujoco.mjtTexture.mjTEXTURE_2D)
    t.file = str(tex)
    mat = spec.add_material(name="room_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "room_tex"
    room = spec.worldbody.add_body(name="room")
    g = room.add_geom(name="room_geom")
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = "room_mesh"
    g.material = "room_mat"
    # Floor at the table-leg height, rotated 180° about z so the robot faces into
    # the room. Shifted +1m in x (≡ moving the workspace −1m in the robot's x) so
    # the table sits on open floor instead of in the room's furniture.
    g.pos = [_TABLE_CENTER[0] + 1.0, _TABLE_CENTER[1], -TABLE_HEIGHT]
    g.quat = [0.0, 0.0, 0.0, 1.0]
    g.group = 2
    g.contype, g.conaffinity = 0, 0


def add_studio(spec: mujoco.MjSpec) -> None:
    """Add the room + table to a scene spec (a ``SceneCfg.spec_fn``)."""
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]

    key = spec.worldbody.add_light()
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.pos = [0.4, 0.4, 1.6]
    key.dir = [-0.25, -0.25, -1.0]
    key.castshadow = True
    key.diffuse = [0.7, 0.7, 0.7]

    _add_room(spec)

    from pick_place_challenge.polyhaven import wood_texture_path

    # Wood material for the table top (textured mesh) + dark legs/edge.
    wood_tex = spec.add_texture(name="wood_tex", type=mujoco.mjtTexture.mjTEXTURE_2D)
    wood_tex.file = str(wood_texture_path())
    wood_mat = spec.add_material(name="wood_mat", reflectance=0.05)
    wood_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wood_tex"
    _add_material(spec, "edge_mat", (0.18, 0.12, 0.08, 1.0))  # table edge / collider
    _add_material(spec, "leg_mat", (0.20, 0.20, 0.22, 1.0))
    fh = -TABLE_HEIGHT

    table = spec.worldbody.add_body(name="table")
    cx, cy = _TABLE_CENTER
    hx, hy = _TABLE_HALF

    def _box(name, pos, half, material, collide=False):
        g = table.add_geom(name=name)
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = list(pos)
        g.size = list(half)
        g.material = material
        g.group = 2
        if not collide:
            g.contype, g.conaffinity = 0, 0

    # Collider slab (top face at z=0) — gives the table thickness + physics.
    _box(
        "table_top",
        (cx, cy, -_TOP_THICK),
        (hx, hy, _TOP_THICK),
        "edge_mat",
        collide=True,
    )
    # Wood-textured surface mesh laid on the slab top (visual only; shows in Viser).
    quad = spec.add_mesh(name="table_top_mesh", file=str(_quad_obj_path()))
    quad.scale = [hx, hy, 1.0]
    quad.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    top = table.add_geom(name="table_top_visual")
    top.type = mujoco.mjtGeom.mjGEOM_MESH
    top.meshname = "table_top_mesh"
    top.pos = [cx, cy, 0.001]
    top.material = "wood_mat"
    top.group = 2
    top.contype, top.conaffinity = 0, 0

    # Legs span from just under the table top (z=0) down to the floor (z=fh),
    # so the feet sit exactly on the room floor.
    leg_half_h = abs(fh) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            _box(
                f"leg_{sx}_{sy}",
                (cx + sx * (hx - _LEG), cy + sy * (hy - _LEG), fh + leg_half_h),
                (_LEG, _LEG, leg_half_h),
                "leg_mat",
            )
