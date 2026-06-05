"""A studio "room + table" backdrop injected via SceneCfg.spec_fn.

The room is built as **textured geometry** (six inward-facing walls carrying the
HDRI's cube faces) rather than a MuJoCo skybox. Skyboxes render in the native
viewer and camera sensors but NOT in the Viser browser viewer (which only draws
geometry + geom textures); geometry walls render everywhere.

Everything here is decor — only the table top (at ``z = 0``) collides, so the
task/reward math is unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from pick_place_challenge.polyhaven import hdri_skybox_files

# A unit quad (2x2 in its local XY plane, +z normal) with UVs. Viser only
# textures *mesh* geoms, so the room walls must be meshes, not planes/boxes.
_QUAD_OBJ = """v -1 -1 0
v 1 -1 0
v 1 1 0
v -1 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""


def _quad_obj_path() -> Path:
    path = Path.home() / ".cache" / "pick_place_challenge" / "wall_quad.obj"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_QUAD_OBJ)
    return path


TABLE_HEIGHT: float = 0.4  # table top at z=0, floor at z=-TABLE_HEIGHT
_TABLE_CENTER = (0.3, 0.0)
_TABLE_HALF = (0.45, 0.4)
_TOP_THICK = 0.02
_LEG = 0.03

_ROOM_HALF = 2.2  # half-size of the cubic room
_ROOM_CX = 0.2  # room centered roughly on the workspace in x


def _add_material(spec, name, rgba, reflectance=0.0):
    mat = spec.add_material(name=name, reflectance=reflectance)
    mat.rgba = rgba
    return mat


def _quat_from_axes(normal, up):
    """w,x,y,z quat for a plane whose local +z = normal, local +y = up."""
    nz = _unit(normal)
    uy = _unit(up)
    ux = _unit(_cross(uy, nz))  # local +x (right)
    uy = _cross(nz, ux)  # re-orthogonalize up
    m = [
        [ux[0], uy[0], nz[0]],
        [ux[1], uy[1], nz[1]],
        [ux[2], uy[2], nz[2]],
    ]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = (
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2
        w, x, y, z = (
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2
        w, x, y, z = (
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    else:
        s = math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2
        w, x, y, z = (
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        )
    return (w, x, y, z)


def _unit(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _add_room_box(spec: mujoco.MjSpec) -> None:
    """Six textured walls from the HDRI cube faces (renders in Viser too)."""
    faces = hdri_skybox_files()  # order: R, L, U, D, F, B
    fh = -TABLE_HEIGHT
    s = _ROOM_HALF
    cz = fh + s  # room vertical center
    cx = _ROOM_CX

    # (face file, center, inward-normal, in-plane up, hflip, vflip)
    walls = [
        (faces[3], (cx, 0, fh), (0, 0, 1), (0, 1, 0), False, False),  # D -> floor
        (
            faces[2],
            (cx, 0, fh + 2 * s),
            (0, 0, -1),
            (0, 1, 0),
            False,
            False,
        ),  # U -> ceiling
        (faces[0], (cx + s, 0, cz), (-1, 0, 0), (0, 0, 1), False, False),  # R -> +x wall
        (faces[1], (cx - s, 0, cz), (1, 0, 0), (0, 0, 1), False, False),  # L -> -x wall
        (faces[4], (cx, s, cz), (0, -1, 0), (0, 0, 1), False, False),  # F -> +y wall
        (faces[5], (cx, -s, cz), (0, 1, 0), (0, 0, 1), False, False),  # B -> -y wall
    ]
    quad = spec.add_mesh(name="wall_quad", file=str(_quad_obj_path()))
    quad.scale = [s, s, 1.0]
    quad.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL  # flat mesh, no volume
    room = spec.worldbody.add_body(name="room")
    for i, (tex_file, pos, normal, up, hflip, vflip) in enumerate(walls):
        tex = spec.add_texture(
            name=f"room_tex_{i}", type=mujoco.mjtTexture.mjTEXTURE_2D
        )
        tex.file = str(tex_file)
        tex.hflip, tex.vflip = hflip, vflip
        mat = spec.add_material(name=f"room_mat_{i}")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"room_tex_{i}"
        g = room.add_geom(name=f"wall_{i}")
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = "wall_quad"
        g.pos = list(pos)
        g.quat = list(_quat_from_axes(normal, up))
        g.material = f"room_mat_{i}"
        g.group = 2
        g.contype, g.conaffinity = 0, 0


def add_studio(spec: mujoco.MjSpec) -> None:
    """Add the room + table to a scene spec (a ``SceneCfg.spec_fn``)."""
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]

    key = spec.worldbody.add_light()
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.pos = [0.4, 0.4, 1.6]
    key.dir = [-0.25, -0.25, -1.0]
    key.castshadow = True
    key.diffuse = [0.5, 0.5, 0.5]

    _add_room_box(spec)

    _add_material(spec, "table_mat", (0.30, 0.22, 0.16, 1.0))
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

    _box(
        "table_top",
        (cx, cy, -_TOP_THICK),
        (hx, hy, _TOP_THICK),
        "table_mat",
        collide=True,
    )
    for sx in (-1, 1):
        for sy in (-1, 1):
            _box(
                f"leg_{sx}_{sy}",
                (cx + sx * (hx - _LEG), cy + sy * (hy - _LEG), fh / 2 - _TOP_THICK),
                (_LEG, _LEG, abs(fh) / 2),
                "leg_mat",
            )
