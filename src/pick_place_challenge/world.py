"""A simple studio "room + table" backdrop, injected via SceneCfg.spec_fn.

Decor only: walls, floor, table legs, skybox and lights are visual (no
collision). The single load-bearing collider is the **table top**, whose top
face is at ``z = 0`` — the same height the ground plane used to be, so the
existing task/reward math (object spawns just above z=0, arm mounted at z=0) is
unchanged. Swapping a plane for a table is purely cosmetic.

Not photoreal — MuJoCo isn't a PBR renderer — but a clean, well-lit studio that
makes the (textured, scanned) object the hero. Drop texture PNGs in for wood/
tile if you want more.
"""

from __future__ import annotations

import mujoco

TABLE_HEIGHT: float = 0.4  # table top at z=0, floor at z=-TABLE_HEIGHT
_TABLE_CENTER = (0.3, 0.0)
_TABLE_HALF = (0.45, 0.4)  # half-extent in x, y
_TOP_THICK = 0.02
_LEG = 0.03


def _add_material(spec: mujoco.MjSpec, name: str, rgba, reflectance: float = 0.0):
    mat = spec.add_material(name=name, reflectance=reflectance)
    mat.rgba = rgba
    return mat


def add_studio(spec: mujoco.MjSpec) -> None:
    """Add a room + table to an existing scene spec (a ``SceneCfg.spec_fn``)."""
    # Dim the default headlight so the key light + shadows give the scene some
    # contrast in the camera renders (otherwise everything is flat and washed out).
    spec.visual.headlight.ambient = [0.2, 0.2, 0.2]
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]

    # --- Sky + lights ---
    from pick_place_challenge.polyhaven import hdri_skybox_files

    sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    sky.cubefiles = hdri_skybox_files()
    key = spec.worldbody.add_light()
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.pos = [0.4, 0.4, 1.6]
    key.dir = [-0.25, -0.25, -1.0]
    key.castshadow = True
    key.diffuse = [0.5, 0.5, 0.5]
    fill = spec.worldbody.add_light()
    fill.type = mujoco.mjtLightType.mjLIGHT_POINT
    fill.pos = [-0.6, -0.6, 1.4]
    fill.castshadow = False
    fill.diffuse = [0.3, 0.3, 0.32]

    # --- Materials ---
    spec.add_texture(
        name="floor_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=(0.26, 0.27, 0.30),
        rgb2=(0.21, 0.22, 0.25),
        width=300,
        height=300,
    )
    floor_mat = spec.add_material(name="floor_mat", reflectance=0.15)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    floor_mat.texrepeat = [6, 6]
    floor_mat.texuniform = True
    _add_material(spec, "table_mat", (0.30, 0.22, 0.16, 1.0), reflectance=0.0)
    _add_material(spec, "leg_mat", (0.20, 0.20, 0.22, 1.0))

    fh = -TABLE_HEIGHT  # floor height

    # --- Floor (visual only) — grounds the table; the HDRI skybox is the room ---
    room = spec.worldbody.add_body(name="room")

    def _box(body, name, pos, half, material, collide=False):
        g = body.add_geom(name=name)
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = list(pos)
        g.size = list(half)
        g.material = material
        g.group = 2
        if not collide:
            g.contype = 0
            g.conaffinity = 0
        return g

    _box(room, "floor", (0.2, 0.0, fh - 0.01), (2.0, 2.0, 0.01), "floor_mat")

    # --- Table: collidable top at z=0 + visual legs ---
    table = spec.worldbody.add_body(name="table")
    cx, cy = _TABLE_CENTER
    hx, hy = _TABLE_HALF
    _box(
        table,
        "table_top",
        (cx, cy, -_TOP_THICK),
        (hx, hy, _TOP_THICK),
        "table_mat",
        collide=True,
    )
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx = cx + sx * (hx - _LEG)
            ly = cy + sy * (hy - _LEG)
            _box(
                table,
                f"leg_{sx}_{sy}",
                (lx, ly, fh / 2 - _TOP_THICK),
                (_LEG, _LEG, abs(fh) / 2),
                "leg_mat",
            )
