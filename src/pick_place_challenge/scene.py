"""The physical scene: assets, robot, objects, and the room + table.

Everything that builds MuJoCo geometry lives here, in one place, so it's easy to
find and hack. Organized top-to-bottom:

    1. Asset fetching  — pull/convert/cache external assets (CC0 / CC-BY).
    2. Robot           — Franka Panda + Robotiq 2F-85 entity.
    3. Pick objects    — the ball (manipuland) and the bowl (target).
    4. Room + table    — the visual backdrop and the work surface.

Nothing is vendored; assets are fetched on first use into ``~/.cache`` and
reused thereafter. The only collider that matters for the task is the table top
at ``z = 0``; everything else is visual decor.
"""

from __future__ import annotations

import json
import math
import subprocess
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np
import objaverse
import robot_descriptions.panda_mj_description as _panda_desc
import robot_descriptions.robotiq_2f85_mj_description as _robotiq_desc
import trimesh
from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from PIL import Image

_CACHE = Path.home() / ".cache" / "pick_place_challenge"

# ============================================================================
# 1. Asset fetching
# ============================================================================

# --- Google Scanned Objects (the bowl), via a blob-filtered sparse git checkout
# of just the dirs we want (the full repo is ~1 GB / 1030 objects). CC-BY-4.0.
_GSO_REPO = "https://github.com/kevinzakka/mujoco_scanned_objects.git"
_GSO_COMMIT = "6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c"  # pinned
_GSO_DIR = _CACHE / "mujoco_scanned_objects"

# The bowl used as the target, plus a few extra graspable objects to swap in.
CURATED_OBJECTS: tuple[str, ...] = (
    "Cole_Hardware_Deep_Bowl_Good_Earth_1075",  # the task's bowl
    "Cole_Hardware_Bowl_Scirocco_YellowBlue",
    "Cole_Hardware_Mug_Classic_Blue",
    "Dino_3",
    "Elephant",
    "Great_Dinos_Triceratops_Toy",
)


def _scanned_object_dir(name: str) -> Path:
    return _GSO_DIR / "models" / name


def ensure_scanned_objects(names: Sequence[str] = CURATED_OBJECTS) -> None:
    """Sparse-checkout the named Scanned-Object dirs into the cache. Idempotent."""
    if all((_scanned_object_dir(n) / "model.xml").exists() for n in names):
        return
    print(f"[pick-place-challenge] Fetching {len(names)} scanned object(s)...")

    def run(*args: str) -> None:
        subprocess.run(args, check=True, capture_output=True, text=True)

    try:
        if not (_GSO_DIR / ".git").exists():
            _GSO_DIR.parent.mkdir(parents=True, exist_ok=True)
            run(
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                _GSO_REPO,
                str(_GSO_DIR),
            )
            run("git", "-C", str(_GSO_DIR), "sparse-checkout", "init", "--cone")
        run(
            "git",
            "-C",
            str(_GSO_DIR),
            "sparse-checkout",
            "set",
            *[f"models/{n}" for n in names],
        )
        run("git", "-C", str(_GSO_DIR), "checkout", _GSO_COMMIT)
    except FileNotFoundError as e:
        raise RuntimeError("git is required to fetch scanned objects.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to fetch scanned objects (network?):\n{e.stderr}"
        ) from e


# --- Poly Haven (the ball mesh + its texture, and the wood table texture). CC0.
_PH_API = "https://api.polyhaven.com"
_PH_UA = {"User-Agent": "pick-place-challenge/0.1 (https://indexrobots.ai)"}
_PH_DIR = _CACHE / "polyhaven"
BALL_ID = "baseball_01"
WOOD_ID = "wood_table_001"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_PH_UA)
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        f.write(r.read())


def _ph_files(asset_id: str) -> dict:
    req = urllib.request.Request(f"{_PH_API}/files/{asset_id}", headers=_PH_UA)
    return json.loads(urllib.request.urlopen(req).read())


def ball_obj_path(asset_id: str = BALL_ID, res: str = "1k") -> Path:
    """Fetch the Poly Haven ball model and return its converted OBJ path."""
    out, obj = (
        _PH_DIR / "models" / asset_id,
        _PH_DIR / "models" / asset_id / f"{asset_id}.obj",
    )
    if obj.exists():
        return obj
    print(f"[pick-place-challenge] Fetching ball mesh '{asset_id}'...")
    gltf = _ph_files(asset_id)["gltf"][res]["gltf"]
    raw = out / "src"
    _download(gltf["url"], raw / f"{asset_id}.gltf")
    for rel, info in gltf["include"].items():
        _download(info["url"], raw / rel)
    scene = trimesh.load(raw / f"{asset_id}.gltf")
    (scene.to_geometry() if hasattr(scene, "to_geometry") else scene).export(obj)
    return obj


def ball_texture_path(asset_id: str = BALL_ID, res: str = "1k") -> Path:
    """The ball's diffuse texture as PNG (MuJoCo only loads PNG)."""
    ball_obj_path(asset_id, res)
    tex = _PH_DIR / "models" / asset_id / "src" / "textures"
    png = tex / f"{asset_id}_diff_{res}.png"
    if not png.exists():
        Image.open(tex / f"{asset_id}_diff_{res}.jpg").convert("RGB").save(png)
    return png


def wood_texture_path(asset_id: str = WOOD_ID, res: str = "2k") -> Path:
    """Fetch a Poly Haven wood diffuse texture (PNG) for the table top."""
    png = _PH_DIR / "textures" / f"{asset_id}_diff_{res}.png"
    if not png.exists():
        print(f"[pick-place-challenge] Fetching wood texture '{asset_id}'...")
        _download(_ph_files(asset_id)["Diffuse"][res]["png"]["url"], png)
    return png


# --- Objaverse (the room mesh). Swap ROOM_UID for any Objaverse uid. CC-BY.
ROOM_UID = "581238dc5fda4dc990571cdc02827783"  # "Cozy living room baked"
_ROOM_DIR = _CACHE / "rooms"
_CEILING_HEIGHT = 2.8  # meters, after scaling


def _interior_floor_z(mesh: trimesh.Trimesh) -> float:
    """Height of the interior floor: the largest up-facing horizontal surface.

    The mesh's lowest point is usually exterior/under-slab geometry (faces
    *down*), not the floor you stand on. The floor is the big up-facing surface,
    so take the area-weighted modal z among up-facing faces (furniture tops also
    face up but have far less area).
    """
    up = mesh.face_normals[:, 2] > 0.9
    if not up.any():
        return float(mesh.bounds[0][2])
    z, area = mesh.triangles_center[up, 2], mesh.area_faces[up]
    totals: dict[int, float] = {}
    for b, a in zip(np.round(z / 0.02).astype(int), area, strict=True):
        totals[int(b)] = totals.get(int(b), 0.0) + float(a)
    return max(totals, key=totals.get) * 0.02


def room_obj_path(uid: str = ROOM_UID) -> tuple[Path, Path]:
    """Fetch + convert the room mesh; return (obj_path, texture_png_path).

    The exported mesh is Z-up, scaled to ``_CEILING_HEIGHT``, with its interior
    floor at z=0 and footprint centered at the origin in x/y.
    """
    out = _ROOM_DIR / uid
    obj, tex = out / "room.obj", out / "room.png"
    if obj.exists() and tex.exists():
        return obj, tex
    print(f"[pick-place-challenge] Fetching room mesh '{uid}' (Objaverse)...")
    out.mkdir(parents=True, exist_ok=True)
    glb = list(objaverse.load_objects([uid]).values())[0]
    mesh = trimesh.load(glb).to_geometry()
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    )
    lo, hi = mesh.bounds
    mesh.apply_scale(_CEILING_HEIGHT / float(hi[2] - lo[2]))
    lo, hi = mesh.bounds
    mesh.apply_translation(
        [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -_interior_floor_z(mesh)]
    )
    mesh.visual.material.baseColorTexture.convert("RGB").save(tex)
    mesh.export(obj)
    return obj, tex


def fetch_all_assets() -> None:
    """Console entry point: pre-fetch every external asset."""
    ensure_scanned_objects()
    ball_texture_path()
    wood_texture_path()
    room_obj_path()
    print(f"[pick-place-challenge] All assets ready in {_CACHE}")


# ============================================================================
# 2. Robot — Franka Panda + Robotiq 2F-85
# ============================================================================

# Hand-less Panda (sibling of panda.xml in the menagerie cache) exposes an empty
# ``attachment`` body on link7; the 2F-85 is grafted onto it.
_PANDA_NOHAND_XML = Path(_panda_desc.MJCF_PATH).parent / "panda_nohand.xml"
_ROBOTIQ_XML = Path(_robotiq_desc.MJCF_PATH)
# 45° about +Z aligns the jaws to the flange (a quat replacement, not a compose).
_ATTACHMENT_QUAT_WXYZ = (0.9238795, 0.0, 0.0, 0.3826834)

GRASP_SITE = "pinch"  # the 2F-85's jaw-center site — a natural grasp reference
GRIPPER_TENDON = "split"  # tendon driven by the single fingers_actuator
GRIPPER_CTRL_MAX = 255.0  # ctrl: 0 = open .. 255 = closed
ARM_ACTION_SCALE = 0.5  # position-action scale for the 7 arm joints

# Menagerie's Panda "home" pose; gripper joints rest open (".*": 0).
_HOME_ARM = {
    "joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": -1.57079,
    "joint5": 0.0, "joint6": 1.57079, "joint7": -0.7853,
}  # fmt: skip

# Reuse the MJCFs' tuned XML actuators (arm position servos + the gripper tendon)
# rather than re-deriving gains — XmlActuator wraps them in place.
_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=(r"joint[1-7]",),
            transmission_type=TransmissionType.JOINT,
            command_field="position",
        ),
        XmlActuatorCfg(
            target_names_expr=(GRIPPER_TENDON,),
            transmission_type=TransmissionType.TENDON,
            command_field="position",
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
)


def build_franka_robotiq_spec() -> mujoco.MjSpec:
    """An ``MjSpec`` for a Franka arm with a Robotiq 2F-85 grafted on."""
    arm = mujoco.MjSpec.from_file(str(_PANDA_NOHAND_XML))
    arm.body("attachment").quat = list(_ATTACHMENT_QUAT_WXYZ)
    gripper = mujoco.MjSpec.from_file(str(_ROBOTIQ_XML))
    arm.attach(gripper, site="attachment_site", prefix="")
    return arm


def franka_robotiq_cfg() -> EntityCfg:
    """mjlab Entity config: fixed-base Franka + Robotiq 2F-85 at the origin."""
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            #joint_pos={**_HOME_ARM, ".*": 0.0},
            joint_pos={
            **_HOME_ARM,
            ".*": 0.0,
            "right_driver_joint": 0.55,
            "left_driver_joint": 0.55,
        },
            joint_vel={".*": 0.0},
        ),
        spec_fn=build_franka_robotiq_spec,
        articulation=_ARTICULATION,
    )


# ============================================================================
# 3. Pick objects — the ball (manipuland) and the bowl (target)
# ============================================================================

BALL_DIAMETER = 0.072  # ~tennis/baseball scale; fits the 85 mm gripper
BOWL_NAME = "Cole_Hardware_Deep_Bowl_Good_Earth_1075"
BOWL_WIDTH = 0.16  # wide enough to drop the ball into


def _scanned_mesh_extent(model_xml: str) -> float:
    m = mujoco.MjModel.from_xml_path(model_xml)
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "model")
    adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[adr : adr + num]
    return float((v.max(axis=0) - v.min(axis=0)).max())


def ball_spec() -> mujoco.MjSpec:
    """A real ball mesh (textured visual) with a sphere collider, free-jointed."""
    obj = str(ball_obj_path())
    tmp = mujoco.MjSpec()
    tmp.add_mesh(name="b", file=obj)
    tm = tmp.compile()
    extent = float(tm.geom_rbound[0] * 2.0) if tm.ngeom else BALL_DIAMETER
    scale = BALL_DIAMETER / extent if extent > 0 else 1.0

    spec = mujoco.MjSpec()
    mesh = spec.add_mesh(name="ball_mesh", file=obj)
    mesh.scale = [scale, scale, scale]
    tex = spec.add_texture(name="ball_tex", type=mujoco.mjtTexture.mjTEXTURE_2D)
    tex.file = str(ball_texture_path())
    mat = spec.add_material(name="ball_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ball_tex"
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint()
    vis = body.add_geom(name="ball_visual")
    vis.type = mujoco.mjtGeom.mjGEOM_MESH
    vis.meshname = "ball_mesh"
    vis.group = 2
    vis.contype, vis.conaffinity = 0, 0
    vis.material = "ball_mat"
    col = body.add_geom(name="ball_collision")
    col.type = mujoco.mjtGeom.mjGEOM_SPHERE
    col.size = [BALL_DIAMETER / 2.0, 0, 0]
    col.group = 3
    col.mass = 0.05
    col.friction = [1.0, 0.01, 0.001]
    return spec


def bowl_spec() -> mujoco.MjSpec:
    """A scanned bowl, rescaled, fixed (its cavity holds the ball)."""
    ensure_scanned_objects([BOWL_NAME])
    model_xml = str(_scanned_object_dir(BOWL_NAME) / "model.xml")
    scale = BOWL_WIDTH / _scanned_mesh_extent(model_xml)
    spec = mujoco.MjSpec.from_file(model_xml)
    for mesh in spec.meshes:
        mesh.scale = (np.array(mesh.scale) * scale).tolist()
    return spec

# ============================================================================
# 3.1. Walls and floor
# ============================================================================
# Enclosure: half-extents in MuJoCo box coordinates.
_WALL_X_HALF = 2.9
_WALL_Y_HALF = 3.0
_WALL_THICK = 0.01
_WALL_Z_CENTER = 1.10
_WALL_Z_HALF = 1.50
_FLOOR_Z = -0.41
_ROOF_Z = 2.61


def add_bounce_enclosure(spec: mujoco.MjSpec) -> None:
    """Visible fixed walls/floor/roof that collide with the ball."""
    mat = spec.add_material(name="bounce_wall_mat")
    mat.rgba = [0.15, 0.35, 0.95, 1.0]

    def wall(name: str, pos: tuple[float, float, float], size: tuple[float, float, float]) -> None:
        body = spec.worldbody.add_body(name=name, pos=pos)
        geom = body.add_geom(name=f"{name}_geom")
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = list(size)
        geom.material = "bounce_wall_mat"
        geom.group = 3
        geom.solref = [0.01, 0.4]
        geom.friction = [0.8, 0.02, 0.001]

    wall("l_bounce_wall", (0.0, 3.0, _WALL_Z_CENTER), (_WALL_X_HALF, _WALL_THICK, _WALL_Z_HALF))
    wall("r_bounce_wall", (0.0, -3.0, _WALL_Z_CENTER), (_WALL_X_HALF, _WALL_THICK, _WALL_Z_HALF))
    wall("f_bounce_wall", (2.9, 0.0, _WALL_Z_CENTER), (_WALL_THICK, _WALL_Y_HALF, _WALL_Z_HALF))
    wall("b_bounce_wall", (-2.9, 0.0, _WALL_Z_CENTER), (_WALL_THICK, _WALL_Y_HALF, _WALL_Z_HALF))
    wall("u_bounce_wall", (0.0, 0.0, _ROOF_Z), (_WALL_X_HALF, _WALL_Y_HALF, _WALL_THICK))
    wall("d_bounce_wall", (0.0, 0.0, _FLOOR_Z), (_WALL_X_HALF, _WALL_Y_HALF, _WALL_THICK))

# ============================================================================
# 4. Room + table — the SceneCfg.spec_fn backdrop
# ============================================================================

TABLE_HEIGHT = 0.4  # table top at z=0, room floor at z=-TABLE_HEIGHT
_TABLE_CENTER = (0.3, 0.0)
_TABLE_HALF = (0.45, 0.4)
_TOP_THICK = 0.02
_LEG = 0.03
# Unit quad (2x2, +z normal) with UVs — Viser only textures *mesh* geoms, so the
# visible table top / room walls must be meshes, not boxes/planes.
_QUAD_OBJ = "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"


def _quad_obj_path() -> Path:
    path = _CACHE / "table_quad.obj"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_QUAD_OBJ)
    return path


def _add_material(spec, name, rgba, reflectance=0.0):
    mat = spec.add_material(name=name, reflectance=reflectance)
    mat.rgba = rgba
    return mat


def add_studio(spec: mujoco.MjSpec) -> None:
    """Add the room + table to a scene spec (used as ``SceneCfg.spec_fn``).

    The room is a real modeled mesh (not a skybox/panorama — a panorama can't be
    a room you sit a robot in). It's visual-only; the only collider is the table
    top at z=0, so the task is unchanged.
    """
    #   walls
    add_bounce_enclosure(spec)

    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    key = spec.worldbody.add_light()
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.pos = [0.4, 0.4, 1.6]
    key.dir = [-0.25, -0.25, -1.0]
    key.castshadow = True
    key.diffuse = [0.7, 0.7, 0.7]

    # # --- Room mesh (visual only). Floor dropped to the table-leg height, rotated
    # # 180° about z so the robot faces in, shifted +1m in x so the table lands on
    # # open floor rather than in the room's furniture.
    # obj, tex = room_obj_path()
    # rmesh = spec.add_mesh(name="room_mesh", file=str(obj))
    # rmesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    # rtex = spec.add_texture(name="room_tex", type=mujoco.mjtTexture.mjTEXTURE_2D)
    # rtex.file = str(tex)
    # rmat = spec.add_material(name="room_mat")
    # rmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "room_tex"
    # room = spec.worldbody.add_body(name="room")
    # rg = room.add_geom(name="room_geom")
    # rg.type = mujoco.mjtGeom.mjGEOM_MESH
    # rg.meshname = "room_mesh"
    # rg.material = "room_mat"
    # rg.pos = [_TABLE_CENTER[0] + 1.0, _TABLE_CENTER[1], -TABLE_HEIGHT]
    # rg.quat = [0.0, 0.0, 0.0, 1.0]
    # rg.group = 2
    # rg.contype, rg.conaffinity = 0, 0

    # --- Table: wood-textured top mesh on a thin collider slab + dark legs.
    wtex = spec.add_texture(name="wood_tex", type=mujoco.mjtTexture.mjTEXTURE_2D)
    wtex.file = str(wood_texture_path())
    wmat = spec.add_material(name="wood_mat", reflectance=0.05)
    wmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wood_tex"
    _add_material(spec, "edge_mat", (0.18, 0.12, 0.08, 1.0))
    _add_material(spec, "leg_mat", (0.20, 0.20, 0.22, 1.0))

    cx, cy = _TABLE_CENTER
    hx, hy = _TABLE_HALF
    fh = -TABLE_HEIGHT
    table = spec.worldbody.add_body(name="table")

    def box(name, pos, half, material, collide=False):
        g = table.add_geom(name=name)
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos, g.size, g.material, g.group = list(pos), list(half), material, 2
        if not collide:
            g.contype, g.conaffinity = 0, 0

    # Collider slab (top face at z=0) gives thickness + physics.
    box(
        "table_top",
        (cx, cy, -_TOP_THICK),
        (hx, hy, _TOP_THICK),
        "edge_mat",
        collide=True,
    )
    # Wood surface mesh laid on the slab (visual only; shows in Viser).
    tmesh = spec.add_mesh(name="table_top_mesh", file=str(_quad_obj_path()))
    tmesh.scale = [hx, hy, 1.0]
    tmesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    top = table.add_geom(name="table_top_visual")
    top.type = mujoco.mjtGeom.mjGEOM_MESH
    top.meshname = "table_top_mesh"
    top.pos = [cx, cy, 0.001]
    top.material = "wood_mat"
    top.group = 2
    top.contype, top.conaffinity = 0, 0
    # Legs span from the table top (z=0) to the floor (z=fh): feet on the floor.
    leg_h = abs(fh) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(
                f"leg_{sx}_{sy}",
                (cx + sx * (hx - _LEG), cy + sy * (hy - _LEG), fh + leg_h),
                (_LEG, _LEG, leg_h),
                "leg_mat",
            )
