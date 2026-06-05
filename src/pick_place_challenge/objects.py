"""Pickable / receptacle objects for the place-the-ball-in-the-bowl task.

- The **ball** is a real Poly Haven mesh (``baseball_01``), free-jointed, with a
  sphere collider for clean rolling.
- The **bowl** is a Google Scanned Object whose convex decomposition keeps its
  cavity, so the ball actually nests inside. It is fixed (a static target).

Both are rescaled to sizes that work for the Robotiq 2F-85 and the task.
"""

from __future__ import annotations

import mujoco
import numpy as np

from pick_place_challenge.assets import ensure_objects, object_dir
from pick_place_challenge.polyhaven import ball_diffuse_path, ball_obj_path

BALL_DIAMETER: float = 0.072  # ~tennis/baseball scale; fits the 85 mm gripper
BOWL_NAME: str = "Cole_Hardware_Deep_Bowl_Good_Earth_1075"
BOWL_WIDTH: float = 0.16  # wide enough to drop the ball in comfortably


def _mesh_extent(model_xml: str) -> float:
    m = mujoco.MjModel.from_xml_path(model_xml)
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "model")
    adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[adr : adr + num]
    return float((v.max(axis=0) - v.min(axis=0)).max())


def get_ball_spec(diameter: float = BALL_DIAMETER) -> mujoco.MjSpec:
    """A real ball mesh (visual) with a sphere collider, free-jointed."""
    obj = str(ball_obj_path())
    # Measure the mesh so the visual + sphere collider match the target diameter.
    tmp = mujoco.MjSpec()
    tmp.add_mesh(name="b", file=obj)
    tm = tmp.compile()
    extent = float(tm.geom_rbound[0] * 2.0) if tm.ngeom else diameter
    scale = diameter / extent if extent > 0 else 1.0
    r = diameter / 2.0

    spec = mujoco.MjSpec()
    mesh = spec.add_mesh(name="ball_mesh", file=obj)
    mesh.scale = [scale, scale, scale]
    # Real diffuse texture from Poly Haven, mapped via the mesh's UVs.
    tex = spec.add_texture(name="ball_tex", type=mujoco.mjtTexture.mjTEXTURE_2D)
    tex.file = str(ball_diffuse_path())
    ball_mat = spec.add_material(name="ball_mat")
    ball_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ball_tex"
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
    col.size = [r, 0, 0]
    col.group = 3
    col.mass = 0.05
    col.friction = [1.0, 0.01, 0.001]
    return spec


def get_bowl_spec(name: str = BOWL_NAME, width: float = BOWL_WIDTH) -> mujoco.MjSpec:
    """A scanned bowl, rescaled, with NO free joint (it's a fixed target)."""
    ensure_objects([name])
    model_xml = str(object_dir(name) / "model.xml")
    scale = width / _mesh_extent(model_xml)
    spec = mujoco.MjSpec.from_file(model_xml)
    for mesh in spec.meshes:
        mesh.scale = (np.array(mesh.scale) * scale).tolist()
    return spec


def make_ball_spec_fn():
    def spec_fn() -> mujoco.MjSpec:
        return get_ball_spec()

    return spec_fn


def make_bowl_spec_fn():
    def spec_fn() -> mujoco.MjSpec:
        return get_bowl_spec()

    return spec_fn
