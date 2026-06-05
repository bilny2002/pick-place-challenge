"""Load a Google Scanned Object as a pickable MuJoCo entity.

Each object ships as an obj2mjcf model: a textured visual mesh (geom group 2,
no collision) plus a 32-piece V-HACD convex decomposition (group 3, collidable).
We add a free joint so it can be picked up, and rescale the whole object to a
common size so the Robotiq 2F-85 can always close on it.
"""

from __future__ import annotations

import mujoco
import numpy as np

from pick_place_challenge.assets import DEFAULT_OBJECT, ensure_objects, object_dir

# Longest object dimension after rescaling, in meters. The 2F-85 opens ~85 mm,
# so this keeps every object graspable regardless of its real-world size.
TARGET_MAX_DIM: float = 0.09


def _visual_mesh_extent(model_xml: str) -> float:
    """Largest bounding-box dimension of the 'model' visual mesh at unit scale."""
    m = mujoco.MjModel.from_xml_path(model_xml)
    mesh_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "model")
    adr = m.mesh_vertadr[mesh_id]
    num = m.mesh_vertnum[mesh_id]
    verts = m.mesh_vert[adr : adr + num]
    return float((verts.max(axis=0) - verts.min(axis=0)).max())


def get_scanned_object_spec(
    name: str = DEFAULT_OBJECT, target_max_dim: float = TARGET_MAX_DIM
) -> mujoco.MjSpec:
    """Return an ``MjSpec`` for one scanned object, free-jointed and rescaled."""
    ensure_objects([name])
    model_xml = str(object_dir(name) / "model.xml")

    extent = _visual_mesh_extent(model_xml)
    scale = target_max_dim / extent if extent > 0 else 1.0

    spec = mujoco.MjSpec.from_file(model_xml)
    for mesh in spec.meshes:
        mesh.scale = (np.array(mesh.scale) * scale).tolist()
    # The single body is named "model"; give it a free joint so it's pickable.
    spec.body("model").add_freejoint()
    return spec


def make_object_spec_fn(name: str = DEFAULT_OBJECT):
    """A no-arg ``spec_fn`` for an mjlab ``EntityCfg``, bound to ``name``."""

    def spec_fn() -> mujoco.MjSpec:
        return get_scanned_object_spec(name)

    return spec_fn
