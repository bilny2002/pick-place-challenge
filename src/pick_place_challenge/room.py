"""Import a real modeled room (Objaverse) as a textured backdrop mesh.

A panorama/HDRI can't be a room you place a robot inside (its "floor" is smeared
across every wall). So we use an actual room mesh: a bright, baked-lighting
living room (Objaverse ``581238dc…``, "Cozy living room baked", CC-BY). We
download the glTF, merge its sub-meshes into one mesh with a baked texture
atlas, rotate it Z-up, scale it to a sane ceiling height, and drop its floor to
the table's floor height. Visual-only. Swap ``ROOM_UID`` for any Objaverse uid.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import objaverse
import trimesh


def _interior_floor_z(mesh: trimesh.Trimesh) -> float:
    """Height of the interior floor: the largest up-facing horizontal surface.

    The mesh's lowest point is usually exterior/under-slab geometry (which faces
    *down*), not the floor you stand on. The floor is the big surface whose
    normals point up, so we take the area-weighted-most-common z among up-facing
    faces. Furniture tops also face up but have far less area.
    """
    normals = mesh.face_normals
    up = normals[:, 2] > 0.9
    if not up.any():
        return float(mesh.bounds[0][2])
    z = mesh.triangles_center[up, 2]
    area = mesh.area_faces[up]
    bins = np.round(z / 0.02).astype(int)  # 2 cm bins
    totals: dict[int, float] = {}
    for b, a in zip(bins, area, strict=True):
        totals[int(b)] = totals.get(int(b), 0.0) + float(a)
    return max(totals, key=totals.get) * 0.02


ROOM_UID = "581238dc5fda4dc990571cdc02827783"
_CACHE = Path.home() / ".cache" / "pick_place_challenge" / "rooms"
_CEILING_HEIGHT = 2.8  # meters from floor to ceiling after scaling


def room_assets(uid: str = ROOM_UID) -> tuple[Path, Path, dict]:
    """Return (obj_path, texture_png_path, meta) for the room, fetching once.

    The exported mesh is Z-up, scaled to ``_CEILING_HEIGHT``, with its floor at
    z=0 and centered at the origin in x/y. ``meta`` holds the floor footprint.
    """
    out = _CACHE / uid
    obj, tex, meta_path = out / "room.obj", out / "room.png", out / "meta.json"
    if obj.exists() and tex.exists() and meta_path.exists():
        return obj, tex, json.loads(meta_path.read_text())

    print(f"[pick-place-challenge] Fetching room mesh '{uid}' (Objaverse)...")
    out.mkdir(parents=True, exist_ok=True)
    glb = list(objaverse.load_objects([uid]).values())[0]
    mesh = trimesh.load(glb).to_geometry()

    # glTF is Y-up; rotate to MuJoCo Z-up.
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    )
    lo, hi = mesh.bounds
    mesh.apply_scale(_CEILING_HEIGHT / float(hi[2] - lo[2]))
    lo, hi = mesh.bounds
    # Interior floor -> z=0 (not the mesh min-z, which is exterior under-slab
    # geometry); center the footprint in x/y.
    floor_z = _interior_floor_z(mesh)
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -floor_z])

    image = mesh.visual.material.baseColorTexture
    image.convert("RGB").save(tex)
    mesh.export(obj)

    lo, hi = mesh.bounds
    meta = {"half_x": float((hi[0] - lo[0]) / 2), "half_y": float((hi[1] - lo[1]) / 2)}
    meta_path.write_text(json.dumps(meta))
    return obj, tex, meta
