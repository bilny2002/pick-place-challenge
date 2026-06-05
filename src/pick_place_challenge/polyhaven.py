"""Fetch CC0 assets from Poly Haven (https://polyhaven.com) on demand.

- A real ball mesh (``baseball_01``), converted glTF -> OBJ via trimesh.
- An indoor HDRI, converted equirectangular -> 6 cube faces for a MuJoCo skybox.

Everything is cached under ``~/.cache/pick_place_challenge`` and fetched once.
Poly Haven assets are CC0 (public domain).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import py360convert
import trimesh
from PIL import Image

_API = "https://api.polyhaven.com"
_UA = {"User-Agent": "pick-place-challenge/0.1 (https://indexrobots.ai)"}
_CACHE = Path.home() / ".cache" / "pick_place_challenge" / "polyhaven"

BALL_ID = "baseball_01"
HDRI_ID = "art_studio"  # a bright indoor room; swap for any Poly Haven HDRI id


def _get(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        f.write(r.read())


def _api(path: str) -> dict:
    req = urllib.request.Request(f"{_API}{path}", headers=_UA)
    return json.loads(urllib.request.urlopen(req).read())


# ---------------------------------------------------------------- ball mesh


def ball_obj_path(asset_id: str = BALL_ID, res: str = "1k") -> Path:
    """Fetch a Poly Haven ball model and return a path to an OBJ (with texture)."""
    out_dir = _CACHE / "models" / asset_id
    obj = out_dir / f"{asset_id}.obj"
    if obj.exists():
        return obj

    print(f"[pick-place-challenge] Fetching ball mesh '{asset_id}'...")
    gltf_info = _api(f"/files/{asset_id}")["gltf"][res]["gltf"]
    raw = out_dir / "src"
    _get(gltf_info["url"], raw / f"{asset_id}.gltf")
    for rel, info in gltf_info["include"].items():
        _get(info["url"], raw / rel)

    scene = trimesh.load(raw / f"{asset_id}.gltf")
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene
    mesh.export(obj)
    return obj


def ball_diffuse_path(asset_id: str = BALL_ID, res: str = "1k") -> Path:
    """Path to the ball's diffuse texture as PNG (MuJoCo requires PNG)."""
    ball_obj_path(asset_id, res)  # ensure fetched
    tex_dir = _CACHE / "models" / asset_id / "src" / "textures"
    png = tex_dir / f"{asset_id}_diff_{res}.png"
    if not png.exists():
        Image.open(tex_dir / f"{asset_id}_diff_{res}.jpg").convert("RGB").save(png)
    return png


# ---------------------------------------------------------------- HDRI skybox

# py360convert returns Y-up faces keyed F/R/B/L/U/D; MuJoCo skybox is Z-up and
# wants 6 files. This mapping + flips were tuned by rendering until the room sat
# upright with the horizon level.
_FACE_ORDER = ("R", "L", "U", "D", "F", "B")


def hdri_skybox_files(asset_id: str = HDRI_ID, face: int = 1024) -> list[str]:
    """Fetch an indoor HDRI and return 6 cube-face PNG paths (MuJoCo skybox order)."""
    out_dir = _CACHE / "hdri" / asset_id
    faces = [out_dir / f"face_{i}.png" for i in range(6)]
    if all(f.exists() for f in faces):
        return [str(f) for f in faces]

    print(f"[pick-place-challenge] Fetching HDRI '{asset_id}' and building skybox...")
    src = out_dir / "equirect.jpg"
    _get(_api(f"/files/{asset_id}")["tonemapped"]["url"], src)

    equirect = np.asarray(Image.open(src).convert("RGB"))
    cube = py360convert.e2c(equirect, face_w=face, cube_format="dict")
    # World +Z is up in MuJoCo, so the equirect "up" face is the world ceiling.
    for path, key in zip(faces, _FACE_ORDER, strict=True):
        Image.fromarray(cube[key].astype(np.uint8)).save(path)
    return [str(f) for f in faces]


# ---------------------------------------------------------------- CLI


def fetch_cli() -> None:
    """Console entry point: pre-fetch the ball mesh and HDRI skybox."""
    ball_obj_path()
    hdri_skybox_files()
    print(f"[pick-place-challenge] Poly Haven assets ready in {_CACHE}")
