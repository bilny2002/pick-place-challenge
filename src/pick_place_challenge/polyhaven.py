"""Fetch CC0 assets from Poly Haven (https://polyhaven.com) on demand.

Currently just the ball mesh (``baseball_01``), converted glTF -> OBJ via
trimesh and cached under ``~/.cache/pick_place_challenge``. Poly Haven assets are
CC0 (public domain).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import trimesh
from PIL import Image

_API = "https://api.polyhaven.com"
_UA = {"User-Agent": "pick-place-challenge/0.1 (https://indexrobots.ai)"}
_CACHE = Path.home() / ".cache" / "pick_place_challenge" / "polyhaven"

BALL_ID = "baseball_01"


def _get(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        f.write(r.read())


def _api(path: str) -> dict:
    req = urllib.request.Request(f"{_API}{path}", headers=_UA)
    return json.loads(urllib.request.urlopen(req).read())


def ball_obj_path(asset_id: str = BALL_ID, res: str = "1k") -> Path:
    """Fetch a Poly Haven ball model and return a path to an OBJ."""
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


WOOD_ID = "wood_table_001"


def wood_texture_path(asset_id: str = WOOD_ID, res: str = "2k") -> Path:
    """Fetch a Poly Haven wood diffuse texture (PNG) for the table top."""
    png = _CACHE / "textures" / f"{asset_id}_diff_{res}.png"
    if png.exists():
        return png
    print(f"[pick-place-challenge] Fetching wood texture '{asset_id}'...")
    url = _api(f"/files/{asset_id}")["Diffuse"][res]["png"]["url"]
    _get(url, png)
    return png


def fetch_cli() -> None:
    """Console entry point: pre-fetch the ball mesh and wood texture."""
    ball_obj_path()
    wood_texture_path()
    print(f"[pick-place-challenge] Poly Haven assets ready in {_CACHE}")
