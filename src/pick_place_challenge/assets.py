"""Fetch a curated handful of Google Scanned Objects on demand.

The full https://github.com/kevinzakka/mujoco_scanned_objects repo is ~1 GB
(1030 objects). We only want a few, so we do a blob-filtered *sparse* checkout
of just the curated object directories into a local cache — same spirit as
``robot_descriptions`` fetching the robot MJCFs. First use clones ~20 MB; after
that it's cached and offline.

Assets: CC-BY-4.0 (Google Scanned Objects). MJCF wrappers: MIT (kevinzakka).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = "https://github.com/kevinzakka/mujoco_scanned_objects.git"
# Pinned for reproducibility (main @ 2024-era snapshot).
_COMMIT = "6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c"
_CACHE = Path.home() / ".cache" / "pick_place_challenge" / "mujoco_scanned_objects"

# The bowl used as the place target, plus a few extra graspable objects
# candidates can swap in. All are rescaled at load time.
CURATED_OBJECTS: tuple[str, ...] = (
    "Cole_Hardware_Deep_Bowl_Good_Earth_1075",  # the task's bowl
    "Cole_Hardware_Bowl_Scirocco_YellowBlue",
    "Cole_Hardware_Mug_Classic_Blue",
    "Dino_3",
    "Elephant",
    "Great_Dinos_Triceratops_Toy",
)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def object_dir(name: str) -> Path:
    """Path to a fetched object directory (call :func:`ensure_objects` first)."""
    return _CACHE / "models" / name


def ensure_objects(names: tuple[str, ...] | list[str] = CURATED_OBJECTS) -> Path:
    """Sparse-checkout the named object dirs into the cache. Idempotent."""
    missing = [n for n in names if not (object_dir(n) / "model.xml").exists()]
    if not missing:
        return _CACHE

    print(f"[pick-place-challenge] Fetching {len(missing)} scanned object(s)...")
    try:
        if not (_CACHE / ".git").exists():
            _CACHE.parent.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    _REPO,
                    str(_CACHE),
                ]
            )
            _run(["git", "-C", str(_CACHE), "sparse-checkout", "init", "--cone"])
        # Always request the full curated set so the cache is consistent.
        _run(
            [
                "git",
                "-C",
                str(_CACHE),
                "sparse-checkout",
                "set",
                *[f"models/{n}" for n in names],
            ]
        )
        _run(["git", "-C", str(_CACHE), "checkout", _COMMIT])
    except FileNotFoundError as e:  # git not installed
        raise RuntimeError(
            "git is required to fetch scanned objects. Install git, or pre-fetch with "
            "`uv run pick-place-fetch-assets`."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to fetch scanned objects (network?). git said:\n{e.stderr}"
        ) from e

    still_missing = [n for n in names if not (object_dir(n) / "model.xml").exists()]
    if still_missing:
        raise RuntimeError(f"Object(s) not found after fetch: {still_missing}")
    return _CACHE


def fetch_cli() -> None:
    """Console entry point: pre-fetch all curated objects."""
    ensure_objects()
    print(f"[pick-place-challenge] {len(CURATED_OBJECTS)} objects ready in {_CACHE}")
