"""Render a showcase GIF: random actions with a camera orbiting the scene.

    uv run python scripts/showcase.py                 # -> docs/showcase.gif
    uv run python scripts/showcase.py --frames 180 --width 640 --height 480

Uses MuJoCo's offscreen renderer (EGL) with a free orbit camera, syncing state
from the running env each frame. Needs a GPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402
from PIL import Image  # noqa: E402

import pick_place_challenge.task  # noqa: E402, F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402


@dataclass(frozen=True)
class Args:
    task: str = "Mjlab-PlaceBall-Franka-State-v0"
    out: str = "docs/showcase.gif"
    frames: int = 110
    width: int = 400
    height: int = 400
    fps: int = 20
    colors: int = 128  # GIF palette size (smaller file)
    distance: float = 1.5
    elevation: float = -18.0
    device: str = "cuda"


def main(args: Args) -> None:
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.lookat = [0.3, 0.0, 0.08]
    cam.distance = args.distance
    cam.elevation = args.elevation

    env.reset()
    action_dim = env.action_space.shape[-1]
    frames: list[Image.Image] = []
    for i in range(args.frames):
        env.step(2.0 * torch.rand(1, action_dim, device=args.device) - 1.0)
        data.qpos[:] = env.sim.wp_data.qpos.numpy()[0]
        if model.nmocap:
            data.mocap_pos[:] = env.sim.wp_data.mocap_pos.numpy()[0]
            data.mocap_quat[:] = env.sim.wp_data.mocap_quat.numpy()[0]
        mujoco.mj_forward(model, data)
        cam.azimuth = 90.0 + 360.0 * i / args.frames  # one full orbit
        renderer.update_scene(data, camera=cam)
        frames.append(Image.fromarray(renderer.render()))
    env.close()

    # Quantize all frames to one shared palette — much smaller GIF, no flicker.
    palette = frames[len(frames) // 2].quantize(colors=args.colors)
    quant = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quant[0].save(
        out,
        save_all=True,
        append_images=quant[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({len(frames)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main(tyro.cli(Args))
