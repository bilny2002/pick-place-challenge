"""Minimal worked example: drive the env with a random policy.

A readable starting point for the mjlab env API — reset, step random actions,
read rewards/observations. With ``--render`` it also saves a filmstrip from the
scene camera so you can see what happened.

    uv run python scripts/random_rollout.py --task state --steps 200
    uv run python scripts/random_rollout.py --task pixels --render out.png

For an *interactive* viewer with a random policy, prefer mjlab's player:

    uv run play Mjlab-PickPlace-Franka-State-v0 --agent random --viewer native
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

import pick_place_challenge.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

_TASKS = {
    "state": "Mjlab-PickPlace-Franka-State-v0",
    "pixels": "Mjlab-PickPlace-Franka-Pixels-v0",
}


@dataclass(frozen=True)
class Args:
    task: str = "state"
    """Which variant: 'state' or 'pixels'."""
    steps: int = 200
    """Number of environment steps."""
    num_envs: int = 4
    """Parallel environments."""
    device: str = "cuda"
    """Torch/warp device ('cuda' or 'cpu')."""
    render: str | None = None
    """If set (and task=pixels), save a scene-camera filmstrip PNG here."""


def main(args: Args) -> None:
    cfg = load_env_cfg(_TASKS[args.task])
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    action_dim = env.action_space.shape[-1]

    obs, _ = env.reset()
    frames: list[torch.Tensor] = []
    reward_sum = torch.zeros(args.num_envs, device=args.device)
    for step in range(args.steps):
        action = 2.0 * torch.rand(args.num_envs, action_dim, device=args.device) - 1.0
        obs, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward
        if args.render and "camera" in obs and step % 10 == 0:
            frames.append(obs["camera"]["scene_rgb"][0].detach().clone())

    print(f"Ran {args.steps} steps x {args.num_envs} envs on {args.device}.")
    print(f"Mean episode-so-far reward: {reward_sum.mean().item():.3f}")

    if args.render and frames:
        from PIL import Image

        strip = torch.cat(frames, dim=2)  # concat along width: (3, H, W*n)
        img = (strip.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(img).save(args.render)
        print(f"Saved filmstrip to {args.render}")

    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
