"""Smoke tests: the model compiles, both tasks register, and the env steps.

The spec/compile test runs anywhere (pure MuJoCo). The env-stepping tests need
an NVIDIA GPU (mjlab runs on MuJoCo-Warp) and are skipped without one.
"""

from __future__ import annotations

import pytest
import torch

import pick_place_challenge.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from pick_place_challenge.robots.franka_robotiq import (
    GRASP_SITE,
    GRIPPER_TENDON,
    build_franka_robotiq_spec,
)

_STATE = "Mjlab-PickPlace-Franka-State-v0"
_PIXELS = "Mjlab-PickPlace-Franka-Pixels-v0"

_needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="mjlab env requires an NVIDIA GPU (MuJoCo-Warp)",
)


def test_spec_compiles_with_arm_and_gripper() -> None:
    """Franka + Robotiq assemble and compile; grasp site and tendon are present."""
    model = build_franka_robotiq_spec().compile()
    # 7 arm joints + 8 gripper linkage joints; 7 arm actuators + 1 tendon actuator.
    assert model.nq == 15
    assert model.nu == 8
    site_names = {model.site(i).name for i in range(model.nsite)}
    assert GRASP_SITE in site_names
    tendon_names = {model.tendon(i).name for i in range(model.ntendon)}
    assert GRIPPER_TENDON in tendon_names


def test_both_tasks_registered() -> None:
    tasks = set(list_tasks())
    assert {_STATE, _PIXELS} <= tasks


@_needs_gpu
@pytest.mark.parametrize("task_id", [_STATE, _PIXELS])
def test_env_resets_and_steps(task_id: str) -> None:
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    try:
        obs, _ = env.reset()
        action_dim = env.action_space.shape[-1]
        assert action_dim == 8  # 7 arm + 1 gripper
        for _ in range(5):
            action = 2.0 * torch.rand(2, action_dim, device="cuda") - 1.0
            obs, reward, terminated, truncated, _ = env.step(action)
        assert torch.isfinite(reward).all()
        assert torch.isfinite(env.scene["robot"].data.joint_pos).all()
    finally:
        env.close()


@_needs_gpu
def test_pixels_env_has_camera_observations() -> None:
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg(_PIXELS)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    try:
        obs, _ = env.reset()
        cam = obs["camera"]
        for key in ("scene_rgb", "wrist_rgb"):
            assert cam[key].shape == (2, 3, 84, 84)
        # The wrist view must not be the degenerate all-white render.
        assert cam["wrist_rgb"].float().mean() < 0.95
    finally:
        env.close()
