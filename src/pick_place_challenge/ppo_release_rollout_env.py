"""PPO-only vec-env wrapper that masks robot actions after ball release."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from mjlab.managers.scene_entity_config import SceneEntityCfg

from pick_place_challenge import scene


class PpoPostReleaseActionMaskWrapper:
    """Freeze policy-controlled motion after release while rewards keep running.

    The wrapped env is the RSL-RL vec-env wrapper used by mjlab. Once the ball is
    detected as released, this wrapper replaces policy actions for that env with a
    fixed neutral/open-gripper action. The simulation still advances one normal
    PPO step at a time, so post-release trajectory rewards continue accumulating.
    """

    def __init__(
        self,
        env,
        *,
        object_name: str = "ball",
        robot_name: str = "robot",
        grasp_site_name: str = scene.GRASP_SITE,
        height_threshold: float = 0.02,
        velocity_threshold: float = 0.05,
    ) -> None:
        self.env = env
        self.object_name = object_name
        self.height_threshold = height_threshold
        self.velocity_threshold = velocity_threshold
        self._released: torch.Tensor | None = None
        self.asset_cfg = SceneEntityCfg(robot_name, site_names=(grasp_site_name,))
        self.asset_cfg.resolve(self.unwrapped.scene)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def get_observations(self) -> TensorDict:
        return self.env.get_observations()

    def reset(self):
        obs, extras = self.env.reset()
        self._reset_release_state()
        return obs, extras

    def close(self) -> None:
        self.env.close()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self._ensure_release_state()
        assert self._released is not None

        self._released |= self._released_mask()
        masked_actions = actions.clone()
        if torch.any(self._released):
            neutral_actions = self._neutral_actions(actions)
            masked_actions[self._released] = neutral_actions[self._released]

        obs, rewards, dones, extras = self.env.step(masked_actions)

        extras.setdefault("log", {})
        extras["log"]["release_mask/released_rate"] = self._released.float().mean()
        extras["log"]["release_mask/action_masked_rate"] = self._released.float().mean()

        done_mask = dones.bool()
        if torch.any(done_mask):
            self._released[done_mask] = False

        return obs, rewards, dones, extras

    def _released_mask(self) -> torch.Tensor:
        robot = self.unwrapped.scene[self.asset_cfg.name]
        ball = self.unwrapped.scene[self.object_name]

        palm_pos = robot.data.site_pos_w[:, self.asset_cfg.site_ids].squeeze(1)
        palm_vel = robot.data.site_lin_vel_w[:, self.asset_cfg.site_ids].squeeze(1)
        ball_pos = ball.data.root_link_pos_w
        ball_vel = ball.data.root_link_lin_vel_w

        height_separated = (
            torch.abs(ball_pos[:, 2] - palm_pos[:, 2]) > self.height_threshold
        )
        velocity_separated = (
            torch.linalg.norm(ball_vel - palm_vel, dim=-1) > self.velocity_threshold
        )
        return height_separated & velocity_separated

    def _neutral_actions(self, actions: torch.Tensor) -> torch.Tensor:
        neutral = torch.zeros_like(actions, device=self.device)
        if neutral.shape[-1] > 0:
            neutral[..., -1] = -1.0
        return neutral

    def _ensure_release_state(self) -> None:
        if self._released is None or len(self._released) != self.num_envs:
            self._reset_release_state()

    def _reset_release_state(self) -> None:
        self._released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


PpoReleaseRolloutWrapper = PpoPostReleaseActionMaskWrapper
