"""MDP terms for the place-the-ball-in-the-bowl task.

Deliberately simple — there's no expert policy and the reward is meant to be
replaced. The "goal" is just the bowl's position (the bowl is a fixed entity),
so no command term is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _target_pos_w(env: ManagerBasedRlEnv, target_name: str, z_offset: float):
    """World position of the bowl's opening (its origin, lifted by z_offset)."""
    bowl: Entity = env.scene[target_name]
    pos = bowl.data.root_link_pos_w.clone()
    pos[:, 2] += z_offset
    return pos


def object_to_target_distance(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_name: str,
    z_offset: float = 0.04,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Vector from the ball to the bowl opening, in the robot base frame."""
    robot: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]
    vec_w = _target_pos_w(env, target_name, z_offset) - obj.data.root_link_pos_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def target_position(
    env: ManagerBasedRlEnv,
    target_name: str,
    z_offset: float = 0.04,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Bowl opening position in the robot base frame."""
    robot: Entity = env.scene[asset_cfg.name]
    rel = _target_pos_w(env, target_name, z_offset) - robot.data.root_link_pos_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), rel)


def reach_and_place_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_name: str,
    reaching_std: float = 0.1,
    placing_std: float = 0.1,
    z_offset: float = 0.04,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """reaching(ee->ball) * (1 + placing(ball->bowl)). Gates placing on reaching."""
    robot: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]
    ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    obj_pos_w = obj.data.root_link_pos_w
    reach_err = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
    reaching = torch.exp(-reach_err / reaching_std**2)
    place_err = torch.sum(
        torch.square(_target_pos_w(env, target_name, z_offset) - obj_pos_w), dim=-1
    )
    placing = torch.exp(-place_err / placing_std**2)
    return reaching * (1.0 + placing)


def placed_in_bowl(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_name: str,
    radius: float = 0.06,
    max_height: float = 0.07,
) -> torch.Tensor:
    """True when the ball is within the bowl's rim radius and below its rim."""
    obj: Entity = env.scene[object_name]
    bowl: Entity = env.scene[target_name]
    obj_pos = obj.data.root_link_pos_w
    bowl_pos = bowl.data.root_link_pos_w
    horizontal = torch.linalg.norm(obj_pos[:, :2] - bowl_pos[:, :2], dim=-1)
    return (horizontal < radius) & (obj_pos[:, 2] < bowl_pos[:, 2] + max_height)
