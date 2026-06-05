"""Register the place-the-ball-in-the-bowl tasks into mjlab's registry.

Importing this module makes ``Mjlab-PlaceBall-Franka-*`` discoverable by
``play`` / ``train`` / ``list_envs``.
"""

from mjlab.tasks.manipulation.config.yam.rl_cfg import (
    yam_lift_cube_ppo_runner_cfg,
    yam_lift_cube_vision_ppo_runner_cfg,
)
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from pick_place_challenge.tasks.pick_place_env_cfg import (
    franka_ball_in_bowl_env_cfg,
    franka_ball_in_bowl_vision_env_cfg,
)


def _state_rl_cfg():
    cfg = yam_lift_cube_ppo_runner_cfg()
    cfg.experiment_name = "franka_place_ball_state"
    return cfg


def _pixels_rl_cfg():
    cfg = yam_lift_cube_vision_ppo_runner_cfg()
    cfg.experiment_name = "franka_place_ball_pixels"
    return cfg


register_mjlab_task(
    task_id="Mjlab-PlaceBall-Franka-State-v0",
    env_cfg=franka_ball_in_bowl_env_cfg(),
    play_env_cfg=franka_ball_in_bowl_env_cfg(play=True),
    rl_cfg=_state_rl_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PlaceBall-Franka-Pixels-v0",
    env_cfg=franka_ball_in_bowl_vision_env_cfg(),
    play_env_cfg=franka_ball_in_bowl_vision_env_cfg(play=True),
    rl_cfg=_pixels_rl_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)
