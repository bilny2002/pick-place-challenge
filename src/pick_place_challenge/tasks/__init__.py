"""Register the Franka pick-and-place tasks into mjlab's task registry.

Importing this module is what makes ``Mjlab-PickPlace-Franka-*`` discoverable by
``play`` / ``train`` / ``list_envs``.
"""

from mjlab.tasks.manipulation.config.yam.rl_cfg import (
    yam_lift_cube_ppo_runner_cfg,
    yam_lift_cube_vision_ppo_runner_cfg,
)
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from pick_place_challenge.tasks.pick_place_env_cfg import (
    franka_pick_place_env_cfg,
    franka_pick_place_vision_env_cfg,
)


def _state_rl_cfg():
    cfg = yam_lift_cube_ppo_runner_cfg()
    cfg.experiment_name = "franka_pick_place_state"
    return cfg


def _pixels_rl_cfg():
    cfg = yam_lift_cube_vision_ppo_runner_cfg()
    cfg.experiment_name = "franka_pick_place_pixels"
    return cfg


register_mjlab_task(
    task_id="Mjlab-PickPlace-Franka-State-v0",
    env_cfg=franka_pick_place_env_cfg(),
    play_env_cfg=franka_pick_place_env_cfg(play=True),
    rl_cfg=_state_rl_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-PickPlace-Franka-Pixels-v0",
    env_cfg=franka_pick_place_vision_env_cfg(),
    play_env_cfg=franka_pick_place_vision_env_cfg(play=True),
    rl_cfg=_pixels_rl_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)
