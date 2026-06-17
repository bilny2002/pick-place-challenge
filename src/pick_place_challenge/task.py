"""The "place the ball in the bowl" task: MDP, env config, and registration.

This is the one place the environment and its (single) MDP are wired together.
Importing this module registers two variants that differ only in observation:

    Mjlab-PlaceBall-Franka-State-v0   — low-dim state (joint state, EE→ball,
                                        ball→bowl, bowl position).
    Mjlab-PlaceBall-Franka-Pixels-v0  — wrist + scene RGB; the ball is not given
                                        as state (you must see it).

Built on mjlab's ``lift_cube`` manipulation skeleton (actions / sim / viewer),
with the lift command + rewards replaced by the reach-and-place MDP below. The
reward is intentionally simple — rip it out and write your own.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonLengthActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manip_mdp
from mjlab.tasks.manipulation.config.yam.rl_cfg import (
    yam_lift_cube_ppo_runner_cfg,
    yam_lift_cube_vision_ppo_runner_cfg,
)
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply, quat_inv
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pick_place_challenge import scene
from pick_place_challenge.runner import BankshotCheckpointRunner

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
FRONT_WALL_CENTER = (2.9, 0.0, 0.0)
HER_DISTANCE_THRESHOLD = 0.10
HER_SUCCESS_REWARD = 0.0
HER_FAILURE_REWARD = -1.0
_BALL_GEOMS = {"ball_collision", "ball_visual"}
_FRONT_WALL_GEOMS = {"f_bounce_wall_geom"}
_SIDE_WALL_GEOMS = {
    "l_bounce_wall_geom",
    "r_bounce_wall_geom",
    "f_bounce_wall_geom",
    "b_bounce_wall_geom",
}
_GROUND_GEOMS = {"d_bounce_wall_geom"}
_TABLE_GEOMS = {"table_top"}


# ============================================================================
# MDP — the single task MDP (goal = the bowl's position; no command term)
# ============================================================================


def _target_pos_w(env: ManagerBasedRlEnv, target_name: str, z_offset: float):
    """World position of the bowl opening (its origin lifted by ``z_offset``)."""
    pos = env.scene[target_name].data.root_link_pos_w.clone()
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
    """reaching(ee→ball) * (1 + placing(ball→bowl)) — gates placing on reaching."""
    robot: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]
    ee = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    obj_pos = obj.data.root_link_pos_w
    reaching = torch.exp(-torch.sum((ee - obj_pos) ** 2, dim=-1) / reaching_std**2)
    place_err = torch.sum(
        (_target_pos_w(env, target_name, z_offset) - obj_pos) ** 2, dim=-1
    )
    placing = torch.exp(-place_err / placing_std**2)
    return reaching * (1.0 + placing)


def ball_movement_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    max_speed: float = 6.0,
) -> torch.Tensor:
    """Small bounded PPO reward for making the ball move."""
    obj: Entity = env.scene[object_name]
    speed = torch.linalg.norm(obj.data.root_link_lin_vel_w, dim=-1)
    return torch.clamp(speed / max_speed, 0.0, 1.0)


def early_joint_velocity_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    duration_s: float = 1.0,
    max_speed: float = 6.0,
) -> torch.Tensor:
    """Reward fast arm motion during the opening throw window."""
    robot: Entity = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    speed = torch.linalg.norm(joint_vel, dim=-1)
    active = (env.episode_length_buf.float() * env.step_dt) <= duration_s
    return torch.clamp(speed / max_speed, 0.0, 1.0) * active.float()


def gripper_open_action(env: ManagerBasedRlEnv, open_action_threshold: float = -0.5):
    """True when the current gripper command is opening the gripper."""
    actions = mdp.last_action(env)
    return actions[:, -1] <= open_action_threshold


def ball_released(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg,
    height_threshold: float = 0.02,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Detect when the ball has separated from the gripper palm.

    The user-level condition is "ball centroid height != palm height and ball
    velocity != palm velocity"; thresholds make that robust to sim noise.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[object_name]

    palm_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    palm_vel = robot.data.site_lin_vel_w[:, asset_cfg.site_ids].squeeze(1)
    ball_pos = ball.data.root_link_pos_w
    ball_vel = ball.data.root_link_lin_vel_w

    height_separated = torch.abs(ball_pos[:, 2] - palm_pos[:, 2]) > height_threshold
    velocity_separated = (
        torch.linalg.norm(ball_vel - palm_vel, dim=-1) > velocity_threshold
    )
    return height_separated & velocity_separated


def placed_in_bowl(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_name: str,
    radius: float = 0.06,
    max_height: float = 0.07,
) -> torch.Tensor:
    """True when the ball is within the bowl's rim radius and below its rim."""
    obj_pos = env.scene[object_name].data.root_link_pos_w
    bowl_pos = env.scene[target_name].data.root_link_pos_w
    horizontal = torch.linalg.norm(obj_pos[:, :2] - bowl_pos[:, :2], dim=-1)
    return (horizontal < radius) & (obj_pos[:, 2] < bowl_pos[:, 2] + max_height)


# ============================================================================
# Ball gripped
# ============================================================================


def reset_ball_to_gripper(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    object_name: str = "ball",
    robot_name: str = "robot",
    asset_cfg: SceneEntityCfg = _ROBOT,
    z_offset: float = 0.0,
) -> None:
    env_ids = mdp.resolve_env_ids(env, env_ids)

    robot: Entity = env.scene[robot_name]
    ball: Entity = env.scene[object_name]

    # Make sure the pinch site reflects the just-reset robot joint state.
    env.sim.forward()

    ball_pose = ball.data.default_root_state[env_ids, :7].clone()
    pinch_pos = robot.data.site_pos_w[env_ids, asset_cfg.site_ids].squeeze(1)

    ball_pose[:, :3] = pinch_pos
    ball_pose[:, 2] += z_offset

    ball.write_root_link_pose_to_sim(ball_pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
    )


# ============================================================================
# HER observation terms
# ============================================================================


def full_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    return torch.cat(
        [
            robot.data.joint_pos,
            robot.data.joint_vel,
            ball.data.root_link_pos_w,
            desired_goal(env),
            mdp.last_action(env),
        ],
        dim=-1,
    )


def achieved_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
    # The achieved goal should be the wall contact point if contact happened,
    # otherwise use current ball position as fallback.
    return env.scene["ball"].data.root_link_pos_w


def desired_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
    # Center point of front wall.
    # Your front wall is x = +2.9, y = 0, z around ball/contact height.
    return torch.tensor(FRONT_WALL_CENTER, device=env.device).repeat(env.num_envs, 1)


def bankshot_sparse_reward_np(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    info: dict[str, Any],
    distance_threshold: float = HER_DISTANCE_THRESHOLD,
    success_reward: float = HER_SUCCESS_REWARD,
    failure_reward: float = HER_FAILURE_REWARD,
) -> float | np.ndarray:
    """HER-compatible sparse reward for wall-bankshot goals.

    ``achieved_goal`` should be the ball's wall contact point when available.
    For normal training ``desired_goal`` is the front-wall center; HER can pass a
    substituted wall contact point. Goal-independent contact facts come from
    ``info``.
    """
    achieved = np.asarray(achieved_goal, dtype=np.float32)
    desired = np.asarray(desired_goal, dtype=np.float32)
    single = achieved.ndim == 1

    distance = np.linalg.norm(achieved - desired, axis=-1)
    hit_wall = np.asarray(info.get("hit_wall", False), dtype=bool)
    hit_front_wall = np.asarray(info.get("hit_front_wall", False), dtype=bool)
    hit_table_first = np.asarray(info.get("hit_table_first", False), dtype=bool)

    success = (
        hit_wall & ~hit_table_first & (hit_front_wall | (distance < distance_threshold))
    )
    reward = np.where(success, success_reward, failure_reward)
    return reward.item() if single else reward


def _tensor_like_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _iter_named_contacts(env: ManagerBasedRlEnv):
    contact = getattr(getattr(env.sim, "data", None), "contact", None)
    contact = getattr(contact, "struct", None)
    if contact is None or not hasattr(contact, "geom"):
        return []

    try:
        geoms = _tensor_like_to_numpy(contact.geom)
        if hasattr(contact, "worldid"):
            world_ids = _tensor_like_to_numpy(contact.worldid)
        else:
            world_ids = np.zeros(np.reshape(geoms, (-1, 2)).shape[0], dtype=int)
    except Exception:
        return []

    if geoms.size == 0:
        return []

    names = {
        idx: env.sim.mj_model.geom(idx).name for idx in range(env.sim.mj_model.ngeom)
    }
    events = []
    geom_pairs = np.reshape(geoms, (-1, 2))
    world_ids = np.reshape(world_ids, (-1,))
    for idx, pair in enumerate(geom_pairs):
        geom_a = names.get(int(pair[0]))
        geom_b = names.get(int(pair[1]))
        if geom_a is None or geom_b is None:
            continue
        events.append(
            (int(world_ids[idx]), geom_a.split("/")[-1], geom_b.split("/")[-1])
        )
    return events


class FrontWallHitMetric:
    """Tracks whether each env has hit the front wall during the episode."""

    def __init__(self) -> None:
        self._hit_front_wall: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._hit_front_wall is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._hit_front_wall[env_ids] = False

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if self._hit_front_wall is None or len(self._hit_front_wall) != env.num_envs:
            self._hit_front_wall = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

        for env_id, geom_a, geom_b in _iter_named_contacts(env):
            if env_id < 0 or env_id >= env.num_envs:
                continue
            names = {geom_a, geom_b}
            if (names & _BALL_GEOMS) and (names & _FRONT_WALL_GEOMS):
                self._hit_front_wall[env_id] = True
        return self._hit_front_wall.float()


class BallReleasedMetric:
    """Tracks whether each env has released the ball during the episode."""

    def __init__(self) -> None:
        self._released: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._released is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._released[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        object_name: str,
        asset_cfg: SceneEntityCfg,
        height_threshold: float = 0.02,
        velocity_threshold: float = 0.05,
    ) -> torch.Tensor:
        if self._released is None or len(self._released) != env.num_envs:
            self._released = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

        self._released |= ball_released(
            env,
            object_name=object_name,
            asset_cfg=asset_cfg,
            height_threshold=height_threshold,
            velocity_threshold=velocity_threshold,
        )
        return self._released.float()


class EarlyGripperOpenReward:
    """One-time reward for opening the gripper early."""

    def __init__(self) -> None:
        self._rewarded: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._rewarded is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._rewarded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        deadline_s: float = 1.0,
        open_action_threshold: float = -0.5,
    ) -> torch.Tensor:
        if self._rewarded is None or len(self._rewarded) != env.num_envs:
            self._rewarded = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

        elapsed = env.episode_length_buf.float() * env.step_dt
        should_reward = (
            (elapsed <= deadline_s)
            & gripper_open_action(env, open_action_threshold)
            & ~self._rewarded
        )
        self._rewarded |= should_reward
        return should_reward.float()


class GripperNotOpenedPenalty:
    """One-time penalty when the gripper has not opened by a deadline."""

    def __init__(self) -> None:
        self._opened: torch.Tensor | None = None
        self._penalized: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._opened is None or self._penalized is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._opened[env_ids] = False
        self._penalized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        deadline_s: float = 2.0,
        open_action_threshold: float = -0.5,
    ) -> torch.Tensor:
        if (
            self._opened is None
            or self._penalized is None
            or len(self._opened) != env.num_envs
        ):
            self._opened = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
            self._penalized = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

        self._opened |= gripper_open_action(env, open_action_threshold)
        elapsed = env.episode_length_buf.float() * env.step_dt
        should_penalize = (elapsed >= deadline_s) & ~self._opened & ~self._penalized
        self._penalized |= should_penalize
        return should_penalize.float()


class BallContactSequenceReward:
    """One-time reward for episode contact milestones."""

    def __init__(self) -> None:
        self._hit_table: torch.Tensor | None = None
        self._hit_ground: torch.Tensor | None = None
        self._rewarded: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._hit_table is None or self._hit_ground is None or self._rewarded is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._hit_table[env_ids] = False
        self._hit_ground[env_ids] = False
        self._rewarded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        mode: str,
    ) -> torch.Tensor:
        if (
            self._hit_table is None
            or self._hit_ground is None
            or self._rewarded is None
            or len(self._hit_table) != env.num_envs
        ):
            self._hit_table = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
            self._hit_ground = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
            self._rewarded = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

        reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

        for env_id, geom_a, geom_b in _iter_named_contacts(env):
            if env_id < 0 or env_id >= env.num_envs:
                continue

            names = {geom_a, geom_b}
            if not (names & _BALL_GEOMS):
                continue

            hit_table_now = bool(names & _TABLE_GEOMS)
            hit_ground_now = bool(names & _GROUND_GEOMS)
            hit_wall_now = bool(names & _SIDE_WALL_GEOMS)

            hit_bad_surface_first = bool(
                self._hit_table[env_id] or self._hit_ground[env_id]
            )
            if mode == "wall_without_table_or_ground_first":
                should_reward = (
                    hit_wall_now
                    and not hit_bad_surface_first
                    and not hit_table_now
                    and not hit_ground_now
                )
            elif mode == "table":
                should_reward = hit_table_now
            elif mode == "ground":
                should_reward = hit_ground_now
            else:
                raise ValueError(f"Unknown ball contact reward mode: {mode}")

            if should_reward and not bool(self._rewarded[env_id]):
                reward[env_id] = 1.0
                self._rewarded[env_id] = True

            if hit_table_now:
                self._hit_table[env_id] = True
            if hit_ground_now:
                self._hit_ground[env_id] = True

        return reward


class NoGroundContactAfterTimeout:
    """Terminates envs whose ball did not touch the bounce floor in time."""

    def __init__(self) -> None:
        self._hit_ground: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._hit_ground is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._hit_ground[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        timeout_s: float = 5.0,
    ) -> torch.Tensor:
        if self._hit_ground is None or len(self._hit_ground) != env.num_envs:
            self._hit_ground = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

        for env_id, geom_a, geom_b in _iter_named_contacts(env):
            if env_id < 0 or env_id >= env.num_envs:
                continue
            names = {geom_a, geom_b}
            if (names & _BALL_GEOMS) and (names & _GROUND_GEOMS):
                self._hit_ground[env_id] = True

        elapsed = env.episode_length_buf.float() * env.step_dt
        return (elapsed >= timeout_s) & ~self._hit_ground


class BallWallContactTermination:
    """Terminates envs once the ball touches any side bounce wall."""

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        for env_id, geom_a, geom_b in _iter_named_contacts(env):
            if env_id < 0 or env_id >= env.num_envs:
                continue
            names = {geom_a, geom_b}
            if (names & _BALL_GEOMS) and (names & _SIDE_WALL_GEOMS):
                terminated[env_id] = True

        return terminated


# ============================================================================
# Env config (state + pixels)
# ============================================================================

_GRIPPER_BODY = "base"  # 2F-85 base body (parent for the wrist camera)
_BOWL_POS = (0.40, 0.16, 0.0)  # fixed bowl location on the table
_BALL_POS = (0.30, -0.10, 0.06)  # nominal ball spawn (randomized at reset)


def _grasp_site() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=(scene.GRASP_SITE,))


def state_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """State-based place-ball-in-bowl."""
    cfg = make_lift_cube_env_cfg()

    cfg.scene.entities = {
        "robot": scene.franka_robotiq_cfg(),
        "ball": scene.EntityCfg(
            spec_fn=scene.ball_spec,
            init_state=scene.EntityCfg.InitialStateCfg(pos=_BALL_POS),
        ),
        "bowl": scene.EntityCfg(
            spec_fn=scene.bowl_spec,
            init_state=scene.EntityCfg.InitialStateCfg(pos=_BOWL_POS),
        ),
    }
    cfg.scene.terrain = None
    cfg.scene.spec_fn = scene.add_studio
    cfg.scene.sensors = ()  # drop the base task's YAM-specific contact sensor

    # Actions: 7 arm joints (position) + gripper tendon (0=open..255=closed).
    arm = cfg.actions["joint_pos"]
    assert isinstance(arm, JointPositionActionCfg)
    arm.actuator_names = (".*",)
    arm.scale = scene.ARM_ACTION_SCALE
    cfg.actions["gripper"] = TendonLengthActionCfg(
        entity_name="robot",
        actuator_names=(scene.GRIPPER_TENDON,),
        scale=scene.GRIPPER_CTRL_MAX / 2.0,
        offset=scene.GRIPPER_CTRL_MAX / 2.0,
    )

    actor = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "ee_to_ball": ObservationTermCfg(
            func=manip_mdp.ee_to_object_distance,
            params={"object_name": "ball", "asset_cfg": _grasp_site()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ball_to_bowl": ObservationTermCfg(
            func=object_to_target_distance,
            params={"object_name": "ball", "target_name": "bowl"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "bowl_position": ObservationTermCfg(
            func=target_position, params={"target_name": "bowl"}
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(actor, enable_corruption=True),
        "critic": ObservationGroupCfg(dict(actor), enable_corruption=False),
        # HER/GoalEnv observations are intentionally disabled for PPO-only
        # training. Keep the helper functions above for later HER experiments.
        # "goal": ObservationGroupCfg(
        #     {
        #         "observation": ObservationTermCfg(func=full_observation),
        #         "achieved_goal": ObservationTermCfg(func=achieved_goal),
        #         "desired_goal": ObservationTermCfg(func=desired_goal),
        #     },
        #     concatenate_terms=False,
        # ),
    }

    cfg.commands = {}
    cfg.rewards = {
        # "reach_place": RewardTermCfg(
        #     func=reach_and_place_reward,
        #     weight=1.0,
        #     params={
        #         "object_name": "ball",
        #         "target_name": "bowl",
        #         "reaching_std": 0.1,
        #         "placing_std": 0.1,
        #         "asset_cfg": _grasp_site(),
        #     },
        # ),
        "ball_movement": RewardTermCfg(
            func=ball_movement_reward,
            weight=3.0,
            params={"object_name": "ball", "max_speed": 6.0},
        ),
        "early_gripper_open": RewardTermCfg(
            func=EarlyGripperOpenReward(),
            weight=0.2,
            params={"deadline_s": 2.0, "open_action_threshold": -0.5},
        ),
        "early_joint_velocity": RewardTermCfg(
            func=early_joint_velocity_reward,
            weight=0.08,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("joint4")
                ),
                "duration_s": 2.0,
                "max_speed": 6.0,
            },
        ),
        "ball_wall_without_table_or_ground_first": RewardTermCfg(
            func=BallContactSequenceReward(),
            weight=6.0,
            params={"mode": "wall_without_table_or_ground_first"},
        ),
        "gripper_not_opened_by_2s": RewardTermCfg(
            func=GripperNotOpenedPenalty(),
            weight=-0.6,
            params={"deadline_s": 2.0, "open_action_threshold": -0.5},
        ),
        "ball_hit_table_penalty": RewardTermCfg(
            func=BallContactSequenceReward(),
            weight=-1.0,
            params={"mode": "table"},
        ),
        "ball_hit_ground_penalty": RewardTermCfg(
            func=BallContactSequenceReward(),
            weight=-1.0,
            params={"mode": "ground"},
        ),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "ball_hit_wall": TerminationTermCfg(
            func=BallWallContactTermination(),
        ),
        "no_ground_contact_after_5s": TerminationTermCfg(
            func=NoGroundContactAfterTimeout(),
            params={"timeout_s": 5.0},
        ),
    }
    cfg.curriculum = {}
    cfg.metrics = {
        "front_wall_hit_rate": MetricsTermCfg(
            func=FrontWallHitMetric(),
            reduce="last",
        ),
        "ball_released_rate": MetricsTermCfg(
            func=BallReleasedMetric(),
            params={
                "object_name": "ball",
                "asset_cfg": _grasp_site(),
                "height_threshold": 0.02,
                "velocity_threshold": 0.05,
            },
            reduce="last",
        ),
    }
    cfg.events = {
        # "reset_ball": EventTermCfg(
        #     func=mdp.reset_root_state_uniform,
        #     mode="reset",
        #     params={
        #         "pose_range": {"x": (-0.05, 0.08), "y": (-0.08, 0.08)},
        #         "velocity_range": {},
        #         "asset_cfg": SceneEntityCfg("ball"),
        #     },
        # ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "reset_ball_to_gripper": EventTermCfg(
            func=reset_ball_to_gripper,
            mode="reset",
            params={
                "object_name": "ball",
                "robot_name": "robot",
                "asset_cfg": _grasp_site(),
                "z_offset": 0.0,
            },
        ),
    }
    cfg.viewer.body_name = "link0"
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 400)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg


def _look_at_quat(eye, target):
    """w,x,y,z quat orienting a MuJoCo camera (looks −z, up +y) at ``target``."""
    fx, fy, fz = (t - e for e, t in zip(eye, target, strict=True))
    fn = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / fn, fy / fn, fz / fn
    rx, ry, rz = fy, -fx, 0.0
    rn = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    rx, ry, rz = rx / rn, ry / rn, rz / rn
    ux, uy, uz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx
    m = [[rx, ux, -fx], [ry, uy, -fy], [rz, uz, -fz]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x, y, z = (
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x, y, z = 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x, y, z = (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x, y, z = (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    return (w, x, y, z)


def _camera(name, eye, target, parent_body=None):
    return CameraSensorCfg(
        name=name,
        camera_name=None,
        parent_body=parent_body,
        pos=eye,
        quat=_look_at_quat(eye, target),
        width=84,
        height=84,
        data_types=("rgb",),
        use_shadows=False,
        use_textures=True,
    )


def pixels_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Image-based variant: wrist + scene RGB; ball not given as state."""
    cfg = state_env_cfg(play=play)
    cfg.scene.sensors = (
        _camera("scene_cam", (1.05, 0.55, 0.6), (0.33, 0.02, 0.05)),
        _camera(
            "wrist_cam", (0.14, 0.0, 0.04), (0.0, 0.0, 0.13), f"robot/{_GRIPPER_BODY}"
        ),
    )
    cfg.observations["camera"] = ObservationGroupCfg(
        terms={
            "scene_rgb": ObservationTermCfg(
                func=manip_mdp.camera_rgb, params={"sensor_name": "scene_cam"}
            ),
            "wrist_rgb": ObservationTermCfg(
                func=manip_mdp.camera_rgb, params={"sensor_name": "wrist_cam"}
            ),
        },
        enable_corruption=False,
        concatenate_terms=False,
    )
    # Drop privileged ball state; keep the (fixed) bowl position.
    for group in ("actor", "critic"):
        cfg.observations[group].terms.pop("ee_to_ball", None)
        cfg.observations[group].terms.pop("ball_to_bowl", None)
    return cfg


# ============================================================================
# Registration
# ============================================================================


def _rl_cfg(vision: bool, name: str):
    cfg = (
        yam_lift_cube_vision_ppo_runner_cfg()
        if vision
        else yam_lift_cube_ppo_runner_cfg()
    )
    cfg.experiment_name = name
    return cfg


register_mjlab_task(
    task_id="Mjlab-PlaceBall-Franka-State-v0",
    env_cfg=state_env_cfg(),
    play_env_cfg=state_env_cfg(play=True),
    rl_cfg=_rl_cfg(False, "franka_place_ball_state"),
    runner_cls=BankshotCheckpointRunner,
)
register_mjlab_task(
    task_id="Mjlab-PlaceBall-Franka-Pixels-v0",
    env_cfg=pixels_env_cfg(),
    play_env_cfg=pixels_env_cfg(play=True),
    rl_cfg=_rl_cfg(True, "franka_place_ball_pixels"),
    runner_cls=BankshotCheckpointRunner,
)
