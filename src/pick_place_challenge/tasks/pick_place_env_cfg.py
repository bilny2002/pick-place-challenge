"""Franka + Robotiq "place the ball in the bowl" env configs.

Pick up a real scanned ball off the table and drop it into a scanned bowl. Two
variants differ only in observation:

* **state**  — joint state, EE->ball, ball->bowl, bowl position.
* **pixels** — wrist + scene RGB; the ball is no longer given as state (you must
  see it), only the (fixed) bowl position is provided.

Built on mjlab's lift_cube skeleton (actions / sim / viewer), with the lift
command and rewards replaced by simple reach-and-place terms (see
``pick_place_challenge.mdp``). The reward is intentionally basic — replace it.
"""

from __future__ import annotations

import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonLengthActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manip_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pick_place_challenge import mdp as ppc_mdp
from pick_place_challenge.objects import make_ball_spec_fn, make_bowl_spec_fn
from pick_place_challenge.robots.franka_robotiq import (
    ARM_ACTION_SCALE,
    GRASP_SITE,
    GRIPPER_CTRL_MAX,
    GRIPPER_TENDON,
    get_franka_robotiq_robot_cfg,
)
from pick_place_challenge.world import add_studio

_GRIPPER_BODY = "base"  # 2F-85 base body (parent for the wrist camera)
_BOWL_POS = (0.40, 0.16, 0.0)  # fixed bowl location on the table
_BALL_POS = (0.30, -0.10, 0.06)  # nominal ball spawn (randomized at reset)


def _grasp_site() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=(GRASP_SITE,))


def franka_ball_in_bowl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """State-based place-ball-in-bowl."""
    cfg = make_lift_cube_env_cfg()

    cfg.scene.entities = {
        "robot": get_franka_robotiq_robot_cfg(),
        "ball": EntityCfg(
            spec_fn=make_ball_spec_fn(),
            init_state=EntityCfg.InitialStateCfg(pos=_BALL_POS),
        ),
        "bowl": EntityCfg(
            spec_fn=make_bowl_spec_fn(),
            init_state=EntityCfg.InitialStateCfg(pos=_BOWL_POS),
        ),
    }
    cfg.scene.terrain = None
    cfg.scene.spec_fn = add_studio
    cfg.scene.sensors = ()  # drop the base task's YAM-specific contact sensor

    # Actions: 7 arm joints + gripper tendon (0=open..255=closed).
    arm_action = cfg.actions["joint_pos"]
    assert isinstance(arm_action, JointPositionActionCfg)
    arm_action.actuator_names = (".*",)
    arm_action.scale = ARM_ACTION_SCALE
    cfg.actions["gripper"] = TendonLengthActionCfg(
        entity_name="robot",
        actuator_names=(GRIPPER_TENDON,),
        scale=GRIPPER_CTRL_MAX / 2.0,
        offset=GRIPPER_CTRL_MAX / 2.0,
    )

    # Observations.
    actor_terms = {
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
            func=ppc_mdp.object_to_target_distance,
            params={"object_name": "ball", "target_name": "bowl"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "bowl_position": ObservationTermCfg(
            func=ppc_mdp.target_position, params={"target_name": "bowl"}
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
    }

    cfg.commands = {}

    cfg.rewards = {
        "reach_place": RewardTermCfg(
            func=ppc_mdp.reach_and_place_reward,
            weight=1.0,
            params={
                "object_name": "ball",
                "target_name": "bowl",
                "reaching_std": 0.1,
                "placing_std": 0.1,
                "asset_cfg": _grasp_site(),
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
    }

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "placed": TerminationTermCfg(
            func=ppc_mdp.placed_in_bowl,
            params={"object_name": "ball", "target_name": "bowl"},
        ),
    }
    cfg.curriculum = {}

    cfg.events = {
        "reset_ball": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.05, 0.08), "y": (-0.08, 0.08)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("ball"),
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
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
    """w,x,y,z quat orienting a MuJoCo camera (looks -z, up +y) at target."""
    fx, fy, fz = (t - e for e, t in zip(eye, target, strict=True))
    fn = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / fn, fy / fn, fz / fn
    ux, uy, uz = 0.0, 0.0, 1.0
    rx, ry, rz = fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux
    rn = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    rx, ry, rz = rx / rn, ry / rn, rz / rn
    ux, uy, uz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx
    m00, m01, m02 = rx, ux, -fx
    m10, m11, m12 = ry, uy, -fy
    m20, m21, m22 = rz, uz, -fz
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return (w, x, y, z)


def franka_ball_in_bowl_vision_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Image-based variant: wrist + scene RGB, ball not given as state."""
    cfg = franka_ball_in_bowl_env_cfg(play=play)

    scene_eye = (1.05, 0.55, 0.6)
    scene_cam = CameraSensorCfg(
        name="scene_cam",
        pos=scene_eye,
        quat=_look_at_quat(scene_eye, (0.33, 0.02, 0.05)),
        width=84,
        height=84,
        data_types=("rgb",),
        use_shadows=False,
        use_textures=True,
    )
    wrist_eye = (0.14, 0.0, 0.04)
    wrist_cam = CameraSensorCfg(
        name="wrist_cam",
        parent_body=f"robot/{_GRIPPER_BODY}",
        pos=wrist_eye,
        quat=_look_at_quat(wrist_eye, (0.0, 0.0, 0.13)),
        width=84,
        height=84,
        data_types=("rgb",),
        use_shadows=False,
        use_textures=True,
    )
    cfg.scene.sensors = (scene_cam, wrist_cam)

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
        terms = cfg.observations[group].terms
        terms.pop("ee_to_ball", None)
        terms.pop("ball_to_bowl", None)

    return cfg
