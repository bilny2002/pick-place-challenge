"""Franka + Robotiq pick-and-place env configs (state and pixels variants).

Built on mjlab's `lift_cube` manipulation task: we reuse its reward / command /
observation MDP terms and swap the YAM arm for a Franka + Robotiq 2F-85. Two
variants are exposed, differing only in the observation:

* **state**  — privileged low-dim state (joint state, EE↔cube, cube↔goal).
* **pixels** — wrist + scene RGB cameras, with privileged object/goal state removed.

There is intentionally no expert policy. The default reward is a simple staged
reach/lift shaping that candidates are free to replace.
"""

from __future__ import annotations

import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonLengthActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.config.yam.env_cfgs import get_cube_spec
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg

from pick_place_challenge.robots.franka_robotiq import (
    ARM_ACTION_SCALE,
    GRASP_SITE,
    GRIPPER_CTRL_MAX,
    GRIPPER_TENDON,
    get_franka_robotiq_robot_cfg,
)

# The 2F-85 base body (after attach) — parent for the wrist camera.
_GRIPPER_BODY = "base"


def franka_pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """State-based Franka pick-and-place."""
    cfg = make_lift_cube_env_cfg()

    # Robot + a single free-floating cube to pick.
    cfg.scene.entities = {
        "robot": get_franka_robotiq_robot_cfg(),
        "cube": EntityCfg(spec_fn=get_cube_spec),
    }

    # Actions: 7 arm joints (position) + gripper tendon. The tendon "length"
    # target is written straight to ctrl (0=open .. 255=closed); map a [-1, 1]
    # action onto that range.
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

    # Point the EE-relative terms at the 2F-85 jaw-center ("pinch") site.
    cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
        GRASP_SITE,
    )
    cfg.rewards["lift"].params["asset_cfg"].site_names = (GRASP_SITE,)

    # Drop YAM-specific fingertip-friction randomization and the YAM EE-ground
    # contact sensor/termination (link names differ). Keep it simple for v1.
    for key in (
        "fingertip_friction_slide",
        "fingertip_friction_spin",
        "fingertip_friction_roll",
    ):
        cfg.events.pop(key, None)
    cfg.scene.sensors = ()
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.curriculum = {}

    # Track the arm base in the viewer; give the solver headroom for the extra
    # gripper contacts.
    cfg.viewer.body_name = "link0"
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 150)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        assert cfg.commands is not None
        cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

    return cfg


def _look_at_quat(
    eye: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """w,x,y,z quat orienting a MuJoCo camera (looks down -z, up +y) at target."""
    fx, fy, fz = (t - e for e, t in zip(eye, target, strict=True))
    fn = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / fn, fy / fn, fz / fn  # forward (camera -z)
    # Right = forward × world_up, Up = right × forward.
    ux, uy, uz = 0.0, 0.0, 1.0
    rx, ry, rz = fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux
    rn = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    rx, ry, rz = rx / rn, ry / rn, rz / rn
    ux, uy, uz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx
    # Camera axes: x=right, y=up, z=-forward. Build rotation matrix columns.
    m00, m01, m02 = rx, ux, -fx
    m10, m11, m12 = ry, uy, -fy
    m20, m21, m22 = rz, uz, -fz
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (w, x, y, z)


def franka_pick_place_vision_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Image-based Franka pick-and-place: wrist + scene RGB, no privileged state."""
    cfg = franka_pick_place_env_cfg(play=play)

    # A fixed scene camera looking at the workspace, and a wrist camera bolted to
    # the gripper base looking out along the approach axis.
    scene_cam = CameraSensorCfg(
        name="scene_cam",
        pos=(0.9, 0.0, 0.7),
        quat=_look_at_quat((0.9, 0.0, 0.7), (0.35, 0.0, 0.1)),
        width=84,
        height=84,
        data_types=("rgb",),
        use_shadows=False,
        use_textures=True,
    )
    # Side-mounted on the gripper base, looking down the approach axis so both
    # jaws frame the workspace below (verified by rendering — a centered view of
    # the grasp region rather than the inside of the gripper body).
    _wrist_pos = (0.14, 0.0, 0.04)
    wrist_cam = CameraSensorCfg(
        name="wrist_cam",
        parent_body=f"robot/{_GRIPPER_BODY}",
        pos=_wrist_pos,
        quat=_look_at_quat(_wrist_pos, (0.0, 0.0, 0.13)),
        width=84,
        height=84,
        data_types=("rgb",),
        use_shadows=False,
        use_textures=True,
    )
    cfg.scene.sensors = (scene_cam, wrist_cam)

    # Camera observation group.
    cfg.observations["camera"] = ObservationGroupCfg(
        terms={
            "scene_rgb": ObservationTermCfg(
                func=manipulation_mdp.camera_rgb, params={"sensor_name": "scene_cam"}
            ),
            "wrist_rgb": ObservationTermCfg(
                func=manipulation_mdp.camera_rgb, params={"sensor_name": "wrist_cam"}
            ),
        },
        enable_corruption=False,
        concatenate_terms=False,
    )

    # Remove privileged object/goal state from the proprio group; keep only the
    # goal position (the agent must infer the cube from pixels).
    actor = cfg.observations["actor"]
    actor.terms.pop("ee_to_cube", None)
    actor.terms.pop("cube_to_goal", None)
    actor.terms["goal_position"] = ObservationTermCfg(
        func=manipulation_mdp.target_position,
        params={
            "command_name": "lift_height",
            "asset_cfg": SceneEntityCfg("robot", site_names=(GRASP_SITE,)),
        },
    )

    return cfg
