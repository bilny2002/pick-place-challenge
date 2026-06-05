"""Franka Panda + Robotiq 2F-85, assembled at runtime from Menagerie MJCFs.

Assets come from ``robot_descriptions`` (which downloads ``mujoco_menagerie``),
so nothing is vendored into this repo. The arm is loaded hand-less
(``panda_nohand.xml``) and the 2F-85 gripper is grafted onto link7's
``attachment`` body — the same recipe used in Index Robotics' internal sim.
"""

from pathlib import Path

import mujoco
import robot_descriptions.panda_mj_description as _panda_desc
import robot_descriptions.robotiq_2f85_mj_description as _robotiq_desc
from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# Hand-less Panda — sibling of panda.xml in the menagerie cache. Exposes an
# empty ``attachment`` body on link7 with an ``attachment_site`` to graft onto.
PANDA_NOHAND_XML: Path = Path(_panda_desc.MJCF_PATH).parent / "panda_nohand.xml"
ROBOTIQ_XML: Path = Path(_robotiq_desc.MJCF_PATH)

# 45° about +Z: aligns the 2F-85 jaws to the flange the way our cell mounts them.
# This is a quat *replacement* of the panda_nohand attachment default, not a compose.
ATTACHMENT_QUAT_WXYZ: tuple[float, float, float, float] = (
    0.9238795,
    0.0,
    0.0,
    0.3826834,
)


def build_franka_robotiq_spec() -> mujoco.MjSpec:
    """Return an ``MjSpec`` for a Franka arm with a Robotiq 2F-85 grafted on."""
    arm = mujoco.MjSpec.from_file(str(PANDA_NOHAND_XML))
    arm.body("attachment").quat = list(ATTACHMENT_QUAT_WXYZ)
    gripper = mujoco.MjSpec.from_file(str(ROBOTIQ_XML))
    arm.attach(gripper, site="attachment_site", prefix="")
    return arm


# ----------------------------------------------------------------------------
# mjlab Entity config
# ----------------------------------------------------------------------------

# The 2F-85 has a "pinch" site at the jaw center — a natural grasp reference.
GRASP_SITE: str = "pinch"

# Arm position-actuator joints and the gripper's tendon (closed loop linkage
# driven by a single `fingers_actuator`, ctrl 0=open .. 255=closed).
ARM_JOINT_EXPR: tuple[str, ...] = (r"joint[1-7]",)
GRIPPER_TENDON: str = "split"
GRIPPER_CTRL_MAX: float = 255.0

# Menagerie's Panda "home" pose (joint4/6/7 bent); gripper joints rest open.
_HOME_ARM = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": -1.57079,
    "joint5": 0.0,
    "joint6": 1.57079,
    "joint7": -0.7853,
}


# Reuse the Franka's tuned XML position actuators and the 2F-85's tendon
# actuator rather than re-deriving gains — XmlActuator wraps them in place.
ARM_ACTUATORS = XmlActuatorCfg(
    target_names_expr=ARM_JOINT_EXPR,
    transmission_type=TransmissionType.JOINT,
    command_field="position",
)
GRIPPER_ACTUATOR = XmlActuatorCfg(
    target_names_expr=(GRIPPER_TENDON,),
    transmission_type=TransmissionType.TENDON,
    command_field="position",
)
ARTICULATION = EntityArticulationInfoCfg(
    actuators=(ARM_ACTUATORS, GRIPPER_ACTUATOR),
    soft_joint_pos_limit_factor=0.9,
)


def get_franka_robotiq_robot_cfg() -> EntityCfg:
    """mjlab Entity config: fixed-base Franka + Robotiq 2F-85 at the origin."""
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={**_HOME_ARM, ".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        spec_fn=build_franka_robotiq_spec,
        articulation=ARTICULATION,
    )


# Per-joint position-action scale for the 7 arm joints (gripper handled by its
# own tendon action term).
ARM_ACTION_SCALE: float = 0.5


if __name__ == "__main__":
    import mujoco.viewer

    mujoco.viewer.launch(build_franka_robotiq_spec().compile())
