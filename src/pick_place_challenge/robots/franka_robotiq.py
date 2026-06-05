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

# Hand-less Panda — sibling of panda.xml in the menagerie cache. Exposes an
# empty ``attachment`` body on link7 with an ``attachment_site`` to graft onto.
PANDA_NOHAND_XML: Path = Path(_panda_desc.MJCF_PATH).parent / "panda_nohand.xml"
ROBOTIQ_XML: Path = Path(_robotiq_desc.MJCF_PATH)

# 45° about +Z: aligns the 2F-85 jaws to the flange the way our cell mounts them.
# This is a quat *replacement* of the panda_nohand attachment default, not a compose.
ATTACHMENT_QUAT_WXYZ: tuple[float, float, float, float] = (0.9238795, 0.0, 0.0, 0.3826834)


def build_franka_robotiq_spec() -> mujoco.MjSpec:
  """Return an ``MjSpec`` for a Franka arm with a Robotiq 2F-85 grafted on."""
  arm = mujoco.MjSpec.from_file(str(PANDA_NOHAND_XML))
  arm.body("attachment").quat = list(ATTACHMENT_QUAT_WXYZ)
  gripper = mujoco.MjSpec.from_file(str(ROBOTIQ_XML))
  arm.attach(gripper, site="attachment_site", prefix="")
  return arm


if __name__ == "__main__":
  import mujoco.viewer

  mujoco.viewer.launch(build_franka_robotiq_spec().compile())
