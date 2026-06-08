"""Open an interactive MuJoCo viewer on the Franka + Robotiq scene.

Pure MuJoCo (CPU) — no GPU, no mjlab env, no policy. The fastest way to confirm
your install works and to look at the robot. Drag bodies to perturb them.

    uv run python scripts/view_scene.py
"""

import mujoco
import mujoco.viewer

from pick_place_challenge.scene import add_studio, build_franka_robotiq_spec


def build_scene_spec() -> mujoco.MjSpec:
    """Franka + Robotiq in the studio room + table (same world as the env)."""
    spec = build_franka_robotiq_spec()
    add_studio(spec)

    # A red cube on the table as a stand-in pick target (the env uses a scanned
    # object; this keeps the CPU viewer dependency-free and offline).
    cube = spec.worldbody.add_body(name="cube", pos=(0.35, 0.0, 0.03))
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.025, 0.025, 0.025),
        mass=0.05,
        rgba=(0.8, 0.2, 0.2, 1.0),
    )
    return spec


def main() -> None:
    model = build_scene_spec().compile()
    data = mujoco.MjData(model)
    key = model.key("home").id if model.nkey else -1
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    print("Launching MuJoCo viewer (close the window to exit)...")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
