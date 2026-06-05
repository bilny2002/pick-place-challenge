"""Open an interactive MuJoCo viewer on the Franka + Robotiq scene.

Pure MuJoCo (CPU) — no GPU, no mjlab env, no policy. The fastest way to confirm
your install works and to look at the robot. Drag bodies to perturb them.

    uv run python scripts/view_scene.py
"""

import mujoco
import mujoco.viewer

from pick_place_challenge.robots.franka_robotiq import build_franka_robotiq_spec


def build_scene_spec() -> mujoco.MjSpec:
    """Franka + Robotiq with a checkered floor and a red cube to look at."""
    spec = build_franka_robotiq_spec()

    # Checkered ground plane + light.
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=(0.2, 0.3, 0.4),
        rgb2=(0.1, 0.15, 0.2),
        width=300,
        height=300,
    )
    mat = spec.add_material(name="grid")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texuniform = True
    mat.texrepeat = [4, 4]
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(2.0, 2.0, 0.05),
        material="grid",
    )
    light = spec.worldbody.add_light()
    light.pos = [0.0, 0.0, 2.0]
    light.dir = [0.0, 0.0, -1.0]

    # A red cube sitting in front of the arm.
    cube = spec.worldbody.add_body(name="cube", pos=(0.35, 0.0, 0.02))
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.02, 0.02, 0.02),
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
