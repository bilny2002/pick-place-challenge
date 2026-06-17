"""Open an interactive MuJoCo viewer on the Franka + Robotiq scene.

Pure MuJoCo (CPU) — no GPU, no mjlab env, no policy. The fastest way to confirm
your install works and to look at the robot. Drag bodies to perturb them.

    uv run python scripts/view_scene.py
"""

import mujoco
import mujoco.viewer

from pick_place_challenge.scene import add_studio, build_franka_robotiq_spec, bowl_spec


def build_scene_spec() -> mujoco.MjSpec:
    """Franka + Robotiq in the studio room + table (same world as the env)."""
    spec = build_franka_robotiq_spec()
    add_studio(spec)

    #################################################################
    # left wall to bounce off
    # position help:
    # x: forward/back on the table
    # y: left/right across the table
    # z: height
    #################################################################
    # l_bounce_wall = spec.worldbody.add_body(name="l_bounce_wall", pos=(0.0, 3.0, 1.10)) #pos=(0.0, 3.0, 1.01)
    # l_bounce_wall.add_geom(
    #     name="l_bounce_wall_geom",
    #     type=mujoco.mjtGeom.mjGEOM_BOX,
    #     size=(2.9, 0.01, 1.50), #(3.0, 0.01, 1.5)
    #     rgba=(0.15, 0.35, 0.95, 1.0),
    #     friction=(0.8, 0.02, 0.001),
    #     solref=(0.01, 0.4),
    # )
    # ################################################################
    # # right wall
    # ################################################################

    # r_bounce_wall = spec.worldbody.add_body(name="r_bounce_wall", pos=(0.0, -3.0, 1.10)) #(0.0, -3.0, 1.01)
    # r_bounce_wall.add_geom(
    #     name="r_bounce_wall_geom",
    #     type=mujoco.mjtGeom.mjGEOM_BOX,
    #     size=(2.9, 0.01, 1.50), #(3.0, 0.01, 1.5)
    #     rgba=(0.15, 0.35, 0.95, 1.0),
    #     friction=(0.8, 0.02, 0.001),
    #     solref=(0.01, 0.4),
    # )

    # ################################################################
    # # front wall
    # ################################################################

    # f_bounce_wall = spec.worldbody.add_body(name="f_bounce_wall", pos=(2.9, 0.0, 1.10)) #(3.0, 0.0, 1.01)
    # f_bounce_wall.add_geom(
    #     name="f_bounce_wall_geom",
    #     type=mujoco.mjtGeom.mjGEOM_BOX,
    #     size=(0.01, 3.0, 1.50), #(0.01, 3.01, 1.5)
    #     rgba=(0.15, 0.35, 0.95, 1.0),
    #     friction=(0.8, 0.02, 0.001),
    #     solref=(0.01, 0.4),
    # )

    # ################################################################
    # # back wall
    # ################################################################

    # b_bounce_wall = spec.worldbody.add_body(name="b_bounce_wall", pos=(-2.9, 0.0, 1.10)) #(-3.0, 0.0, 1.01)
    # b_bounce_wall.add_geom(
    #     name="b_bounce_wall_geom",
    #     type=mujoco.mjtGeom.mjGEOM_BOX,
    #     size=(0.01, 3.0, 1.50), #(0.01, 3.01, 1.5)
    #     rgba=(0.15, 0.35, 0.95, 1.0),
    #     friction=(0.8, 0.02, 0.001),
    #     solref=(0.01, 0.4),
    # )

    # ################################################################
    # # roof/ up
    # ################################################################

    # u_bounce_wall = spec.worldbody.add_body(name="u_bounce_wall", pos=(0, 0.0, 2.61)) #(0, 0.0, 3)
    # u_bounce_wall.add_geom(
    #     name="u_bounce_wall_geom",
    #     type=mujoco.mjtGeom.mjGEOM_BOX,
    #     size=(2.9, 3.0, 0.01), # (3.0, 3.0, 0.01)
    #     rgba=(0.15, 0.35, 0.95, 1.0),
    #     friction=(0.8, 0.02, 0.001),
    #     solref=(0.01, 0.4),
    # )

    # ################################################################
    # # floor/ down
    # ################################################################

    # d_bounce_wall = spec.worldbody.add_body(name="d_bounce_wall", pos=(0.0, 0.0, -0.41)) #(0.0, 0.0, -0.4)
    # d_bounce_wall.add_geom(
    #     name="d_bounce_wall_geom",
    #     type=mujoco.mjtGeom.mjGEOM_BOX,
    #     size=(2.9, 3.0, 0.01), #(3.0, 3.0, 0.01)
    #     rgba=(0.15, 0.35, 0.95, 1.0),
    #     friction=(0.8, 0.02, 0.001),
    #     solref=(0.01, 0.4),
    # )

    #################################################################
    # bowl to land in
    #################################################################
    bowl = bowl_spec()
    bowl_frame = spec.worldbody.add_frame(name="bowl_frame", pos=(0.40, 0.16, 0.0))
    bowl_frame.attach_body(bowl.worldbody.first_body(), prefix="bowl_")

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
