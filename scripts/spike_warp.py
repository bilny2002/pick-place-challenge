"""IND-1281 spike: does Franka + Robotiq 2F-85 simulate in MuJoCo-Warp?

The risk is the Robotiq linkage: two ``<connect>`` loop-closure equalities, a
``<joint>`` polycoef equality coupling the drivers, and a fixed tendon driving a
single ``fingers_actuator``. MuJoCo-Warp does not support every MJCF feature, so
we (1) compile the combined model, (2) push it through mujoco_warp, (3) step it
on the GPU while actuating the gripper, and (4) check the fingers actually close
without blowing up.

Run: `uv run python scripts/spike_warp.py`
"""

import mujoco
import numpy as np

from pick_place_challenge.scene import build_franka_robotiq_spec


def describe(model: mujoco.MjModel) -> None:
    eq_types = [mujoco.mjtEq(model.eq_type[i]).name for i in range(model.neq)]
    print(
        f"  nq={model.nq} nv={model.nv} nu={model.nu} "
        f"nbody={model.nbody} ngeom={model.ngeom}"
    )
    print(f"  neq={model.neq} eq_types={eq_types}")
    print(f"  ntendon={model.ntendon} nwrap={model.nwrap}")
    actuators = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    print(f"  actuators={actuators}")


def main() -> None:
    print("== Building Franka + Robotiq 2F-85 spec ==")
    spec = build_franka_robotiq_spec()
    model = spec.compile()
    print("Compiled to MjModel OK.")
    describe(model)

    # Locate the gripper tendon actuator (ctrlrange 0..255, 255 = closed).
    act_name = "fingers_actuator"
    act_id = mujoco.mj_id2name and mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
    )
    if act_id < 0:
        raise SystemExit(f"actuator {act_name!r} not found — attach failed?")
    lo, hi = model.actuator_ctrlrange[act_id]
    print(f"\nGripper actuator {act_name!r}: ctrlrange=({lo}, {hi})")

    # --- 1) CPU MuJoCo reference (sanity) ---
    print("\n== CPU MuJoCo reference step ==")
    data = mujoco.MjData(model)
    data.ctrl[act_id] = hi  # command "close"
    for _ in range(500):
        mujoco.mj_step(model, data)
    print(
        f"  after 500 steps: qpos finite={np.isfinite(data.qpos).all()} "
        f"max|qvel|={np.abs(data.qvel).max():.3f}"
    )

    # --- 2) MuJoCo-Warp on GPU (the actual question) ---
    print("\n== MuJoCo-Warp GPU step ==")
    import warp as wp
    import mujoco_warp as mjw

    wp.init()
    print(f"  warp device: {wp.get_device()}")

    m = mjw.put_model(model)
    d = mjw.put_data(model, mujoco.MjData(model))
    # Command "close" across the (batched) warp data.
    ctrl = d.ctrl.numpy()
    ctrl[:, act_id] = hi
    d.ctrl = wp.array(ctrl, dtype=wp.float32)

    for _ in range(500):
        mjw.step(m, d)
    wp.synchronize()

    qpos = d.qpos.numpy()
    qvel = d.qvel.numpy()
    finite = np.isfinite(qpos).all() and np.isfinite(qvel).all()
    print(
        f"  after 500 warp steps: qpos finite={finite} "
        f"max|qvel|={np.abs(qvel).max():.3f}"
    )

    # Driver joint should have moved toward closed under the tendon actuator.
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
    adr = model.jnt_qposadr[jid]
    print(f"  right_driver_joint qpos = {qpos[0, adr]:.4f} (0=open, ~0.8=closed)")

    print("\n== VERDICT ==")
    if finite:
        print("  PASS: Franka + Robotiq 2F-85 steps in MuJoCo-Warp on GPU.")
    else:
        print("  FAIL: non-finite state — Warp likely mishandles the linkage.")


if __name__ == "__main__":
    main()
