# pick-place-challenge

A small, self-contained robotics sandbox: a **Franka Panda + Robotiq 2F-85**
doing **pick-and-place** of a real **scanned object** on a table, in
[mjlab](https://github.com/mujocolab/mjlab) (GPU-accelerated MuJoCo,
Isaac-Lab-style manager API).

**This is not a graded take-home.** There's no target metric and no expert policy
to beat. It's a thing to play with. Pick a direction that interests you, go down
the rabbit hole, and show us something cool. If it's interesting, we'll want to
talk. Email us anytime to ask what we'd find impressive — genuinely, ask.

---

## Quickstart (~60 seconds)

```bash
uv sync                                   # install everything (locked)
uv run python scripts/view_scene.py       # look at the robot + table (CPU, no GPU needed)
```

The pickable objects (a handful of [Google Scanned Objects](https://github.com/kevinzakka/mujoco_scanned_objects))
are fetched on first use into `~/.cache`. To pre-fetch them: `uv run pick-place-fetch-assets`.

Drive it with a random policy and watch it in an interactive viewer:

```bash
uv run play Mjlab-PickPlace-Franka-State-v0 --agent random --viewer native
# headless / over SSH? use:  --viewer viser   (opens a browser viewer)
```

List the tasks this repo registers:

```bash
uv run pick-place-envs
```

A minimal, readable env-loop example to copy from:

```bash
uv run python scripts/random_rollout.py --task state --steps 200
```

---

## The two environments

Same task, same robot, same reward — they differ **only in the observation**:

| Task ID | Observation |
|---|---|
| `Mjlab-PickPlace-Franka-State-v0` | Low-dim **state**: joint pos/vel, end-effector→cube vector, cube→goal vector, last action. |
| `Mjlab-PickPlace-Franka-Pixels-v0` | **Pixels**: an `84×84` RGB **wrist** camera + `84×84` RGB **scene** camera, plus proprioception and the goal position. No privileged object/goal state — you have to see the cube. |

**Action** (both): 8-D — 7 arm joint-position targets + 1 gripper command
(`-1` open … `+1` closed; the Robotiq's tendon does the rest).

**Reward** (both): a simple staged reach/lift shaping. It's deliberately basic —
**rip it out and write your own** if you're doing RL.

Switch tasks by swapping the ID, e.g.:

```bash
uv run play Mjlab-PickPlace-Franka-Pixels-v0 --agent random --viewer native
```

---

## Make something cool

Open-ended. Some rabbit holes, roughly easy → hard — **you don't have to pick from this list**:

- **Scripted / classical control.** Write an IK or operational-space controller
  and a heuristic grasp that actually picks the cube up. (Great on CPU.)
- **Behavior cloning.** Collect demos (teleop or your scripted policy), train a
  policy, evaluate it. Bonus: from pixels.
- **Reinforcement learning.** `uv run train Mjlab-PickPlace-Franka-State-v0 --env.scene.num-envs 2048`
  (mjlab ships RSL-RL + PPO). Then `uv run play … --agent trained …`. Pixels is the hard mode.
- **Reward / task design.** The default reward is a placeholder. Design one that
  produces a crisp pick-and-*place* into a target region.
- **Perception.** Estimate the cube pose from the wrist camera; close the loop on it.
- **Performance.** Profile the env, push `--env.scene.num-envs` as high as it goes,
  make rendering or stepping faster.
- **Make it harder / more realistic.** Add a table, clutter, multiple objects,
  domain randomization, distractors, a different gripper.

Surprise us. The "what" matters less than that it's thoughtful and works.

---

## Hardware

- **CPU / macOS is fine** for viewing the scene, the random/scripted paths, BC
  inference, and small-scale env stepping — MuJoCo-Warp has a CPU backend.
- **An NVIDIA GPU** makes large-scale **RL training** practical (thousands of
  parallel envs). The `--agent trained` and big `--env.scene.num-envs` paths
  effectively want one. You can do a lot without it.

Scripts and tasks default to `cuda`; pass `--device cpu` to `random_rollout.py`,
or `--device cpu` style flags where mjlab exposes them, to stay on CPU.

---

## How it's built (for the curious)

- `src/pick_place_challenge/robots/franka_robotiq.py` — assembles the arm +
  gripper at runtime from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  (pulled by `robot_descriptions`; nothing vendored). The 2F-85's full mimic
  linkage is kept — it simulates correctly in MuJoCo-Warp.
- `src/pick_place_challenge/objects.py` + `assets.py` — fetch and load a curated
  set of scanned objects (free-jointed, auto-rescaled so the gripper can close on
  any of them). Swap the pick target via `DEFAULT_OBJECT` in `assets.py`.
- `src/pick_place_challenge/world.py` — the studio room + table backdrop (decor is
  visual-only; the table top at `z=0` is the one collider). Drop in wood/tile
  texture PNGs here for more realism.
- `src/pick_place_challenge/tasks/` — the two env configs + registration. They
  reuse mjlab's `lift_cube` manipulation MDP terms.
- `scripts/` — `view_scene.py` (pure-MuJoCo viewer), `random_rollout.py`
  (env-loop example), `spike_warp.py` (the Warp compatibility check).
- `tests/test_scene_smoke.py` — `uv run pytest`.

You're free to change anything in here. Have fun.
