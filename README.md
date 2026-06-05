# pick-place-challenge

A small, self-contained robotics sandbox: a **Franka Panda + Robotiq 2F-85**
that must **pick up a ball and place it in a bowl** on a table, inside a real
modeled room (an Objaverse parking garage), using
[mjlab](https://github.com/mujocolab/mjlab) (GPU-accelerated MuJoCo,
Isaac-Lab-style manager API). The ball (a Poly Haven mesh), bowl (a scanned
object whose cavity actually holds the ball), and room are real assets, not
primitives.

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

Assets — the [Poly Haven](https://polyhaven.com) ball mesh, the scanned bowl from
[mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects),
and the [Objaverse](https://objaverse.allenai.org) room mesh — are fetched on
first use into `~/.cache` (the room is ~tens of MB, so first launch takes a
moment). Pre-fetch with `uv run pick-place-fetch-assets` and
`uv run pick-place-fetch-ph`.

Drive it with a random policy and watch it in an interactive viewer:

```bash
uv run play Mjlab-PlaceBall-Franka-State-v0 --agent random --viewer native
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

Same task (place the ball in the bowl), same robot, same reward — they differ
**only in the observation**:

| Task ID | Observation |
|---|---|
| `Mjlab-PlaceBall-Franka-State-v0` | Low-dim **state**: joint pos/vel, end-effector→ball vector, ball→bowl vector, bowl position, last action. |
| `Mjlab-PlaceBall-Franka-Pixels-v0` | **Pixels**: an `84×84` RGB **wrist** camera + `84×84` RGB **scene** camera, plus proprioception and the (fixed) bowl position. The ball is *not* given as state — you have to see it. |

**Action** (both): 8-D — 7 arm joint-position targets + 1 gripper command
(`-1` open … `+1` closed; the Robotiq's tendon does the rest).

**Reward** (both): a simple reach-then-place shaping (reach the ball × bring it to
the bowl). It's deliberately basic — **rip it out and write your own** if doing RL.
Success = ball resting inside the bowl rim.

Switch tasks by swapping the ID, e.g.:

```bash
uv run play Mjlab-PlaceBall-Franka-Pixels-v0 --agent random --viewer native
```

---

## Make something cool

Open-ended. Some rabbit holes, roughly easy → hard — **you don't have to pick from this list**:

- **Scripted / classical control.** Write an IK or operational-space controller
  and a heuristic grasp that actually picks the ball up and drops it in. (Great on CPU.)
- **Behavior cloning.** Collect demos (teleop or your scripted policy), train a
  policy, evaluate it. Bonus: from pixels.
- **Reinforcement learning.** `uv run train Mjlab-PlaceBall-Franka-State-v0 --env.scene.num-envs 2048`
  (mjlab ships RSL-RL + PPO). Then `uv run play … --agent trained …`. Pixels is the hard mode.
- **Reward / task design.** The default reward is a placeholder. Design one that
  yields a clean grasp-and-place; randomize the bowl location (it's fixed by default).
- **Perception.** Estimate the ball pose from the wrist camera; close the loop on it.
- **Performance.** Profile the env, push `--env.scene.num-envs` as high as it goes,
  make rendering or stepping faster.
- **Make it harder / more realistic.** Move/clutter the bowl, add distractor
  objects, domain-randomize textures and lighting, swap the room or the gripper.

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
- `src/pick_place_challenge/objects.py` — the ball (a real Poly Haven mesh with a
  sphere collider) and the bowl (a scanned object whose convex decomposition keeps
  its cavity, so the ball nests inside). Tweak `BALL_DIAMETER`, `BOWL_NAME`.
- `src/pick_place_challenge/polyhaven.py` / `assets.py` / `room.py` — on-demand
  fetch of the Poly Haven ball, the scanned bowl, and the Objaverse room mesh.
  Swap the room via `GARAGE_UID` in `room.py` (any Objaverse uid).
- `src/pick_place_challenge/world.py` — the table + the room. The room is a real
  textured mesh (visual-only), so it renders in the Viser browser viewer as well
  as the native viewer/cameras. The table top at `z=0` is the only collider; the
  rest is decor.
- `src/pick_place_challenge/mdp.py` — the reach-and-place reward, ball→bowl
  observation, and the "placed in bowl" success check.
- `src/pick_place_challenge/tasks/` — the two env configs + registration, built on
  mjlab's manipulation skeleton (Franka swapped in for the YAM arm).
- `scripts/` — `view_scene.py` (pure-MuJoCo viewer), `random_rollout.py`
  (env-loop example), `spike_warp.py` (the Warp compatibility check).
- `tests/test_scene_smoke.py` — `uv run pytest`.

You're free to change anything in here. Have fun.
