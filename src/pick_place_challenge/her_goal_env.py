"""GoalEnv-style adapter for HER-compatible bankshot reward recomputation."""

from __future__ import annotations

from typing import Any

import numpy as np

from pick_place_challenge.task import (
    FRONT_WALL_CENTER,
    HER_DISTANCE_THRESHOLD,
    HER_FAILURE_REWARD,
    HER_SUCCESS_REWARD,
    bankshot_sparse_reward_np,
)

_BALL_GEOMS = {"ball_collision", "ball_visual"}
_FRONT_WALL_GEOMS = {"f_bounce_wall_geom"}
_WALL_GEOMS = {
    "f_bounce_wall_geom",
    "l_bounce_wall_geom",
    "r_bounce_wall_geom",
    "b_bounce_wall_geom",
    "u_bounce_wall_geom",
    "d_bounce_wall_geom",
}
_TABLE_GEOMS = {"table_top", "table_top_visual"}


def _as_numpy(value: Any) -> np.ndarray:
    """Convert torch/NumPy/scalar inputs into a NumPy array for GoalEnv rewards."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _tensor_like_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


class HerGoalEnvWrapper:
    """Expose a GoalEnv-like API around the mjlab manager-based environment.

    The wrapped environment must provide a non-concatenated ``"goal"`` observation
    group containing ``observation``, ``achieved_goal``, and ``desired_goal``.
    """

    def __init__(
        self,
        env,
        *,
        distance_threshold: float = HER_DISTANCE_THRESHOLD,
        success_reward: float = HER_SUCCESS_REWARD,
        failure_reward: float = HER_FAILURE_REWARD,
    ) -> None:
        self.env = env
        self.distance_threshold = distance_threshold
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self._tracker_ready = False
        self._hit_wall: np.ndarray | None = None
        self._hit_front_wall: np.ndarray | None = None
        self._hit_table_first: np.ndarray | None = None
        self._wall_contact_position: np.ndarray | None = None
        self._first_contact_type: np.ndarray | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return bankshot_sparse_reward_np(
            _as_numpy(achieved_goal),
            _as_numpy(desired_goal),
            info,
            distance_threshold=self.distance_threshold,
            success_reward=self.success_reward,
            failure_reward=self.failure_reward,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        goal_obs = self._to_goal_obs(obs)
        self._reset_contact_tracker(goal_obs)
        return goal_obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        goal_obs = self._to_goal_obs(obs)
        if not self._tracker_ready:
            self._reset_contact_tracker(goal_obs)
        self._update_contact_tracker()
        self._apply_tracked_achieved_goal(goal_obs)
        info = self._merge_contact_info(info, goal_obs)
        reward = self.compute_reward(
            goal_obs["achieved_goal"], goal_obs["desired_goal"], info
        )
        info["is_success"] = reward == self.success_reward
        return goal_obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def _to_goal_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if {"observation", "achieved_goal", "desired_goal"} <= set(obs):
            return {
                "observation": obs["observation"],
                "achieved_goal": obs["achieved_goal"],
                "desired_goal": obs["desired_goal"],
            }
        if "goal" in obs:
            goal_obs = obs["goal"]
            return {
                "observation": goal_obs["observation"],
                "achieved_goal": goal_obs["achieved_goal"],
                "desired_goal": goal_obs["desired_goal"],
            }
        return {
            "observation": self._base_observation(obs),
            "achieved_goal": self._current_ball_position(),
            "desired_goal": self._front_wall_goal(),
        }

    def _base_observation(self, obs: dict[str, Any]) -> Any:
        if "actor" in obs:
            return obs["actor"]
        return obs

    def _current_ball_position(self) -> np.ndarray:
        scene = getattr(self.env, "scene", None)
        if scene is None:
            raise KeyError("Cannot infer achieved_goal without env.scene['ball'].")
        return _tensor_like_to_numpy(scene["ball"].data.root_link_pos_w)

    def _front_wall_goal(self) -> np.ndarray:
        num_envs = int(getattr(self.env, "num_envs", 1))
        return np.tile(np.asarray(FRONT_WALL_CENTER, dtype=np.float32), (num_envs, 1))

    def _merge_contact_info(
        self, info: dict[str, Any], goal_obs: dict[str, Any]
    ) -> dict[str, Any]:
        merged = self._tracked_contact_info(goal_obs)
        merged.update(dict(info))
        return merged

    def _reset_contact_tracker(self, goal_obs: dict[str, Any]) -> None:
        fallback_pos = _tensor_like_to_numpy(goal_obs["achieved_goal"])
        fallback_pos = np.asarray(fallback_pos, dtype=np.float32)
        if fallback_pos.ndim == 1:
            fallback_pos = fallback_pos[None, :]

        num_envs = int(getattr(self.env, "num_envs", fallback_pos.shape[0]))
        self._hit_wall = np.zeros(num_envs, dtype=bool)
        self._hit_front_wall = np.zeros(num_envs, dtype=bool)
        self._hit_table_first = np.zeros(num_envs, dtype=bool)
        self._wall_contact_position = np.zeros((num_envs, 3), dtype=np.float32)
        self._wall_contact_position[: fallback_pos.shape[0]] = fallback_pos
        self._first_contact_type = np.full(num_envs, None, dtype=object)
        self._tracker_ready = True

    def _update_contact_tracker(self) -> None:
        if (
            self._hit_wall is None
            or self._hit_front_wall is None
            or self._hit_table_first is None
            or self._wall_contact_position is None
            or self._first_contact_type is None
        ):
            return

        for env_id, geom_a, geom_b, pos in self._iter_contact_events():
            if env_id < 0 or env_id >= len(self._hit_wall):
                continue
            names = {geom_a, geom_b}
            if not names & _BALL_GEOMS:
                continue

            if names & _TABLE_GEOMS:
                if self._first_contact_type[env_id] is None:
                    self._first_contact_type[env_id] = "table"
                    self._hit_table_first[env_id] = True

            if names & _WALL_GEOMS:
                if self._first_contact_type[env_id] is None:
                    self._first_contact_type[env_id] = (
                        "front_wall" if names & _FRONT_WALL_GEOMS else "other_wall"
                    )
                self._hit_wall[env_id] = True
                self._wall_contact_position[env_id] = np.asarray(pos, dtype=np.float32)

            if names & _FRONT_WALL_GEOMS:
                self._hit_front_wall[env_id] = True

    def _apply_tracked_achieved_goal(self, goal_obs: dict[str, Any]) -> None:
        if self._hit_wall is None or self._wall_contact_position is None:
            return
        achieved = _tensor_like_to_numpy(goal_obs["achieved_goal"]).copy()
        single = achieved.ndim == 1
        achieved_2d = achieved[None, :] if single else achieved
        hit = self._hit_wall[: achieved_2d.shape[0]]
        achieved_2d[hit] = self._wall_contact_position[: achieved_2d.shape[0]][hit]
        goal_obs["achieved_goal"] = achieved_2d[0] if single else achieved_2d

    def _tracked_contact_info(self, goal_obs: dict[str, Any]) -> dict[str, Any]:
        if not self._tracker_ready or self._hit_wall is None:
            self._reset_contact_tracker(goal_obs)

        assert self._hit_wall is not None
        assert self._hit_front_wall is not None
        assert self._hit_table_first is not None
        assert self._wall_contact_position is not None
        assert self._first_contact_type is not None
        return {
            "hit_wall": self._hit_wall.copy(),
            "hit_front_wall": self._hit_front_wall.copy(),
            "hit_table_first": self._hit_table_first.copy(),
            "wall_contact_position": self._wall_contact_position.copy(),
            "first_contact_type": self._first_contact_type.copy(),
        }

    def _iter_contact_events(self):
        contact = getattr(getattr(self.env, "sim", None), "data", None)
        contact = getattr(getattr(contact, "contact", None), "struct", None)
        if contact is None or not hasattr(contact, "geom"):
            return []

        try:
            geoms = _tensor_like_to_numpy(contact.geom)
            positions = _tensor_like_to_numpy(contact.pos)
            if hasattr(contact, "worldid"):
                world_ids = _tensor_like_to_numpy(contact.worldid)
            else:
                world_ids = np.zeros(np.reshape(geoms, (-1, 2)).shape[0], dtype=int)
        except Exception:
            return []

        if geoms.size == 0:
            return []

        names = self._geom_names()
        events = []
        geom_pairs = np.reshape(geoms, (-1, 2))
        positions = np.reshape(positions, (-1, 3))
        world_ids = np.reshape(world_ids, (-1,))
        for idx, pair in enumerate(geom_pairs):
            geom_a = names.get(int(pair[0]))
            geom_b = names.get(int(pair[1]))
            if geom_a is None or geom_b is None:
                continue
            events.append(
                (
                    int(world_ids[idx]),
                    geom_a.split("/")[-1],
                    geom_b.split("/")[-1],
                    positions[idx],
                )
            )
        return events

    def _geom_names(self) -> dict[int, str]:
        model = getattr(getattr(self.env, "sim", None), "mj_model", None)
        if model is None:
            return {}
        return {idx: model.geom(idx).name for idx in range(model.ngeom)}
