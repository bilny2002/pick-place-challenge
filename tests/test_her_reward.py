from __future__ import annotations

import numpy as np

from pick_place_challenge.her_goal_env import HerGoalEnvWrapper


class _FakeGeom:
    def __init__(self, name: str):
        self.name = name


class _FakeModel:
    _names = {
        0: "ball_collision",
        1: "table_top",
        2: "f_bounce_wall_geom",
        3: "l_bounce_wall_geom",
    }
    ngeom = len(_names)

    def geom(self, idx: int):
        return _FakeGeom(self._names[idx])


class _FakeContact:
    def __init__(self, geom, pos, worldid):
        self.geom = np.asarray(geom)
        self.pos = np.asarray(pos, dtype=np.float32)
        self.worldid = np.asarray(worldid)


class _FakeContactData:
    def __init__(self, contact: _FakeContact):
        self.contact = type("_ContactBridge", (), {"struct": contact})()


class _FakeSim:
    def __init__(self, contact: _FakeContact):
        self.mj_model = _FakeModel()
        self.data = _FakeContactData(contact)


class _FakeGoalEnv:
    action_space = object()
    observation_space = object()

    def __init__(self, *, info=None, contact: _FakeContact | None = None, num_envs=1):
        self.info = info or {}
        self.num_envs = num_envs
        self.closed = False
        self.sim = _FakeSim(contact or _FakeContact([], [], []))
        self.obs = {
            "goal": {
                "observation": np.tile(np.array([1.0, 2.0, 3.0]), (num_envs, 1)),
                "achieved_goal": np.tile(np.array([2.9, 0.0, 0.0]), (num_envs, 1)),
                "desired_goal": np.tile(np.array([2.9, 0.0, 0.0]), (num_envs, 1)),
            }
        }

    def reset(self, **kwargs):
        return self.obs, {"reset": True}

    def step(self, action):
        return self.obs, np.array(-123.0), False, False, dict(self.info)

    def close(self):
        self.closed = True


def test_step_reward_matches_compute_reward() -> None:
    env = HerGoalEnvWrapper(
        _FakeGoalEnv(info={"hit_wall": True, "hit_front_wall": True})
    )

    obs, reward, terminated, truncated, info = env.step(np.zeros(1))

    assert not terminated
    assert not truncated
    np.testing.assert_array_equal(
        reward, env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
    )


def test_compute_reward_works_with_substituted_goal() -> None:
    env = HerGoalEnvWrapper(_FakeGoalEnv())

    achieved = np.array([0.0, 3.0, 0.0])
    substituted_goal = np.array([0.0, 3.0, 0.0])
    reward = env.compute_reward(
        achieved,
        substituted_goal,
        {"hit_wall": True, "hit_front_wall": False},
    )

    np.testing.assert_array_equal(reward, [env.success_reward])


def test_compute_reward_batched() -> None:
    env = HerGoalEnvWrapper(_FakeGoalEnv())

    achieved = np.array([[2.9, 0.0, 0.0], [0.0, 3.0, 0.0]])
    desired = np.array([[2.9, 0.0, 0.0], [2.9, 0.0, 0.0]])
    reward = env.compute_reward(
        achieved,
        desired,
        {
            "hit_wall": np.array([True, True]),
            "hit_front_wall": np.array([True, False]),
        },
    )

    assert reward.shape == (2,)
    np.testing.assert_array_equal(
        reward, np.array([env.success_reward, env.failure_reward])
    )


def test_front_wall_contact_succeeds() -> None:
    env = HerGoalEnvWrapper(_FakeGoalEnv())

    reward = env.compute_reward(
        np.array([2.9, 0.0, 0.0]),
        np.array([2.9, 0.0, 0.0]),
        {"hit_wall": True, "hit_front_wall": True},
    )

    assert reward == env.success_reward


def test_no_wall_contact_fails_even_at_target() -> None:
    env = HerGoalEnvWrapper(_FakeGoalEnv())

    reward = env.compute_reward(
        np.array([2.9, 0.0, 0.0]),
        np.array([2.9, 0.0, 0.0]),
        {"hit_wall": False, "hit_front_wall": False},
    )

    np.testing.assert_array_equal(reward, [env.failure_reward])


def test_table_first_contact_fails_even_if_front_wall_hit() -> None:
    env = HerGoalEnvWrapper(_FakeGoalEnv())

    reward = env.compute_reward(
        np.array([2.9, 0.0, 0.0]),
        np.array([2.9, 0.0, 0.0]),
        {
            "hit_wall": True,
            "hit_front_wall": True,
            "hit_table_first": True,
        },
    )

    assert reward == env.failure_reward


def test_reset_returns_goalenv_observation() -> None:
    env = HerGoalEnvWrapper(_FakeGoalEnv())

    obs, info = env.reset()

    assert info == {"reset": True}
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}


def test_wrapper_delegates_existing_env_attributes_and_close() -> None:
    base_env = _FakeGoalEnv()
    env = HerGoalEnvWrapper(base_env)

    assert env.action_space is base_env.action_space
    assert env.observation_space is base_env.observation_space
    env.close()
    assert base_env.closed


def test_tracker_records_front_wall_contact_position_as_achieved_goal() -> None:
    contact = _FakeContact(
        geom=[[0, 2]],
        pos=[[2.9, 0.2, 0.1]],
        worldid=[0],
    )
    env = HerGoalEnvWrapper(_FakeGoalEnv(contact=contact))
    env.reset()

    obs, reward, _, _, info = env.step(np.zeros(1))

    np.testing.assert_allclose(obs["achieved_goal"], [[2.9, 0.2, 0.1]])
    np.testing.assert_array_equal(info["hit_wall"], [True])
    np.testing.assert_array_equal(info["hit_front_wall"], [True])
    assert reward == env.success_reward


def test_tracker_remembers_table_first_before_later_front_wall() -> None:
    base_env = _FakeGoalEnv(
        contact=_FakeContact(
            geom=[[0, 1]],
            pos=[[0.0, 0.0, 0.0]],
            worldid=[0],
        )
    )
    env = HerGoalEnvWrapper(base_env)
    env.reset()
    env.step(np.zeros(1))

    base_env.sim.data.contact.struct = _FakeContact(
        geom=[[0, 2]],
        pos=[[2.9, 0.0, 0.0]],
        worldid=[0],
    )
    _obs, reward, _terminated, _truncated, info = env.step(np.zeros(1))

    np.testing.assert_array_equal(info["hit_wall"], [True])
    np.testing.assert_array_equal(info["hit_front_wall"], [True])
    np.testing.assert_array_equal(info["hit_table_first"], [True])
    assert reward == env.failure_reward


def test_tracker_keeps_parallel_env_contacts_separate() -> None:
    contact = _FakeContact(
        geom=[[0, 2], [0, 3]],
        pos=[[2.9, 0.0, 0.0], [0.0, 3.0, 0.0]],
        worldid=[0, 1],
    )
    env = HerGoalEnvWrapper(_FakeGoalEnv(contact=contact, num_envs=2))
    env.reset()

    obs, reward, _terminated, _truncated, info = env.step(np.zeros((2, 1)))

    np.testing.assert_allclose(obs["achieved_goal"], [[2.9, 0.0, 0.0], [0.0, 3.0, 0.0]])
    np.testing.assert_array_equal(info["hit_wall"], [True, True])
    np.testing.assert_array_equal(info["hit_front_wall"], [True, False])
    np.testing.assert_array_equal(reward, [env.success_reward, env.failure_reward])
