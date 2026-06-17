"""Project-specific RSL-RL runner hooks."""

from __future__ import annotations

import os
import statistics
import time

import torch
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from rsl_rl.utils import check_nan

from pick_place_challenge.ppo_release_rollout_env import (
    PpoPostReleaseActionMaskWrapper,
)


class BankshotCheckpointRunner(ManipulationOnPolicyRunner):
    """Save a one-time checkpoint once front-wall hit rate reaches 50%."""

    front_wall_checkpoint_threshold = 0.5
    front_wall_checkpoint_name = "model_front_wall_50.pt"

    def __init__(self, *args, **kwargs) -> None:
        if args:
            args = (PpoPostReleaseActionMaskWrapper(args[0]), *args[1:])
        super().__init__(*args, **kwargs)
        self._saved_front_wall_checkpoint = False

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"]["rnd_cfg"]
                        else None
                    )
                    self.logger.process_env_step(
                        rewards, dones, extras, intrinsic_rewards
                    )

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self._maybe_save_front_wall_checkpoint(it)

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight
                if self.cfg["algorithm"]["rnd_cfg"]
                else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )  # type: ignore[arg-type]
            self.logger.stop_logging_writer()

    def _maybe_save_front_wall_checkpoint(self, iteration: int) -> None:
        if self._saved_front_wall_checkpoint:
            return
        if self.logger.disable_logs or self.logger.log_dir is None:
            return

        hit_rate = self._current_front_wall_hit_rate()
        if hit_rate is None or hit_rate < self.front_wall_checkpoint_threshold:
            return

        path = os.path.join(self.logger.log_dir, self.front_wall_checkpoint_name)
        self.save(
            path,
            infos={
                "front_wall_hit_rate": hit_rate,
                "front_wall_checkpoint_threshold": self.front_wall_checkpoint_threshold,
                "front_wall_checkpoint_iteration": iteration,
            },
        )
        self._saved_front_wall_checkpoint = True
        print(
            "[INFO] Saved front-wall checkpoint "
            f"at iteration {iteration}: hit_rate={hit_rate:.3f}, path={path}"
        )

    def _current_front_wall_hit_rate(self) -> float | None:
        key = "Episode_Metrics/front_wall_hit_rate"
        values = []
        for ep_info in self.logger.ep_extras:
            if key not in ep_info:
                continue
            value = ep_info[key]
            if isinstance(value, torch.Tensor):
                values.extend(value.detach().cpu().reshape(-1).tolist())
            else:
                values.append(float(value))
        if not values:
            return None
        return float(statistics.mean(values))
