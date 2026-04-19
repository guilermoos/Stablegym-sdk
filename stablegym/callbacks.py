"""
Training callbacks for StableGym SDK.

Provides checkpoint saving, reward logging, and custom callback hooks
during the RL training process.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional

from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """
    Save model checkpoints at regular intervals during training.

    Args:
        save_freq: Save checkpoint every N timesteps.
        save_path: Directory path to save checkpoints.
        name_prefix: Prefix for checkpoint filenames.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "./models/checkpoints",
        name_prefix: str = "checkpoint",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(str(path))
            if self.verbose > 0:
                self.logger.record("checkpoint/saved_at", self.num_timesteps)
        return True


class RewardLoggerCallback(BaseCallback):
    """
    Log episode rewards and training metrics at regular intervals.

    Args:
        log_freq: Log every N timesteps.
        verbose: Verbosity level.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_rewards = []
        self._episode_count = 0
        self._start_time = None

    def _init_callback(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                self._episode_count += 1
                reward = info["episode"]["r"]
                self._episode_rewards.append(reward)

        if self.num_timesteps % self.log_freq == 0:
            elapsed = time.time() - self._start_time if self._start_time else 0
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0

            if self._episode_rewards:
                recent_rewards = self._episode_rewards[-100:]
                mean_reward = sum(recent_rewards) / len(recent_rewards)
                print(
                    f"  [Step {self.num_timesteps:>8}] "
                    f"Episodes: {self._episode_count:>4} | "
                    f"Mean reward (last 100): {mean_reward:>8.2f} | "
                    f"FPS: {fps:.0f}"
                )
            else:
                print(f"  [Step {self.num_timesteps:>8}] FPS: {fps:.0f}")

        return True

    def _on_training_end(self) -> None:
        if self._episode_rewards:
            avg_reward = sum(self._episode_rewards) / len(self._episode_rewards)
            print(f"\n  Training complete - Average episode reward: {avg_reward:.2f}")
