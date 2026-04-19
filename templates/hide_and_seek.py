"""
Hide and Seek Survival Template for StableGym SDK.

A custom pygame-based environment where an agent must evade
multiple hunting enemies on a grid with obstacles.

Features:
- Multi-enemy AI with tracking behavior
- Wall obstacle navigation
- Survival-based reward structure
- Progressive difficulty with enemy count

Environment: Custom HideSeekEnv (pygame rendering)
Algorithm: DQN (recommended) or PPO/A2C
"""

import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class HideSeekEnv(gym.Env):
    """
    Hide and Seek environment - Agent must evade hunting enemies.

    The agent navigates a grid while enemies actively track it.
    Survival time determines the score.

    Observation: Normalized [agent_x, agent_y, enemy1_x, enemy1_y, ...] (shape: (12,))
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """

    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, render_mode=None, fps=15, size=15, max_steps=150, n_enemies=5):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.n_enemies = n_enemies
        self.render_fps = fps if fps > 0 else self.metadata["render_fps"]

        self.action_space = spaces.Discrete(4)
        # Observation: [agent_x, agent_y, e1_x, e1_y, e2_x, e2_y, ...]
        obs_dim = 2 + 2 * min(n_enemies, 5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 40

        # Generate wall obstacles (border walls + internal structures)
        self.walls = self._generate_walls()

        self.agent_pos = None
        self.enemies = None
        self.current_step = 0

    def _generate_walls(self):
        """Generate wall positions for the environment."""
        walls = [
            # Central barriers
            [4, 4], [4, 5], [4, 6],
            [10, 4], [10, 5], [10, 6],
            [6, 10], [7, 10], [8, 10],
            [4, 12], [11, 11],
        ]
        # Add boundary filtering
        walls = [w for w in walls if 0 <= w[0] < self.size and 0 <= w[1] < self.size]
        return walls

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.agent_pos = [0, 0]

        # Place enemies in corners and edges
        enemy_positions = [
            [self.size - 1, self.size - 1],
            [self.size - 1, 0],
            [0, self.size - 1],
            [self.size // 2, self.size - 1],
            [self.size - 3, self.size // 2],
        ]
        self.enemies = enemy_positions[:self.n_enemies]

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def _get_obs(self):
        obs = [*self.agent_pos]
        for enemy in self.enemies[:5]:
            obs.extend(enemy)
        # Pad if fewer enemies
        while len(obs) < self.observation_space.shape[0]:
            obs.append(0.0)
        return np.array(obs, dtype=np.float32) / (self.size - 1)

    def _is_wall(self, pos):
        return pos in self.walls or not (0 <= pos[0] < self.size) or not (0 <= pos[1] < self.size)

    def step(self, action):
        self.current_step += 1

        # Agent movement
        new_agent = list(self.agent_pos)
        if action == 0:
            new_agent[1] -= 1
        elif action == 1:
            new_agent[0] += 1
        elif action == 2:
            new_agent[1] += 1
        elif action == 3:
            new_agent[0] -= 1

        if not self._is_wall(new_agent):
            self.agent_pos = new_agent

        # Enemy AI - tracking behavior
        for idx in range(len(self.enemies)):
            if random.random() < 0.6:
                ex, ey = self.enemies[idx]
                ax, ay = self.agent_pos
                diff_x = ax - ex
                diff_y = ay - ey
                new_enemy = [ex, ey]

                if abs(diff_x) > abs(diff_y):
                    new_enemy[0] += 1 if diff_x > 0 else -1
                elif abs(diff_y) > 0:
                    new_enemy[1] += 1 if diff_y > 0 else -1

                if not self._is_wall(new_enemy):
                    self.enemies[idx] = new_enemy

        terminated = False
        truncated = False
        reward = 1.0  # Survival reward

        # Caught by enemy
        if self.agent_pos in self.enemies:
            reward = -50.0
            terminated = True
        elif self.current_step >= self.max_steps:
            reward = 100.0  # Survived!
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            window_size = self.size * self.cell_size
            self.window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption(f"Hide and Seek - {self.size}x{self.size}")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.size * self.cell_size, self.size * self.cell_size))
        canvas.fill((240, 245, 240))

        # Grid lines
        for x in range(self.size):
            for y in range(self.size):
                pygame.draw.rect(
                    canvas, (180, 190, 180),
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), 1
                )

        # Walls
        for w in self.walls:
            pygame.draw.rect(
                canvas, (40, 40, 40),
                (w[0] * self.cell_size, w[1] * self.cell_size, self.cell_size, self.cell_size)
            )

        # Enemies (red)
        for en in self.enemies:
            pygame.draw.rect(
                canvas, (200, 30, 30),
                (en[0] * self.cell_size, en[1] * self.cell_size, self.cell_size, self.cell_size)
            )

        # Agent (blue)
        pygame.draw.rect(
            canvas, (30, 100, 220),
            (self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size,
             self.cell_size, self.cell_size)
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


def env_factory(render_mode=None, fps=0, **kwargs):
    """
    Factory function for Hide and Seek environment.

    Args:
        render_mode: 'human' for visual rendering, None for headless.
        fps: Frames per second cap for rendering.
        **kwargs: Additional arguments passed to HideSeekEnv.

    Returns:
        gym.Env: HideSeekEnv instance.
    """
    return HideSeekEnv(render_mode=render_mode, fps=fps, **kwargs)


TEMPLATE_CONFIG = {
    "id": "hide_and_seek",
    "env_factory": env_factory,
    "algorithm": "DQN",
    "policy": "MlpPolicy",
    "steps": 200_000,
    "fps_visual": 15,
    "net_arch": [256, 256],
    "exploration_fraction": 0.6,
    "learning_rate": 5e-4,
    "batch_size": 256,
    "target_update_interval": 1000,
    "save_freq": 25_000,
    "verbose": 1,
}
