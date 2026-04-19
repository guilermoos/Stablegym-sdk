"""
Grid World Navigation Template for StableGym SDK.

A custom pygame-based environment where an agent must navigate
a grid to reach a goal while avoiding obstacles.

Features:
- Heatmap visualization showing visited cells
- Obstacle avoidance
- Sparse reward structure
- Customizable grid size and difficulty

Environment: Custom GridEnv (pygame rendering)
Algorithm: DQN (default) or PPO/A2C
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class GridEnv(gym.Env):
    """
    Grid World environment with obstacle navigation.

    The agent starts at the top-left corner and must reach
    the bottom-right goal while avoiding obstacles.

    Observation: Normalized [x, y] position (shape: (2,))
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """

    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, render_mode=None, fps=15, size=6, max_steps=100):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.render_fps = fps if fps > 0 else self.metadata["render_fps"]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 60

        # Environment state
        self.goal_pos = [size - 1, size - 1]
        self.obstacles = [[2, 2], [3, 2], [size // 2, size - 2]]
        self.agent_pos = [0, 0]
        self.current_step = 0
        self.heatmap = np.zeros((size, size), dtype=int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.agent_pos = [0, 0]
        self.heatmap.fill(0)
        self.heatmap[self.agent_pos[0], self.agent_pos[1]] += 1

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32) / (self.size - 1)

    def step(self, action):
        self.current_step += 1
        new_pos = list(self.agent_pos)

        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        if action == 0:
            new_pos[1] -= 1
        elif action == 1:
            new_pos[0] += 1
        elif action == 2:
            new_pos[1] += 1
        elif action == 3:
            new_pos[0] -= 1

        # Boundary check
        new_pos[0] = max(0, min(self.size - 1, new_pos[0]))
        new_pos[1] = max(0, min(self.size - 1, new_pos[1]))

        reward = -1.0
        terminated = False
        truncated = False

        # Obstacle collision
        if new_pos in self.obstacles:
            reward = -10.0
        else:
            self.agent_pos = new_pos

        # Goal reached
        if self.agent_pos == self.goal_pos:
            reward = 100.0
            terminated = True

        # Max steps reached
        if self.current_step >= self.max_steps:
            truncated = True

        self.heatmap[self.agent_pos[0], self.agent_pos[1]] += 1

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
            pygame.display.set_caption(f"Grid World - {self.size}x{self.size}")
            self.clock = pygame.time.Clock()

        # Create canvas
        canvas = pygame.Surface((self.size * self.cell_size, self.size * self.cell_size))
        canvas.fill((255, 255, 255))

        # Draw heatmap
        max_v = np.max(self.heatmap)
        if max_v > 0:
            for x in range(self.size):
                for y in range(self.size):
                    v = self.heatmap[x, y]
                    if v > 0:
                        alpha = int((v / max_v) * 180)
                        hs = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        hs.fill((255, 100, 100, alpha))
                        canvas.blit(hs, (x * self.cell_size, y * self.cell_size))

        # Draw grid lines
        for x in range(self.size):
            for y in range(self.size):
                pygame.draw.rect(
                    canvas, (200, 200, 200),
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), 1
                )

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(
                canvas, (60, 60, 60),
                (obs[0] * self.cell_size, obs[1] * self.cell_size, self.cell_size, self.cell_size)
            )

        # Draw goal
        pygame.draw.rect(
            canvas, (50, 205, 50),
            (self.goal_pos[0] * self.cell_size, self.goal_pos[1] * self.cell_size,
             self.cell_size, self.cell_size)
        )

        # Draw agent
        pygame.draw.rect(
            canvas, (0, 100, 255),
            (self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size,
             self.cell_size, self.cell_size)
        )

        # Update display
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
    Factory function for Grid World environment.

    Args:
        render_mode: 'human' for visual rendering, None for headless.
        fps: Frames per second cap for rendering.
        **kwargs: Additional arguments passed to GridEnv.

    Returns:
        gym.Env: GridEnv instance.
    """
    return GridEnv(render_mode=render_mode, fps=fps, **kwargs)


TEMPLATE_CONFIG = {
    "id": "grid_world",
    "env_factory": env_factory,
    "algorithm": "DQN",
    "policy": "MlpPolicy",
    "steps": 100_000,
    "fps_visual": 15,
    "net_arch": [128, 128],
    "exploration_fraction": 0.5,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "target_update_interval": 1000,
    "save_freq": 20_000,
    "verbose": 1,
}
