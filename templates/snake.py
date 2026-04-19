"""
Snake Game Template for StableGym SDK.

A custom pygame-based implementation of the classic Snake game.
The agent controls a snake that must eat food while avoiding
walls and its own tail.

Features:
- Growing snake body mechanics
- Score-based reward shaping
- Collision detection (walls and self)
- Progressive difficulty as snake grows

Environment: Custom SnakeEnv (pygame rendering)
Algorithm: DQN (recommended) or PPO/A2C
"""

import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class SnakeEnv(gym.Env):
    """
    Snake game environment.

    The agent controls a snake that grows by eating food.
    Game ends if the snake hits a wall or itself.

    Observation: [head_x, head_y, food_x, food_y, direction, tail_length] (shape: (6,))
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """

    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, render_mode=None, fps=15, grid_size=12, max_steps=500):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_fps = fps if fps > 0 else self.metadata["render_fps"]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 40

        # Game state
        self.snake = []
        self.direction = 1  # Default: moving right
        self.food_pos = None
        self.current_step = 0
        self.score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.score = 0

        # Initialize snake in the center
        start_x = self.grid_size // 2
        start_y = self.grid_size // 2
        self.snake = [[start_x, start_y], [start_x - 1, start_y], [start_x - 2, start_y]]
        self.direction = 1  # Moving right

        # Place first food
        self.food_pos = self._place_food()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def _get_obs(self):
        head = self.snake[0]
        obs = [
            head[0] / (self.grid_size - 1),
            head[1] / (self.grid_size - 1),
            self.food_pos[0] / (self.grid_size - 1),
            self.food_pos[1] / (self.grid_size - 1),
            self.direction / 3.0,
            len(self.snake) / self.max_steps,
        ]
        return np.array(obs, dtype=np.float32)

    def _place_food(self):
        """Place food at a random position not occupied by the snake."""
        while True:
            pos = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]
            if pos not in self.snake:
                return pos

    def _is_collision(self, pos):
        """Check if position collides with walls or snake body."""
        # Wall collision
        if pos[0] < 0 or pos[0] >= self.grid_size or pos[1] < 0 or pos[1] >= self.grid_size:
            return True
        # Self collision (exclude tail which will move)
        if pos in self.snake[:-1]:
            return True
        return False

    def step(self, action):
        self.current_step += 1

        # Prevent 180-degree turns
        if abs(action - self.direction) != 2:
            self.direction = action

        # Calculate new head position
        head = list(self.snake[0])
        if self.direction == 0:    # Up
            head[1] -= 1
        elif self.direction == 1:  # Right
            head[0] += 1
        elif self.direction == 2:  # Down
            head[1] += 1
        elif self.direction == 3:  # Left
            head[0] -= 1

        terminated = False
        reward = 0.1  # Small survival reward

        # Check collision
        if self._is_collision(head):
            reward = -10.0
            terminated = True
        else:
            self.snake.insert(0, head)

            # Check if food eaten
            if head == self.food_pos:
                reward = 10.0 + len(self.snake) * 0.5  # Growing reward
                self.score += 1
                self.food_pos = self._place_food()
            else:
                self.snake.pop()  # Remove tail if no food eaten

        # Max steps
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {"score": self.score}

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            window_size = self.grid_size * self.cell_size
            self.window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption(f"Snake - Score: {self.score}")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        canvas.fill((20, 30, 20))  # Dark green background

        # Grid lines
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pygame.draw.rect(
                    canvas, (30, 45, 30),
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), 1
                )

        # Draw snake body
        for i, segment in enumerate(self.snake):
            color = (0, 200, 100) if i == 0 else (0, 150, 80)  # Head brighter
            pygame.draw.rect(
                canvas, color,
                (segment[0] * self.cell_size + 1, segment[1] * self.cell_size + 1,
                 self.cell_size - 2, self.cell_size - 2),
                border_radius=4,
            )

        # Draw food
        pygame.draw.rect(
            canvas, (255, 80, 80),
            (self.food_pos[0] * self.cell_size + 2, self.food_pos[1] * self.cell_size + 2,
             self.cell_size - 4, self.cell_size - 4),
            border_radius=6,
        )

        # Update window title with score
        pygame.display.set_caption(f"Snake - Score: {self.score} | Steps: {self.current_step}")

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
    Factory function for Snake environment.

    Args:
        render_mode: 'human' for visual rendering, None for headless.
        fps: Frames per second cap for rendering.
        **kwargs: Additional arguments passed to SnakeEnv.

    Returns:
        gym.Env: SnakeEnv instance.
    """
    return SnakeEnv(render_mode=render_mode, fps=fps, **kwargs)


TEMPLATE_CONFIG = {
    "id": "snake_game",
    "env_factory": env_factory,
    "algorithm": "DQN",
    "policy": "MlpPolicy",
    "steps": 300_000,
    "fps_visual": 15,
    "net_arch": [128, 128, 64],
    "exploration_fraction": 0.5,
    "learning_rate": 5e-4,
    "batch_size": 128,
    "target_update_interval": 1000,
    "save_freq": 25_000,
    "verbose": 1,
}
