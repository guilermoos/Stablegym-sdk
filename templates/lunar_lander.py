"""
Lunar Lander Template for StableGym SDK.

A classic continuous control problem where the agent must land
a spacecraft safely on the moon's surface.

Features:
- Continuous/Discrete action space support
- Physics-based simulation
- Reward shaping for fuel efficiency and soft landing
- Built-in Gymnasium environment

Environment: Gymnasium LunarLander-v2 (built-in)
Algorithm: PPO (recommended) or DQN/A2C
"""

import gymnasium as gym


def env_factory(render_mode=None, fps=0, continuous=False, **kwargs):
    """
    Factory function for LunarLander-v2 environment.

    Args:
        render_mode: 'human' for visual rendering, None for headless.
        fps: Frames per second cap (not used by LunarLander, kept for API compatibility).
        continuous: If True, use continuous action space (LunarLander-v2).
        **kwargs: Additional arguments passed to gym.make.

    Returns:
        gym.Env: LunarLander-v2 environment instance.
    """
    env_id = "LunarLander-v2"
    return gym.make(env_id, render_mode=render_mode, continuous=continuous, **kwargs)


TEMPLATE_CONFIG = {
    "id": "lunar_lander",
    "env_factory": env_factory,
    "algorithm": "PPO",
    "policy": "MlpPolicy",
    "steps": 500_000,
    "fps_visual": 60,
    "net_arch": [256, 256],
    "learning_rate": 3e-4,
    "batch_size": 128,
    "n_steps": 2048,
    "n_epochs": 10,
    "save_freq": 50_000,
    "verbose": 1,
}
