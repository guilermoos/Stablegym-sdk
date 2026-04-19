"""
CartPole-v1 Template for StableGym SDK.

A classic control problem where a pole is attached to a cart that
moves along a frictionless track. The goal is to keep the pole upright
by applying forces to the cart.

Environment: Gymnasium CartPole-v1 (built-in)
Algorithm: DQN (default) or PPO/A2C
"""

import gymnasium as gym


def env_factory(render_mode=None, fps=0, **kwargs):
    """
    Factory function for CartPole-v1 environment.

    Args:
        render_mode: 'human' for visual rendering, None for headless.
        fps: Frames per second cap (not used by CartPole, kept for API compatibility).
        **kwargs: Additional arguments (not used).

    Returns:
        gym.Env: CartPole-v1 environment instance.
    """
    return gym.make("CartPole-v1", render_mode=render_mode)


TEMPLATE_CONFIG = {
    "id": "cartpole_v1",
    "env_factory": env_factory,
    "algorithm": "DQN",
    "policy": "MlpPolicy",
    "steps": 50_000,
    "fps_visual": 60,
    "net_arch": [64, 64],
    "exploration_fraction": 0.4,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "target_update_interval": 500,
    "save_freq": 10_000,
    "verbose": 1,
}
