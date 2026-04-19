"""
StableGym SDK - A generalist framework for Reinforcement Learning
with Stable-Baselines3 and Pygame environments.

This SDK provides a unified interface for creating, training, and
evaluating RL agents across diverse environment templates.
"""

__version__ = "1.0.0"
__author__ = "StableGym Team"

from .sdk import StableGymSDK, GymEnvFactory
from .callbacks import CheckpointCallback, RewardLoggerCallback
from .utils import get_device, set_seed, resolve_path

__all__ = [
    "StableGymSDK",
    "GymEnvFactory",
    "CheckpointCallback",
    "RewardLoggerCallback",
    "get_device",
    "set_seed",
    "resolve_path",
]
