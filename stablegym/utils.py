"""
Utility functions for StableGym SDK.

Provides device detection, path resolution, seed management,
and other helper functions used across the SDK.
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_device(requested: str = "auto") -> str:
    """
    Resolve the computation device.

    Args:
        requested: 'auto', 'cuda', or 'cpu'.

    Returns:
        Resolved device string compatible with Stable-Baselines3.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_str: str) -> str:
    """
    Resolve a path string to an absolute path.

    Supports:
    - Relative paths (resolved against current working directory)
    - Home directory expansion (~)
    - Absolute paths (returned as-is)

    Args:
        path_str: Path string (relative or absolute).

    Returns:
        Absolute path string.
    """
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def ensure_dir(path_str: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path_str: Directory path.

    Returns:
        Absolute path to the directory.
    """
    path = Path(resolve_path(path_str))
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
