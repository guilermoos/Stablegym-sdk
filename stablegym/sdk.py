"""
Core SDK module - Abstracts Stable-Baselines3 and environment management
into a generalist, template-agnostic interface.
"""

import os
import sys
import time
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from .utils import get_device, resolve_path, set_seed
from .callbacks import CheckpointCallback, RewardLoggerCallback


# Registry of supported RL algorithms
ALGORITHM_REGISTRY = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
}


class GymEnvFactory:
    """
    Generic environment factory for creating vectorized Gymnasium environments.
    Works with any environment that follows the Gymnasium API.
    """

    def __init__(self, env_factory: Callable, **default_kwargs):
        """
        Args:
            env_factory: Callable that returns a gym.Env instance.
                         Signature: factory(render_mode=None, fps=0, **kwargs)
            **default_kwargs: Default arguments passed to env_factory.
        """
        self.env_factory = env_factory
        self.default_kwargs = default_kwargs

    def create(self, render_mode: Optional[str] = None, fps: int = 0, **override_kwargs):
        """Create a single environment instance."""
        kwargs = {**self.default_kwargs, **override_kwargs}
        return self.env_factory(render_mode=render_mode, fps=fps, **kwargs)

    def create_vec(self, render_mode: Optional[str] = None, fps: int = 0, n_envs: int = 1, **override_kwargs):
        """Create a vectorized environment ( DummyVecEnv ) for stable-baselines3."""
        def _make_env():
            def _init():
                return self.create(render_mode=render_mode, fps=fps, **override_kwargs)
            return _init

        if n_envs == 1:
            return DummyVecEnv([_make_env()])
        return DummyVecEnv([_make_env() for _ in range(n_envs)])


class StableGymSDK:
    """
    Generalist SDK for Reinforcement Learning.

    Provides a unified interface for:
    - Loading templates dynamically
    - Training agents with multiple algorithms
    - Running inference
    - Managing models and checkpoints

    The SDK is template-agnostic: any environment following the
    Gymnasium API can be used without SDK modifications.
    """

    def __init__(
        self,
        device: str = "auto",
        seed: Optional[int] = None,
        models_dir: str = "./models",
    ):
        """
        Args:
            device: Computation device - 'auto', 'cuda', 'cpu'.
            seed: Random seed for reproducibility.
            models_dir: Directory to save/load trained models.
        """
        self.device = get_device(device)
        self.seed = seed
        self.models_dir = Path(resolve_path(models_dir))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            set_seed(seed)

        self._current_model: Optional[BaseAlgorithm] = None
        self._current_config: Optional[Dict[str, Any]] = None
        self._algorithm: Optional[Type[BaseAlgorithm]] = None

    # ------------------------------------------------------------------ #
    # Template Loading
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_template(template_path: str) -> Dict[str, Any]:
        """
        Dynamically load a template module and extract its TEMPLATE_CONFIG.

        Args:
            template_path: Relative or absolute path to the template .py file.

        Returns:
            Dictionary containing the template configuration.

        Expected TEMPLATE_CONFIG keys:
            - id (str): Unique identifier for the template.
            - env_factory (callable): Factory function for the environment.
            - algorithm (str, optional): RL algorithm name ('DQN', 'PPO', 'A2C'). Default: 'DQN'.
            - steps (int): Total training timesteps.
            - fps_visual (int): FPS cap for visual inference.
            - net_arch (list): Neural network architecture layers.
            - exploration_fraction (float, optional): For DQN exploration. Default: 0.4.
            - learning_rate (float, optional): Learning rate. Default: 1e-3.
            - batch_size (int, optional): Batch size. Default: 128.
            - n_envs (int, optional): Number of parallel envs. Default: 1.
            - save_freq (int, optional): Checkpoint frequency in steps. Default: 10000.
            - policy (str, optional): Policy network type. Default: 'MlpPolicy'.
            - env_kwargs (dict, optional): Additional kwargs for env_factory.
        """
        path = resolve_path(template_path)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Template file not found: {path}")

        module_name = f"stablegym_template_{Path(path).stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Failed to load template from '{path}': {e}")

        if not hasattr(module, "TEMPLATE_CONFIG"):
            raise ValueError(f"Template '{path}' must define 'TEMPLATE_CONFIG' dictionary.")

        config = module.TEMPLATE_CONFIG

        # Validate required fields
        required = ["id", "env_factory", "steps"]
        for key in required:
            if key not in config:
                raise ValueError(f"TEMPLATE_CONFIG missing required key: '{key}'")

        return config

    # ------------------------------------------------------------------ #
    # Model Building
    # ------------------------------------------------------------------ #

    def build_model(
        self,
        config: Dict[str, Any],
        env: Union[gym.Env, DummyVecEnv],
        algorithm: Optional[str] = None,
    ) -> BaseAlgorithm:
        """
        Build an RL model from the given configuration.

        Args:
            config: Template configuration dictionary.
            env: Gymnasium environment (single or vectorized).
            algorithm: Algorithm override (e.g., 'PPO', 'A2C').
                       If None, uses config["algorithm"] or defaults to 'DQN'.

        Returns:
            Configured Stable-Baselines3 model instance.
        """
        algo_name = algorithm or config.get("algorithm", "DQN").upper()

        if algo_name not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{algo_name}'. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )

        self._algorithm = ALGORITHM_REGISTRY[algo_name]
        self._current_config = config

        # Common hyperparameters
        policy = config.get("policy", "MlpPolicy")
        learning_rate = config.get("learning_rate", 1e-3)
        net_arch = config.get("net_arch", [64, 64])
        batch_size = config.get("batch_size", 128)
        verbose = config.get("verbose", 1)

        # Algorithm-specific parameters
        model_kwargs: Dict[str, Any] = {
            "policy": policy,
            "env": env,
            "verbose": verbose,
            "device": self.device,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "policy_kwargs": {"net_arch": net_arch},
            "seed": self.seed,
        }

        # Add algorithm-specific kwargs
        if algo_name == "DQN":
            model_kwargs["exploration_fraction"] = config.get("exploration_fraction", 0.4)
            model_kwargs["target_update_interval"] = config.get("target_update_interval", 500)
        elif algo_name in ("PPO", "A2C"):
            model_kwargs["n_steps"] = config.get("n_steps", 2048)
            if algo_name == "PPO":
                model_kwargs["n_epochs"] = config.get("n_epochs", 10)

        model = self._algorithm(**model_kwargs)
        self._current_model = model

        print(f"[SDK] Model built: {algo_name} | Device: {self.device} | Policy: {policy}")
        print(f"[SDK] Network architecture: {net_arch} | LR: {learning_rate}")

        return model

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        config: Dict[str, Any],
        visual: bool = False,
        algorithm: Optional[str] = None,
        save_checkpoints: bool = True,
    ) -> BaseAlgorithm:
        """
        Train an RL agent using the provided template configuration.

        Args:
            config: Template configuration dictionary.
            visual: If True, enables human rendering during training.
            algorithm: Algorithm name override.
            save_checkpoints: Whether to save periodic checkpoints.

        Returns:
            Trained Stable-Baselines3 model.
        """
        env_id = config["id"]
        steps = config["steps"]
        fps = config.get("fps_visual", 30) if visual else 0
        render_mode = "human" if visual else None
        n_envs = config.get("n_envs", 1)

        mode_str = "VISUAL/UNLOCKED" if visual else "BLIND/MAX_COMPUTE"
        print(f"\n[SDK] Training Mode: {mode_str} | Template: {env_id}")
        print(f"[SDK] Total timesteps: {steps} | Device: {self.device}")

        # Create environment
        env_factory = GymEnvFactory(config["env_factory"])
        env = env_factory.create_vec(render_mode=render_mode, fps=fps, n_envs=n_envs)

        # Build model
        model = self.build_model(config, env, algorithm=algorithm)

        # Setup callbacks
        callbacks = []
        if save_checkpoints:
            save_freq = config.get("save_freq", 10000)
            callbacks.append(CheckpointCallback(
                save_freq=save_freq,
                save_path=str(self.models_dir / env_id),
                name_prefix="checkpoint",
            ))
        callbacks.append(RewardLoggerCallback(log_freq=1000))

        # Train
        start_time = time.time()
        try:
            model.learn(
                total_timesteps=steps,
                callback=callbacks if callbacks else None,
                progress_bar=True,
            )

            # Save final model
            final_path = self.models_dir / f"{env_id}.zip"
            model.save(str(final_path))

            elapsed = time.time() - start_time
            print(f"\n[SDK] Training completed successfully!")
            print(f"[SDK] Model saved to: {final_path}")
            print(f"[SDK] Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        except KeyboardInterrupt:
            print("\n[SDK] Training interrupted by user. Saving partial model...")
            partial_path = self.models_dir / f"{env_id}_partial.zip"
            model.save(str(partial_path))
            print(f"[SDK] Partial model saved to: {partial_path}")

        finally:
            env.close()

        return model

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def infer(
        self,
        config: Dict[str, Any],
        model_path: Optional[str] = None,
        episodes: Optional[int] = None,
        deterministic: bool = True,
    ) -> None:
        """
        Run inference using a trained model.

        Args:
            config: Template configuration dictionary.
            model_path: Path to saved model. If None, uses default path.
            episodes: Number of episodes to run. None = infinite loop.
            deterministic: Use deterministic policy.
        """
        env_id = config["id"]
        fps = config.get("fps_visual", 30)
        render_mode = "human"

        # Resolve model path
        if model_path is None:
            model_path = self.models_dir / f"{env_id}.zip"
        else:
            model_path = Path(resolve_path(model_path))

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at '{model_path}'.\n"
                f"Tip: Run training first with: python -m stablegym --train --template <path>"
            )

        print(f"\n[SDK] Inference Mode | Template: {env_id} | Model: {model_path}")

        # Determine algorithm from config
        algo_name = config.get("algorithm", "DQN").upper()
        if algo_name not in ALGORITHM_REGISTRY:
            algo_name = "DQN"

        # Create environment
        env_factory = GymEnvFactory(config["env_factory"])
        env = env_factory.create(render_mode=render_mode, fps=fps)

        # Load model
        model_class = ALGORITHM_REGISTRY[algo_name]
        model = model_class.load(str(model_path), env=env, device=self.device)

        obs, info = env.reset()
        episode_count = 0
        total_reward = 0.0

        print(f"[SDK] Running inference... Press Ctrl+C to stop.\n")

        try:
            while episodes is None or episode_count < episodes:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    episode_count += 1
                    print(f"  Episode {episode_count} finished | Total reward: {total_reward:.2f}")
                    total_reward = 0.0
                    time.sleep(0.5)
                    obs, info = env.reset()

        except KeyboardInterrupt:
            print(f"\n[SDK] Inference stopped by user. Total episodes: {episode_count}")

        finally:
            env.close()

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def list_templates(self, templates_dir: str = "./templates") -> None:
        """List all available templates in the given directory."""
        path = Path(resolve_path(templates_dir))

        if not path.exists():
            print(f"[SDK] Templates directory not found: {path}")
            return

        print(f"\n[SDK] Available templates in '{templates_dir}':\n")
        for file in sorted(path.glob("*.py")):
            if file.name.startswith("_"):
                continue
            try:
                config = self.load_template(str(file))
                algo = config.get("algorithm", "DQN")
                steps = config.get("steps", "N/A")
                print(f"  - {file.name}")
                print(f"    ID: {config['id']} | Algorithm: {algo} | Steps: {steps}")
            except Exception as e:
                print(f"  - {file.name} [Error loading: {e}]")
        print()

    def evaluate(
        self,
        config: Dict[str, Any],
        model_path: Optional[str] = None,
        n_eval_episodes: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate a trained model over multiple episodes.

        Args:
            config: Template configuration dictionary.
            model_path: Path to saved model. None = auto-resolve.
            n_eval_episodes: Number of evaluation episodes.

        Returns:
            Dictionary with evaluation metrics (mean_reward, std_reward, etc.).
        """
        env_id = config["id"]

        if model_path is None:
            model_path = self.models_dir / f"{env_id}.zip"
        else:
            model_path = Path(resolve_path(model_path))

        algo_name = config.get("algorithm", "DQN").upper()
        if algo_name not in ALGORITHM_REGISTRY:
            algo_name = "DQN"

        # Create vectorized env for evaluation
        env_factory = GymEnvFactory(config["env_factory"])
        env = env_factory.create_vec(n_envs=1)

        model_class = ALGORITHM_REGISTRY[algo_name]
        model = model_class.load(str(model_path), env=env, device=self.device)

        episode_rewards = []

        for ep in range(n_eval_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                done = done[0] if isinstance(done, (list, np.ndarray)) else done

            episode_rewards.append(ep_reward)
            print(f"  Eval episode {ep + 1}/{n_eval_episodes}: reward = {ep_reward:.2f}")

        env.close()

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        max_reward = float(np.max(episode_rewards))
        min_reward = float(np.min(episode_rewards))

        results = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "n_episodes": n_eval_episodes,
        }

        print(f"\n[SDK] Evaluation Results:")
        print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"  Min / Max: {min_reward:.2f} / {max_reward:.2f}")

        return results
