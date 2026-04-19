"""
Command-Line Interface for StableGym SDK.

Provides a unified CLI for training, inference, evaluation,
and template management across all RL environments.

Usage Examples:
    # Training
    python -m stablegym --train --template templates/grid.py
    python -m stablegym --train --template templates/grid.py --visual
    python -m stablegym --train --template templates/cartpole.py --gpu
    python -m stablegym --train --template templates/cartpole.py --cpu

    # Inference
    python -m stablegym --infer --template templates/cartpole.py

    # Evaluation
    python -m stablegym --eval --template templates/cartpole.py --episodes 10

    # List available templates
    python -m stablegym --list
"""

import argparse
import sys

from .sdk import StableGymSDK


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="stablegym",
        description="StableGym SDK - Reinforcement Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --train --template templates/grid.py
  %(prog)s --train --template templates/cartpole.py --gpu --visual
  %(prog)s --infer --template templates/hide_and_seek.py
  %(prog)s --eval --template templates/grid.py --episodes 20
  %(prog)s --list
        """,
    )

    # Mutually exclusive action group
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--train", "--training",
        action="store_true",
        help="Train a new RL agent",
    )
    action_group.add_argument(
        "--infer", "--inference",
        action="store_true",
        help="Run inference with a trained model",
    )
    action_group.add_argument(
        "--eval", "--evaluate",
        action="store_true",
        help="Evaluate a trained model over multiple episodes",
    )
    action_group.add_argument(
        "--list",
        action="store_true",
        help="List all available templates",
    )

    # Template path
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Path to the template Python file (relative or absolute)",
    )

    # Device selection (GPU / CPU)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--gpu",
        action="store_const",
        const="cuda",
        dest="device",
        default="auto",
        help="Force GPU (CUDA) usage for training/inference",
    )
    device_group.add_argument(
        "--cpu",
        action="store_const",
        const="cpu",
        dest="device",
        default="auto",
        help="Force CPU usage for training/inference",
    )

    # Training options
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Enable visual rendering during training (slower)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["DQN", "PPO", "A2C"],
        help="Override the RL algorithm specified in the template",
    )

    # Evaluation options
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )

    # Model path override
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a specific model file to load",
    )

    # Templates directory
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="./templates",
        help="Directory containing templates (default: ./templates)",
    )

    # Models directory
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory to save/load models (default: ./models)",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments for consistency."""
    if not args.list and not args.template:
        print("[Error] --template is required (except when using --list).")
        sys.exit(1)

    if args.visual and not args.train:
        print("[Warning] --visual only applies to training mode. Ignoring.")


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Handle --list action
    if args.list:
        sdk = StableGymSDK(models_dir=args.models_dir)
        sdk.list_templates(templates_dir=args.templates_dir)
        return

    # Initialize SDK
    sdk = StableGymSDK(
        device=args.device,
        seed=args.seed,
        models_dir=args.models_dir,
    )

    # Load template
    try:
        config = sdk.load_template(args.template)
    except Exception as e:
        print(f"[Error] Failed to load template: {e}")
        sys.exit(1)

    # Execute action
    try:
        if args.train:
            sdk.train(
                config=config,
                visual=args.visual,
                algorithm=args.algorithm,
            )

        elif args.infer:
            sdk.infer(
                config=config,
                model_path=args.model,
                episodes=None,  # Infinite loop until Ctrl+C
                deterministic=True,
            )

        elif args.eval:
            sdk.evaluate(
                config=config,
                model_path=args.model,
                n_eval_episodes=args.episodes,
            )

    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Info] Operation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"[Error] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
