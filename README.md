# StableGym SDK

A generalist framework for Reinforcement Learning that bridges **Stable-Baselines3** and **Pygame** environments through a unified, template-agnostic SDK. Create, train, and evaluate RL agents across diverse scenarios with a single command-line interface.

---

## Features

- **Template-agnostic SDK**: Any Gymnasium-compatible environment works out of the box
- **Multi-algorithm support**: DQN, PPO, A2C - configurable per template
- **Hardware control**: Explicit `--gpu` and `--cpu` flags
- **Visual training**: Optional rendering during training for debugging
- **Checkpoint system**: Automatic periodic model saving
- **Rich template library**: 5 built-in environments from classic control to custom games
- **Pure relative paths**: Works seamlessly in any directory structure

---

## Project Structure

```
stablegym-rl/
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── stablegym/                  # SDK package
│   ├── __init__.py             # Package exports
│   ├── __main__.py             # Module entry point
│   ├── sdk.py                  # Core SDK (StableGymSDK, GymEnvFactory)
│   ├── cli.py                  # Command-line interface
│   ├── callbacks.py            # Training callbacks
│   └── utils.py                # Utilities (device, seed, paths)
├── templates/                  # Environment templates
│   ├── __init__.py
│   ├── cartpole.py             # Classic CartPole-v1
│   ├── grid.py                 # Grid World navigation
│   ├── hide_and_seek.py        # Evasion survival game
│   ├── lunar_lander.py         # Lunar landing control
│   └── snake.py                # Snake game
└── models/                     # Saved models directory
    └── .gitkeep
```

---

## Installation

### 1. Clone and setup virtual environment

```bash
git clone https://github.com/yourusername/stablegym-rl.git
cd stablegym-rl

# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install as package (optional)

```bash
pip install -e .
```

This enables the `stablegym` command globally.

---

## Quick Start

### List available templates

```bash
python -m stablegym --list
```

### Training

Train with GPU (auto-detected):
```bash
python -m stablegym --train --template templates/grid.py
```

Train with explicit GPU:
```bash
python -m stablegym --train --template templates/cartpole.py --gpu
```

Train with CPU only:
```bash
python -m stablegym --train --template templates/hide_and_seek.py --cpu
```

Train with visual rendering:
```bash
python -m stablegym --train --template templates/snake.py --visual
```

Train with algorithm override:
```bash
python -m stablegym --train --template templates/cartpole.py --algorithm PPO --gpu
```

### Inference

Run a trained model:
```bash
python -m stablegym --infer --template templates/grid.py
```

### Evaluation

Evaluate over multiple episodes:
```bash
python -m stablegym --eval --template templates/cartpole.py --episodes 20
```

### Full CLI Reference

```bash
python -m stablegym --help
```

| Flag | Description |
|------|-------------|
| `--train` | Train a new agent |
| `--infer` | Run inference |
| `--eval` | Evaluate trained model |
| `--list` | List all templates |
| `--template PATH` | Path to template file (relative) |
| `--gpu` | Force CUDA/GPU usage |
| `--cpu` | Force CPU usage |
| `--visual` | Enable rendering during training |
| `--algorithm {DQN,PPO,A2C}` | Override algorithm |
| `--episodes N` | Number of eval episodes (default: 10) |
| `--model PATH` | Specific model file path |
| `--templates-dir PATH` | Custom templates directory |
| `--models-dir PATH` | Custom models directory |
| `--seed N` | Random seed for reproducibility |

---

## Templates

### CartPole (`templates/cartpole.py`)
Classic control problem. Balance a pole on a moving cart.
- **Algorithm**: DQN
- **Steps**: 50,000
- **Difficulty**: Beginner

### Grid World (`templates/grid.py`)
Navigate a grid to reach a goal while avoiding obstacles. Features heatmap visualization.
- **Algorithm**: DQN
- **Steps**: 100,000
- **Difficulty**: Beginner

### Hide and Seek (`templates/hide_and_seek.py`)
Evade multiple tracking enemies in a maze with walls. Survival-based rewards.
- **Algorithm**: DQN
- **Steps**: 200,000
- **Difficulty**: Intermediate

### Lunar Lander (`templates/lunar_lander.py`)
Physics-based spacecraft landing. Continuous control challenge.
- **Algorithm**: PPO
- **Steps**: 500,000
- **Difficulty**: Advanced

### Snake (`templates/snake.py`)
Classic snake game. Eat food, grow longer, avoid walls and self-collision.
- **Algorithm**: DQN
- **Steps**: 300,000
- **Difficulty**: Intermediate

---

## Creating Custom Templates

Create a new Python file in the `templates/` folder:

```python
"""My Custom Environment Template"""

import gymnasium as gym


def env_factory(render_mode=None, fps=0, **kwargs):
    """Factory function - must accept render_mode and fps."""
    return gym.make("MyEnv-v0", render_mode=render_mode)


TEMPLATE_CONFIG = {
    "id": "my_custom_env",           # Unique identifier
    "env_factory": env_factory,      # Environment factory function
    "algorithm": "PPO",              # RL algorithm: DQN, PPO, or A2C
    "policy": "MlpPolicy",           # Policy network type
    "steps": 100_000,                # Training timesteps
    "fps_visual": 30,                # Rendering FPS during inference
    "net_arch": [128, 128],          # Neural network hidden layers
    "learning_rate": 3e-4,           # Learning rate
    "batch_size": 128,               # Batch size
    "save_freq": 10_000,             # Checkpoint frequency
    "verbose": 1,                    # Verbosity level
}
```

Train your custom template:
```bash
python -m stablegym --train --template templates/my_custom_env.py
```

---

## SDK Usage (Programmatic)

Use the SDK directly in Python scripts:

```python
from stablegym import StableGymSDK

# Initialize SDK
sdk = StableGymSDK(device="auto", seed=42)

# Load template
config = sdk.load_template("templates/cartpole.py")

# Train
model = sdk.train(config, visual=False)

# Evaluate
results = sdk.evaluate(config, n_eval_episodes=10)

# Run inference
sdk.infer(config, episodes=5)
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any modern x86_64 | Multi-core |
| RAM | 4 GB | 8 GB+ |
| GPU | Optional | NVIDIA with CUDA |
| OS | Linux/macOS/Windows | Linux |

---

## License

This project is licensed under the MIT License.

---

## Contributing

Contributions are welcome! To add a new template:

1. Create a new `.py` file in `templates/`
2. Implement the `env_factory` function and `TEMPLATE_CONFIG`
3. Test with: `python -m stablegym --train --template templates/your_template.py`
4. Submit a pull request
