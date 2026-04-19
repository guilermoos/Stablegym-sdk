"""
Templates package for StableGym SDK.

Each template defines a TEMPLATE_CONFIG dictionary with environment
configuration, training hyperparameters, and rendering settings.

To create a new template, copy any existing template and modify:
    - id: Unique identifier for your environment
    - env_factory: Factory function that returns a gym.Env instance
    - steps: Total training timesteps
    - net_arch: Neural network hidden layers
    - Other hyperparameters as needed
"""
