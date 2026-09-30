"""
Base algorithm interface for Multi-Agent Reinforcement Learning.

Provides abstract base class for all MARL algorithms to ensure
consistent API across A2C, MAPPO, QMIX, MADDPG, etc.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BaseAlgorithm(ABC):
    """Abstract base class for all MARL algorithms."""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            device: Device for computation
            **kwargs: Algorithm-specific parameters
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        self.training_step = 0

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute training loss from batch of experiences.

        Args:
            batch: Dictionary containing:
                - observations: (batch_size, num_agents, obs_dim)
                - actions: (batch_size, num_agents, action_dim)
                - rewards: (batch_size, num_agents)
                - next_observations: (batch_size, num_agents, obs_dim)
                - dones: (batch_size, num_agents)
                - Additional algorithm-specific fields

        Returns:
            Dictionary of losses:
                - total_loss: Total loss for backpropagation
                - Additional component losses for logging
        """
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            batch: Dictionary of batched experiences

        Returns:
            Dictionary of training metrics for logging
        """
        pass

    @abstractmethod
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Get action(s) from policy given observation(s).

        Args:
            observation: (num_agents, obs_dim) or (batch, num_agents, obs_dim)
            deterministic: Whether to use deterministic (greedy) action selection
            **kwargs: Algorithm-specific arguments

        Returns:
            actions: (num_agents, action_dim) or (batch, num_agents, action_dim)
            info: Optional dictionary with additional information
                  (log_probs, values, messages, etc.)
        """
        pass

    def save(self, path: str):
        """
        Save algorithm state to disk.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'training_step': self.training_step,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
        }

        # Save model states (implemented by subclass)
        checkpoint.update(self._get_state_dict())

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """
        Load algorithm state from disk.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.training_step = checkpoint['training_step']
        self.learning_rate = checkpoint['learning_rate']
        self.gamma = checkpoint['gamma']

        # Load model states (implemented by subclass)
        self._load_state_dict(checkpoint)

        logger.info(f"Loaded checkpoint from {path}")

    @abstractmethod
    def _get_state_dict(self) -> Dict[str, Any]:
        """
        Get state dict for saving.

        Returns:
            Dictionary with model states and optimizer states
        """
        pass

    @abstractmethod
    def _load_state_dict(self, checkpoint: Dict[str, Any]):
        """
        Load state dict from checkpoint.

        Args:
            checkpoint: Loaded checkpoint dictionary
        """
        pass

    def set_training_mode(self, mode: bool = True):
        """
        Set training mode for all models.

        Args:
            mode: True for training, False for evaluation
        """
        pass

    def to(self, device: str):
        """
        Move all models to device.

        Args:
            device: Target device ('cpu', 'cuda', etc.)
        """
        self.device = device


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for on-policy algorithms (A2C, PPO, MAPPO).
    Assumes rollout buffer is used for data collection.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Args:
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            device: Computation device
            **kwargs: Additional parameters
        """
        super().__init__(learning_rate, gamma, device, **kwargs)
        self.gae_lambda = gae_lambda

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: (num_steps, num_agents) rewards
            values: (num_steps, num_agents) value estimates
            dones: (num_steps, num_agents) done flags
            next_value: (num_agents,) value of next state

        Returns:
            advantages: (num_steps, num_agents) advantages
            returns: (num_steps, num_agents) returns
        """
        num_steps = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros_like(next_value)

        # Compute backwards
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_val = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            # TD residual
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]

            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for off-policy algorithms (QMIX, MADDPG, SAC).
    Uses experience replay buffer.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Args:
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            device: Computation device
            **kwargs: Additional parameters
        """
        super().__init__(learning_rate, gamma, device, **kwargs)
        self.tau = tau

    def soft_update(self, target: nn.Module, source: nn.Module):
        """
        Soft update target network parameters.
        θ_target = τ * θ_source + (1 - τ) * θ_target

        Args:
            target: Target network
            source: Source network
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update(self, target: nn.Module, source: nn.Module):
        """
        Hard update target network parameters.
        θ_target = θ_source

        Args:
            target: Target network
            source: Source network
        """
        target.load_state_dict(source.state_dict())


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    activation: str = 'relu',
    output_activation: Optional[str] = None
) -> nn.Module:
    """
    Build multi-layer perceptron.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        activation: Activation function ('relu', 'tanh', 'silu')
        output_activation: Output activation (None, 'tanh', 'sigmoid')

    Returns:
        MLP module
    """
    activation_fn = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'silu': nn.SiLU,
    }[activation]

    layers = []

    # Input layer
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(activation_fn())

    # Hidden layers
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation_fn())

    # Output layer
    layers.append(nn.Linear(hidden_dim, output_dim))

    if output_activation == 'tanh':
        layers.append(nn.Tanh())
    elif output_activation == 'sigmoid':
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)
