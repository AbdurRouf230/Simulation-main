"""
Multi-Agent Proximal Policy Optimization (MAPPO).

MAPPO combines PPO with centralized training and decentralized execution (CTDE).
- Decentralized actors: Each agent sees only local observations
- Centralized critic: Critic sees global state for better value estimation
- PPO clipping: Ensures stable policy updates
- GAE: Generalized Advantage Estimation for variance reduction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Dict, Tuple, Optional, Any
import logging

from .base_algorithm import OnPolicyAlgorithm, build_mlp

logger = logging.getLogger(__name__)


class MAPPOActor(nn.Module):
    """
    Decentralized actor network for MAPPO.
    Each agent uses this to select actions based on local observations.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        continuous: bool = True
    ):
        """
        Args:
            obs_dim: Dimension of local observation
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            continuous: Whether action space is continuous
        """
        super().__init__()
        self.continuous = continuous

        # Policy network
        self.policy_net = build_mlp(
            obs_dim, hidden_dim, action_dim,
            num_layers=num_layers,
            activation='tanh'
        )

        # For continuous actions, also output log std
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor.

        Args:
            obs: (batch, obs_dim) or (batch, num_agents, obs_dim)

        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        # Get policy output
        policy_out = self.policy_net(obs)

        if self.continuous:
            # Gaussian policy
            mean = policy_out
            std = torch.exp(self.log_std)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            # Categorical policy
            logits = policy_out
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions.

        Args:
            obs: (batch, obs_dim) observations
            actions: (batch, action_dim) actions to evaluate

        Returns:
            log_probs: (batch,) log probabilities
            entropy: (batch,) entropy of distribution
        """
        policy_out = self.policy_net(obs)

        if self.continuous:
            mean = policy_out
            std = torch.exp(self.log_std)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = policy_out
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        return log_probs, entropy


class MAPPOCritic(nn.Module):
    """
    Centralized critic network for MAPPO.
    Uses global state to estimate value function.
    """

    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        """
        Args:
            global_state_dim: Dimension of global state
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()

        self.value_net = build_mlp(
            global_state_dim, hidden_dim, 1,
            num_layers=num_layers,
            activation='tanh'
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of global state.

        Args:
            global_state: (batch, global_state_dim)

        Returns:
            value: (batch, 1) estimated value
        """
        return self.value_net(global_state)


class MAPPOPolicy(nn.Module):
    """
    Combined actor-critic policy for MAPPO.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        continuous: bool = True
    ):
        """
        Args:
            obs_dim: Local observation dimension
            action_dim: Action dimension
            global_state_dim: Global state dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            continuous: Continuous action space
        """
        super().__init__()

        self.actor = MAPPOActor(
            obs_dim, action_dim, hidden_dim, num_layers, continuous
        )
        self.critic = MAPPOCritic(
            global_state_dim, hidden_dim, num_layers
        )

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from policy.

        Args:
            obs: (batch, obs_dim) local observations
            deterministic: Use mean action (for evaluation)

        Returns:
            action: (batch, action_dim)
            log_prob: (batch,)
        """
        if deterministic and hasattr(self.actor, 'continuous') and self.actor.continuous:
            # For continuous, return mean
            mean = self.actor.policy_net(obs)
            return mean, torch.zeros(mean.shape[0], device=mean.device)
        else:
            return self.actor(obs)

    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate.

        Args:
            global_state: (batch, global_state_dim)

        Returns:
            value: (batch, 1)
        """
        return self.critic(global_state)


class MAPPOTrainer(OnPolicyAlgorithm):
    """
    Multi-Agent PPO trainer implementing centralized training, decentralized execution.

    Key features:
    - PPO clipping (epsilon=0.2)
    - GAE for advantage estimation (lambda=0.95)
    - Value function clipping
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        num_agents: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        continuous: bool = True,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Args:
            obs_dim: Local observation dimension
            action_dim: Action dimension
            global_state_dim: Global state dimension
            num_agents: Number of agents
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            continuous: Continuous action space
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
            device: Device
        """
        super().__init__(learning_rate, gamma, gae_lambda, device, **kwargs)

        self.num_agents = num_agents
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Create policy
        self.policy = MAPPOPolicy(
            obs_dim, action_dim, global_state_dim,
            hidden_dim, num_layers, continuous
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss.

        Args:
            batch: Dictionary with:
                - local_obs: (batch, num_agents, obs_dim)
                - global_state: (batch, global_state_dim)
                - actions: (batch, num_agents, action_dim)
                - old_log_probs: (batch, num_agents)
                - advantages: (batch, num_agents)
                - returns: (batch, num_agents)
                - old_values: (batch, num_agents)

        Returns:
            Dictionary of losses
        """
        local_obs = batch['local_obs']
        global_state = batch['global_state']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        old_values = batch.get('old_values')

        batch_size, num_agents = local_obs.shape[:2]

        # Flatten batch and agents for processing
        local_obs_flat = local_obs.reshape(-1, local_obs.shape[-1])
        actions_flat = actions.reshape(-1, actions.shape[-1])
        old_log_probs_flat = old_log_probs.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)

        # Evaluate actions
        new_log_probs, entropy = self.policy.actor.evaluate_actions(
            local_obs_flat, actions_flat
        )

        # Compute ratio for PPO
        ratio = torch.exp(new_log_probs - old_log_probs_flat)

        # Normalize advantages
        advantages_norm = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # PPO policy loss with clipping
        policy_loss_1 = advantages_norm * ratio
        policy_loss_2 = advantages_norm * torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Value loss
        values = self.policy.critic(global_state).squeeze(-1)  # (batch,)

        # Expand values to match agents
        values_expanded = values.unsqueeze(1).expand(-1, num_agents).reshape(-1)

        if old_values is not None:
            # Value clipping (from PPO paper)
            old_values_flat = old_values.reshape(-1)
            values_clipped = old_values_flat + torch.clamp(
                values_expanded - old_values_flat,
                -self.clip_epsilon, self.clip_epsilon
            )
            value_loss_1 = F.mse_loss(values_expanded, returns_flat)
            value_loss_2 = F.mse_loss(values_clipped, returns_flat)
            value_loss = torch.max(value_loss_1, value_loss_2)
        else:
            value_loss = F.mse_loss(values_expanded, returns_flat)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss +
            self.entropy_coef * entropy_loss
        )

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': -entropy_loss,
            'approx_kl': ((ratio - 1) - torch.log(ratio)).mean(),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            batch: Batch of experiences

        Returns:
            Training metrics
        """
        # Compute losses
        losses = self.compute_loss(batch)

        # Optimize
        self.optimizer.zero_grad()
        losses['total_loss'].backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.max_grad_norm
        )

        self.optimizer.step()

        self.training_step += 1

        # Return metrics
        metrics = {
            'total_loss': losses['total_loss'].item(),
            'policy_loss': losses['policy_loss'].item(),
            'value_loss': losses['value_loss'].item(),
            'entropy': losses['entropy'].item(),
            'approx_kl': losses['approx_kl'].item(),
            'grad_norm': grad_norm.item(),
        }

        return metrics

    def get_action(
        self,
        observation: torch.Tensor,
        global_state: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Get actions for all agents.

        Args:
            observation: (num_agents, obs_dim) local observations
            global_state: (global_state_dim,) global state (for value)
            deterministic: Use deterministic actions
            **kwargs: Additional arguments

        Returns:
            actions: (num_agents, action_dim)
            info: Dictionary with log_probs, values, etc.
        """
        self.policy.eval()

        with torch.no_grad():
            # Get actions
            actions, log_probs = self.policy.get_action(observation, deterministic)

            # Get value if global state provided
            if global_state is not None:
                if global_state.dim() == 1:
                    global_state = global_state.unsqueeze(0)
                values = self.policy.get_value(global_state)
            else:
                values = None

        info = {
            'log_probs': log_probs,
            'values': values,
        }

        return actions, info

    def _get_state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        return {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

    def _load_state_dict(self, checkpoint: Dict[str, Any]):
        """Load state dict from checkpoint."""
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def set_training_mode(self, mode: bool = True):
        """Set training mode."""
        self.policy.train(mode)

    def to(self, device: str):
        """Move to device."""
        super().to(device)
        self.policy = self.policy.to(device)
        return self
