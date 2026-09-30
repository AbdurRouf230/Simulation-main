"""
MADDPG: Multi-Agent Deep Deterministic Policy Gradient.

MADDPG extends DDPG to multi-agent environments using centralized training
with decentralized execution (CTDE). Each agent has its own actor-critic,
but critics are trained with global information during training.

Key features:
- Continuous action spaces
- Centralized critics (see all observations and actions)
- Decentralized actors (see only local observations)
- Experience replay with target networks
- Ornstein-Uhlenbeck noise for exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .base_algorithm import OffPolicyAlgorithm, build_mlp


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2
    ):
        """
        Args:
            size: Dimension of noise
            mu: Long-term mean
            theta: Mean reversion rate
            sigma: Noise scale
            dt: Time step
        """
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state."""
        self.state = np.ones(self.size) * self.mu

    def sample(self) -> np.ndarray:
        """Sample noise."""
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.state += dx
        return self.state


class MADDPGActor(nn.Module):
    """Decentralized actor network for MADDPG."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_action: float = 1.0
    ):
        """
        Args:
            obs_dim: Local observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            max_action: Maximum action value
        """
        super().__init__()
        self.max_action = max_action

        self.actor = build_mlp(
            obs_dim, hidden_dim, action_dim,
            num_layers=num_layers,
            activation='relu',
            output_activation='tanh'
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor.

        Args:
            obs: (batch, obs_dim) local observations

        Returns:
            actions: (batch, action_dim) actions in [-max_action, max_action]
        """
        return self.max_action * self.actor(obs)


class MADDPGCritic(nn.Module):
    """Centralized critic network for MADDPG."""

    def __init__(
        self,
        total_obs_dim: int,
        total_action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        """
        Args:
            total_obs_dim: Sum of all agents' observation dimensions
            total_action_dim: Sum of all agents' action dimensions
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()

        self.critic = build_mlp(
            total_obs_dim + total_action_dim, hidden_dim, 1,
            num_layers=num_layers,
            activation='relu'
        )

    def forward(
        self,
        all_obs: torch.Tensor,
        all_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through critic.

        Args:
            all_obs: (batch, total_obs_dim) concatenated observations
            all_actions: (batch, total_action_dim) concatenated actions

        Returns:
            q_value: (batch, 1) Q-value
        """
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.critic(x)


class MADDPGAgent(nn.Module):
    """Single MADDPG agent with actor and critic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        total_obs_dim: int,
        total_action_dim: int,
        hidden_dim: int = 256,
        max_action: float = 1.0
    ):
        """
        Args:
            obs_dim: Local observation dimension
            action_dim: Local action dimension
            total_obs_dim: Global observation dimension (sum of all agents)
            total_action_dim: Global action dimension (sum of all agents)
            hidden_dim: Hidden dimension
            max_action: Maximum action value
        """
        super().__init__()

        self.actor = MADDPGActor(obs_dim, action_dim, hidden_dim, max_action=max_action)
        self.critic = MADDPGCritic(total_obs_dim, total_action_dim, hidden_dim)

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from actor."""
        return self.actor(obs)

    def get_q_value(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """Get Q-value from critic."""
        return self.critic(all_obs, all_actions)


class MADDPGTrainer(OffPolicyAlgorithm):
    """
    MADDPG trainer for multi-agent continuous control.

    Key features:
    - Centralized training, decentralized execution
    - Continuous action spaces with OU noise exploration
    - Target networks with soft updates
    - Experience replay
    """

    def __init__(
        self,
        obs_dims: List[int],
        action_dims: List[int],
        num_agents: int,
        hidden_dim: int = 256,
        max_action: float = 1.0,
        learning_rate_actor: float = 1e-4,
        learning_rate_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        noise_theta: float = 0.15,
        noise_sigma: float = 0.2,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Args:
            obs_dims: List of observation dimensions for each agent
            action_dims: List of action dimensions for each agent
            num_agents: Number of agents
            hidden_dim: Hidden dimension
            max_action: Maximum action value
            learning_rate_actor: Actor learning rate
            learning_rate_critic: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            noise_theta: OU noise theta parameter
            noise_sigma: OU noise sigma parameter
            device: Device
        """
        super().__init__(learning_rate_actor, gamma, tau, device, **kwargs)

        self.num_agents = num_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.total_obs_dim = sum(obs_dims)
        self.total_action_dim = sum(action_dims)
        self.max_action = max_action

        # Create agents
        self.agents = nn.ModuleList()
        self.target_agents = nn.ModuleList()

        for i in range(num_agents):
            agent = MADDPGAgent(
                obs_dims[i], action_dims[i],
                self.total_obs_dim, self.total_action_dim,
                hidden_dim, max_action
            ).to(device)

            target_agent = MADDPGAgent(
                obs_dims[i], action_dims[i],
                self.total_obs_dim, self.total_action_dim,
                hidden_dim, max_action
            ).to(device)

            self.agents.append(agent)
            self.target_agents.append(target_agent)

            # Initialize target networks
            self.hard_update(target_agent, agent)

        # Optimizers
        self.actor_optimizers = [
            torch.optim.Adam(agent.actor.parameters(), lr=learning_rate_actor)
            for agent in self.agents
        ]

        self.critic_optimizers = [
            torch.optim.Adam(agent.critic.parameters(), lr=learning_rate_critic)
            for agent in self.agents
        ]

        # Exploration noise
        self.noise_generators = [
            OUNoise(action_dims[i], theta=noise_theta, sigma=noise_sigma)
            for i in range(num_agents)
        ]

    def get_action(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        add_noise: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Get actions for all agents.

        Args:
            observations: (num_agents, obs_dim) or list of (obs_dim,) tensors
            deterministic: Use deterministic actions
            add_noise: Add exploration noise

        Returns:
            actions: (num_agents, action_dim) actions
            info: Additional information
        """
        actions = []

        for i, agent in enumerate(self.agents):
            agent.eval()

            with torch.no_grad():
                if isinstance(observations, list):
                    obs = observations[i]
                else:
                    obs = observations[i]

                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)

                action = agent.get_action(obs)
                action = action.squeeze(0)  # Remove batch dimension

                # Add exploration noise
                if not deterministic and add_noise:
                    noise = torch.tensor(
                        self.noise_generators[i].sample(),
                        dtype=torch.float32,
                        device=action.device
                    )
                    action = action + noise
                    action = torch.clamp(action, -self.max_action, self.max_action)

                actions.append(action)

        actions = torch.stack(actions)

        info = {
            'noise_level': self.noise_generators[0].sigma if add_noise else 0.0,
        }

        return actions, info

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute MADDPG training loss.

        Args:
            batch: Dictionary with:
                - observations: (batch, num_agents, obs_dim)
                - actions: (batch, num_agents, action_dim)
                - rewards: (batch, num_agents)
                - next_observations: (batch, num_agents, obs_dim)
                - dones: (batch, num_agents)

        Returns:
            Dictionary of losses for each agent
        """
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']

        batch_size = observations.shape[0]

        # Flatten observations and actions
        all_obs = observations.view(batch_size, -1)  # (batch, total_obs_dim)
        all_actions = actions.view(batch_size, -1)  # (batch, total_action_dim)
        all_next_obs = next_observations.view(batch_size, -1)

        losses = {}

        for agent_idx in range(self.num_agents):
            # === Critic Loss ===

            # Current Q-values
            q_current = self.agents[agent_idx].get_q_value(all_obs, all_actions)

            # Target Q-values
            with torch.no_grad():
                # Get next actions from target actors
                next_actions = []
                for i, target_agent in enumerate(self.target_agents):
                    next_obs_i = next_observations[:, i, :]
                    next_action_i = target_agent.get_action(next_obs_i)
                    next_actions.append(next_action_i)

                all_next_actions = torch.cat(next_actions, dim=-1)

                # Target Q-value
                q_next = self.target_agents[agent_idx].get_q_value(
                    all_next_obs, all_next_actions
                )

                # TD target
                target_q = rewards[:, agent_idx:agent_idx+1] + \
                          self.gamma * (1 - dones[:, agent_idx:agent_idx+1]) * q_next

            critic_loss = F.mse_loss(q_current, target_q)

            # === Actor Loss ===

            # Get actions from current actors
            current_actions = []
            for i, agent in enumerate(self.agents):
                if i == agent_idx:
                    # Use current agent's actor
                    current_action = agent.get_action(observations[:, i, :])
                else:
                    # Use other agents' actions from batch (detached)
                    current_action = actions[:, i, :].detach()
                current_actions.append(current_action)

            all_current_actions = torch.cat(current_actions, dim=-1)

            # Actor loss (negative Q-value)
            actor_q = self.agents[agent_idx].get_q_value(all_obs, all_current_actions)
            actor_loss = -actor_q.mean()

            losses[f'critic_loss_{agent_idx}'] = critic_loss
            losses[f'actor_loss_{agent_idx}'] = actor_loss
            losses[f'q_mean_{agent_idx}'] = q_current.mean()

        # Total loss (for reporting)
        total_critic_loss = sum(losses[f'critic_loss_{i}'] for i in range(self.num_agents))
        total_actor_loss = sum(losses[f'actor_loss_{i}'] for i in range(self.num_agents))
        losses['total_critic_loss'] = total_critic_loss
        losses['total_actor_loss'] = total_actor_loss

        return losses

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        losses = self.compute_loss(batch)

        # Train each agent
        for agent_idx in range(self.num_agents):
            # Update critic
            self.critic_optimizers[agent_idx].zero_grad()
            losses[f'critic_loss_{agent_idx}'].backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.agents[agent_idx].critic.parameters(), 1.0
            )
            self.critic_optimizers[agent_idx].step()

            # Update actor
            self.actor_optimizers[agent_idx].zero_grad()
            losses[f'actor_loss_{agent_idx}'].backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.agents[agent_idx].actor.parameters(), 1.0
            )
            self.actor_optimizers[agent_idx].step()

            # Soft update target networks
            self.soft_update(
                self.target_agents[agent_idx].actor,
                self.agents[agent_idx].actor
            )
            self.soft_update(
                self.target_agents[agent_idx].critic,
                self.agents[agent_idx].critic
            )

        self.training_step += 1

        # Return metrics
        metrics = {
            'total_critic_loss': losses['total_critic_loss'].item(),
            'total_actor_loss': losses['total_actor_loss'].item(),
        }

        for i in range(self.num_agents):
            metrics[f'critic_loss_{i}'] = losses[f'critic_loss_{i}'].item()
            metrics[f'actor_loss_{i}'] = losses[f'actor_loss_{i}'].item()
            metrics[f'q_mean_{i}'] = losses[f'q_mean_{i}'].item()

        return metrics

    def _get_state_dict(self) -> Dict[str, any]:
        """Get state dict for saving."""
        state_dict = {
            'agents_state_dict': [agent.state_dict() for agent in self.agents],
            'target_agents_state_dict': [agent.state_dict() for agent in self.target_agents],
            'actor_optimizers_state_dict': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers_state_dict': [opt.state_dict() for opt in self.critic_optimizers],
        }
        return state_dict

    def _load_state_dict(self, checkpoint: Dict[str, any]):
        """Load state dict from checkpoint."""
        for i, (agent, target_agent) in enumerate(zip(self.agents, self.target_agents)):
            agent.load_state_dict(checkpoint['agents_state_dict'][i])
            target_agent.load_state_dict(checkpoint['target_agents_state_dict'][i])
            self.actor_optimizers[i].load_state_dict(
                checkpoint['actor_optimizers_state_dict'][i]
            )
            self.critic_optimizers[i].load_state_dict(
                checkpoint['critic_optimizers_state_dict'][i]
            )

    def set_training_mode(self, mode: bool = True):
        """Set training mode."""
        for agent in self.agents:
            agent.train(mode)

    def to(self, device: str):
        """Move to device."""
        super().to(device)
        for i in range(self.num_agents):
            self.agents[i] = self.agents[i].to(device)
            self.target_agents[i] = self.target_agents[i].to(device)
        return self

    def reset_noise(self):
        """Reset exploration noise."""
        for noise_gen in self.noise_generators:
            noise_gen.reset()