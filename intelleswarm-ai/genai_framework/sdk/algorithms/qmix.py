"""
QMIX: Monotonic Value Function Factorization for Deep Multi-Agent Reinforcement Learning.

QMIX learns a centralized but factored Q-function that represents the joint action-value
while maintaining monotonicity constraints. This allows for decentralized execution
while benefiting from centralized training.

Key components:
- Individual Q-networks for each agent (decentralized execution)
- Mixing network that combines individual Q-values (centralized training)
- Hypernetwork that generates mixing weights
- Monotonicity constraint: ∂Q_tot/∂Q_i ≥ 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np

from .base_algorithm import OffPolicyAlgorithm, build_mlp


class QMIXAgentNetwork(nn.Module):
    """Individual Q-network for each agent in QMIX."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        """
        Args:
            obs_dim: Local observation dimension
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()

        self.q_network = build_mlp(
            obs_dim, hidden_dim, action_dim,
            num_layers=num_layers,
            activation='relu'
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.

        Args:
            obs: (batch, obs_dim) local observations

        Returns:
            q_values: (batch, action_dim) Q-values for all actions
        """
        return self.q_network(obs)


class QMIXHyperNetwork(nn.Module):
    """Hypernetwork that generates weights for the mixing network."""

    def __init__(
        self,
        state_dim: int,
        num_agents: int,
        hidden_dim: int = 32,
        mixing_embed_dim: int = 32
    ):
        """
        Args:
            state_dim: Global state dimension
            num_agents: Number of agents
            hidden_dim: Hidden dimension for hypernetwork
            mixing_embed_dim: Embedding dimension for mixing network
        """
        super().__init__()
        self.num_agents = num_agents
        self.mixing_embed_dim = mixing_embed_dim

        # Hypernetwork for first layer weights (positive constraint via abs)
        self.hyper_w1 = build_mlp(
            state_dim, hidden_dim, num_agents * mixing_embed_dim,
            num_layers=2, activation='relu'
        )

        # Hypernetwork for first layer biases
        self.hyper_b1 = build_mlp(
            state_dim, hidden_dim, mixing_embed_dim,
            num_layers=2, activation='relu'
        )

        # Hypernetwork for second layer weights (positive constraint via abs)
        self.hyper_w2 = build_mlp(
            state_dim, hidden_dim, mixing_embed_dim,
            num_layers=2, activation='relu'
        )

        # Hypernetwork for second layer bias
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate mixing network weights from global state.

        Args:
            state: (batch, state_dim) global state

        Returns:
            w1: (batch, num_agents, mixing_embed_dim) first layer weights
            b1: (batch, mixing_embed_dim) first layer bias
            w2: (batch, mixing_embed_dim) second layer weights
            b2: (batch, 1) second layer bias
        """
        batch_size = state.shape[0]

        # Generate weights (apply abs for monotonicity constraint)
        w1 = torch.abs(self.hyper_w1(state))  # Positive weights
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)

        b1 = self.hyper_b1(state)  # (batch, mixing_embed_dim)

        w2 = torch.abs(self.hyper_w2(state))  # Positive weights
        w2 = w2.view(batch_size, self.mixing_embed_dim)

        b2 = self.hyper_b2(state)  # (batch, 1)

        return w1, b1, w2, b2


class QMIXMixingNetwork(nn.Module):
    """Mixing network that combines individual Q-values."""

    def __init__(
        self,
        state_dim: int,
        num_agents: int,
        mixing_embed_dim: int = 32
    ):
        """
        Args:
            state_dim: Global state dimension
            num_agents: Number of agents
            mixing_embed_dim: Embedding dimension for mixing
        """
        super().__init__()
        self.num_agents = num_agents
        self.mixing_embed_dim = mixing_embed_dim

        self.hyper_net = QMIXHyperNetwork(
            state_dim, num_agents, mixing_embed_dim=mixing_embed_dim
        )

    def forward(
        self,
        agent_q_values: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix individual Q-values into joint Q-value.

        Args:
            agent_q_values: (batch, num_agents) selected Q-values for each agent
            state: (batch, state_dim) global state

        Returns:
            q_tot: (batch, 1) mixed total Q-value
        """
        batch_size = agent_q_values.shape[0]

        # Get mixing weights from hypernetwork
        w1, b1, w2, b2 = self.hyper_net(state)

        # First layer: Q-values -> embedding
        # agent_q_values: (batch, num_agents) -> (batch, num_agents, 1)
        agent_q_values = agent_q_values.unsqueeze(-1)

        # Matrix multiplication: (batch, 1, num_agents) @ (batch, num_agents, mixing_embed_dim)
        hidden = torch.bmm(agent_q_values.transpose(1, 2), w1)  # (batch, 1, mixing_embed_dim)
        hidden = hidden.squeeze(1)  # (batch, mixing_embed_dim)
        hidden = hidden + b1  # Add bias
        hidden = F.elu(hidden)  # Non-linearity

        # Second layer: embedding -> scalar
        q_tot = torch.sum(hidden * w2, dim=-1, keepdim=True) + b2  # (batch, 1)

        return q_tot


class QMIXNetwork(nn.Module):
    """Complete QMIX network combining agent networks and mixing network."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 64,
        mixing_embed_dim: int = 32
    ):
        """
        Args:
            obs_dim: Local observation dimension
            state_dim: Global state dimension
            action_dim: Number of actions per agent
            num_agents: Number of agents
            hidden_dim: Hidden dimension for agent networks
            mixing_embed_dim: Embedding dimension for mixing
        """
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim

        # Individual agent Q-networks
        self.agent_networks = nn.ModuleList([
            QMIXAgentNetwork(obs_dim, action_dim, hidden_dim)
            for _ in range(num_agents)
        ])

        # Mixing network
        self.mixing_net = QMIXMixingNetwork(
            state_dim, num_agents, mixing_embed_dim
        )

    def forward(
        self,
        local_obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through QMIX network.

        Args:
            local_obs: (batch, num_agents, obs_dim) local observations
            state: (batch, state_dim) global state
            actions: (batch, num_agents) actions taken (if None, return all Q-values)

        Returns:
            q_tot: (batch, 1) total Q-value (if actions provided)
            agent_q_values: (batch, num_agents, action_dim) individual Q-values
        """
        batch_size = local_obs.shape[0]

        # Get Q-values from each agent
        agent_q_values = []
        for i, agent_net in enumerate(self.agent_networks):
            q_vals = agent_net(local_obs[:, i, :])  # (batch, action_dim)
            agent_q_values.append(q_vals)

        agent_q_values = torch.stack(agent_q_values, dim=1)  # (batch, num_agents, action_dim)

        if actions is None:
            # Return all Q-values for action selection
            return None, agent_q_values

        # Select Q-values for taken actions
        selected_q_values = agent_q_values.gather(
            2, actions.unsqueeze(-1)
        ).squeeze(-1)  # (batch, num_agents)

        # Mix Q-values
        q_tot = self.mixing_net(selected_q_values, state)  # (batch, 1)

        return q_tot, agent_q_values


class QMIXTrainer(OffPolicyAlgorithm):
    """
    QMIX trainer implementing monotonic value function factorization.

    Key features:
    - Centralized training with decentralized execution
    - Monotonic mixing network ensures consistency
    - Experience replay with target networks
    - Epsilon-greedy exploration
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 64,
        mixing_embed_dim: int = 32,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.001,
        target_update_interval: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Args:
            obs_dim: Local observation dimension
            state_dim: Global state dimension
            action_dim: Number of actions
            num_agents: Number of agents
            hidden_dim: Hidden dimension
            mixing_embed_dim: Mixing embedding dimension
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            target_update_interval: Hard update interval for target network
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay steps
            device: Device
        """
        super().__init__(learning_rate, gamma, tau, device, **kwargs)

        self.num_agents = num_agents
        self.action_dim = action_dim
        self.target_update_interval = target_update_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_network = QMIXNetwork(
            obs_dim, state_dim, action_dim, num_agents,
            hidden_dim, mixing_embed_dim
        ).to(device)

        self.target_network = QMIXNetwork(
            obs_dim, state_dim, action_dim, num_agents,
            hidden_dim, mixing_embed_dim
        ).to(device)

        # Initialize target network
        self.hard_update(self.target_network, self.q_network)

        # Optimizer
        self.optimizer = torch.optim.RMSprop(
            self.q_network.parameters(),
            lr=learning_rate,
            alpha=0.99,
            eps=1e-5
        )

        self.training_step = 0

    def get_epsilon(self) -> float:
        """Get current exploration epsilon."""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 np.exp(-self.training_step / self.epsilon_decay)
        return epsilon

    def get_action(
        self,
        local_obs: torch.Tensor,
        deterministic: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Get actions using epsilon-greedy policy.

        Args:
            local_obs: (num_agents, obs_dim) local observations
            deterministic: Use greedy actions (no exploration)

        Returns:
            actions: (num_agents,) selected actions
            info: Additional information
        """
        self.q_network.eval()

        with torch.no_grad():
            if local_obs.dim() == 2:
                local_obs = local_obs.unsqueeze(0)  # Add batch dimension

            _, agent_q_values = self.q_network(local_obs, None, None)
            agent_q_values = agent_q_values[0]  # Remove batch dimension

            if deterministic or np.random.random() > self.get_epsilon():
                # Greedy actions
                actions = torch.argmax(agent_q_values, dim=-1)
            else:
                # Random actions
                actions = torch.randint(0, self.action_dim, (self.num_agents,))

        info = {
            'epsilon': self.get_epsilon(),
            'q_values': agent_q_values,
        }

        return actions, info

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute QMIX training loss.

        Args:
            batch: Dictionary with:
                - local_obs: (batch, num_agents, obs_dim)
                - state: (batch, state_dim)
                - actions: (batch, num_agents)
                - rewards: (batch,)
                - next_local_obs: (batch, num_agents, obs_dim)
                - next_state: (batch, state_dim)
                - dones: (batch,)

        Returns:
            Dictionary of losses
        """
        local_obs = batch['local_obs']
        state = batch['state']
        actions = batch['actions']
        rewards = batch['rewards']
        next_local_obs = batch['next_local_obs']
        next_state = batch['next_state']
        dones = batch['dones']

        # Current Q-values
        q_tot_current, _ = self.q_network(local_obs, state, actions)
        q_tot_current = q_tot_current.squeeze(-1)  # (batch,)

        # Target Q-values
        with torch.no_grad():
            # Get next actions using main network (Double DQN style)
            _, next_agent_q_values = self.q_network(next_local_obs, None, None)
            next_actions = torch.argmax(next_agent_q_values, dim=-1)  # (batch, num_agents)

            # Evaluate next actions with target network
            q_tot_next, _ = self.target_network(next_local_obs, next_state, next_actions)
            q_tot_next = q_tot_next.squeeze(-1)  # (batch,)

            # TD target
            td_target = rewards + self.gamma * (1 - dones) * q_tot_next

        # TD loss
        td_loss = F.mse_loss(q_tot_current, td_target)

        return {
            'total_loss': td_loss,
            'td_loss': td_loss,
            'q_mean': q_tot_current.mean(),
            'target_mean': td_target.mean(),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        losses = self.compute_loss(batch)

        # Optimize
        self.optimizer.zero_grad()
        losses['total_loss'].backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)

        self.optimizer.step()

        # Update target network
        if self.training_step % self.target_update_interval == 0:
            self.hard_update(self.target_network, self.q_network)

        self.training_step += 1

        # Return metrics
        metrics = {
            'td_loss': losses['td_loss'].item(),
            'q_mean': losses['q_mean'].item(),
            'target_mean': losses['target_mean'].item(),
            'grad_norm': grad_norm.item(),
            'epsilon': self.get_epsilon(),
        }

        return metrics

    def _get_state_dict(self) -> Dict[str, any]:
        """Get state dict for saving."""
        return {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

    def _load_state_dict(self, checkpoint: Dict[str, any]):
        """Load state dict from checkpoint."""
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def set_training_mode(self, mode: bool = True):
        """Set training mode."""
        self.q_network.train(mode)

    def to(self, device: str):
        """Move to device."""
        super().to(device)
        self.q_network = self.q_network.to(device)
        self.target_network = self.target_network.to(device)
        return self