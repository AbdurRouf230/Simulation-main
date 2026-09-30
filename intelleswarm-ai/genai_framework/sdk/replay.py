"""
Experience Replay Buffer for Multi-Agent Reinforcement Learning.

Provides efficient storage and sampling of multi-agent experiences.
Supports both uniform and prioritized experience replay.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class Transition:
    """Single transition in multi-agent environment."""

    def __init__(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        messages: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        priority: float = 1.0
    ):
        """
        Args:
            observations: (num_agents, obs_dim) observations
            actions: (num_agents, action_dim) actions
            rewards: (num_agents,) rewards
            next_observations: (num_agents, obs_dim) next observations
            dones: (num_agents,) done flags
            messages: (num_agents, msg_dim) agent messages
            adjacency: (num_agents, num_agents) communication graph
            priority: Priority for sampling (used in PER)
        """
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones
        self.messages = messages
        self.adjacency = adjacency
        self.priority = priority


class ExperienceBuffer:
    """
    Stores (s, a, r, s', done, msg, adj) tuples for MARL.
    Supports priority sampling for QMIX/MADDPG.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        device: str = 'cpu',
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        """
        Args:
            capacity: Maximum buffer size
            device: Device to store tensors
            prioritized: Whether to use prioritized experience replay (PER)
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent
            beta_increment: Increment beta each sample to anneal bias
        """
        self.capacity = capacity
        self.device = device
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.position = 0

    def push(self, transition: Transition):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(transition.priority)
        else:
            # Overwrite oldest
            self.buffer[self.position] = transition
            self.priorities[self.position] = transition.priority

        self.position = (self.position + 1) % self.capacity

    def push_batch(self, transitions: List[Transition]):
        """Add multiple transitions at once."""
        for transition in transitions:
            self.push(transition)

    def sample(
        self,
        batch_size: int,
        priority: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray]]:
        """
        Sample batch from buffer.

        Args:
            batch_size: Number of transitions to sample
            priority: Whether to use priority-based sampling
        Returns:
            batch: Dictionary of batched tensors
            weights: Importance sampling weights (if prioritized)
            indices: Sampled indices (if prioritized, for updating priorities)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")

        # Sample indices
        if priority and self.prioritized:
            indices, weights = self._sample_prioritized(batch_size)
        else:
            indices = random.sample(range(len(self.buffer)), batch_size)
            weights = None

        # Gather transitions
        transitions = [self.buffer[idx] for idx in indices]

        # Stack into batch
        batch = self._stack_transitions(transitions)

        return batch, weights, indices if priority else None

    def _sample_prioritized(self, batch_size: int) -> Tuple[List[int], torch.Tensor]:
        """Sample using prioritized experience replay."""
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probabilities
        )

        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return indices.tolist(), weights

    def _stack_transitions(self, transitions: List[Transition]) -> Dict[str, torch.Tensor]:
        """Stack list of transitions into batched tensors."""
        batch = {
            'observations': torch.stack([t.observations for t in transitions]).to(self.device),
            'actions': torch.stack([t.actions for t in transitions]).to(self.device),
            'rewards': torch.stack([t.rewards for t in transitions]).to(self.device),
            'next_observations': torch.stack([t.next_observations for t in transitions]).to(self.device),
            'dones': torch.stack([t.dones for t in transitions]).to(self.device),
        }

        # Optional fields
        if transitions[0].messages is not None:
            batch['messages'] = torch.stack([t.messages for t in transitions]).to(self.device)

        if transitions[0].adjacency is not None:
            batch['adjacency'] = torch.stack([t.adjacency for t in transitions]).to(self.device)

        return batch

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for prioritized replay."""
        if not self.prioritized:
            return

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.buffer[idx].priority = priority

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.position = 0


class RolloutBuffer:
    """
    On-policy rollout buffer for PPO/A2C style algorithms.
    Stores complete trajectories with advantages and returns.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Args:
            device: Device to store tensors
        """
        self.device = device
        self.reset()

    def reset(self):
        """Clear buffer for new rollout."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.messages = []
        self.adjacency = []

    def push(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
        message: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None
    ):
        """Add single step to rollout."""
        self.observations.append(observation.cpu())
        self.actions.append(action.cpu())
        self.rewards.append(reward.cpu())
        self.values.append(value.cpu())
        self.log_probs.append(log_prob.cpu())
        self.dones.append(done.cpu())

        if message is not None:
            self.messages.append(message.cpu())
        if adjacency is not None:
            self.adjacency.append(adjacency.cpu())

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).

        Args:
            last_value: Value estimate for final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        Returns:
            returns: (num_steps, num_agents) discounted returns
            advantages: (num_steps, num_agents) GAE advantages
        """
        advantages = []
        last_gae = torch.zeros_like(last_value)

        # Compute advantages backwards
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            # TD residual
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]

            # GAE
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages.insert(0, last_gae)

        advantages = torch.stack(advantages)
        returns = advantages + torch.stack(self.values)

        return returns, advantages

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all rollout data as batched tensors."""
        batch = {
            'observations': torch.stack(self.observations).to(self.device),
            'actions': torch.stack(self.actions).to(self.device),
            'rewards': torch.stack(self.rewards).to(self.device),
            'values': torch.stack(self.values).to(self.device),
            'log_probs': torch.stack(self.log_probs).to(self.device),
            'dones': torch.stack(self.dones).to(self.device),
        }

        if len(self.messages) > 0:
            batch['messages'] = torch.stack(self.messages).to(self.device)

        if len(self.adjacency) > 0:
            batch['adjacency'] = torch.stack(self.adjacency).to(self.device)

        return batch

    def __len__(self) -> int:
        """Return number of steps in rollout."""
        return len(self.observations)


class MultiAgentReplayBuffer:
    """
    Specialized replay buffer for centralized training with decentralized execution (CTDE).
    Stores both local observations and global state.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        num_agents: int = 4,
        device: str = 'cpu'
    ):
        """
        Args:
            capacity: Maximum buffer size
            num_agents: Number of agents
            device: Device to store tensors
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.device = device

        self.local_obs = deque(maxlen=capacity)
        self.global_state = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_local_obs = deque(maxlen=capacity)
        self.next_global_state = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def push(
        self,
        local_obs: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_local_obs: torch.Tensor,
        next_global_state: torch.Tensor,
        dones: torch.Tensor
    ):
        """Add CTDE transition."""
        self.local_obs.append(local_obs.cpu())
        self.global_state.append(global_state.cpu())
        self.actions.append(actions.cpu())
        self.rewards.append(rewards.cpu())
        self.next_local_obs.append(next_local_obs.cpu())
        self.next_global_state.append(next_global_state.cpu())
        self.dones.append(dones.cpu())

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch for CTDE training."""
        if len(self.local_obs) < batch_size:
            raise ValueError(f"Not enough samples: {len(self.local_obs)} < {batch_size}")

        indices = random.sample(range(len(self.local_obs)), batch_size)

        batch = {
            'local_obs': torch.stack([self.local_obs[i] for i in indices]).to(self.device),
            'global_state': torch.stack([self.global_state[i] for i in indices]).to(self.device),
            'actions': torch.stack([self.actions[i] for i in indices]).to(self.device),
            'rewards': torch.stack([self.rewards[i] for i in indices]).to(self.device),
            'next_local_obs': torch.stack([self.next_local_obs[i] for i in indices]).to(self.device),
            'next_global_state': torch.stack([self.next_global_state[i] for i in indices]).to(self.device),
            'dones': torch.stack([self.dones[i] for i in indices]).to(self.device),
        }

        return batch

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.local_obs)

    def clear(self):
        """Clear all transitions."""
        self.local_obs.clear()
        self.global_state.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_local_obs.clear()
        self.next_global_state.clear()
        self.dones.clear()
