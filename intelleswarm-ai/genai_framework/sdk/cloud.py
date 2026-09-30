# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
from __future__ import annotations
from typing import Dict

import torch
import torch.nn.functional as fn

from .edge import WorldModelVAE
from .swarm import MultiAgentPolicy, GraphCommModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SwarmTrainer:
    """
    Multi-algorithm trainer for generative world model + multi-agent RL.
    Supports A2C, MAPPO, QMIX, and MADDPG algorithms.
    This is cloud-side only.
    """

    def __init__(
        self,
        world_model: WorldModelVAE = None,
        policy: MultiAgentPolicy = None,
        comm_module: GraphCommModule = None,
        algorithm: str = 'a2c',
        num_agents: int = 4,
        obs_dim: int = 32,
        action_dim: int = 4,
        global_state_dim: int = 128,
        **algorithm_kwargs
    ):
        """
        Initialize SwarmTrainer with specified algorithm.

        Args:
            world_model: WorldModelVAE instance (for A2C)
            policy: MultiAgentPolicy instance (for A2C)
            comm_module: GraphCommModule instance
            algorithm: Algorithm type ('a2c', 'mappo', 'qmix', 'maddpg')
            num_agents: Number of agents in swarm
            obs_dim: Observation dimension
            action_dim: Action dimension
            global_state_dim: Global state dimension (for CTDE algorithms)
            **algorithm_kwargs: Additional algorithm-specific parameters
        """
        self.algorithm = algorithm.lower()
        self.num_agents = num_agents

        # Initialize based on algorithm type
        if self.algorithm == 'a2c':
            # Original A2C implementation
            if world_model is None or policy is None:
                raise ValueError("A2C requires world_model and policy")

            self.world_model = world_model.to(device)
            self.policy = policy.to(device)
            self.comm_module = comm_module.to(device) if comm_module else None

            # Optimizers for A2C
            self.world_opt = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
            self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
            self.trainer = None

        elif self.algorithm == 'mappo':
            # MAPPO implementation
            from .algorithms.mappo import MAPPOTrainer

            self.world_model = world_model.to(device) if world_model else None
            self.comm_module = comm_module.to(device) if comm_module else None

            # Create MAPPO trainer
            self.trainer = MAPPOTrainer(
                obs_dim=obs_dim,
                action_dim=action_dim,
                global_state_dim=global_state_dim,
                num_agents=num_agents,
                device=str(device),
                **algorithm_kwargs
            )
            self.policy = self.trainer.policy  # For compatibility

            # Optional world model optimizer
            if self.world_model is not None:
                self.world_opt = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
            else:
                self.world_opt = None

        elif self.algorithm == 'qmix':
            # QMIX implementation
            from .algorithms.qmix import QMIXTrainer

            self.world_model = world_model.to(device) if world_model else None
            self.comm_module = comm_module.to(device) if comm_module else None

            # Create QMIX trainer
            self.trainer = QMIXTrainer(
                obs_dim=obs_dim,
                state_dim=global_state_dim,
                action_dim=action_dim,
                num_agents=num_agents,
                device=str(device),
                **algorithm_kwargs
            )
            self.policy = self.trainer.q_network  # For compatibility

            # Optional world model optimizer
            if self.world_model is not None:
                self.world_opt = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
            else:
                self.world_opt = None

        elif self.algorithm == 'maddpg':
            # MADDPG implementation
            from .algorithms.maddpg import MADDPGTrainer

            self.world_model = world_model.to(device) if world_model else None
            self.comm_module = comm_module.to(device) if comm_module else None

            # For MADDPG, we need obs_dims and action_dims as lists
            obs_dims = algorithm_kwargs.get('obs_dims', [obs_dim] * num_agents)
            action_dims = algorithm_kwargs.get('action_dims', [action_dim] * num_agents)

            # Create MADDPG trainer
            self.trainer = MADDPGTrainer(
                obs_dims=obs_dims,
                action_dims=action_dims,
                num_agents=num_agents,
                device=str(device),
                **{k: v for k, v in algorithm_kwargs.items()
                   if k not in ['obs_dims', 'action_dims']}
            )
            self.policy = self.trainer.agents  # For compatibility

            # Optional world model optimizer
            if self.world_model is not None:
                self.world_opt = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
            else:
                self.world_opt = None

        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Supported: 'a2c', 'mappo', 'qmix' (planned), 'maddpg' (planned)"
            )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute one training step using the selected algorithm.

        Args:
            batch: Dictionary with training data. Required keys depend on algorithm:
                A2C:
                  - obs: (B, obs_dim)
                  - next_obs: (B, obs_dim)
                  - msg: (B, msg_dim)
                  - actions: (B, action_dim)
                  - rewards: (B,)
                MAPPO:
                  - local_obs: (B, num_agents, obs_dim)
                  - global_state: (B, global_state_dim)
                  - actions: (B, num_agents, action_dim)
                  - old_log_probs: (B, num_agents)
                  - advantages: (B, num_agents)
                  - returns: (B, num_agents)

        Returns:
            Dictionary of training metrics
        """
        if self.algorithm == 'a2c':
            return self._train_step_a2c(batch)
        elif self.algorithm == 'mappo':
            return self._train_step_mappo(batch)
        elif self.algorithm == 'qmix':
            return self._train_step_qmix(batch)
        elif self.algorithm == 'maddpg':
            return self._train_step_maddpg(batch)
        else:
            raise NotImplementedError(f"Training for {self.algorithm} not implemented")

    def _train_step_a2c(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Original A2C training step."""
        obs = batch["obs"].to(device)
        next_obs = batch["next_obs"].to(device)
        msg = batch["msg"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)

        metrics = {}

        # ---- World model update ----
        if self.world_model is not None:
            self.world_opt.zero_grad()
            recon, mu, logvar = self.world_model(obs)
            recon_loss = fn.mse_loss(recon, next_obs)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            world_loss = recon_loss + 1e-3 * kl
            world_loss.backward()
            self.world_opt.step()

            metrics["loss_world"] = float(world_loss.item())
            metrics["recon_loss"] = float(recon_loss.item())
            metrics["kl_loss"] = float(kl.item())

        # ---- Policy update (simplified A2C-style) ----
        self.policy_opt.zero_grad()
        with torch.no_grad():
            if self.world_model is not None:
                _, mu, logvar = self.world_model(obs)
                z = self.world_model.reparameterize(mu, logvar)
            else:
                z = torch.zeros(obs.shape[0], 32, device=device)  # Dummy latent

        logits, values = self.policy(obs, z, msg)

        if actions.dtype == torch.long:
            logp = fn.log_softmax(logits, dim=-1)
            chosen_logp = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        else:
            # Continuous actions
            chosen_logp = -((logits - actions) ** 2).mean(dim=-1)

        advantage = rewards - values.squeeze(-1).detach()
        policy_loss = -(chosen_logp * advantage).mean()
        value_loss = fn.mse_loss(values.squeeze(-1), rewards)
        total_policy_loss = policy_loss + 0.5 * value_loss

        total_policy_loss.backward()
        self.policy_opt.step()

        metrics.update({
            "loss_policy": float(policy_loss.item()),
            "loss_value": float(value_loss.item()),
            "total_policy_loss": float(total_policy_loss.item()),
        })

        return metrics

    def _train_step_mappo(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """MAPPO training step."""
        metrics = {}

        # Optional world model update
        if self.world_model is not None and "obs" in batch and "next_obs" in batch:
            obs = batch["obs"].to(device)
            next_obs = batch["next_obs"].to(device)

            self.world_opt.zero_grad()
            recon, mu, logvar = self.world_model(obs)
            recon_loss = fn.mse_loss(recon, next_obs)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            world_loss = recon_loss + 1e-3 * kl
            world_loss.backward()
            self.world_opt.step()

            metrics["loss_world"] = float(world_loss.item())

        # MAPPO policy update
        mappo_metrics = self.trainer.train_step(batch)
        metrics.update(mappo_metrics)

        return metrics

    def _train_step_qmix(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """QMIX training step."""
        metrics = {}

        # Optional world model update
        if self.world_model is not None and "obs" in batch and "next_obs" in batch:
            obs = batch["obs"].to(device)
            next_obs = batch["next_obs"].to(device)

            self.world_opt.zero_grad()
            recon, mu, logvar = self.world_model(obs)
            recon_loss = fn.mse_loss(recon, next_obs)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            world_loss = recon_loss + 1e-3 * kl
            world_loss.backward()
            self.world_opt.step()

            metrics["loss_world"] = float(world_loss.item())

        # QMIX policy update
        qmix_metrics = self.trainer.train_step(batch)
        metrics.update(qmix_metrics)

        return metrics

    def _train_step_maddpg(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """MADDPG training step."""
        metrics = {}

        # Optional world model update
        if self.world_model is not None and "obs" in batch and "next_obs" in batch:
            obs = batch["obs"].to(device)
            next_obs = batch["next_obs"].to(device)

            self.world_opt.zero_grad()
            recon, mu, logvar = self.world_model(obs)
            recon_loss = fn.mse_loss(recon, next_obs)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            world_loss = recon_loss + 1e-3 * kl
            world_loss.backward()
            self.world_opt.step()

            metrics["loss_world"] = float(world_loss.item())

        # MADDPG policy update
        maddpg_metrics = self.trainer.train_step(batch)
        metrics.update(maddpg_metrics)

        return metrics
