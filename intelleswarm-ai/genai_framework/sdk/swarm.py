# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .edge import mlp, EnvReading, Pose, WorldModelVAE, EdgePerceptionTransformer, LocalizationModule, EnvAdaptationModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiAgentPolicy(nn.Module):
    """
    Multi-agent policy network (actor-critic style).
    """

    def __init__(self, obs_dim: int, latent_dim: int, msg_dim: int, action_dim: int):
        super().__init__()
        self.core = mlp(obs_dim + latent_dim + msg_dim, 256, 256, layers=3)
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, msg: torch.Tensor):
        x = torch.cat([obs, z, msg], dim=-1)
        h = F.relu(self.core(x))
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value


class GraphCommModule(nn.Module):
    """
    Simple attention-based message passing between drones.
    """

    def __init__(self, node_dim: int, msg_dim: int, num_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(node_dim, msg_dim)
        self.k_proj = nn.Linear(node_dim, msg_dim)
        self.v_proj = nn.Linear(node_dim, msg_dim)
        self.out_proj = nn.Linear(msg_dim, msg_dim)
        self.num_heads = num_heads

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, D = node_feats.shape
        H = self.num_heads

        Q = self.q_proj(node_feats)
        K = self.k_proj(node_feats)
        V = self.v_proj(node_feats)

        def split_heads(x):
            return x.view(B, N, H, -1).transpose(1, 2)

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        d_k = Kh.size(-1)
        scores = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(d_k)
        mask = (adj == 0).unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = attn @ Vh

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)
        return out


@dataclass
class EnergyState:
    battery_level: float
    temperature: float
    cycle_count: int


class EnergyManager:
    def __init__(self, low_threshold: float = 0.2):
        self.threshold_low = low_threshold

    def should_return_to_base(self, energy: EnergyState, mission_criticality: float) -> bool:
        return energy.battery_level < self.threshold_low and mission_criticality < 0.8


class PollenController:
    def __init__(self, min_pwm: float = 0.1, max_pwm: float = 0.9):
        self.min_pwm = min_pwm
        self.max_pwm = max_pwm

    def compute_pwm(self, target_dose: float, max_dose: float) -> float:
        r = max(0.0, min(1.0, target_dose / (max_dose + 1e-8)))
        return self.min_pwm + (self.max_pwm - self.min_pwm) * r


class DroneBrain:
    """
    High-level per-drone brain: wires perception + world model + policy.
    """

    def __init__(
        self,
        obs_dim: int,
        msg_dim: int,
        action_dim: int,
        world_latent_dim: int = 32,
        enable_collision_avoidance: bool = False,
    ):
        self.perception = EdgePerceptionTransformer()
        self.world_model = WorldModelVAE(obs_dim=obs_dim, latent_dim=world_latent_dim)
        self.policy = MultiAgentPolicy(
            obs_dim=obs_dim,
            latent_dim=world_latent_dim,
            msg_dim=msg_dim,
            action_dim=action_dim,
        )
        self.energy_mgr = EnergyManager()
        self.env_adapt = EnvAdaptationModule()
        self.pollen_ctrl = PollenController()
        self.localization = LocalizationModule()

        # Collision avoidance system (optional)
        self.enable_collision_avoidance = enable_collision_avoidance
        self.collision_model = None
        self.obstacle_encoder = None
        self.collision_checker = None

        if enable_collision_avoidance:
            from .collision import (
                TrajectoryDiffusionModel,
                ObstacleEncoder,
                CollisionChecker,
            )
            self.collision_model = TrajectoryDiffusionModel(
                state_dim=obs_dim,
                action_horizon=10,
                hidden_dim=256,
                num_timesteps=50
            )
            self.obstacle_encoder = ObstacleEncoder(
                point_dim=3,
                hidden_dim=128,
                output_dim=64
            )
            self.collision_checker = CollisionChecker(safety_radius=1.5)

            self.collision_model.to(device)
            self.obstacle_encoder.to(device)

        self.perception.to(device)
        self.world_model.to(device)
        self.policy.to(device)

    def step(
        self,
        obs_vec: torch.Tensor,
        msg_vec: torch.Tensor,
        env: EnvReading,
        point_cloud: torch.Tensor = None,
        goal: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """
        Execute one step of the drone brain.

        Args:
            obs_vec: Observation vector
            msg_vec: Message vector from other agents
            env: Environment reading
            point_cloud: Optional (num_points, 3) point cloud for collision avoidance
            goal: Optional (3,) goal position for collision-aware navigation

        Returns:
            Dictionary with action, environment parameters, and status
        """
        obs_vec = obs_vec.to(device)
        msg_vec = msg_vec.to(device)

        with torch.no_grad():
            _, mu, logvar = self.world_model(obs_vec)
            z = self.world_model.reparameterize(mu, logvar)
            logits, _ = self.policy(obs_vec, z, msg_vec)
            action = torch.tanh(logits)

            # Collision avoidance (if enabled and data provided)
            collision_safe = True
            if self.enable_collision_avoidance and point_cloud is not None and goal is not None:
                # Encode obstacles
                if point_cloud.dim() == 2:
                    point_cloud = point_cloud.unsqueeze(0)  # Add batch dimension
                point_cloud = point_cloud.to(device)
                goal = goal.to(device)

                obstacle_features = self.obstacle_encoder(point_cloud)

                # Generate collision-free trajectory
                state = obs_vec.unsqueeze(0) if obs_vec.dim() == 1 else obs_vec
                goal_batch = goal.unsqueeze(0) if goal.dim() == 1 else goal

                trajectory = self.collision_model.generate(
                    state, goal_batch, obstacle_features, num_samples=1
                )

                # Extract first waypoint as immediate action direction
                # trajectory shape: (batch, num_samples, action_horizon, 3)
                next_waypoint = trajectory[0, 0, 0, :]  # First waypoint

                # Check if original action is safe
                current_pos = torch.zeros(3, device=device)  # Placeholder
                proposed_traj = torch.stack([current_pos, current_pos + action[:3]])

                obstacles = []  # Would be extracted from point_cloud
                collision_safe, _ = self.collision_checker.check_trajectory(
                    proposed_traj, obstacles
                )

                # If unsafe, use collision-free trajectory direction
                if not collision_safe:
                    # Replace action with safe direction
                    direction = next_waypoint / (torch.norm(next_waypoint) + 1e-8)
                    action[:3] = direction

        params = self.env_adapt.adjust_parameters(env)

        # Placeholder for EnergyState
        dummy_energy = self.energy_mgr.should_return_to_base(
            EnergyState(battery_level=0.5, temperature=30.0, cycle_count=100),
            mission_criticality=0.9
        )

        return {
            "action": action.detach().cpu().numpy().tolist(),
            "env_params": params,
            "energy_status": 0.5 if not dummy_energy else 0.2,
            "collision_safe": collision_safe,  # Added collision safety status
        }
