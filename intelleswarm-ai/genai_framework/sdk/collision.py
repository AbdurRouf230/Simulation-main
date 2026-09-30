"""
Collision Avoidance System for IntelleSwarm.

This module provides diffusion-based trajectory generation for collision-free navigation.
Key components:
- TrajectoryDiffusionModel: Conditional diffusion for safe trajectories
- ObstacleEncoder: Processes depth/LiDAR into obstacle features
- CollisionChecker: Geometric collision detection
- TrajectoryOptimizer: Enforces dynamics constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,) tensor of timestep indices
        Returns:
            (batch_size, dim) position embeddings
        """
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class UNetBlock(nn.Module):
    """Basic U-Net block for trajectory denoising."""

    def __init__(self, in_dim: int, out_dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
        )
        self.conv1 = nn.Linear(in_dim, out_dim)
        self.conv2 = nn.Linear(out_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_dim != out_dim:
            self.residual = nn.Linear(in_dim, out_dim)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_dim) input features
            t_emb: (batch, time_dim) time embeddings
        Returns:
            (batch, out_dim) output features
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add time embedding
        h = h + self.time_mlp(t_emb)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = F.silu(h)

        return h + self.residual(x)


class TrajectoryDiffusionModel(nn.Module):
    """
    Conditional diffusion model for collision-free trajectory generation.
    Based on Denoising Diffusion Probabilistic Models (DDPM).

    Generates safe trajectories by iteratively denoising random noise,
    conditioned on current state, goal, and obstacle features.
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_horizon: int = 10,
        hidden_dim: int = 256,
        num_timesteps: int = 50,
        dropout: float = 0.1
    ):
        """
        Args:
            state_dim: Dimension of state representation
            action_horizon: Number of waypoints to generate
            hidden_dim: Hidden dimension for U-Net
            num_timesteps: Number of diffusion timesteps
            dropout: Dropout rate
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.trajectory_dim = action_horizon * 3  # 3D waypoints
        self.num_timesteps = num_timesteps

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Condition encoder (state + goal + obstacles)
        condition_dim = state_dim + 3 + 64  # state + goal(3D) + obstacle_features(64)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # U-Net encoder
        self.encoder = nn.ModuleList([
            UNetBlock(self.trajectory_dim, hidden_dim, hidden_dim, dropout),
            UNetBlock(hidden_dim, hidden_dim, hidden_dim, dropout),
            UNetBlock(hidden_dim, hidden_dim, hidden_dim, dropout),
        ])

        # U-Net decoder
        self.decoder = nn.ModuleList([
            UNetBlock(hidden_dim + hidden_dim, hidden_dim, hidden_dim, dropout),
            UNetBlock(hidden_dim + hidden_dim, hidden_dim, hidden_dim, dropout),
            UNetBlock(hidden_dim + hidden_dim, hidden_dim, hidden_dim, dropout),
        ])

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.trajectory_dim),
        )

        # Noise schedule (linear beta schedule)
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        state: torch.Tensor,
        goal: torch.Tensor,
        obstacle_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in noisy trajectory.

        Args:
            noisy_trajectory: (batch, action_horizon * 3) noisy waypoints
            timestep: (batch,) diffusion timestep
            state: (batch, state_dim) current state
            goal: (batch, 3) goal position
            obstacle_features: (batch, 64) encoded obstacles
        Returns:
            (batch, action_horizon * 3) predicted noise
        """
        batch_size = noisy_trajectory.shape[0]

        # Time embedding
        t_emb = self.time_emb(timestep)

        # Condition embedding
        condition = torch.cat([state, goal, obstacle_features], dim=-1)
        cond_emb = self.condition_encoder(condition)

        # U-Net forward pass
        h = noisy_trajectory
        encoder_outputs = []

        # Encoder
        for block in self.encoder:
            h = block(h, t_emb)
            encoder_outputs.append(h)

        # Decoder with skip connections
        for i, block in enumerate(self.decoder):
            skip = encoder_outputs[-(i+1)]
            h = torch.cat([h, skip], dim=-1)
            h = block(h, t_emb)

        # Add condition
        h = h + cond_emb

        # Output
        noise_pred = self.output(h)
        return noise_pred

    @torch.no_grad()
    def generate(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        obstacle_features: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Generate collision-free trajectories using DDPM sampling.

        Args:
            state: (batch, state_dim) current state
            goal: (batch, 3) goal position
            obstacle_features: (batch, 64) encoded obstacles
            num_samples: Number of trajectory samples to generate
        Returns:
            (batch, num_samples, action_horizon, 3) generated trajectories
        """
        batch_size = state.shape[0]
        device = state.device

        # Start with random noise
        trajectory = torch.randn(
            batch_size, num_samples, self.trajectory_dim, device=device
        )

        # Expand state, goal, obstacle for multiple samples
        state_exp = state.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, self.state_dim)
        goal_exp = goal.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, 3)
        obs_exp = obstacle_features.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, 64)

        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            timestep = torch.full((batch_size * num_samples,), t, device=device, dtype=torch.long)

            # Reshape for model
            traj_flat = trajectory.reshape(-1, self.trajectory_dim)

            # Predict noise
            noise_pred = self(traj_flat, timestep, state_exp, goal_exp, obs_exp)

            # DDPM update
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)

            beta = self.betas[t]
            noise = torch.randn_like(traj_flat) if t > 0 else 0

            traj_flat = (traj_flat - beta / torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(1 - beta)
            traj_flat = traj_flat + torch.sqrt(beta) * noise

            trajectory = traj_flat.reshape(batch_size, num_samples, self.trajectory_dim)

        # Reshape to waypoints
        trajectory = trajectory.reshape(batch_size, num_samples, self.action_horizon, 3)
        return trajectory


class ObstacleEncoder(nn.Module):
    """
    Processes depth/LiDAR point cloud into obstacle features.
    Uses voxel-based encoding for efficient representation.
    """

    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 64,
        voxel_size: float = 0.5,
        max_points: int = 1024
    ):
        """
        Args:
            point_dim: Dimension of each point (3 for xyz)
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            voxel_size: Size of voxel grid
            max_points: Maximum points to process
        """
        super().__init__()
        self.point_dim = point_dim
        self.max_points = max_points
        self.voxel_size = voxel_size

        # Point feature encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Aggregate features
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Encode point cloud into obstacle features.

        Args:
            point_cloud: (batch, num_points, 3) point cloud
        Returns:
            (batch, output_dim) obstacle features
        """
        batch_size = point_cloud.shape[0]

        # Subsample if too many points
        if point_cloud.shape[1] > self.max_points:
            indices = torch.randperm(point_cloud.shape[1])[:self.max_points]
            point_cloud = point_cloud[:, indices, :]

        # Encode each point
        point_features = self.point_encoder(point_cloud)  # (batch, num_points, hidden_dim)

        # Max pooling aggregation
        aggregated = torch.max(point_features, dim=1)[0]  # (batch, hidden_dim)

        # Final encoding
        obstacle_features = self.aggregator(aggregated)  # (batch, output_dim)

        return obstacle_features


class CollisionChecker:
    """
    Fast geometric collision detection with safety margins.
    Minimum separation: 1.5m by default.
    """

    def __init__(self, safety_radius: float = 1.5):
        """
        Args:
            safety_radius: Minimum safe distance in meters
        """
        self.safety_radius = safety_radius

    def check_trajectory(
        self,
        trajectory: torch.Tensor,
        obstacles: List[torch.Tensor],
        safety_radius: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Check if trajectory is collision-free.

        Args:
            trajectory: (action_horizon, 3) waypoints
            obstacles: List of (3,) obstacle positions
            safety_radius: Override default safety radius
        Returns:
            (is_safe, closest_approach) tuple
        """
        if safety_radius is None:
            safety_radius = self.safety_radius

        if len(obstacles) == 0:
            return True, float('inf')

        # Stack obstacles
        obstacles_tensor = torch.stack(obstacles)  # (num_obstacles, 3)

        # Compute distances from each waypoint to each obstacle
        # (action_horizon, 1, 3) - (1, num_obstacles, 3) -> (action_horizon, num_obstacles)
        distances = torch.norm(
            trajectory.unsqueeze(1) - obstacles_tensor.unsqueeze(0),
            dim=-1
        )

        # Find closest approach
        closest_approach = torch.min(distances).item()

        # Check if safe
        is_safe = closest_approach >= safety_radius

        return is_safe, closest_approach

    def check_inter_agent_collision(
        self,
        trajectories: List[torch.Tensor],
        safety_radius: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Check for collisions between multiple agent trajectories.

        Args:
            trajectories: List of (action_horizon, 3) trajectories
            safety_radius: Override default safety radius
        Returns:
            (is_safe, closest_approach) tuple
        """
        if safety_radius is None:
            safety_radius = self.safety_radius

        if len(trajectories) < 2:
            return True, float('inf')

        min_distance = float('inf')

        # Check all pairs
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                # Pairwise distances at each timestep
                distances = torch.norm(
                    trajectories[i] - trajectories[j],
                    dim=-1
                )
                pair_min = torch.min(distances).item()
                min_distance = min(min_distance, pair_min)

        is_safe = min_distance >= safety_radius
        return is_safe, min_distance


class TrajectoryOptimizer:
    """
    Post-processes diffusion samples to enforce dynamics constraints.
    Ensures feasibility with respect to velocity and acceleration limits.
    """

    def __init__(
        self,
        max_velocity: float = 5.0,
        max_acceleration: float = 2.0,
        dt: float = 0.1
    ):
        """
        Args:
            max_velocity: Maximum velocity in m/s
            max_acceleration: Maximum acceleration in m/s^2
            dt: Time step between waypoints in seconds
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.dt = dt

    def optimize(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Enforce dynamics constraints on trajectory.

        Args:
            trajectory: (action_horizon, 3) waypoints
        Returns:
            (action_horizon, 3) feasible trajectory
        """
        optimized = trajectory.clone()

        for i in range(1, len(optimized)):
            # Compute velocity
            velocity = (optimized[i] - optimized[i-1]) / self.dt
            speed = torch.norm(velocity)

            # Clip velocity
            if speed > self.max_velocity:
                velocity = velocity / speed * self.max_velocity
                optimized[i] = optimized[i-1] + velocity * self.dt

            # Check acceleration if not first step
            if i > 1:
                prev_velocity = (optimized[i-1] - optimized[i-2]) / self.dt
                acceleration = (velocity - prev_velocity) / self.dt
                accel_magnitude = torch.norm(acceleration)

                if accel_magnitude > self.max_acceleration:
                    # Limit acceleration
                    acceleration = acceleration / accel_magnitude * self.max_acceleration
                    new_velocity = prev_velocity + acceleration * self.dt
                    optimized[i] = optimized[i-1] + new_velocity * self.dt

        return optimized

    def smooth_trajectory(self, trajectory: torch.Tensor, smoothing_factor: float = 0.5) -> torch.Tensor:
        """
        Apply smoothing to trajectory using exponential moving average.

        Args:
            trajectory: (action_horizon, 3) waypoints
            smoothing_factor: Smoothing strength (0 = no smoothing, 1 = max smoothing)
        Returns:
            (action_horizon, 3) smoothed trajectory
        """
        if smoothing_factor == 0:
            return trajectory

        smoothed = trajectory.clone()

        for i in range(1, len(smoothed) - 1):
            smoothed[i] = (
                smoothing_factor * (smoothed[i-1] + smoothed[i+1]) / 2.0 +
                (1 - smoothing_factor) * smoothed[i]
            )

        return smoothed
