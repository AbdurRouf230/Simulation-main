# intelleswarm-ai/genai_framework/sdk/edge.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# COPYRIGHT (C) 2025, IntelleSwarm Corporation, USA. All Right Reserved.
# Author: Zahid Rahman
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import math
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods = []
    last = in_dim
    for _ in range(layers - 1):
        mods.append(nn.Linear(last, hidden))
        mods.append(nn.ReLU(inplace=True))
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


class EdgePerceptionTransformer(nn.Module):
    """
    Tiny transformer-like model for onboard perception.
    """

    def __init__(
            self,
            img_channels: int = 3,
            patch_size: int = 8,
            emb_dim: int = 128,
            num_heads: int = 4,
            depth: int = 2,
            num_classes: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        self.proj = nn.Conv2d(img_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, emb_dim))  # overprovision

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Linear(emb_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, H, W)
        returns: (logits, cls_feature)
        """
        B = x.size(0)
        x = self.proj(x)  # (B, emb, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, emb)
        n_tokens = x.size(1)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, x], dim=1)  # (B, 1+N, emb)
        pos = self.pos_embed[:, : tokens.size(1), :]
        tokens = tokens + pos

        enc = self.encoder(tokens)
        cls_feat = enc[:, 0, :]
        logits = self.head(cls_feat)
        return logits, cls_feat


class WorldModelVAE(nn.Module):
    """
    Latent generative world model (VAE).
    """

    def __init__(self, obs_dim: int, latent_dim: int = 32):
        super().__init__()
        self.enc = mlp(obs_dim, 128, latent_dim * 2, layers=3)
        self.dec = mlp(latent_dim, 128, obs_dim, layers=3)

    def encode(self, obs: torch.Tensor):
        stats = self.enc(obs)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.dec(z)

    def forward(self, obs: torch.Tensor):
        mu, logvar = self.encode(obs)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# Small structs reused elsewhere but convenient to define here
@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class EnvReading:
    wind_speed: float
    humidity: float
    temperature: float


class LocalizationModule:
    """
    Placeholder interface for VIO/RTK-SLAM integration.
    """

    def update(self, imu_data: Dict[str, float], vision_feats: torch.Tensor) -> Pose:
        # TODO: hook to real SLAM / RTK
        return Pose(0.0, 0.0, 0.0, 0.0)


class EnvAdaptationModule:
    def adjust_parameters(self, env: EnvReading) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if env.wind_speed > 10.0:
            params["max_speed"] = 0.7
            params["hover_gain"] = 1.2
        else:
            params["max_speed"] = 1.0
            params["hover_gain"] = 1.0
        return params
