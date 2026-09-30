# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
from __future__ import annotations
from typing import Dict, Any
import os
import hmac
import hashlib

import torch
import torch.nn as nn

from .edge import mlp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TelemetryEncoder(nn.Module):
    """
    Small autoencoder for telemetry compression.
    """

    def __init__(self, raw_dim: int, latent_dim: int = 32):
        super().__init__()
        self.enc = mlp(raw_dim, 128, latent_dim, layers=3)
        self.dec = mlp(latent_dim, 128, raw_dim, layers=3)

    def forward(self, x: torch.Tensor):
        z = self.enc(x)
        recon = self.dec(z)
        return z, recon


class TelemetryClient:
    def __init__(self, encoder: TelemetryEncoder):
        self.encoder = encoder.to(device)

    def prepare_payload(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        keys = sorted(telemetry.keys())
        x = torch.tensor([[telemetry[k] for k in keys]], dtype=torch.float32, device=device)
        z, _ = self.encoder(x)
        return {
            "keys": keys,
            "latent": z.detach().cpu().tolist(),
        }

    def decode_payload(self, payload: Dict[str, Any]) -> Dict[str, float]:
        keys = payload["keys"]
        z = torch.tensor(payload["latent"], dtype=torch.float32, device=device)
        recon = self.encoder.dec(z)[0].detach().cpu().tolist()
        return {k: v for k, v in zip(keys, recon)}


class SecurityModule:
    """
    Basic HMAC-based integrity for messages.
    """

    def __init__(self, secret: bytes | None = None):
        self.secret = secret or os.urandom(32)

    def sign(self, payload: bytes) -> str:
        return hmac.new(self.secret, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature)
