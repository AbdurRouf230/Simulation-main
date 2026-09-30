"""
Multi-Agent Reinforcement Learning Algorithms.

This module provides implementations of various MARL algorithms:
- MAPPO: Multi-Agent Proximal Policy Optimization
- QMIX: Value Function Factorization
- MADDPG: Multi-Agent Deep Deterministic Policy Gradient
"""

from .base_algorithm import BaseAlgorithm, OnPolicyAlgorithm, OffPolicyAlgorithm, build_mlp
from .mappo import MAPPOActor, MAPPOCritic, MAPPOTrainer
from .qmix import QMIXNetwork, QMIXTrainer
from .maddpg import MADDPGActor, MADDPGCritic, MADDPGTrainer

__all__ = [
    # Base classes
    "BaseAlgorithm",
    "OnPolicyAlgorithm",
    "OffPolicyAlgorithm",
    "build_mlp",

    # MAPPO
    "MAPPOActor",
    "MAPPOCritic",
    "MAPPOTrainer",

    # QMIX
    "QMIXNetwork",
    "QMIXTrainer",

    # MADDPG
    "MADDPGActor",
    "MADDPGCritic",
    "MADDPGTrainer",
]