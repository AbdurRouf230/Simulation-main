# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from .edge import Pose


@dataclass
class GeoFence:
    vertices: List[Tuple[float, float]]  # simple 2D polygon


class SafetyManager:
    """
    Enforces geo-fence + basic link-quality failsafe.
    """

    def __init__(self, fences: List[GeoFence]):
        self.fences = fences

    def inside_any_fence(self, x: float, y: float) -> bool:
        for fence in self.fences:
            inside = False
            verts = fence.vertices
            n = len(verts)
            for i in range(n):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % n]
                if ((y1 > y) != (y2 > y)) and (
                    x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-8) + x1
                ):
                    inside = not inside
            if inside:
                return True
        return False

    def should_trigger_rth(self, pose: Pose, link_quality: float) -> bool:
        in_fence = self.inside_any_fence(pose.x, pose.y)
        if not in_fence or link_quality < 0.2:
            return True
        return False
