# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
from __future__ import annotations
from typing import Dict, Any


class GroundStationBackend:
    """
    Simple in-memory mission + telemetry store.
    """

    def __init__(self):
        self.active_missions: Dict[str, Dict[str, Any]] = {}

    def create_mission(self, mission_id: str, config: Dict[str, Any]) -> None:
        self.active_missions[mission_id] = {
            "config": config,
            "status": "created",
            "events": [],
        }

    def update_status(self, mission_id: str, status: str) -> None:
        if mission_id in self.active_missions:
            self.active_missions[mission_id]["status"] = status

    def ingest_telemetry(self, mission_id: str, drone_id: str, payload: Dict[str, Any]) -> None:
        if mission_id in self.active_missions:
            self.active_missions[mission_id]["events"].append(
                {"drone_id": drone_id, "payload": payload}
            )


class IntelleSwarmSDK:
    """
    External entry point (for BUET, defense, ag partners).
    """

    def __init__(self):
        self.backend = GroundStationBackend()

    def create_mission(self, mission_id: str, config: Dict[str, Any]) -> None:
        self.backend.create_mission(mission_id, config)

    def push_telemetry(self, mission_id: str, drone_id: str, telemetry: Dict[str, Any]) -> None:
        self.backend.ingest_telemetry(mission_id, drone_id, telemetry)

    def get_mission_state(self, mission_id: str) -> Dict[str, Any]:
        return self.backend.active_missions.get(mission_id, {})
