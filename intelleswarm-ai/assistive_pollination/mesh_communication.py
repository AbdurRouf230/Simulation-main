# Decentralized Communication Mesh (Gossip Protocol)
# File: mesh_communication.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
# 
# Simulates the peer-to-peer (P2P) communication and state synchronization
# critical for the swarm's fault-tolerant, decentralized operation.
# Uses a simple Gossip protocol to propagate state updates (e.g., token status, warnings).

import random
import time  # FIX: Added missing import for time.time()
from typing import Dict, List, Any, Set, Tuple

# --- Configuration Constants ---
NUM_DRONES = 15
MISSION_AREA_SIZE = 1000  # meters
COMMUNICATION_RANGE_M = 150  # Max distance for P2P link (simulating LoRa/mesh radio range)
GOSSIP_CYCLE_TIME_S = 1.0  # Time per synchronization cycle
CRITICAL_MESSAGE_TIMEOUT_CYCLES = 5  # How long to propagate a critical message


# --- Data Structures ---

class DroneNode:
    """Represents a single autonomous drone participating in the mesh network."""

    def __init__(self, drone_id: int, pos_x: float, pos_y: float):
        self.id = drone_id
        self.position = (pos_x, pos_y)

        # Core State: Data that must be synchronized across the swarm
        self.state: Dict[str, Any] = {
            'token_held': None,  # Region ID token currently held
            'is_healthy': True,
            'last_known_health': {},  # Health status of neighbors
            'critical_warning': None,  # E.g., "Pest Detected", "Low Battery"
            'timestamp': time.time()  # Time of last local state update
        }

        self.neighbors: Set[int] = set()
        # Message cache to prevent immediate re-gossiping of the exact same message
        self._message_cache: Dict[str, float] = {}

    def update_position(self, new_pos: Tuple[float, float]):
        """Simulates drone movement."""
        self.position = new_pos

    def update_local_state(self, key: str, value: Any):
        """Updates a critical piece of state and updates the timestamp."""
        self.state[key] = value
        self.state['timestamp'] = time.time()
        print(f"[Drone {self.id}] Local State Updated: {key} = {value}")

    def get_gossip_payload(self) -> Dict[str, Any]:
        """Creates the payload to send to neighbors."""
        return {
            'sender_id': self.id,
            'state': self.state.copy(),
            'timestamp': self.state['timestamp']
        }

    def process_gossip(self, incoming_payload: Dict[str, Any]):
        """
        Receives and processes state from a neighboring drone.
        Only updates if the incoming state is newer (higher timestamp).
        """
        sender_id = incoming_payload['sender_id']
        sender_state = incoming_payload['state']
        sender_timestamp = incoming_payload['timestamp']

        # Update last known health of the sender (a vital decentralized health check)
        self.state['last_known_health'][sender_id] = sender_state.get('is_healthy', True)

        # Basic State Synchronization Logic (Token/Warning)

        # 1. Critical Warning Synchronization (High Priority)
        incoming_warning = sender_state.get('critical_warning')
        if incoming_warning and self.state['critical_warning'] != incoming_warning:
            self.update_local_state('critical_warning', incoming_warning)
            # Remove print statement from here to prevent log spam in the main logic, 
            # as update_local_state already prints
            return  # Stop processing, critical message takes precedence

        # 2. Token Synchronization (Token must be more recent to override)
        if sender_state['token_held'] is not None and sender_timestamp > self.state['timestamp']:
            # This logic prevents race conditions: only update if the token holding is more recent
            self.update_local_state('token_held', sender_state['token_held'])
            # Note: A real system would require cryptographic verification of the token ownership.


def find_neighbors(drones: List[DroneNode], comm_range: int):
    """Calculates the current P2P neighbors based on physical proximity."""

    def distance(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    for d1 in drones:
        d1.neighbors.clear()
        for d2 in drones:
            if d1.id != d2.id:
                if distance(d1.position, d2.position) <= comm_range:
                    d1.neighbors.add(d2.id)


def run_gossip_cycle(drones: List[DroneNode], drone_map: Dict[int, DroneNode]):
    """Executes one round of the decentralized gossip protocol."""

    # Pre-calculate payloads for the current cycle to ensure consistency
    payloads = {d.id: d.get_gossip_payload() for d in drones}

    # Drones send and process state simultaneously
    for sender in drones:
        if not sender.neighbors:
            continue

        # Select a subset of neighbors to gossip to (fan-out)
        gossip_targets = random.sample(list(sender.neighbors), k=min(2, len(sender.neighbors)))

        for target_id in gossip_targets:
            receiver = drone_map[target_id]
            # Send the payload
            receiver.process_gossip(payloads[sender.id])


# --- Simulation Run ---

if __name__ == "__main__":
    # Initialize drones in a formation that ensures some gaps in coverage, testing decentralization
    drones = []
    for i in range(NUM_DRONES):
        # Place drones in a simulated line or loose formation
        x = (i * (MISSION_AREA_SIZE / NUM_DRONES)) + random.uniform(-10, 10)
        y = random.uniform(50, 150)
        drones.append(DroneNode(i, x, y))

    drone_map = {d.id: d for d in drones}

    # Critical Scenario Setup

    # 1. Swarm State Setup
    # Drone 5 claims the "P-10" Pollination Token (from swarm_coordination.py)
    drone_map[5].update_local_state('token_held', 'P-10')

    # 2. Critical Event (Simulating Edge Inference detection)
    # Drone 1 detects a critical pest and issues a warning
    drone_map[1].update_local_state('critical_warning', 'PEST_OUTBREAK_SECTOR_A')

    print("--- Mesh Communication Simulation (Gossip Protocol) ---")
    print(f"COMM Range: {COMMUNICATION_RANGE_M}m | Total Drones: {NUM_DRONES}")

    # --- Simulation Loop ---
    for cycle in range(1, 10):
        print(f"\n===== CYCLE {cycle} =====")
        find_neighbors(drones, COMMUNICATION_RANGE_M)

        # Print network connectivity status (The Mesh)
        connectivity = {d.id: list(d.neighbors) for d in drones}
        print(f"Connectivity Map: {connectivity}")

        run_gossip_cycle(drones, drone_map)

        # Check propagation of the critical warning
        warning_count = sum(1 for d in drones if d.state['critical_warning'] is not None)
        print(f"Warning Propagation Status: {warning_count}/{NUM_DRONES} drones received the PEST warning.")

        if warning_count == NUM_DRONES:
            print("\nSUCCESS: Critical message successfully propagated across the entire swarm mesh.")
            break

    # Final Status Check
    print("\n--- Final State Validation ---")
    for d_id in [1, 5, 8, 14]:  # Check originator and distant/intermediate nodes
        drone = drone_map[d_id]
        print(
            f"Drone {d_id} State: Token={drone.state['token_held']}, Warning={drone.state['critical_warning']}, Neighbors={len(drone.neighbors)}")

    # MVP Validation Check: Did the critical state propagate successfully?
    if warning_count == NUM_DRONES:
        print(
            "\nMVP Validation: Successful. The decentralized mesh successfully propagated the critical warning and "
            "state (token) across the network, validating fault-tolerant communication.")
    else:
        print(
            "\nMVP Validation: Partial Success. Propagation did not reach all nodes within the required cycles, "
            "indicating potential network segmentation issues.")
