# Swarm Fault Tolerance and Re-tasking
# File: fault_tolerance.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
#
# Simulates the decentralized mechanism for detecting a drone failure (going 'is_healthy=False')
# and the subsequent, autonomous re-tasking of its mission token to a nearby, healthy drone.

import time
import random
from typing import Dict, List, Any, Set, Tuple, Optional

# --- Configuration Constants ---
NUM_DRONES = 15
MISSION_AREA_SIZE = 1000  # meters
COMMUNICATION_RANGE_M = 150
HEALTH_CHECK_TIMEOUT_CYCLES = 3  # Cycles without an update means a drone is considered failed (in seconds)


# --- Core Data Structures (Simplified for simulation) ---

class DroneNode:
    """Represents a single autonomous drone participating in the swarm."""

    def __init__(self, drone_id: int, pos_x: float, pos_y: float):
        self.id = drone_id
        self.position = (pos_x, pos_y)
        self.neighbors: Set[int] = set()

        # State inherited from mesh_communication.py
        self.state: Dict[str, Any] = {
            'token_held': None,  # The ID of the mission region this drone is responsible for
            'is_healthy': True,  # Local health status
            'last_state_time': time.time(),  # Time of last local state update (used for health checking)
            'active_mission': None  # Mission ID, linked to the token_held
        }

        # External Knowledge (Learned from Mesh Gossip)
        self.known_neighbor_states: Dict[int, Dict[str, Any]] = {}

    def update_local_state(self, key: str, value: Any):
        """Updates a critical piece of state and resets the health time stamp."""
        self.state[key] = value
        self.state['last_state_time'] = time.time()

    def receive_neighbor_state(self, sender_id: int, state_payload: Dict[str, Any]):
        """Simulates receiving a gossip packet and updating knowledge."""
        # This stores the sender's full state, including the 'timestamp' (proof of life).
        self.known_neighbor_states[sender_id] = state_payload

    def initiate_failure(self):
        """Simulates a catastrophic hardware failure."""
        self.state['is_healthy'] = False
        self.state['active_mission'] = None
        # Note: A failed drone cannot communicate, so its neighbors will discover the failure via timeout.


class SwarmFaultTolerance:
    """
    Manages the core logic for detecting failures and re-tasking tokens.
    This logic resides within every healthy drone.
    """

    def __init__(self, drones: List[DroneNode]):
        self.drones = drones
        self.drone_map = {d.id: d for d in drones}
        self.current_cycle = 0

    def _gossip_step(self):
        """Simulates a communication cycle (passing state between neighbors)."""
        # In a real system, this is a separate module (mesh_communication.py).
        # Here, we simulate broadcasting the sender's state to all its neighbors.
        for sender in self.drones:
            if not sender.state['is_healthy']:
                continue  # Dead drones don't broadcast

            payload = {
                'token_held': sender.state['token_held'],
                'is_healthy': sender.state['is_healthy'],
                'last_state_time': sender.state['last_state_time'],
                'active_mission': sender.state['active_mission'],
                'timestamp': time.time()  # Always send the current time as proof of life
            }

            for neighbor_id in sender.neighbors:
                if self.drone_map[neighbor_id].state['is_healthy']:
                    self.drone_map[neighbor_id].receive_neighbor_state(sender.id, payload)

    def _check_health_and_recover(self, current_drone: DroneNode):
        """
        Core Decentralized Logic for Failure Detection and Token Recovery.
        Executed by every healthy drone in every cycle.
        """
        if not current_drone.state['is_healthy']:
            return

        for neighbor_id, neighbor_state in current_drone.known_neighbor_states.items():

            # --- 1. Failure Detection (Health Check Timeout) ---
            # FIX: Use the 'timestamp' (time message was sent) for health check timeout
            # instead of 'last_state_time' (time drone last updated its internal state).
            time_since_last_update = time.time() - neighbor_state['timestamp']

            # Use a time-based check to determine if the neighbor is effectively dead/unresponsive
            is_unresponsive = (time_since_last_update > HEALTH_CHECK_TIMEOUT_CYCLES)

            # Check if neighbor was known to hold a token AND is unresponsive
            if neighbor_state['token_held'] is not None and is_unresponsive:
                failed_drone_id = neighbor_id
                failed_token = neighbor_state['token_held']

                # --- 2. Token Recovery and Re-tasking ---

                # Decentralized decision: only claim if this drone is the nearest or has a tie-breaking ID.
                # Simplification: Assume the first drone to detect and attempt recovery succeeds.

                if current_drone.state['token_held'] is None:
                    print(
                        f"[{current_drone.id}][RECOVERY] Detected failure of Drone {failed_drone_id}. Token {failed_token} is orphaned.")

                    # Claim the orphaned token (Decentralized Auction/Claim)
                    current_drone.update_local_state('token_held', failed_token)
                    current_drone.update_local_state('active_mission', f"RE-TASK_{failed_token}")

                    # In a real system, this next line would be a network broadcast to release the token.
                    # In this simulation, we mark the token as orphaned in the failed drone's state 
                    # to prevent other detection loops from triggering for the same token.
                    self.drone_map[failed_drone_id].state['token_held'] = None

                    print(f"[{current_drone.id}][SUCCESS] Claimed orphaned token {failed_token}. Re-tasking initiated.")
                    return  # Only claim one token per cycle

    def run_cycle(self):
        """Runs one full cycle of the fault tolerance system."""
        self.current_cycle += 1

        # 1. Update neighbor list (based on position)
        self._find_neighbors(COMMUNICATION_RANGE_M)

        # 2. Gossip and exchange state
        self._gossip_step()

        # 3. All healthy drones check for failures and attempt recovery
        for drone in self.drones:
            self._check_health_and_recover(drone)

    def _find_neighbors(self, comm_range: int):
        """Calculates the current P2P neighbors based on physical proximity."""

        def distance(pos1, pos2):
            return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

        for d1 in self.drones:
            d1.neighbors.clear()
            for d2 in self.drones:
                if d1.id != d2.id:
                    if distance(d1.position, d2.position) <= comm_range:
                        d1.neighbors.add(d2.id)


# --- Simulation Run ---

if __name__ == "__main__":
    random.seed(42)

    # Initialize drones in a cluster to ensure high connectivity
    drones = []
    for i in range(NUM_DRONES):
        x = random.uniform(400, 600)
        y = random.uniform(400, 600)
        drones.append(DroneNode(i, x, y))

    # 1. Initial Setup: Drone 5 is tasked with a critical mission (Token P-15)
    DRONE_TO_KILL = 5
    MISSION_TOKEN = 'P-15'

    # Manually assign the token and active mission to the victim drone
    drones[DRONE_TO_KILL].update_local_state('token_held', MISSION_TOKEN)
    drones[DRONE_TO_KILL].update_local_state('active_mission', 'Pollinate Sector P-15')

    fault_system = SwarmFaultTolerance(drones)

    print("--- Swarm Fault Tolerance & Re-tasking Simulation ---")
    print(f"Scenario: Drone {DRONE_TO_KILL} (Holding Token '{MISSION_TOKEN}') will fail.")

    # Run initial cycles to ensure all neighbors know Drone 5 holds the token
    for i in range(1, HEALTH_CHECK_TIMEOUT_CYCLES + 1):
        fault_system.run_cycle()
        print(f"\n[INFO] Cycle {i}: Drone {DRONE_TO_KILL} is active. State is gossiped.")
        time.sleep(0.001)  # Advance time slightly for health check logic

    print("\n==============================================")
    print(f"!!!! CATASTROPHIC FAILURE: DRONE {DRONE_TO_KILL} CRASHES !!!!")
    print("==============================================")

    # 2. Failure Event
    drones[DRONE_TO_KILL].initiate_failure()
    # Also simulate that the last state time is now OLD, triggering the timeout
    # Set the 'last_state_time' far in the past to guarantee failure detection by the check
    drones[DRONE_TO_KILL].state['last_state_time'] = time.time() - (HEALTH_CHECK_TIMEOUT_CYCLES + 1) * 10

    # 3. Recovery Cycles
    recovering_drone_id = None
    for i in range(1, HEALTH_CHECK_TIMEOUT_CYCLES + 3):
        fault_system.run_cycle()

        # Check for recovery
        for drone in fault_system.drones:
            if drone.id != DRONE_TO_KILL and drone.state['token_held'] == MISSION_TOKEN:
                recovering_drone_id = drone.id
                break

        if recovering_drone_id is not None:
            break

    # Final Validation
    print("\n--- FINAL STATE VALIDATION ---")
    print(
        f"Original Drone {DRONE_TO_KILL} (Status): Healthy={fault_system.drones[DRONE_TO_KILL].state['is_healthy']}, Token={fault_system.drones[DRONE_TO_KILL].state['token_held']}")

    if recovering_drone_id is not None:
        print(f"RECOVERY SUCCESS: Drone {recovering_drone_id} assumed mission '{MISSION_TOKEN}'.")
        print(
            "MVP Validation: Successful. Decentralized system successfully detected failure and re-tasked the "
            "critical mission token without central command.")
    else:
        print("RECOVERY FAILURE: Mission token was not re-tasked within the required cycles.")
