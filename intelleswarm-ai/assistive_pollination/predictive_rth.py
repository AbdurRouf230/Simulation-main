# Battery Life and Predictive RTH (Return to Home) Logic
# File: predictive_rth.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
#
# Implements the critical safety and efficiency logic that predicts the time to
# battery exhaustion and initiates an autonomous Return To Home (RTH) sequence
# to the nearest charging base, ensuring the drone never crashes due to power loss.

import time
import random
from typing import Dict, Any, Tuple, Optional

# --- Configuration Constants (Units in Wh/second for simplicity) ---
MAX_BATTERY_WH = 50.0  # Total battery capacity (Watt-hours)
SAFETY_MARGIN_PERCENT = 0.20  # 20% of required RTH energy is added as a safety buffer

# Power Consumption Rates (Wh/second)
CONSUMPTION_RATE = {
    "HOVER": 0.05,  # Low power state (e.g., waiting for decision)
    "SEARCH": 0.10,  # Cruising velocity for coverage
    "POLLINATE": 0.08,  # Precision hover/dispense (higher than HOVER due to thruster adjustments)
    "RTH_TRAVEL": 0.12,  # High-speed travel back to base
}


# --- Core Data Structures ---

class PowerManager:
    """Tracks current battery level and calculates instantaneous consumption."""

    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.battery_wh = MAX_BATTERY_WH
        self.current_mode = "HOVER"
        self.last_update_time = time.time()
        self.total_energy_consumed = 0.0

    def update_consumption(self, mode: str, mission_time_s: float):
        """
        Updates the battery based on time spent in a specific mode.
        :param mode: Current operation mode (e.g., 'SEARCH', 'POLLINATE').
        :param mission_time_s: Duration spent in this mode since last update.
        """
        self.current_mode = mode
        rate = CONSUMPTION_RATE.get(mode, CONSUMPTION_RATE["HOVER"])
        energy_used = rate * mission_time_s

        self.battery_wh -= energy_used
        self.total_energy_consumed += energy_used

        # Ensure battery does not go below zero
        if self.battery_wh < 0:
            self.battery_wh = 0
            print(f"[CRITICAL] Drone {self.drone_id} ran out of power!")

    def get_status(self) -> Dict[str, float]:
        """Returns current battery status metrics."""
        return {
            'level_wh': self.battery_wh,
            'level_percent': (self.battery_wh / MAX_BATTERY_WH) * 100,
            'current_rate': CONSUMPTION_RATE.get(self.current_mode, CONSUMPTION_RATE["HOVER"])
        }


class PredictiveRTH:
    """
    Implements the autonomous RTH decision-making logic.
    This logic needs inputs from:
    1. PowerManager (Current Battery Level)
    2. GNC/Swarm Coordination (Distance to Base)
    """

    def __init__(self, drone_id: int, base_location: Tuple[float, float], initial_location: Tuple[float, float]):
        self.drone_id = drone_id
        self.base_location = base_location
        self.current_location = initial_location
        self.power_manager = PowerManager(drone_id)

    def _calculate_distance_to_base(self) -> float:
        """Simulates GNC calculating the distance to the charging base (meters)."""
        x1, y1 = self.current_location
        x2, y2 = self.base_location
        distance_m = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return distance_m

    def _estimate_time_to_base(self, distance_m: float, avg_speed_mps: float = 5.0) -> float:
        """Estimates the flight time required for RTH (seconds)."""
        if avg_speed_mps <= 0: return float('inf')
        return distance_m / avg_speed_mps

    def predict_and_decide(self, current_location: Tuple[float, float]) -> bool:
        """
        The core predictive logic.
        Triggers RTH if predicted consumption exceeds remaining power + safety margin.
        :returns: True if RTH is initiated, False otherwise.
        """
        self.current_location = current_location

        distance_m = self._calculate_distance_to_base()
        time_to_base_s = self._estimate_time_to_base(distance_m)

        # 1. Calculate required RTH energy
        rth_rate = CONSUMPTION_RATE["RTH_TRAVEL"]
        energy_required_rth = rth_rate * time_to_base_s

        # 2. Add safety margin (e.g., for wind, path deviation, hover-to-land)
        safety_buffer = energy_required_rth * SAFETY_MARGIN_PERCENT
        total_energy_needed = energy_required_rth + safety_buffer

        current_battery = self.power_manager.get_status()['level_wh']

        # 3. Decision Gate: Is the remaining battery less than what's needed for a safe return?
        if current_battery < total_energy_needed:
            # --- RTH Command Triggered ---
            print("\n!!! PREDICTIVE RTH TRIGGERED !!!")
            print(f"  [Required]: {total_energy_needed:.2f} Wh (RTH + Safety)")
            print(f"  [Remaining]: {current_battery:.2f} Wh")
            print(f"  [Distance]: {distance_m:.1f} meters. Initiating RTH_TRAVEL mode.")
            self.power_manager.current_mode = "RTH_TRAVEL"
            return True

        return False


# --- Simulation Run ---

if __name__ == "__main__":
    random.seed(42)
    BASE = (50, 50)  # Central charging station
    START_POS = (10, 10)  # Drone starts far from base

    rth_system = PredictiveRTH(drone_id=10, base_location=BASE, initial_location=START_POS)

    current_pos = list(START_POS)
    mission_time_s = 0.0

    # Simulates the drone's movement and activity over time
    mission_events = [
        ("SEARCH", 10.0, (15, 15)),  # Search for 10s
        ("POLLINATE", 5.0, (20, 20)),  # Pollinate for 5s
        ("SEARCH", 15.0, (30, 35)),  # Search and move
        ("POLLINATE", 10.0, (40, 50)),  # Heavy activity
        # Simulate a long stretch of high-consumption search far from home
        ("SEARCH", 100.0, (80, 80)),
        ("SEARCH", 100.0, (85, 90)),
        ("POLLINATE", 50.0, (90, 85)),
        ("SEARCH", 100.0, (80, 70)),
        ("POLLINATE", 50.0, (75, 75)),
        ("SEARCH", 100.0, (70, 70)),  # Continue searching until RTH triggers
    ]

    print("--- Battery Life and Predictive RTH Simulation ---")
    print(f"Base: {BASE} | Max Capacity: {MAX_BATTERY_WH} Wh")
    print("-" * 50)

    rth_triggered = False

    for mode, duration, new_pos in mission_events:
        if rth_triggered:
            break

        # 1. Update position (simulates GNC path)
        current_pos = new_pos
        mission_time_s += duration

        # 2. Update power consumption
        rth_system.power_manager.update_consumption(mode, duration)

        # 3. Run RTH prediction and decision logic
        if rth_system.predict_and_decide(current_pos):
            rth_triggered = True

        status = rth_system.power_manager.get_status()
        print(
            f"[{mode:<9} | T={mission_time_s:.0f}s] Batt: {status['level_wh']:.2f} Wh ({status['level_percent']:.1f}%) | Pos: {current_pos}")

    # Final Validation
    print("\n--- FINAL STATE VALIDATION ---")
    final_status = rth_system.power_manager.get_status()

    # Check if the remaining battery is enough for the predicted RTH trip
    dist_to_base = rth_system._calculate_distance_to_base()
    time_to_base = rth_system._estimate_time_to_base(dist_to_base)
    energy_needed_rth = CONSUMPTION_RATE["RTH_TRAVEL"] * time_to_base

    if rth_triggered and final_status['level_wh'] > energy_needed_rth:
        print(
            f"MVP Validation: Successful. RTH triggered autonomously at {final_status['level_percent']:.1f}% remaining, ensuring sufficient energy to return ({final_status['level_wh']:.2f} Wh > {energy_needed_rth:.2f} Wh).")
        print("The predictive model prevented critical power loss.")
    else:
        print(
            "MVP Validation: Failure. RTH prediction failed to trigger at an appropriate time or the remaining "
            "battery was insufficient.")
