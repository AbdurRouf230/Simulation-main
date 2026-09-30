# Weather and Environmental Sensing Integration
# File: weather_sensing.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
# 
# Simulates the drone's ability to ingest local environmental data (wind, temp)
# and translate these measurements into actionable, real-time safety and GNC adjustments.

import random
import time
from typing import Dict, Any, Tuple

# --- Configuration Constants ---
MAX_SAFE_WIND_SPEED_MPS = 8.0  # Meters per second (Threshold for RTH or Emergency Land)
MAX_WIND_ADJUSTMENT_FACTOR = 1.5  # Maximum GNC compensation factor for wind


# --- Data Structures (Simulated Weather Sensor) ---

class EnvironmentalSensor:
    """Simulates the collection of local environmental data."""

    def __init__(self, location: Tuple[float, float]):
        self.location = location
        self.wind_speed_mps = 0.0
        self.wind_direction_deg = 0
        self.temperature_c = 25.0
        self.precipitation_level = 0  # 0=None, 1=Light, 2=Heavy

    def update_readings(self, wind_speed: float, temp: float, precip: int):
        """Manually updates the sensor readings for simulation."""
        self.wind_speed_mps = wind_speed
        self.temperature_c = temp
        self.precipitation_level = precip
        self.wind_direction_deg = random.randint(0, 359)  # Direction is random for simplicity

    def get_readings(self) -> Dict[str, Any]:
        """Returns the current sensor data."""
        return {
            'wind_speed_mps': self.wind_speed_mps,
            'wind_direction_deg': self.wind_direction_deg,
            'temperature_c': self.temperature_c,
            'precipitation_level': self.precipitation_level
        }


# --- Core Safety and Adjustment Logic ---

class WeatherSafetySystem:
    """Processes sensor data and determines flight mode/GNC compensation."""

    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.current_flight_mode = "SEARCHING"
        self.gnc_compensation_factor = 1.0  # Multiplier for thrust/control inputs

    def calculate_gnc_adjustment(self, wind_speed: float) -> float:
        """
        Calculates the GNC thrust compensation required to maintain position
        in adverse wind conditions. Higher wind = higher factor.
        """
        if wind_speed <= 2.0:
            return 1.0  # No adjustment needed (calm)

        # Scale the adjustment factor based on wind strength, capped by MAX_WIND_ADJUSTMENT_FACTOR
        adjustment = 1.0 + (wind_speed / MAX_SAFE_WIND_SPEED_MPS) * (MAX_WIND_ADJUSTMENT_FACTOR - 1.0)

        return min(adjustment, MAX_WIND_ADJUSTMENT_FACTOR)

    def assess_safety_and_adjust_mode(self, readings: Dict[str, Any]) -> str:
        """
        The core logic for environmental safety: determines mission changes.
        :returns: The recommended new flight mode.
        """
        wind_speed = readings['wind_speed_mps']
        precip = readings['precipitation_level']

        # 1. Update GNC compensation based on wind speed
        self.gnc_compensation_factor = self.calculate_gnc_adjustment(wind_speed)

        # 2. Safety Decisions (Highest priority first)

        if precip >= 2:  # Heavy precipitation (Severe rain/hail)
            self.current_flight_mode = "EMERGENCY_LAND"
            return f"CRITICAL: Heavy Precipitation detected. Forcing EMERGENCY_LAND. GNC Factor: {self.gnc_compensation_factor:.2f}"

        elif precip == 1:  # Light precipitation
            self.current_flight_mode = "RTH_TRAVEL"
            return f"HIGH ALERT: Light Precipitation detected. Initiating RTH_TRAVEL. GNC Factor: {self.gnc_compensation_factor:.2f}"

        elif wind_speed > MAX_SAFE_WIND_SPEED_MPS:  # High Wind Check
            self.current_flight_mode = "RTH_TRAVEL"  # Cannot maintain mission, needs RTH
            return f"HIGH ALERT: Wind speed ({wind_speed:.1f} m/s) above safety threshold. Initiating RTH_TRAVEL. GNC Factor: {self.gnc_compensation_factor:.2f}"

        elif wind_speed > MAX_SAFE_WIND_SPEED_MPS * 0.75:  # Moderate Wind Check (Still flyable but taxing)
            self.current_flight_mode = "HOVER" if self.current_flight_mode == "POLLINATING" else "SEARCHING"  # Stay in mission but be cautious
            return f"WARNING: High Wind Speed ({wind_speed:.1f} m/s). Adjusted GNC Factor to {self.gnc_compensation_factor:.2f}."

        else:
            self.current_flight_mode = "SEARCHING"  # Clear conditions
            return f"STATUS: Environmental conditions nominal. GNC Factor: {self.gnc_compensation_factor:.2f}"


# --- Simulation Run ---

if __name__ == "__main__":
    random.seed(42)

    sensor = EnvironmentalSensor(location=(40.7, -74.0))
    safety_system = WeatherSafetySystem(drone_id=12)

    print("--- Weather and Environmental Sensing Simulation ---")
    print(f"Max Safe Wind Speed: {MAX_SAFE_WIND_SPEED_MPS} m/s")
    print("-" * 60)

    # Simulated Environmental Cycles (Wind and Weather Severity Increase)
    scenarios = [
        # Scenario 1: Nominal conditions
        (2.5, 20.0, 0, "Nominal Flight"),
        # Scenario 2: Moderate Wind - Requires GNC compensation
        (6.5, 18.0, 0, "Moderate Wind (GNC Compensation)"),
        # Scenario 3: Wind exceeds safety threshold - RTH triggered
        (9.0, 17.0, 0, "High Wind (RTH Trigger)"),
        # Scenario 4: Heavy Rain/Precipitation - Emergency Land triggered
        (3.0, 15.0, 2, "Heavy Rain (Emergency Land)")
    ]

    for wind, temp, precip, description in scenarios:
        print(f"\n[SCENARIO] {description}")

        # 1. Update sensor readings
        sensor.update_readings(wind, temp, precip)
        readings = sensor.get_readings()

        # 2. Process safety decision
        decision_message = safety_system.assess_safety_and_adjust_mode(readings)

        # 3. Output
        print(f"  Sensor Input: Wind={readings['wind_speed_mps']:.1f} m/s, Precip={readings['precipitation_level']}")
        print(f"  Safety Decision: {decision_message}")
        print(f"  New Flight Mode: {safety_system.current_flight_mode}")
        print(f"  GNC Factor: {safety_system.gnc_compensation_factor:.2f}")

    # Final Validation
    print("\n--- MVP Validation Check ---")

    # Check if the last decision (Heavy Rain) resulted in EMERGENCY_LAND
    final_mode = safety_system.current_flight_mode
    final_gnc_factor = safety_system.gnc_compensation_factor

    if final_mode == "EMERGENCY_LAND" and final_gnc_factor < MAX_WIND_ADJUSTMENT_FACTOR:
        print(
            "MVP Validation: Successful. The Weather Safety System correctly prioritized the critical precipitation warning over wind and forced an EMERGENCY_LAND, demonstrating adaptive safety logic.")
    else:
        print(
            "MVP Validation: Failure. The safety system did not correctly transition to the critical flight mode based on environmental inputs.")
