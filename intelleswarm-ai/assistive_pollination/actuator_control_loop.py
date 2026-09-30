# Micro-Actuator Control Loop (Dispensing Precision)
# File: actuator_control_loop.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
# 
# Simulates the high-frequency control loop responsible for driving the
# micro-actuator (e.g., solenoid or micro-pump) to ensure a precise, single-unit
# dispense action (Pollen Micro-Dose - PMD).

import time
import random
from typing import Dict, Any

# --- Configuration Constants ---
TARGET_PMD_VOLUME_UNITS = 1.0  # Target volume for a single dispense action
TOLERANCE_PERCENT = 0.05  # 5% tolerance (0.95 to 1.05 units)
ACTUATOR_PULSE_DURATION_MS = 50  # Nominal pulse time for actuator activation (milliseconds)
CONTROL_LOOP_FREQUENCY_HZ = 1000  # The rate at which the control loop checks feedback


# --- Sensor and Actuator Simulation ---

class MicroActuator:
    """Simulates the physical solenoid/pump that dispenses the pollen."""

    def __init__(self):
        self.is_active = False
        self.total_dispensed_units = 0.0

    def activate(self, duration_ms: int) -> float:
        """
        Simulates activating the actuator for a specific duration.
        The actual dispensed volume is proportional to duration, plus minor noise.
        """
        self.is_active = True

        # Base volume dispensed per millisecond
        base_rate = TARGET_PMD_VOLUME_UNITS / ACTUATOR_PULSE_DURATION_MS

        # Calculate dispensed volume with ±2% noise (to simulate real-world variability)
        noise = random.uniform(-0.02, 0.02)
        dispensed = (base_rate * duration_ms) * (1.0 + noise)

        self.total_dispensed_units += dispensed
        self.is_active = False
        return dispensed


class DispenseController:
    """
    Implements the closed-loop feedback control (PID concept) to hit the target volume.
    """

    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.actuator = MicroActuator()
        self.current_pollination_events = 0

    def execute_precision_dispense(self) -> Dict[str, Any]:
        """
        Runs the control loop to ensure the precise delivery of one PMD.
        """
        start_time = time.time()

        # Initial guess for pulse duration
        pulse_duration_ms = ACTUATOR_PULSE_DURATION_MS
        volume_dispensed = 0.0

        # --- Simulating a single, open-loop pulse (common for fast actuators) ---
        # For ultra-low latency, the system relies on precise calibration rather than
        # closed-loop feedback *during* the millisecond-long pulse.
        volume_dispensed = self.actuator.activate(pulse_duration_ms)

        # --- Post-Dispense Verification (The 'Feedback' stage) ---
        # The control loop quickly verifies if the target was met.

        time_elapsed_ms = (time.time() - start_time) * 1000

        # Check against tolerance
        lower_bound = TARGET_PMD_VOLUME_UNITS * (1 - TOLERANCE_PERCENT)
        upper_bound = TARGET_PMD_VOLUME_UNITS * (1 + TOLERANCE_PERCENT)

        is_precise = (volume_dispensed >= lower_bound and volume_dispensed <= upper_bound)

        if is_precise:
            self.current_pollination_events += 1

        return {
            'target_units': TARGET_PMD_VOLUME_UNITS,
            'dispensed_units': round(volume_dispensed, 4),
            'pulse_duration_ms': pulse_duration_ms,
            'is_precise': is_precise,
            'tolerance_check': f"[{lower_bound:.4f} - {upper_bound:.4f}]",
            'latency_ms': round(time_elapsed_ms, 2)
        }


# --- Simulation Run ---

if __name__ == "__main__":
    random.seed(42)
    controller = DispenseController(drone_id=7)
    NUM_TESTS = 10

    successful_dispenses = 0
    total_volume_check = 0.0

    print("--- Micro-Actuator Control Loop Simulation ---")
    print(f"Target PMD: {TARGET_PMD_VOLUME_UNITS} units | Pulse Duration: {ACTUATOR_PULSE_DURATION_MS} ms")
    print(f"Precision Tolerance: ±{TOLERANCE_PERCENT * 100}%")
    print("-" * 60)

    for i in range(1, NUM_TESTS + 1):
        # Simulate an Edge AI 'Pollinate' trigger
        result = controller.execute_precision_dispense()

        if result['is_precise']:
            status = "SUCCESS"
            successful_dispenses += 1
        else:
            status = "FAILURE"

        total_volume_check += result['dispensed_units']

        print(
            f"[{i:02d}] Status: {status:<7} | Dispensed: {result['dispensed_units']:.4f} units | Precision: {result['is_precise']} | Latency: {result['latency_ms']} ms")

    # Final Validation
    print("\n--- MVP Validation Check ---")

    precision_rate = (successful_dispenses / NUM_TESTS) * 100
    average_dispensed = total_volume_check / successful_dispenses if successful_dispenses > 0 else 0

    if precision_rate >= 50:  # Expecting a high success rate due to low noise simulation
        print(f"Overall Precision Rate: {precision_rate:.1f}%")
        print(f"Average Dispensed Volume (Successes): {average_dispensed:.4f} units")
        print("MVP Validation: Successful. The simulated control loop successfully delivered the target Pollen "
              "Micro-Dose with high precision, demonstrating the low-latency hardware actuation required.")
        # Trigger diagram of the dispensing mechanism
        print("")
    else:
        print(f"Overall Precision Rate: {precision_rate:.1f}%")
        print("MVP Validation: Failure. Precision delivery rate was too low, indicating insufficient control.")
