# Automated Diagnostic and Self-Repair (OTA Update)
# File: diagnostic_repair.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
# 
# Simulates the drone's internal diagnostic system, its ability to attempt
# local self-repair (e.g., sensor calibration, component reset), and the
# secure Over-The-Air (OTA) firmware update protocol.

import random
import time
from typing import Dict, Any, List

# --- Configuration Constants ---
CURRENT_FW_VERSION = "2.3.1-beta"
NEW_FW_VERSION = "2.4.0-stable"
MAX_REPAIR_ATTEMPTS = 3


# --- Simulated Component State ---

class Component:
    """Represents a simulated hardware or software component."""

    def __init__(self, name: str, health: float):
        self.name = name
        self.health = health  # 0.0 to 1.0 (1.0 is perfect)
        self.is_operational = health >= 0.8
        self.error_count = 0

    def attempt_self_reset(self) -> bool:
        """Simulates a low-level software reset to fix a transient fault."""
        self.error_count += 1
        if self.error_count <= MAX_REPAIR_ATTEMPTS:
            # Assume reset has a 70% chance of temporarily restoring function
            if random.random() > 0.3:
                self.health = 0.95  # Temporary improvement
                self.is_operational = True
                return True
        self.is_operational = False
        return False


# --- Core Diagnostic and Update Logic ---

class DiagnosticSystem:
    """Manages system health monitoring and self-repair decisions."""

    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.firmware_version = CURRENT_FW_VERSION
        self.components: Dict[str, Component] = {
            "GNC_Sensor": Component("GNC_Sensor", 0.98),
            "Micro_Actuator": Component("Micro_Actuator", 0.95),
            "Vision_AI": Component("Vision_AI", 0.82)
        }
        self.logs: List[str] = []

    def _log(self, message: str):
        """Internal logging function."""
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] Drone {self.drone_id}: {message}")
        print(message)

    def run_diagnostics(self):
        """Checks all components and attempts self-repair if needed."""
        self._log("Running deep system diagnostics...")

        for name, comp in self.components.items():
            if comp.health < 0.8 and comp.is_operational:
                # Component is degrading but still running (Warning)
                self._log(f"[WARNING] {name} health is low ({comp.health:.2f}).")

            elif not comp.is_operational:
                self._log(f"[FAULT] {name} non-operational. Health: {comp.health:.2f}. Attempting self-repair...")

                if comp.attempt_self_reset():
                    self._log(f"[REPAIR SUCCESS] {name} restored via software reset (Attempt {comp.error_count}).")
                else:
                    self._log(f"[REPAIR FAILURE] {name} failed self-repair. Requires RTH for manual inspection.")
                    # Trigger mission abort/RTH command
                    return f"CRITICAL_FAILURE: {name}"

        self._log("Diagnostics complete. System operational status checked.")
        return "OK"

    def execute_ota_update(self, new_version: str) -> bool:
        """Simulates receiving and installing a secure OTA firmware package."""
        self._log(f"Initiating OTA update from {self.firmware_version} to {new_version}...")

        # 1. Security Check (Simulated: Assume crypto handshake passed)
        if random.random() < 0.05:  # 5% chance of checksum failure
            self._log("[OTA FAILURE] Checksum validation failed. Aborting update.")
            return False

        # 2. Pre-update safety check (e.g., battery level, stable ground)
        if self.run_diagnostics() != "OK":
            self._log("[OTA FAILURE] Diagnostics flagged non-operational component. Deferring update.")
            return False

        # 3. Installation sequence
        time.sleep(2)  # Simulated download and install time
        self.firmware_version = new_version

        # 4. Post-update self-test
        self._log("[OTA SUCCESS] Firmware updated. Running post-install self-test...")

        # Simulate successful reboot and health check
        for name in self.components:
            self.components[name].health = min(1.0,
                                               self.components[name].health + 0.1)  # Simulate health improvement/reset
            self.components[name].is_operational = True

        self._log(f"New Firmware Version: {self.firmware_version}. All components verified.")
        return True


# --- Simulation Run ---

if __name__ == "__main__":
    random.seed(42)
    diag_system = DiagnosticSystem(drone_id=7)

    print("--- Automated Diagnostic and Self-Repair Simulation ---")
    print(f"Initial Firmware: {diag_system.firmware_version}")
    print("-" * 60)

    # --- SCENARIO 1: Transient Fault and Self-Repair ---

    # Induce a transient failure in the GNC Sensor component
    diag_system.components["GNC_Sensor"].health = 0.5
    diag_system.components["GNC_Sensor"].is_operational = False

    print("\n[CYCLE 1: DIAGNOSTICS WITH FAULT]")
    status = diag_system.run_diagnostics()

    # --- SCENARIO 2: Post-Repair Status Check and OTA Update ---

    if status == "OK" and diag_system.components["GNC_Sensor"].is_operational:
        print("\n[CYCLE 2: SYSTEM IS NOW STABLE]")

        # Attempt the OTA update
        update_success = diag_system.execute_ota_update(NEW_FW_VERSION)

        print("\n--- MVP Validation Check ---")
        if update_success and diag_system.firmware_version == NEW_FW_VERSION:
            print(
                "MVP Validation: Successful. The diagnostic system autonomously recovered from a transient fault, "
                "stabilized, and then executed a secure Over-The-Air (OTA) firmware update, demonstrating resilience "
                "and maintainability.")

        else:
            print(
                "MVP Validation: Failure. The OTA update process failed or the version number was not correctly "
                "updated.")
    else:
        print("\n--- MVP Validation Check ---")
        print("MVP Validation: Failure. The diagnostic system was unable to self-repair the induced fault.")
