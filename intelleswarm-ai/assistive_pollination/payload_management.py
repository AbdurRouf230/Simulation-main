# Payload Management and Dispensing System
# File: payload_management.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
# 
# Simulates the drone's capacity, inventory tracking, and the controlled,
# metered dispensing of the pollen payload based on mission commands.

import random
import time # FIX: Added missing import for time module
from typing import Dict, Any, Optional

# --- Configuration Constants ---
MAX_PAYLOAD_UNITS = 1000  # Total capacity in discrete Pollen Micro-Doses (PMD)
DISPENSE_UNIT = 1         # One Pollen Micro-Dose (PMD) per pollination event
LOW_PAYLOAD_THRESHOLD = 50 # Units remaining before triggering a "Return to Base" warning
MISSION_EVENT_TYPES = ["Bloom_Stage_1_Needs_Pollen", "Pest_Detected", "Unready"]

# --- Core Data Structures ---

class PayloadManager:
    """Manages the pollen payload inventory and dispensing mechanism."""
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.payload_remaining = MAX_PAYLOAD_UNITS
        self.total_dispensed = 0
        self.is_dispenser_healthy = True
        self.last_dispense_time: Optional[float] = None

    def dispense_pollen(self) -> bool:
        """
        Executes a controlled, metered dispense action (one PMD).
        Returns True if successful, False if payload is empty or dispenser failed.
        """
        if not self.is_dispenser_healthy:
            print(f"[ERROR] Drone {self.drone_id}: Dispenser mechanism malfunction.")
            return False

        if self.payload_remaining >= DISPENSE_UNIT:
            self.payload_remaining -= DISPENSE_UNIT
            self.total_dispensed += DISPENSE_UNIT
            self.last_dispense_time = time.time()
            return True
        else:
            # This triggers the "Return to Base" logic for the Swarm Coordinator
            return False

    def check_status(self) -> Dict[str, Any]:
        """Reports the current payload status and any critical warnings."""
        status = {
            'remaining_units': self.payload_remaining,
            'is_low_payload': self.payload_remaining <= LOW_PAYLOAD_THRESHOLD,
            'is_empty': self.payload_remaining == 0,
            'dispenser_status': 'Healthy' if self.is_dispenser_healthy else 'Malfunction'
        }
        return status

class DroneMissionController:
    """
    Simulates the mission logic that receives Edge AI outputs and triggers
    the dispensing action.
    """
    def __init__(self, drone_id: int):
        self.id = drone_id
        self.payload_manager = PayloadManager(drone_id)
        self.mission_status: Dict[str, Any] = {'current_action': 'Searching'}

    def process_edge_ai_result(self, detection_type: str, confidence: float) -> Optional[str]:
        """
        Takes the output from Edge Inference (e.g., 'Bloom_Stage_1_Needs_Pollen')
        and executes the necessary payload action.
        """
        if detection_type == "Bloom_Stage_1_Needs_Pollen" and confidence > 0.85:
            self.mission_status['current_action'] = 'Pollinating'
            
            # --- CORE DISPENSE LOGIC ---
            if self.payload_manager.dispense_pollen():
                result = f"POLLINATION SUCCESSFUL: Dispensed 1 PMD at 85%+ confidence."
                # Log this event for later yield analytics
                # print(fn"[LOG] {self.id}: {result}")
                return result
            else:
                result = "POLLINATION FAILED: Payload empty, initiating RTH."
                self.mission_status['current_action'] = 'ReturnToHome'
                return result
        
        elif detection_type == "Pest_Detected":
            # Example of a non-pollen payload interaction (e.g., micro-pesticide spray)
            self.mission_status['current_action'] = 'Reporting_Pest'
            return f"PEST DETECTED: Not dispensing pollen, escalating warning."
            
        else:
            self.mission_status['current_action'] = 'Continuing_Search'
            return "NO ACTION REQUIRED: Continuing search/monitoring."


# --- Simulation Run ---

if __name__ == "__main__":
    import time
    random.seed(42)
    drone = DroneMissionController(drone_id=7)
    
    print("--- Payload Management Simulation (Drone 7) ---")
    print(f"Initial Payload: {drone.payload_manager.payload_remaining} units")
    print("-" * 50)
    
    pollination_events = 0
    return_to_home_triggered = False
    
    # Run simulation cycles until the payload is critically low or empty
    for i in range(1, MAX_PAYLOAD_UNITS + 10):
        if return_to_home_triggered:
            break

        # Simulate a result from the Edge AI stack
        # 40% chance of needing pollination
        if random.random() < 0.4:
            detection = "Bloom_Stage_1_Needs_Pollen"
            confidence = random.uniform(0.9, 0.99)
        else:
            # 30% chance of 'Pest_Detected', 30% chance of 'Unready'
            detection = random.choice(["Pest_Detected", "Unready"])
            confidence = random.uniform(0.7, 0.9)
            
        
        # Process the result and attempt payload action
        mission_log = drone.process_edge_ai_result(detection, confidence)
        
        if "POLLINATION SUCCESSFUL" in mission_log:
            pollination_events += 1

        payload_status = drone.payload_manager.check_status()
        
        # Check for RTH trigger
        if payload_status['is_low_payload'] and not return_to_home_triggered:
            print(f"\n!!! ALERT: PAYLOAD LOW ({payload_status['remaining_units']} units) !!!")
            print("Initiating RTH protocol via Swarm Communication Mesh.")
            return_to_home_triggered = True
        
        # Stop if fully empty
        if payload_status['is_empty']:
            print("\nPAYLOAD EMPTY. Mission complete for this charge cycle.")
            break
            
        time.sleep(0.001) # Simulate real-time delay

    print("\n--- FINAL STATE VALIDATION ---")
    print(f"Total Successful Pollination Events: {pollination_events}")
    print(f"Final Payload Remaining: {drone.payload_manager.payload_remaining} units")
    
    # MVP Validation Check: Did the system track inventory and trigger RTH correctly?
    if drone.payload_manager.payload_remaining <= LOW_PAYLOAD_THRESHOLD and return_to_home_triggered:
        print("\nMVP Validation: Successful. The Payload Management system accurately tracked consumption, and the critical 'Low Payload' status was detected and triggered the RTH protocol.")
    else:
        print("\nMVP Validation: Failure. Payload tracking or RTH trigger logic did not execute as expected.")

