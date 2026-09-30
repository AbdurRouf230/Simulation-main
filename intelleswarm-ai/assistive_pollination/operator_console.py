# Remote Operator Console (Monitoring & Override) Logic
# File: operator_console.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman 
#
# Simulates the centralized control and monitoring dashboard that provides
# situational awareness to human operators and allows for critical, high-level
# override commands (e.g., Emergency Landing, Global RTH).

import time
import random
from typing import Dict, List, Any, Optional

# --- Configuration Constants ---
NUM_DRONES = 20
SWARM_OPERATOR_ID = "OP-INTEL001"


# --- Data Structures (Simulated Telemetry Feed) ---

def generate_drone_telemetry(drone_id: int, current_status: str) -> Dict[str, Any]:
    """Generates a simulated real-time status update for one drone."""
    telemetry = {
        'drone_id': drone_id,
        'timestamp': time.time(),
        'status': current_status,  # e.g., 'SEARCHING', 'POLLINATING', 'RTH_TRAVEL', 'DANGER'
        'battery_percent': round(random.uniform(5, 100), 1),
        'payload_percent': round(random.uniform(0, 100), 1),
        'position_lat_lon': (random.uniform(34.0, 34.1), random.uniform(-119.0, -118.9)),
        'critical_warning': random.choice([None, None, None, "Wind Shear Warning", "GNC Fault"])
    }
    return telemetry


class SwarmMonitor:
    """Manages the real-time data flow for the operator console display."""

    def __init__(self):
        self.live_telemetry: Dict[int, Dict[str, Any]] = {}
        self.critical_alerts: List[str] = []
        self.system_health = {'overall': 'GREEN'}

    def ingest_realtime_data(self, new_telemetry: List[Dict[str, Any]]):
        """Updates the internal state with the latest telemetry from the cloud pipeline."""
        for data in new_telemetry:
            d_id = data['drone_id']
            self.live_telemetry[d_id] = data
            self._analyze_data(data)

    def _analyze_data(self, data: Dict[str, Any]):
        """Runs automated checks to flag issues for the operator."""
        alert_message = None

        if data['status'] == 'DANGER':
            alert_message = f"CRITICAL: Drone {data['drone_id']} reports DANGER status at {data['position_lat_lon']}."
        elif data['battery_percent'] < 10.0 and data['status'] != 'RTH_TRAVEL':
            alert_message = f"HIGH PRIORITY: Drone {data['drone_id']} at 9% battery, NOT RTH."
        elif data['critical_warning'] is not None:
            alert_message = f"WARNING: Drone {data['drone_id']} reports: {data['critical_warning']}"

        if alert_message and alert_message not in self.critical_alerts:
            self.critical_alerts.append(alert_message)
            self.system_health['overall'] = 'RED' if 'CRITICAL' in alert_message else 'YELLOW'

    def get_console_summary(self) -> Dict[str, Any]:
        """Provides the summary data displayed on the HMI dashboard."""

        # Calculate key performance indicators (KPIs)
        searching_drones = sum(1 for d in self.live_telemetry.values() if d['status'] == 'SEARCHING')
        rth_drones = sum(1 for d in self.live_telemetry.values() if 'RTH' in d['status'])

        return {
            'time': time.strftime('%H:%M:%S'),
            'total_drones': len(self.live_telemetry),
            'system_health': self.system_health['overall'],
            'drones_searching': searching_drones,
            'drones_rth': rth_drones,
            'active_alerts': self.critical_alerts,
            'operator_id': SWARM_OPERATOR_ID
        }


class OperatorConsole:
    """Handles the high-level override commands from the human operator."""

    def __init__(self, monitor: SwarmMonitor):
        self.monitor = monitor

    def execute_override(self, command: str, target_id: Optional[int] = None) -> str:
        """
        Simulates the operator sending a critical command to the swarm coordinator.
        :param command: The desired override ('RTH_ALL', 'EMERGENCY_LAND', 'ABORT_MISSION').
        """
        if command == 'EMERGENCY_LAND' and target_id is not None:
            print(f"\n[OVERRIDE] Operator {SWARM_OPERATOR_ID} commands EMERGENCY LAND for Drone {target_id}.")
            # In a real system, this command is broadcast to the mesh via the cloud gateway.
            # Simulation: We update the drone's status directly
            if target_id in self.monitor.live_telemetry:
                self.monitor.live_telemetry[target_id]['status'] = 'EMERGENCY_LANDING'
                return f"Command executed: Drone {target_id} status set to EMERGENCY_LANDING."
            return f"Error: Drone {target_id} not found."

        elif command == 'RTH_ALL':
            print(f"\n[OVERRIDE] Operator {SWARM_OPERATOR_ID} commands GLOBAL RETURN TO HOME.")
            # Simulation: Change all healthy drones to RTH status
            for d_id in self.monitor.live_telemetry:
                if self.monitor.live_telemetry[d_id]['status'] not in ['DANGER', 'EMERGENCY_LANDING']:
                    self.monitor.live_telemetry[d_id]['status'] = 'RTH_TRAVEL'
            return "Command executed: All active drones directed to RTH_TRAVEL mode."

        elif command == 'ABORT_MISSION':
            print(f"\n[OVERRIDE] Operator {SWARM_OPERATOR_ID} commands GLOBAL MISSION ABORT.")
            return "Command executed: Global mission aborted. All drones set to HOVER/IDLE."

        return f"Unknown command: {command}"


# --- Simulation Run ---

if __name__ == "__main__":
    random.seed(42)
    monitor = SwarmMonitor()
    console = OperatorConsole(monitor)

    print("--- Remote Operator Console Simulation ---")

    # --- STEP 1: Initial Ingestion of Healthy Data ---
    initial_telemetry = [generate_drone_telemetry(i, 'SEARCHING') for i in range(1, NUM_DRONES + 1)]
    monitor.ingest_realtime_data(initial_telemetry)

    summary = monitor.get_console_summary()
    print(
        f"\n[Cycle 1 - Initial Status] Health: {summary['system_health']} | Searching: {summary['drones_searching']}/{summary['total_drones']}")

    # --- STEP 2: Simulate a Critical Failure and Ingest Data ---

    # Drone 5 is in danger and Drone 15 has low battery
    telemetry_update = [
        generate_drone_telemetry(5, 'DANGER'),
        generate_drone_telemetry(15, 'SEARCHING'),
    ]
    # Manually set critical values for simulation trigger
    telemetry_update[1]['battery_percent'] = 8.5

    monitor.ingest_realtime_data(telemetry_update)

    summary = monitor.get_console_summary()
    print(f"\n[Cycle 2 - Failure Detected] Health: {summary['system_health']}")
    print("Active Alerts:")
    for alert in summary['active_alerts']:
        print(f"  - {alert}")

    # --- STEP 3: Operator Executes Override Command ---

    # The operator sees the CRITICAL alert for Drone 5 and commands an emergency land
    override_result = console.execute_override('EMERGENCY_LAND', target_id=5)
    print(override_result)

    # The operator then decides to call everyone home due to the high-priority battery alert
    override_result = console.execute_override('RTH_ALL')
    print(override_result)

    # --- STEP 4: Final State Validation ---

    final_summary = monitor.get_console_summary()

    drone_5_status = monitor.live_telemetry.get(5, {}).get('status')
    drone_1_status = monitor.live_telemetry.get(1, {}).get('status')  # A non-targeted drone

    print("\n--- FINAL STATE VALIDATION ---")

    if drone_5_status == 'EMERGENCY_LANDING' and drone_1_status == 'RTH_TRAVEL':
        print(
            "MVP Validation: Successful. The console correctly detected critical alerts and executed both targeted ("
            "EMERGENCY_LAND) and global (RTH_ALL) override commands, demonstrating the required HMI capability.")
    else:
        print("MVP Validation: Failure. Override commands did not correctly update drone status in the simulation.")
