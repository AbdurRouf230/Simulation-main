# Fault Tolerance Unit Tests
# File: test_fault_tolerance.py
#
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
#
# Contains unit tests to verify the decentralized failure detection and re-tasking logic.

import unittest
from unittest.mock import patch, MagicMock
from typing import List, Dict

# Import components from the target module
from fault_tollerance import DroneNode, SwarmFaultTolerance, HEALTH_CHECK_TIMEOUT_CYCLES, COMMUNICATION_RANGE_M

# Mock time to ensure deterministic test results
MOCK_START_TIME = 1000.0

@patch('fault_tolerance.time.time', return_value=MOCK_START_TIME)
class TestFaultTolerance(unittest.TestCase):

    def setUp(self):
        """Setup drones and the SwarmFaultTolerance system."""
        # Drone 1 (Receiver/Recoverer): Idle, healthy. Position (0,0)
        self.d1 = DroneNode(1, 0.0, 0.0)
        # Drone 2 (Victim): Holds a token. Position (50,0) - within comm range
        self.d2 = DroneNode(2, 50.0, 0.0)
        # Drone 3 (Interferer): Also healthy, holds a different token. Position (100,0)
        self.d3 = DroneNode(3, 100.0, 0.0)
        
        # Drones array and system setup
        self.drones = [self.d1, self.d2, self.d3]
        self.system = SwarmFaultTolerance(self.drones)
        
        # Setup Mission Token for D2
        self.FAILED_TOKEN = 'T-42'
        self.d2.update_local_state('token_held', self.FAILED_TOKEN)
        self.d2.update_local_state('active_mission', 'Mission-42')
        
        # Setup Mission Token for D3
        self.d3.update_local_state('token_held', 'T-10')
        
        # Ensure neighbors are found before running gossip
        self.system._find_neighbors(COMMUNICATION_RANGE_M)
        
        # Run a gossip step to ensure D1 and D3 know D2's initial state
        self.system._gossip_step()

        # D1 should know D2's state (and D3's)
        self.assertIn(2, self.d1.neighbors)
        self.assertIn(2, self.d1.known_neighbor_states)
        self.assertEqual(self.d1.known_neighbor_states[2]['token_held'], self.FAILED_TOKEN)


    # --- Test Core Recovery Logic ---

    @patch('fault_tolerance.time.time', side_effect=[
        MOCK_START_TIME, # Initial mock setup time (unused here)
        MOCK_START_TIME + HEALTH_CHECK_TIMEOUT_CYCLES + 1 # Time has advanced past timeout
    ])
    @patch('fault_tolerance.print', MagicMock())
    def test_recovery_success(self, mock_time, mock_patch):
        """Test that an idle drone successfully detects failure and claims the token."""
        
        # 1. Simulate D2's failure (it stops communicating, so its state is stale)
        self.d2.initiate_failure()
        # To simulate the network timeout, the second time.time() call (used in the check) 
        # is set to be past the timeout threshold.

        # 2. Execute the recovery check on the healthy, idle drone (D1)
        self.system._check_health_and_recover(self.d1)
        
        # 3. Validation
        # D1 should now hold the token
        self.assertEqual(self.d1.state['token_held'], self.FAILED_TOKEN)
        self.assertEqual(self.d1.state['active_mission'], f"RE-TASK_{self.FAILED_TOKEN}")
        # D2's token should be marked as None (orphaned/claimed)
        self.assertIsNone(self.d2.state['token_held'])

    @patch('fault_tolerance.time.time', side_effect=[
        MOCK_START_TIME, 
        MOCK_START_TIME + HEALTH_CHECK_TIMEOUT_CYCLES - 1 # Time has NOT advanced past timeout
    ])
    @patch('fault_tolerance.print', MagicMock())
    def test_no_recovery_if_not_timed_out(self, mock_time, mock_patch):
        """Test that recovery is prevented if the neighbor state is still considered fresh."""
        
        # 1. Simulate D2's failure (stops communicating)
        self.d2.initiate_failure()
        
        # 2. Execute the recovery check (time check should fail)
        self.system._check_health_and_recover(self.d1)
        
        # 3. Validation
        # D1 should remain idle
        self.assertIsNone(self.d1.state['token_held'])
        # D2's token should remain, as the failure wasn't confirmed
        self.assertEqual(self.d2.state['token_held'], self.FAILED_TOKEN)

    @patch('fault_tolerance.time.time', side_effect=[
        MOCK_START_TIME, 
        MOCK_START_TIME + HEALTH_CHECK_TIMEOUT_CYCLES + 1 
    ])
    @patch('fault_tolerance.print', MagicMock())
    def test_no_recovery_if_current_drone_busy(self, mock_time, mock_patch):
        """Test that a busy drone does not attempt to claim a new token, even if a failure is detected."""
        
        # 1. D1 is busy
        self.d1.update_local_state('token_held', 'D1_BUSY')
        
        # 2. Simulate D2's failure
        self.d2.initiate_failure()
        
        # 3. Execute the recovery check
        self.system._check_health_and_recover(self.d1)
        
        # 4. Validation
        # D1 should still hold its original token
        self.assertEqual(self.d1.state['token_held'], 'D1_BUSY')
        # D2's token should NOT be marked as None, as D1 never executed the claim logic
        self.assertEqual(self.d2.state['token_held'], self.FAILED_TOKEN)
        
if __name__ == '__main__':
    unittest.main(exit=False)

