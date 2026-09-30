# Mesh Communication Unit Tests
# File: test_mesh_communication.py
#
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
#
# Contains unit tests to verify the core functionality of the decentralized mesh communication.

import unittest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Tuple

# Import functions and classes from the target module
from mesh_communication import DroneNode, find_neighbors, run_gossip_cycle, COMMUNICATION_RANGE_M


# Mock the logger to prevent test output noise from print statements in the code
@patch('mesh_communication.print')
class TestMeshCommunication(unittest.TestCase):

    def setUp(self):
        """Set up standard drone nodes for testing."""
        # Note: We need to mock time for deterministic timestamping
        with patch('mesh_communication.time.time', return_value=100.0):
            self.drone1 = DroneNode(1, 0.0, 0.0)  # Position (0, 0)
            self.drone2 = DroneNode(2, 50.0, 0.0)  # Position (50, 0)
            self.drone3 = DroneNode(3, 300.0, 0.0)  # Position (300, 0) - Far away
            self.drones = [self.drone1, self.drone2, self.drone3]
            self.drone_map = {d.id: d for d in self.drones}

    # --- Test DroneNode Internal Logic ---

    @patch('mesh_communication.time.time', return_value=101.5)
    def test_update_local_state_updates_timestamp(self, mock_time, mock_print):
        """Verify that local state updates correctly change the timestamp."""
        self.drone1.update_local_state('is_healthy', False)
        self.assertFalse(self.drone1.state['is_healthy'])
        self.assertEqual(self.drone1.state['timestamp'], 101.5)

    def test_get_gossip_payload(self, mock_print):
        """Verify the payload structure is correct."""
        payload = self.drone1.get_gossip_payload()
        self.assertEqual(payload['sender_id'], 1)
        self.assertIn('token_held', payload['state'])
        self.assertEqual(payload['timestamp'], 100.0)

    # --- Test Gossip Processing Logic (Conflict Resolution) ---

    @patch('mesh_communication.time.time', side_effect=[102.0, 102.0])
    def test_process_gossip_token_update_newer(self, mock_time, mock_print):
        """Newer token state should overwrite older state."""

        # Drone 1's state is at time 100.0 (old)

        # Incoming payload from Drone 2 (time 101.0, token P-20)
        incoming = {
            'sender_id': 2,
            'state': {'token_held': 'P-20', 'is_healthy': True},
            'timestamp': 101.0
        }

        self.drone1.process_gossip(incoming)

        # State should be updated because incoming timestamp (101.0) > local timestamp (100.0)
        self.assertEqual(self.drone1.state['token_held'], 'P-20')
        self.assertEqual(self.drone1.state['timestamp'], 102.0)  # Updated by update_local_state call
        self.assertTrue(self.drone1.state['last_known_health'][2])

    def test_process_gossip_token_ignored_older(self, mock_print):
        """Older token state should be ignored."""

        # Setup: Drone 1 claims a token, increasing its local timestamp
        with patch('mesh_communication.time.time', return_value=105.0):
            self.drone1.update_local_state('token_held', 'P-10')
            self.assertEqual(self.drone1.state['timestamp'], 105.0)

        # Incoming payload from Drone 2 (time 103.0, token P-20) - OLDER STATE
        incoming = {
            'sender_id': 2,
            'state': {'token_held': 'P-20', 'is_healthy': True},
            'timestamp': 103.0
        }

        self.drone1.process_gossip(incoming)

        # State should NOT be updated
        self.assertEqual(self.drone1.state['token_held'], 'P-10')
        self.assertEqual(self.drone1.state['timestamp'], 105.0)  # Timestamp should remain the same

    @patch('mesh_communication.time.time', return_value=106.0)
    def test_process_gossip_critical_warning_precedence(self, mock_time, mock_print):
        """A new critical warning should always update local state, regardless of timestamp."""

        # Setup: Drone 1 has an older token state (time 100.0)

        # Incoming payload with a critical warning (even if timestamp is old, it should propagate)
        incoming = {
            'sender_id': 2,
            'state': {'critical_warning': 'PEST', 'timestamp': 99.0},  # Older timestamp
            'timestamp': 99.0
        }

        self.drone1.process_gossip(incoming)

        # Warning should be updated
        self.assertEqual(self.drone1.state['critical_warning'], 'PEST')
        # Local timestamp should be updated by update_local_state()
        self.assertEqual(self.drone1.state['timestamp'], 106.0)

    # --- Test Network Utility Functions ---

    def test_find_neighbors(self, mock_print):
        """Test neighbor discovery based on communication range."""

        # Range is 150m. Drone 1(0,0), Drone 2(50,0), Drone 3(300,0)

        find_neighbors(self.drones, COMMUNICATION_RANGE_M)

        # Drone 1 (0,0) is close to Drone 2 (50,0) -> Distance 50m
        self.assertIn(2, self.drone1.neighbors)
        self.assertNotIn(3, self.drone1.neighbors)
        self.assertEqual(len(self.drone1.neighbors), 1)

        # Drone 3 (300,0) is too far from both
        self.assertEqual(len(self.drone3.neighbors), 0)

    # --- Test Gossip Cycle Execution ---

    @patch('mesh_communication.random.sample', side_effect=[[2], [1], []])
    def test_run_gossip_cycle_propagation(self, mock_sample, mock_print):
        """Test that state propagates between neighbors in a cycle."""

        # 1. Setup Connectivity (d1 <-> d2)
        find_neighbors(self.drones, COMMUNICATION_RANGE_M)

        # 2. Setup initial state for propagation (d1 has a token at T=100.0)
        self.drone1.state['token_held'] = 'P-1'
        self.drone1.state['timestamp'] = 100.0

        # 3. Setup d2's local time to be older
        self.drone2.state['timestamp'] = 90.0
        self.assertIsNone(self.drone2.state['token_held'])

        # 4. Run Cycle
        # mock_sample forces: d1 gossips to d2. d2 gossips to d1. d3 has no neighbors.
        with patch('mesh_communication.time.time', return_value=101.0):
            run_gossip_cycle(self.drones, self.drone_map)

        # 5. Verify Propagation
        # Drone 2 should have received the token from Drone 1
        self.assertEqual(self.drone2.state['token_held'], 'P-1')
        # Drone 2's timestamp should be updated locally due to incoming state
        self.assertEqual(self.drone2.state['timestamp'], 101.0)

        # Drone 1 should not have changed state (it received its own older state)
        self.assertEqual(self.drone1.state['token_held'], 'P-1')
        self.assertEqual(self.drone1.state['timestamp'], 100.0)


if __name__ == '__main__':
    unittest.main(exit=False)
