# Swarm Coordination Unit Tests
# File: test_swarm_coordination.py
#
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
#
# Contains unit tests to verify the core functionality of the swarm system.

import unittest
from unittest.mock import patch, MagicMock
from typing import List

# Import necessary dependencies from the IntelleSwarm project structure
from drone_model import Drone
from environment import SubRegion
from utils import calculate_distance
# The communication protocol state is used to verify network side-effects
from communication_protocol import initialize_token_state, _TOKEN_STATE

# Import functions from the target module
from swarm_coordination import (
    request_token,
    claim_region,
    release_token,
    decide_next_move
)

# Mock logger and constants for isolated unit testing
logger = MagicMock()
SIMULATED_CLOSE_DISTANCE = 5


class TestSwarmCoordination(unittest.TestCase):

    def setUp(self):
        """Setup fresh drone and region objects before each test."""
        self.drone_a = Drone(drone_id=1, pos_x=10.0, pos_y=10.0)
        self.drone_b = Drone(drone_id=2, pos_x=500.0, pos_y=500.0)

        # Setup regions with different distances and initial states
        self.region_near = SubRegion(region_id=101, center_x=12.0, center_y=10.0)
        self.region_far = SubRegion(region_id=102, center_x=800.0, center_y=800.0)
        self.region_covered = SubRegion(region_id=103, center_x=50.0, center_y=50.0)
        self.region_covered.is_covered = True  # Already completed

        self.regions: List[SubRegion] = [self.region_near, self.region_far, self.region_covered]

        # Ensure communication state is clean and initialized for all regions
        initialize_token_state([r.id for r in self.regions])

        # Reset any mocked logger calls
        logger.reset_mock()

    # --- Test Core Function: request_token ---

    @patch('swarm_coordination.request_token_network', return_value=True)
    def test_request_token_success(self, mock_network):
        """Test token request succeeds when drone is active and region is not in history."""
        success = request_token(self.drone_a, self.region_near.id)
        self.assertTrue(success)
        mock_network.assert_called_once_with(self.drone_a.id, self.region_near.id)

    def test_request_token_fails_if_already_covered(self):
        """Test token request fails if the region is already in the drone's history."""
        self.drone_a.coverage_history.add(self.region_near.id)

        with patch('swarm_coordination.request_token_network') as mock_network:
            success = request_token(self.drone_a, self.region_near.id)
            self.assertFalse(success)
            mock_network.assert_not_called()

    # --- Test Core Function: claim_region ---

    def test_claim_region_updates_state(self):
        """Test that claiming a region updates drone, region, and network state correctly."""

        claim_region(self.drone_a, self.region_near)

        # Post-claim checks
        self.assertEqual(self.drone_a.token_held, self.region_near.id)
        self.assertEqual(self.drone_a.current_target, self.region_near.id)
        self.assertEqual(self.region_near.claimed_by, self.drone_a.id)
        self.assertEqual(_TOKEN_STATE[self.region_near.id], self.drone_a.id)
        self.assertEqual(self.drone_a.status_message, f"Claimed Region {self.region_near.id}")

    # --- Test Core Function: release_token ---

    @patch('swarm_coordination.release_token_network')
    def test_release_token_completes_coverage(self, mock_network_release):
        """Test that releasing a token correctly marks region as covered and clears drone state."""

        # Setup: Claim the region first
        claim_region(self.drone_a, self.region_near)

        # Release the token
        release_token(self.drone_a, self.regions)

        # Post-release checks
        self.assertTrue(self.region_near.is_covered)
        self.assertIsNone(self.region_near.claimed_by)
        self.assertIsNone(self.drone_a.token_held)
        self.assertIsNone(self.drone_a.current_target)
        self.assertIn(self.region_near.id, self.drone_a.coverage_history)
        self.assertEqual(self.drone_a.status_message, "Idle")

        # Verify network call
        mock_network_release.assert_called_once_with(self.drone_a.id, self.region_near.id)

    # --- Test Core Function: decide_next_move ---

    def test_decide_next_move_inactive_drone(self):
        """Inactive drone should be grounded and not execute logic."""
        self.drone_a.is_active = False
        decide_next_move(self.drone_a, self.regions)
        self.assertEqual(self.drone_a.status_message, "Grounded")
        self.assertIsNone(self.drone_a.token_held)

    @patch('swarm_coordination.release_token')
    @patch('random.random', return_value=0.1)  # Forces task completion (0.1 < 0.2)
    def test_decide_next_move_completes_task_when_near(self, mock_random, mock_release):
        """Drone holding a token and near the target should attempt task completion."""

        # Setup: Drone is holding token and positioned close to the target center
        claim_region(self.drone_a, self.region_near)
        self.drone_a.update_position(self.region_near.center)  # Distance is 0, which is < 5

        decide_next_move(self.drone_a, self.regions)

        # Should have called release_token
        mock_release.assert_called_once_with(self.drone_a, self.regions)

    @patch('swarm_coordination.release_token')
    @patch('random.random', return_value=0.5)  # Prevents task completion (0.5 !< 0.2)
    def test_decide_next_move_continues_task_when_near_but_no_completion(self, mock_random, mock_release):
        """Drone holding a token should continue if the random completion check fails."""

        # Setup: Drone is holding token and positioned close to the target center
        claim_region(self.drone_a, self.region_near)
        self.drone_a.update_position(self.region_near.center)

        decide_next_move(self.drone_a, self.regions)

        # Should NOT have called release_token
        mock_release.assert_not_called()
        self.assertIsNotNone(self.drone_a.token_held)

    @patch('swarm_coordination.request_token_network', return_value=True)
    def test_decide_next_move_claims_nearest_region(self, mock_network):
        """Drone should claim the nearest unclaimed region when idle."""

        # Drone A is much closer to region_near (101) than region_far (102)
        decide_next_move(self.drone_a, self.regions)

        # Drone A should claim region_near (ID 101)
        self.assertEqual(self.drone_a.token_held, self.region_near.id)
        self.assertEqual(self.region_near.claimed_by, self.drone_a.id)
        mock_network.assert_called_once()

    @patch('swarm_coordination.request_token_network', return_value=False)
    def test_decide_next_move_token_contention(self, mock_network):
        """Drone should remain unclaimed if token request fails (contention)."""

        # Attempt to claim nearest region, but fail the network step
        decide_next_move(self.drone_a, self.regions)

        # Drone A should remain idle/unclaimed
        self.assertIsNone(self.drone_a.token_held)
        self.assertIsNone(self.region_near.claimed_by)
        self.assertEqual(self.drone_a.status_message, f"Requesting Region {self.region_near.id}")
        mock_network.assert_called_once()

    @patch('swarm_coordination.request_token_network', return_value=True)
    def test_decide_next_move_respects_history(self, mock_network):
        """Drone should not attempt to claim a region already in its coverage history."""

        # Add the nearest region to history
        self.drone_a.coverage_history.add(self.region_near.id)

        # Drone should skip region_near (101) and target region_far (102)
        decide_next_move(self.drone_a, self.regions)

        # Drone A should claim region_far (ID 102)
        self.assertEqual(self.drone_a.token_held, self.region_far.id)
        self.assertEqual(self.region_far.claimed_by, self.drone_a.id)

        # Verify the network call was for the farther region
        mock_network.assert_called_once_with(self.drone_a.id, self.region_far.id)

    def test_decide_next_move_no_tasks_left(self):
        """Drone should report 'No tasks left' if all regions are claimed or covered."""

        # Setup: Mark the remaining unclaimed regions as covered
        self.region_near.is_covered = True
        self.region_far.is_covered = True

        decide_next_move(self.drone_a, self.regions)

        self.assertEqual(self.drone_a.status_message, "No tasks left")


if __name__ == '__main__':
    # Running tests without exiting to allow the environment to continue
    unittest.main(exit=False)
