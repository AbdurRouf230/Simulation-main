# Payload Management Unit Tests
# File: test_payload_management.py
#
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
#
# Contains unit tests to verify the core logic of the Payload Management and Dispensing system.

import unittest
from unittest.mock import patch, MagicMock

# Import components and constants from the target module
from payload_management import (
    PayloadManager,
    DroneMissionController,
    MAX_PAYLOAD_UNITS,
    DISPENSE_UNIT,
    LOW_PAYLOAD_THRESHOLD
)

MOCK_TIME = 1000.0


@patch('payload_management.time.time', return_value=MOCK_TIME)
class TestPayloadManager(unittest.TestCase):

    def setUp(self, mock_time):
        """Set up a fresh PayloadManager for each test."""
        self.manager = PayloadManager(drone_id=1)

    # --- Test Dispensing Actions ---

    def test_successful_dispense(self, mock_time):
        """Tests standard successful dispensing."""
        initial_remaining = self.manager.payload_remaining

        success = self.manager.dispense_pollen()

        self.assertTrue(success)
        self.assertEqual(self.manager.payload_remaining, initial_remaining - DISPENSE_UNIT)
        self.assertEqual(self.manager.total_dispensed, DISPENSE_UNIT)
        self.assertEqual(self.manager.last_dispense_time, MOCK_TIME)

    def test_dispense_when_empty(self, mock_time):
        """Tests attempting to dispense when the payload is empty."""
        self.manager.payload_remaining = 0

        success = self.manager.dispense_pollen()

        self.assertFalse(success)
        self.assertEqual(self.manager.payload_remaining, 0)
        self.assertEqual(self.manager.total_dispensed, 0)  # Should not change
        self.assertIsNone(self.manager.last_dispense_time)  # Should remain None or unchanged

    def test_dispense_with_malfunction(self, mock_time):
        """Tests attempting to dispense when the mechanism is unhealthy."""
        self.manager.is_dispenser_healthy = False

        success = self.manager.dispense_pollen()

        self.assertFalse(success)
        self.assertEqual(self.manager.payload_remaining, MAX_PAYLOAD_UNITS)
        self.assertEqual(self.manager.total_dispensed, 0)
        self.assertIsNone(self.manager.last_dispense_time)

    # --- Test Status Checks ---

    def test_status_initial(self, mock_time):
        """Tests the status report immediately after initialization."""
        status = self.manager.check_status()
        self.assertEqual(status['remaining_units'], MAX_PAYLOAD_UNITS)
        self.assertFalse(status['is_low_payload'])
        self.assertFalse(status['is_empty'])
        self.assertEqual(status['dispenser_status'], 'Healthy')

    def test_status_low_payload(self, mock_time):
        """Tests the status report when the payload crosses the low threshold."""
        self.manager.payload_remaining = LOW_PAYLOAD_THRESHOLD
        status = self.manager.check_status()
        self.assertTrue(status['is_low_payload'])

    def test_status_empty(self, mock_time):
        """Tests the status report when the payload is empty."""
        self.manager.payload_remaining = 0
        status = self.manager.check_status()
        self.assertTrue(status['is_empty'])
        self.assertTrue(status['is_low_payload'])  # Empty implies low

    def test_status_malfunction(self, mock_time):
        """Tests the status report when the dispenser is malfunctioning."""
        self.manager.is_dispenser_healthy = False
        status = self.manager.check_status()
        self.assertEqual(status['dispenser_status'], 'Malfunction')


@patch('payload_management.time.time', return_value=MOCK_TIME)
class TestDroneMissionController(unittest.TestCase):

    def setUp(self, mock_time):
        self.controller = DroneMissionController(drone_id=2)

    # Mock the dispense_pollen method to control mission flow
    @patch.object(PayloadManager, 'dispense_pollen', return_value=True)
    def test_successful_pollination_action(self, mock_dispense, mock_time):
        """Test successful pollination trigger with high confidence."""

        result = self.controller.process_edge_ai_result("Bloom_Stage_1_Needs_Pollen", 0.90)

        mock_dispense.assert_called_once()
        self.assertIn("POLLINATION SUCCESSFUL", result)
        self.assertEqual(self.controller.mission_status['current_action'], 'Pollinating')

    def test_failed_pollination_low_confidence(self, mock_time):
        """Test pollination is blocked due to low confidence."""

        # We don't need to patch dispense_pollen here as it shouldn't be called
        result = self.controller.process_edge_ai_result("Bloom_Stage_1_Needs_Pollen", 0.84)

        self.assertIn("NO ACTION REQUIRED", result)
        self.assertEqual(self.controller.mission_status['current_action'], 'Continuing_Search')
        # Check that payload hasn't changed (since dispense_pollen wasn't called)
        self.assertEqual(self.controller.payload_manager.payload_remaining, MAX_PAYLOAD_UNITS)

    @patch.object(PayloadManager, 'dispense_pollen', return_value=False)
    def test_failed_pollination_payload_empty_triggers_rth(self, mock_dispense, mock_time):
        """Test that failure to dispense (e.g., due to empty payload) initiates RTH."""

        # Simulate an attempt that fails because the payload manager is empty/malfunctioned
        result = self.controller.process_edge_ai_result("Bloom_Stage_1_Needs_Pollen", 0.95)

        mock_dispense.assert_called_once()
        self.assertIn("POLLINATION FAILED: Payload empty, initiating RTH.", result)
        self.assertEqual(self.controller.mission_status['current_action'], 'ReturnToHome')

    def test_pest_detection_action(self, mock_time):
        """Test non-pollen action for pest detection."""

        result = self.controller.process_edge_ai_result("Pest_Detected", 0.99)

        self.assertIn("PEST DETECTED", result)
        self.assertEqual(self.controller.mission_status['current_action'], 'Reporting_Pest')
        self.assertEqual(self.controller.payload_manager.payload_remaining, MAX_PAYLOAD_UNITS)

    def test_unready_no_action(self, mock_time):
        """Test default case where no action is taken."""

        result = self.controller.process_edge_ai_result("Unready", 0.99)

        self.assertIn("NO ACTION REQUIRED", result)
        self.assertEqual(self.controller.mission_status['current_action'], 'Continuing_Search')


if __name__ == '__main__':
    unittest.main(exit=False)
