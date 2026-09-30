# Edge Inference Unit Tests
# File: test_edge_inference.py
#
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
#
# Contains unit tests to verify the core functionality of the Edge Inference Simulation.

import unittest
from unittest.mock import patch, MagicMock
import time
import random
from typing import List, Dict

# Import the classes and constants from the target module
from edge_inference import (
    FlowerObservation,
    DroneEdgeStack,
    TARGET_LATENCY_MS,
    IMAGE_CAPTURE_TIME_MS,
    PRE_PROCESSING_TIME_MS,
    SIMULATED_INFERENCE_TIME_MEAN_MS
)


class TestEdgeInference(unittest.TestCase):

    def setUp(self):
        """Set up the DroneEdgeStack instance and base data for testing."""
        self.drone_stack = DroneEdgeStack(drone_id=99)
        self.sample_detection_classes = self.drone_stack.classes

    def test_flower_observation_to_payload(self):
        """Tests that the FlowerObservation object correctly formats its output payload."""

        obs = FlowerObservation(
            obj_id=5,
            classification="Bloom_Stage_1_Needs_Pollen",
            confidence=0.8765,
            center_x=300,
            center_y=450
        )

        payload = obs.to_payload()

        # Verify structure and data types
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload['id'], 5)
        self.assertEqual(payload['type'], "Bloom_Stage_1_Needs_Pollen")
        self.assertEqual(payload['location'], (300, 450))

        # Verify confidence rounding
        self.assertEqual(payload['confidence'], 0.88)

    # --- Test Edge Stack Simulation Logic ---

    @patch('edge_inference.random.gauss', return_value=15.0)  # Mock inference time (15ms)
    @patch('edge_inference.random.randint', return_value=5)  # Mock 5 detections
    @patch('edge_inference.time.time', side_effect=[0.0, 0.0 + (5 + 3 + 15) / 1000.0])  # Mock time measurement
    @patch('edge_inference.time.sleep', return_value=None)  # Mock sleep calls to prevent delay
    def test_inference_cycle_viable(self, mock_sleep, mock_time, mock_randint, mock_gauss):
        """
        Tests a successful, viable cycle where total latency is less than the target.
        Expected Total Latency: 5ms (Capture) + 3ms (Pre-process) + 15ms (Inference) = 23ms
        """

        # Mock random.choice to ensure we get a Bloom Stage for payload filtering
        with patch('edge_inference.random.choice', side_effect=self.sample_detection_classes):
            payload, latency, viable = self.drone_stack.run_inference_cycle()

            # 1. Latency Check
            expected_latency = IMAGE_CAPTURE_TIME_MS + PRE_PROCESSING_TIME_MS + 15.0
            self.assertAlmostEqual(latency, expected_latency, delta=0.01)
            self.assertTrue(viable)
            self.assertTrue(latency <= TARGET_LATENCY_MS)

            # 2. Payload Check (ensures filtering and number of detections)
            # The mocked random.choice returns items from self.sample_detection_classes sequentially.
            # "Bloom_Stage_1_Needs_Pollen" and "Bloom_Stage_2_Nectar_Ready" are included.
            self.assertEqual(len(payload), 2)
            self.assertEqual(payload[0]['type'], "Bloom_Stage_1_Needs_Pollen")

            # 3. Component Check
            self.assertEqual(mock_gauss.call_count, 1)

    @patch('edge_inference.random.gauss', return_value=60.0)  # Mock high inference time (60ms)
    @patch('edge_inference.random.randint', return_value=4)
    @patch('edge_inference.time.time', side_effect=[0.0, 0.0 + (5 + 3 + 60) / 1000.0])  # Mock time measurement
    @patch('edge_inference.time.sleep', return_value=None)
    def test_inference_cycle_non_viable(self, mock_sleep, mock_time, mock_randint, mock_gauss):
        """
        Tests a failed cycle where total latency exceeds the target.
        Expected Total Latency: 5ms (Capture) + 3ms (Pre-process) + 60ms (Inference) = 68ms
        """

        with patch('edge_inference.random.choice', return_value="Unready"):  # Simple mock
            payload, latency, viable = self.drone_stack.run_inference_cycle()

            # 1. Latency Check
            expected_latency = IMAGE_CAPTURE_TIME_MS + PRE_PROCESSING_TIME_MS + 60.0
            self.assertAlmostEqual(latency, expected_latency, delta=0.01)
            self.assertFalse(viable)
            self.assertTrue(latency > TARGET_LATENCY_MS)

            # 2. Payload Check (should be empty if only "Unready" is found)
            self.assertEqual(len(payload), 0)

    @patch('edge_inference.random.gauss', return_value=40.0)
    @patch('edge_inference.random.randint', return_value=5)
    @patch('edge_inference.time.time', side_effect=[0.0, 0.0 + (5 + 3 + 40) / 1000.0])
    @patch('edge_inference.time.sleep', return_value=None)
    def test_payload_filtering(self, mock_sleep, mock_time, mock_randint, mock_gauss):
        """Tests that the final decision payload only includes 'Bloom_Stage' classifications."""

        # Define mock choices for 5 detections: 
        # 1. Bloom_Stage_1_Needs_Pollen (INCLUDED)
        # 2. Unready (EXCLUDED)
        # 3. Pest_Detected (EXCLUDED)
        # 4. Bloom_Stage_2_Nectar_Ready (INCLUDED)
        # 5. Unready (EXCLUDED)
        mock_choices = [
            "Bloom_Stage_1_Needs_Pollen", "Unready", "Pest_Detected",
            "Bloom_Stage_2_Nectar_Ready", "Unready"
        ]

        with patch('edge_inference.random.choice', side_effect=mock_choices):
            payload, _, _ = self.drone_stack.run_inference_cycle()

            # Only 2 out of 5 detections should be in the final payload
            self.assertEqual(len(payload), 2)

            # Verify the types included
            self.assertIn("Bloom_Stage_1_Needs_Pollen", [p['type'] for p in payload])
            self.assertIn("Bloom_Stage_2_Nectar_Ready", [p['type'] for p in payload])

            # Verify excluded types are absent
            self.assertNotIn("Unready", [p['type'] for p in payload])
            self.assertNotIn("Pest_Detected", [p['type'] for p in payload])


if __name__ == '__main__':
    unittest.main(exit=False)
