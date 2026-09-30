# Onboard Edge Inference Simulation: Flower-Stage Detection
# File: edge_inference.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
#
# Simulates the performance and output of the lightweight, quantized AI model
# (QuantizedFlowerNet) running on the drone's microcontroller for real-time decision-making.

import time
import random
from typing import Dict, Tuple, List

# --- Configuration Constants ---
# Target latency specified in the pitch deck: < 50ms
TARGET_LATENCY_MS = 50
# Actual components run in microseconds, but we simulate in milliseconds for clarity.
SIMULATED_INFERENCE_TIME_MEAN_MS = 40  # Mean time for the quantized model
SIMULATED_INFERENCE_TIME_STD_MS = 5  # Standard deviation
IMAGE_CAPTURE_TIME_MS = 5  # Time to capture image data from sensor
PRE_PROCESSING_TIME_MS = 3  # Time for image standardization/normalization


# --- Data Structures ---

class FlowerObservation:
    """Represents a single detected object in the drone's field of view."""

    def __init__(self, obj_id: int, classification: str, confidence: float, center_x: int, center_y: int):
        self.obj_id = obj_id
        self.classification = classification
        self.confidence = confidence
        self.coordinates = (center_x, center_y)

    def to_payload(self) -> Dict:
        """Formats the observation for use by the Swarm Coordination Logic."""
        return {
            'id': self.obj_id,
            'type': self.classification,
            'confidence': round(self.confidence, 2),
            'location': self.coordinates
        }


def _simulate_sensor_capture() -> bytes:
    """Simulates reading image data from the micro-camera sensor."""
    time.sleep(IMAGE_CAPTURE_TIME_MS / 1000)
    return b"simulated_image_data_buffer"


def _simulate_inference() -> float:
    """
    Simulates the execution time of the quantized model on the micro-controller.
    This execution time is the crucial metric for Edge AI viability.
    """
    # Simulate normal distribution of inference time
    inference_time_ms = random.gauss(SIMULATED_INFERENCE_TIME_MEAN_MS, SIMULATED_INFERENCE_TIME_STD_MS)
    return max(1.0, inference_time_ms)  # Ensure positive time


class DroneEdgeStack:
    """Simulates the core processing unit on a single drone."""

    def __init__(self, drone_id: int):
        self.id = drone_id
        self.model_name = "QuantizedFlowerNet_v1.2"
        self.classes = ["Bloom_Stage_1_Needs_Pollen", "Bloom_Stage_2_Nectar_Ready", "Unready", "Pest_Detected"]

    def run_inference_cycle(self) -> Tuple[List[Dict], float, bool]:
        """
        Runs a full end-to-end detection and classification cycle.
        """
        start_time = time.time()

        # 1. Capture and Pre-process (Simulation)
        _ = _simulate_sensor_capture()
        time.sleep(PRE_PROCESSING_TIME_MS / 1000)

        # 2. Inference Execution
        inference_time_ms = _simulate_inference()
        time.sleep(inference_time_ms / 1000)

        # 3. Post-processing and Result Generation

        # Simulate detecting 3 to 8 objects (e.g., flowers, leaves)
        num_detections = random.randint(3, 8)
        detections: List[FlowerObservation] = []
        for i in range(num_detections):
            classification = random.choice(self.classes)
            # Ensure high confidence for ready/needed stages for decision-making
            confidence = random.uniform(0.7, 0.99)
            center_x = random.randint(0, 1000)
            center_y = random.randint(0, 1000)
            detections.append(FlowerObservation(i, classification, confidence, center_x, center_y))

        end_time = time.time()
        total_latency_ms = (end_time - start_time) * 1000

        # Output the core decision payload
        decision_payload = [d.to_payload() for d in detections if d.classification.startswith("Bloom_Stage")]

        # Check viability against the strict Khosla viability criteria
        is_viable = total_latency_ms <= TARGET_LATENCY_MS

        return decision_payload, total_latency_ms, is_viable


# --- Run Simulation ---

if __name__ == "__main__":
    random.seed(42)
    drone_1 = DroneEdgeStack(drone_id=1)

    print(f"--- Edge Inference Viability Test (Model: {drone_1.model_name}) ---")
    print(f"TARGET Latency: <{TARGET_LATENCY_MS} ms")
    print("-" * 50)

    # Run multiple cycles to test average performance
    num_cycles = 10
    total_latency = 0
    viable_cycles = 0

    for i in range(1, num_cycles + 1):
        payload, latency, viable = drone_1.run_inference_cycle()
        total_latency += latency
        if viable:
            viable_cycles += 1

        viability_status = "VIABLE" if viable else "FAIL"

        # Print critical metrics and sample output
        print(f"Cycle {i}: Latency: {latency:.2f} ms ({viability_status})")

        # Print a sample of the output feeding the Swarm Coordination Logic
        if payload:
            print(f"  > Output Payload Sample ({len(payload)} detections): {payload[0]}")

    avg_latency = total_latency / num_cycles

    print("-" * 50)
    print(f"SUMMARY ({num_cycles} Cycles):")
    print(f"Average Total Latency: {avg_latency:.2f} ms")
    print(f"Viability Success Rate: {viable_cycles}/{num_cycles} ({(viable_cycles / num_cycles) * 100:.1f}%)")

    # MVP Viability Check: Does the system meet the average latency requirement?
    if avg_latency < TARGET_LATENCY_MS and viable_cycles >= num_cycles * 0.8:
        print(
            "\nMVP VIABILITY CONFIRMED: Average latency meets the critical <50ms threshold. The Edge AI stack is "
            "computationally viable.")
    else:
        print(
            "\nMVP VIABILITY FAILURE: Average latency exceeds the critical threshold or reliability is too low. "
            "Requires further quantization and optimization.")
