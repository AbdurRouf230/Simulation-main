# Multi-Sensor Fusion (Localization and Mapping)
# File: multi_sensor_fusion.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
# 
# Simulates the drone's ability to fuse data from multiple heterogeneous sensors
# (GPS, IMU/Velocity) using an Extended Kalman Filter (EKF) concept to achieve
# a high-precision, low-latency position estimate for reliable GNC.

import random
import time
from typing import Dict, Any, Tuple

# --- Configuration Constants ---
# Noise levels represent the estimated error variance (lower is better)
GPS_NOISE_VARIANCE = 5.0  # GPS is slow and inherently noisy (high variance)
IMU_NOISE_VARIANCE = 0.5  # IMU/Vision is fast but drifts (low variance, high drift)
FUSION_FREQUENCY_HZ = 10.0  # How often the fusion process runs


# --- Sensor Simulation ---

def get_true_position(t: float) -> Tuple[float, float]:
    """Simulates the true, underlying position of the drone."""
    x = 100 + 5 * t
    y = 50 + 2 * t * t / 10
    return (x, y)


def simulate_gps_reading(true_pos: Tuple[float, float]) -> Tuple[float, float]:
    """Simulates a noisy, low-rate GPS measurement."""
    x_noise = random.uniform(-GPS_NOISE_VARIANCE, GPS_NOISE_VARIANCE)
    y_noise = random.uniform(-GPS_NOISE_VARIANCE, GPS_NOISE_VARIANCE)
    return (true_pos[0] + x_noise, true_pos[1] + y_noise)


def simulate_imu_reading(true_velocity: Tuple[float, float]) -> Tuple[float, float]:
    """Simulates a high-rate IMU/Velocity measurement (fast but prone to drift)."""
    vx_noise = random.uniform(-IMU_NOISE_VARIANCE, IMU_NOISE_VARIANCE)
    vy_noise = random.uniform(-IMU_NOISE_VARIANCE, IMU_NOISE_VARIANCE)
    return (true_velocity[0] + vx_noise, true_velocity[1] + vy_noise)


# --- Core Fusion Logic (Simplified EKF Concept) ---

class SensorFusionEngine:
    """
    Implements a highly simplified, weighted averaging approach to mimic the
    effect of an Extended Kalman Filter (EKF).
    """

    def __init__(self, initial_pos: Tuple[float, float]):
        self.estimated_pos = list(initial_pos)
        self.last_update_time = time.time()
        self.estimated_variance = GPS_NOISE_VARIANCE  # Start with GPS uncertainty

    def fuse_sensors(self, gps_reading: Tuple[float, float], imu_data: Tuple[float, float]) -> Tuple[float, float]:
        """
        Performs the fusion step: blending GPS and IMU data based on confidence (variance).
        
        Note: In a true EKF, the covariance matrix would be updated. Here, we use a
        simple weighted average, where the weight is inversely proportional to the
        sensor's noise variance (lower variance = higher weight/trust).
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # 1. Prediction (Using IMU/Velocity)
        # IMU provides change in position (delta), which we add to the current estimate
        vx_imu, vy_imu = imu_data  # This is velocity, not position

        # Estimate new position based on motion model (prediction step)
        predicted_pos = [
            self.estimated_pos[0] + vx_imu * dt,
            self.estimated_pos[1] + vy_imu * dt
        ]

        # 2. Update (Using GPS measurement)
        # This step corrects the predicted position with the GPS measurement.

        # Calculate the Gain (Weighting Factor - Higher for trusted sensor)
        # The true gain calculation is complex, here we simplify by trusting the sensor
        # with lower noise variance more.

        # Weighting ratio: How much to trust the IMU vs. the GPS (IMU_weight / (IMU_weight + GPS_weight))
        # Trust = 1 / Variance (Weight is proportional to Trust)
        total_trust = (1.0 / IMU_NOISE_VARIANCE) + (1.0 / GPS_NOISE_VARIANCE)

        weight_imu = (1.0 / IMU_NOISE_VARIANCE) / total_trust
        weight_gps = (1.0 / GPS_NOISE_VARIANCE) / total_trust

        # Final Fused Position (Weighted Average of Prediction and Measurement)
        fused_x = (predicted_pos[0] * weight_imu) + (gps_reading[0] * weight_gps)
        fused_y = (predicted_pos[1] * weight_imu) + (gps_reading[1] * weight_gps)

        self.estimated_pos = [fused_x, fused_y]

        return (fused_x, fused_y)


# --- Simulation Run ---

if __name__ == "__main__":
    random.seed(42)
    start_time_sim = 0.0
    initial_true_pos = get_true_position(start_time_sim)

    fusion_engine = SensorFusionEngine(initial_true_pos)

    print("--- Multi-Sensor Fusion (Localization) Simulation ---")
    print(f"True Start Pos: ({initial_true_pos[0]:.2f}, {initial_true_pos[1]:.2f})")
    print(f"GPS Noise Variance: {GPS_NOISE_VARIANCE} | IMU Noise Variance: {IMU_NOISE_VARIANCE}")
    print("-" * 75)
    print(f"{'Time (s)':<10}{'True Pos':<18}{'GPS Reading':<18}{'IMU Velocity':<18}{'Fused Estimate':<18}")
    print("-" * 75)

    # Simulate a 10-second flight path
    num_steps = int(10 * FUSION_FREQUENCY_HZ)
    dt = 1.0 / FUSION_FREQUENCY_HZ  # Time step for each fusion cycle

    # Simple constant velocity model for IMU input simulation
    sim_true_velocity = (5.0, 0.2 * 2 * start_time_sim)  # Velocity components based on true position derivative

    for i in range(1, num_steps + 1):
        t = i * dt
        true_pos = get_true_position(t)

        # Simulate noisy sensor readings
        gps_r = simulate_gps_reading(true_pos)
        imu_v = simulate_imu_reading(sim_true_velocity)

        # Run the fusion engine
        fused_e = fusion_engine.fuse_sensors(gps_r, imu_v)

        # Only print every 10th step for brevity
        if i % 10 == 0:
            print(
                f"{t:<10.1f}({true_pos[0]:.1f},{true_pos[1]:.1f}) ({gps_r[0]:.1f},{gps_r[1]:.1f}) ({imu_v[0]:.1f},{imu_v[1]:.1f}) ({fused_e[0]:.1f},{fused_e[1]:.1f})")

    # Final Validation
    print("\n--- MVP Validation Check ---")

    # Calculate error of the final GPS reading vs the final Fused Estimate
    final_gps_error = ((true_pos[0] - gps_r[0]) ** 2 + (true_pos[1] - gps_r[1]) ** 2) ** 0.5
    final_fused_error = ((true_pos[0] - fused_e[0]) ** 2 + (true_pos[1] - fused_e[1]) ** 2) ** 0.5

    print(f"Final True Position: ({true_pos[0]:.2f}, {true_pos[1]:.2f})")
    print(f"GPS Error Distance: {final_gps_error:.2f} units")
    print(f"Fused Error Distance: {final_fused_error:.2f} units")

    if final_fused_error < final_gps_error and final_fused_error < GPS_NOISE_VARIANCE:
        print(
            "MVP Validation: Successful. The fusion engine produced a position estimate significantly more accurate "
            "than the single, noisy GPS input, fulfilling the requirement for high-precision GNC.")

    else:
        print("MVP Validation: Failure. The sensor fusion did not successfully reduce the overall localization error.")
