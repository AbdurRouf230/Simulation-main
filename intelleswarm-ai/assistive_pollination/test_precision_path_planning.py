# Precision Path Planning Unit Tests
# File: test_precision_path_planning.py
#
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
#
# Contains unit tests for the Potential Field GNC system.

import unittest
from unittest.mock import patch, MagicMock
import math

# Import classes and constants from the target module
from precision_path_planning import (
    Vector2D, 
    Drone, 
    GOAL_ATTRACTION_FACTOR, 
    OBSTACLE_REPULSION_FACTOR, 
    OBSTACLE_DANGER_RADIUS,
    TIMESTEP
)

class TestVector2D(unittest.TestCase):
    """Tests the fundamental 2D vector mathematics."""

    def test_magnitude(self):
        v = Vector2D(3, 4)
        self.assertAlmostEqual(v.magnitude(), 5.0)
        v_zero = Vector2D(0, 0)
        self.assertAlmostEqual(v_zero.magnitude(), 0.0)

    def test_normalize(self):
        v = Vector2D(10, 0)
        v.normalize()
        self.assertAlmostEqual(v.x, 1.0)
        self.assertAlmostEqual(v.y, 0.0)
        
        v_diag = Vector2D(1, 1)
        v_diag.normalize()
        # 1/sqrt(2) approx 0.7071
        self.assertAlmostEqual(v_diag.x, 0.70710678)
        
        v_zero = Vector2D(0, 0)
        v_zero.normalize() # Should not raise error
        self.assertEqual(v_zero.x, 0.0)

    def test_addition(self):
        v1 = Vector2D(1, 2)
        v2 = Vector2D(3, -1)
        v_sum = v1 + v2
        self.assertAlmostEqual(v_sum.x, 4.0)
        self.assertAlmostEqual(v_sum.y, 1.0)

    def test_subtraction(self):
        v1 = Vector2D(5, 5)
        v2 = Vector2D(2, 3)
        v_diff = v1 - v2
        self.assertAlmostEqual(v_diff.x, 3.0)
        self.assertAlmostEqual(v_diff.y, 2.0)

    def test_scalar_multiplication(self):
        v = Vector2D(2, 4)
        v_scaled = v * 2.5
        self.assertAlmostEqual(v_scaled.x, 5.0)
        self.assertAlmostEqual(v_scaled.y, 10.0)

class TestDroneGNC(unittest.TestCase):
    """Tests the Drone's GNC logic using the Potential Field approach."""

    def setUp(self):
        self.drone = Drone(10, (10.0, 10.0))

    def test_set_goal_and_is_at_goal(self):
        """Test goal assignment and reaching the goal state."""
        self.drone.set_goal((10.5, 10.5))
        
        # Not at goal yet (distance is sqrt(0.5^2 + 0.5^2) = 0.707, which is < 1.0 tolerance)
        self.assertTrue(self.drone.is_at_goal(tolerance=1.0))
        
        # Not at goal with strict tolerance
        self.assertFalse(self.drone.is_at_goal(tolerance=0.5))

    def test_idle_if_no_goal(self):
        """Drone should stop if no goal is set or if it's already at the goal."""
        self.drone.goal = None
        self.drone.velocity = Vector2D(5, 5) # Start with velocity
        self.drone.update_motion([])
        self.assertAlmostEqual(self.drone.velocity.magnitude(), 0.0)

    # --- Test Attractive Force ---

    def test_attraction_force_only(self):
        """Drone should accelerate directly towards the goal when no obstacles are present."""
        self.drone.set_goal((110.0, 10.0)) # 100m away on X-axis
        
        self.drone.update_motion([])
        
        # Expected attractive force magnitude: distance * GOAL_ATTRACTION_FACTOR
        expected_force_mag = 100.0 * GOAL_ATTRACTION_FACTOR
        
        # Expected acceleration magnitude (fn=ma, m=1)
        expected_accel_mag = expected_force_mag 
        
        # Expected velocity after 1 timestep: V = a * dt
        expected_v_mag = expected_accel_mag * TIMESTEP
        
        self.assertAlmostEqual(self.drone.velocity.magnitude(), expected_v_mag)
        self.assertAlmostEqual(self.drone.velocity.x, expected_v_mag) # Should be moving purely on X
        self.assertAlmostEqual(self.drone.velocity.y, 0.0)
        
        # Check position update
        expected_pos_x = 10.0 + expected_v_mag * TIMESTEP
        self.assertAlmostEqual(self.drone.position.x, expected_pos_x)

    # --- Test Repulsive Force ---

    @patch('precision_path_planning.random.uniform', return_value=0.5) # Mock random for safety
    def test_repulsion_force_on_danger_edge(self, mock_random):
        """Test repulsion force calculation right at the danger radius (should be zero)."""
        obstacle = Vector2D(10.0 + OBSTACLE_DANGER_RADIUS, 10.0)
        self.drone.set_goal((50, 50)) # Attraction keeps it moving
        
        self.drone.update_motion([obstacle])
        
        # Repulsion magnitude calculation: (1/dist - 1/R_danger)^2
        # dist = R_danger, so 1/dist - 1/R_danger = 0. Repulsion should be 0.
        
        # The resulting acceleration should be purely attractive (F_attraction)
        expected_attraction = (self.drone.goal - self.drone.position).magnitude() * GOAL_ATTRACTION_FACTOR
        resulting_acceleration = (self.drone.velocity / TIMESTEP).magnitude() 
        
        self.assertAlmostEqual(resulting_acceleration, expected_attraction)

    @patch('precision_path_planning.random.uniform', return_value=0.5)
    def test_repulsion_force_dominates_close(self, mock_random):
        """Test that a close obstacle generates a strong repulsion force."""
        
        # Obstacle 1m away (very close)
        dist = 1.0
        obstacle = Vector2D(10.0 + dist, 10.0) 
        self.drone.set_goal((10.0 + 100.0, 10.0)) # Goal is far away (strong attraction)

        # Run one step
        self.drone.update_motion([obstacle])
        
        # Repulsion magnitude should be high
        expected_repulsion_mag = OBSTACLE_REPULSION_FACTOR * (1/dist - 1/OBSTACLE_DANGER_RADIUS)**2
        
        # Attraction magnitude should be high but less than repulsion
        expected_attraction_mag = (self.drone.goal - self.drone.position).magnitude() * GOAL_ATTRACTION_FACTOR
        
        # Since the goal is directly opposite the obstacle, the net force should be repulsion - attraction
        # and the movement should be *away* from the obstacle (negative X-axis)
        
        net_force_x = expected_attraction_mag - expected_repulsion_mag 
        
        # Check that the drone is moving AWAY from the obstacle (X-position should decrease)
        self.assertTrue(self.drone.position.x < 10.0)
        
        # Check that the calculated acceleration magnitude is close to the expected force
        resulting_acceleration = (self.drone.velocity / TIMESTEP).magnitude() 
        self.assertAlmostEqual(resulting_acceleration, abs(net_force_x), places=1)

    @patch('precision_path_planning.random.uniform', return_value=0.5)
    def test_repulsion_force_division_by_zero_fix(self, mock_random):
        """Test the critical fix: drone starts on top of an obstacle (dist=0)."""
        
        # Obstacle is at the same starting position
        obstacle = Vector2D(10.0, 10.0) 
        self.drone.set_goal((50.0, 50.0))
        
        # Run one step
        self.drone.update_motion([obstacle])
        
        # The drone should have moved away from (10, 10) due to the emergency force
        self.assertTrue(self.drone.position.x != 10.0 or self.drone.position.y != 10.0)
        
        # The velocity should be non-zero
        self.assertTrue(self.drone.velocity.magnitude() > 0.0)
        
        # Verify no runtime errors (like DivisionByZero) occurred

    # --- Test Control Logic ---

    def test_max_speed_limit(self):
        """Test that the drone's velocity is capped at max_speed."""
        self.drone.set_goal((1000.0, 10.0)) # A distant goal to generate max force
        
        # Run multiple steps until max speed is reached
        for _ in range(50):
            self.drone.update_motion([])
            
        # Velocity magnitude must not exceed max_speed
        self.assertLessEqual(self.drone.velocity.magnitude(), self.drone.max_speed + 1e-6)
        self.assertAlmostEqual(self.drone.velocity.magnitude(), self.drone.max_speed, places=3) # Should be capped at max_speed

if __name__ == '__main__':
    unittest.main(exit=False)

