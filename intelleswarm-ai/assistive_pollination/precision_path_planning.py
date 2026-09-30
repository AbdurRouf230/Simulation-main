# Precision Path Planning (GNC) - Potential Field Approach
# File: precision_path_planning.py
# --------------------------------------------------------------------------
# Copyright (c) 2025, IntelliSwarm Corporation. All Rights Reserved.
# This software is proprietary and confidential. Unauthorized copying or
# dissemination is strictly prohibited.
# --------------------------------------------------------------------------
# Author: Zahid Rahman
# 
# Simulates the Guidance, Navigation, and Control (GNC) system's ability to
# generate dynamic, collision-free trajectories in real-time, critical for
# both efficient coverage and Swarm-level collision avoidance.
# Uses a simplified 2D Potential Field method for real-time responsiveness.

import math
import random
from typing import Tuple, List, Optional

# --- Configuration Constants ---
SIMULATION_AREA_SIZE = 100 # meters x meters
TIMESTEP = 0.1             # Time step (seconds) for simulation
GOAL_ATTRACTION_FACTOR = 0.8 # Multiplier for the force pulling towards the goal
OBSTACLE_REPULSION_FACTOR = 10.0 # Multiplier for the force pushing away from obstacles
OBSTACLE_DANGER_RADIUS = 10  # Distance (m) where repulsion force begins

# --- Core Data Structures ---

class Vector2D:
    """Simple 2D vector class for position and forces."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def magnitude(self) -> float:
        """Returns the length of the vector."""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        """Scales the vector to a unit length, preserving direction."""
        mag = self.magnitude()
        if mag > 0:
            self.x /= mag
            self.y /= mag

    def __add__(self, other):
        """Vector addition."""
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Vector subtraction."""
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float):
        """Scalar multiplication."""
        return Vector2D(self.x * scalar, self.y * scalar)

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

class Drone:
    """Represents a micro-drone with GNC capabilities."""
    def __init__(self, drone_id: int, start_pos: Tuple[float, float]):
        self.id = drone_id
        self.position = Vector2D(*start_pos)
        self.velocity = Vector2D(0, 0)
        self.max_speed = 5.0 # m/s
        self.goal: Optional[Vector2D] = None
        self.history: List[Tuple[float, float]] = [start_pos]

    def set_goal(self, goal_pos: Tuple[float, float]):
        """Assigns a new target location (e.g., a region from Swarm Coordination)."""
        self.goal = Vector2D(*goal_pos)
        print(f"Drone {self.id} assigned goal: {self.goal}")

    def is_at_goal(self, tolerance: float = 1.0) -> bool:
        """Checks if the drone has reached its assigned target."""
        if self.goal is None:
            return False
        distance = (self.goal - self.position).magnitude()
        return distance < tolerance

    def update_motion(self, obstacles: List[Vector2D]):
        """
        The Core GNC Loop: Computes resulting force and updates position/velocity.
        """
        if self.goal is None or self.is_at_goal():
            self.velocity = Vector2D(0, 0)
            return

        # 1. Guidance: Calculate Potential Field Forces

        # A. Attractive Force (F_goal)
        diff_to_goal = self.goal - self.position
        f_attraction = diff_to_goal * GOAL_ATTRACTION_FACTOR
        f_total = f_attraction

        # B. Repulsive Force (F_obstacle)
        for obstacle in obstacles:
            diff_to_obstacle = obstacle - self.position
            dist = diff_to_obstacle.magnitude()

            if dist < OBSTACLE_DANGER_RADIUS:
                
                if dist == 0:
                    # FIX: Handle division by zero error if drone is exactly on the obstacle.
                    # Apply a large, randomized escape force.
                    repulsion_magnitude = 50.0 
                    f_repulsion = Vector2D(random.uniform(-1, 1), random.uniform(-1, 1))
                else:
                    # Standard repulsion calculation
                    # The repulsion force is inversely proportional to the distance squared
                    repulsion_magnitude = OBSTACLE_REPULSION_FACTOR * (1/dist - 1/OBSTACLE_DANGER_RADIUS)**2
                    # Direction is the negative of the vector to the obstacle (pushing away)
                    f_repulsion = diff_to_obstacle * (-1.0) # Start with vector pointing away

                # Ensure f_repulsion is normalized before applying magnitude (handles randomized vector if dist=0)
                f_repulsion.normalize()
                f_repulsion = f_repulsion * repulsion_magnitude
                
                f_total += f_repulsion
        
        # 2. Control: Convert Force to Acceleration (fn=ma, assume m=1)
        acceleration = f_total 

        # 3. Navigation: Update Velocity and Position
        
        # Update velocity (v = v + a*dt)
        self.velocity += acceleration * TIMESTEP
        
        # Apply maximum speed limit (critical for drone stability)
        if self.velocity.magnitude() > self.max_speed:
            self.velocity.normalize()
            self.velocity = self.velocity * self.max_speed

        # Update position (p = p + v*dt)
        self.position += self.velocity * TIMESTEP
        self.history.append((self.position.x, self.position.y))

