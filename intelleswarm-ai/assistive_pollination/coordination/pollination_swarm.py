"""
Pollination Swarm Coordinator.

Multi-agent swarm coordination system for agricultural pollination
using IntelleSwarm MAPPO, QMIX, and MADDPG algorithms.
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time

# Import IntelleSwarm framework components
import sys
sys.path.append('../..')
from genai_framework.sdk.swarm import DroneBrain, MultiAgentPolicy
from genai_framework.sdk.cloud import SwarmTrainer
from genai_framework.sdk.algorithms.mappo import MAPPOTrainer
from genai_framework.sdk.collision import TrajectoryDiffusionModel, CollisionChecker

# Import mission planning components
from mission.agricultural_planner import MissionPlan, DroneAssignment, PollinationTarget

logger = logging.getLogger(__name__)


class PollinationAction(Enum):
    """Specific actions for pollination drones."""
    NAVIGATE_TO_TARGET = "navigate"
    COLLECT_POLLEN = "collect_pollen"
    TRANSFER_POLLEN = "transfer_pollen"
    HOVER_AND_POLLINATE = "hover_pollinate"
    RETURN_FOR_RECHARGE = "return_recharge"
    WAIT_FOR_PARTNER = "wait_partner"
    FORMATION_ADJUST = "formation_adjust"
    EMERGENCY_LAND = "emergency_land"


@dataclass
class PollinationState:
    """Current state of a pollination drone."""
    drone_id: str
    position_3d: Tuple[float, float, float]  # GPS + altitude
    battery_percent: float
    pollen_load: float  # Amount of pollen currently carrying
    current_target: Optional[PollinationTarget]
    action: PollinationAction
    targets_completed: int
    flight_time_minutes: float
    last_pollination_success: bool
    environmental_conditions: Dict[str, float]


class PollinationObservation:
    """Observation space for pollination reinforcement learning."""

    def __init__(self, state: PollinationState, nearby_drones: List[PollinationState],
                 visible_targets: List[PollinationTarget], environmental_data: Dict[str, float]):
        self.own_state = state
        self.nearby_drones = nearby_drones
        self.visible_targets = visible_targets
        self.environmental_data = environmental_data

    def to_tensor(self) -> torch.Tensor:
        """Convert observation to neural network input tensor."""
        obs_vector = []

        # Own state features (8 features)
        obs_vector.extend([
            self.own_state.position_3d[0] / 180.0,  # Normalized latitude
            self.own_state.position_3d[1] / 180.0,  # Normalized longitude
            self.own_state.position_3d[2] / 100.0,  # Normalized altitude
            self.own_state.battery_percent / 100.0,
            self.own_state.pollen_load / 10.0,  # Normalize pollen load
            self.own_state.targets_completed / 100.0,
            self.own_state.flight_time_minutes / 60.0,  # Normalize to hours
            1.0 if self.own_state.last_pollination_success else 0.0
        ])

        # Current target features (4 features)
        if self.own_state.current_target:
            target = self.own_state.current_target
            obs_vector.extend([
                target.location_3d[0] / 180.0,
                target.location_3d[1] / 180.0,
                target.location_3d[2] / 100.0,
                target.pollination_priority
            ])
        else:
            obs_vector.extend([0.0, 0.0, 0.0, 0.0])

        # Nearby drones features (max 3 drones, 4 features each = 12 features)
        for i in range(3):
            if i < len(self.nearby_drones):
                drone = self.nearby_drones[i]
                obs_vector.extend([
                    drone.position_3d[0] / 180.0,
                    drone.position_3d[1] / 180.0,
                    drone.position_3d[2] / 100.0,
                    drone.pollen_load / 10.0
                ])
            else:
                obs_vector.extend([0.0, 0.0, 0.0, 0.0])

        # Environmental features (8 features)
        obs_vector.extend([
            self.environmental_data.get('temperature', 20) / 40.0,  # 0-40°C
            self.environmental_data.get('humidity', 50) / 100.0,   # 0-100%
            self.environmental_data.get('wind_speed', 0) / 20.0,   # 0-20 m/s
            self.environmental_data.get('wind_direction', 0) / 360.0, # 0-360°
            self.environmental_data.get('light_intensity', 50000) / 100000.0,
            self.environmental_data.get('time_of_day', 12) / 24.0,  # Hour of day
            len(self.visible_targets) / 20.0,  # Normalize target count
            1.0 if self.environmental_data.get('weather_optimal', True) else 0.0
        ])

        return torch.tensor(obs_vector, dtype=torch.float32)


class PollinationSwarm:
    """
    Multi-agent swarm coordinator for agricultural pollination.

    Uses IntelleSwarm MAPPO algorithm to coordinate drone actions
    for efficient field coverage and cross-pollination.
    """

    def __init__(self,
                 max_drones: int = 12,
                 algorithm: str = 'mappo',
                 obs_dim: int = 32,
                 action_dim: int = len(PollinationAction),
                 communication_range: float = 100.0):

        self.max_drones = max_drones
        self.algorithm = algorithm
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.communication_range = communication_range

        # Initialize IntelleSwarm trainer
        self.trainer = SwarmTrainer(
            algorithm=algorithm,
            obs_dim=obs_dim,
            action_dim=action_dim,
            global_state_dim=obs_dim * max_drones,
            num_agents=max_drones,
            lr=3e-4,
            gamma=0.99,
            batch_size=64
        )

        # Collision avoidance system
        self.collision_model = TrajectoryDiffusionModel(
            state_dim=obs_dim,
            action_horizon=10,
            hidden_dim=256
        )
        self.collision_checker = CollisionChecker()

        # Drone brain instances
        self.drone_brains = {}
        for i in range(max_drones):
            self.drone_brains[f"POLL_DRONE_{i+1:02d}"] = DroneBrain(
                obs_dim=obs_dim,
                world_latent_dim=16,  # Fixed: latent_dim -> world_latent_dim
                msg_dim=8,
                action_dim=action_dim,
                enable_collision_avoidance=True  # Fixed: Enable collision avoidance
                # Note: drone_id, device, collision_model are not DroneBrain parameters
            )

        # Mission state tracking
        self.active_mission: Optional[MissionPlan] = None
        self.drone_states: Dict[str, PollinationState] = {}
        self.mission_start_time: float = 0.0
        self.performance_metrics = {
            'targets_completed': 0,
            'successful_pollinations': 0,
            'total_flight_time': 0.0,
            'pollen_transfer_events': 0,
            'coordination_efficiency': 0.0
        }

    async def execute_pollination_mission(self, mission_plan: MissionPlan) -> Dict[str, Any]:
        """
        Execute a complete pollination mission using multi-agent coordination.

        Args:
            mission_plan: Detailed mission plan from AgriculturalMissionPlanner

        Returns:
            Mission execution results and performance metrics
        """
        logger.info(f"Starting pollination mission: {mission_plan.mission_id}")

        self.active_mission = mission_plan
        self.mission_start_time = time.time()

        # Initialize drone states
        self._initialize_drone_states(mission_plan.drone_assignments)

        # Execute mission phases
        results = await self._execute_mission_phases()

        # Generate mission report
        mission_report = self._generate_mission_report(results)

        logger.info(f"Mission {mission_plan.mission_id} completed: "
                   f"{mission_report['targets_completed']} targets, "
                   f"{mission_report['success_rate']:.1f}% success rate")

        return mission_report

    async def _execute_mission_phases(self) -> Dict[str, Any]:
        """Execute the main mission coordination loop."""

        total_phases = 5
        phase_results = {}

        try:
            # Phase 1: Deployment and Formation
            logger.info("Phase 1: Drone deployment and formation")
            await self._phase_deployment()

            # Phase 2: Field Survey and Target Confirmation
            logger.info("Phase 2: Field survey and target confirmation")
            survey_results = await self._phase_field_survey()
            phase_results['survey'] = survey_results

            # Phase 3: Coordinated Pollination Execution
            logger.info("Phase 3: Coordinated pollination execution")
            pollination_results = await self._phase_pollination_execution()
            phase_results['pollination'] = pollination_results

            # Phase 4: Cross-Pollination Coordination
            logger.info("Phase 4: Cross-pollination coordination")
            cross_pollination_results = await self._phase_cross_pollination()
            phase_results['cross_pollination'] = cross_pollination_results

            # Phase 5: Mission Completion and Return
            logger.info("Phase 5: Mission completion and return")
            completion_results = await self._phase_mission_completion()
            phase_results['completion'] = completion_results

        except Exception as e:
            logger.error(f"Mission execution error: {e}")
            # Execute emergency protocols
            await self._emergency_mission_abort()
            phase_results['error'] = str(e)

        return phase_results

    async def _phase_deployment(self):
        """Phase 1: Deploy drones and establish formation."""

        deployment_tasks = []

        for drone_id, state in self.drone_states.items():
            # Create deployment task for each drone
            task = self._deploy_single_drone(drone_id, state)
            deployment_tasks.append(task)

        # Execute all deployments in parallel
        await asyncio.gather(*deployment_tasks)

        # Establish initial formation
        await self._establish_formation()

    async def _deploy_single_drone(self, drone_id: str, state: PollinationState):
        """Deploy a single drone to its starting position."""

        # Simulate drone takeoff and navigation to start position
        if state.current_target:
            start_position = state.current_target.location_3d
        else:
            # Use first assignment target as start position
            assignment = self._get_drone_assignment(drone_id)
            if assignment and assignment.assigned_targets:
                start_position = assignment.assigned_targets[0].location_3d
            else:
                start_position = (37.7749, -122.4194, 50.0)  # Default position

        # Simulate flight to position
        await asyncio.sleep(0.5)  # Deployment time simulation

        # Update drone state
        state.position_3d = start_position
        state.action = PollinationAction.NAVIGATE_TO_TARGET

        logger.debug(f"{drone_id} deployed to position {start_position}")

    async def _establish_formation(self):
        """Establish initial swarm formation for field coverage."""

        # Use MAPPO to determine optimal formation
        observations = []
        for drone_id, state in self.drone_states.items():
            obs = self._create_observation(drone_id, state)
            observations.append(obs.to_tensor())

        # Get coordinated actions from MAPPO
        with torch.no_grad():
            if observations:
                obs_batch = torch.stack(observations)
                actions = self.trainer.get_actions(obs_batch, deterministic=True)

                # Apply formation actions
                for i, (drone_id, state) in enumerate(self.drone_states.items()):
                    if i < len(actions):
                        action_idx = actions[i].item()
                        action = list(PollinationAction)[action_idx % len(PollinationAction)]
                        state.action = action

        await asyncio.sleep(1.0)  # Formation establishment time
        logger.info("Initial formation established")

    async def _phase_field_survey(self) -> Dict[str, Any]:
        """Phase 2: Survey field and confirm targets."""

        survey_tasks = []

        for drone_id, state in self.drone_states.items():
            task = self._survey_assigned_area(drone_id, state)
            survey_tasks.append(task)

        survey_results = await asyncio.gather(*survey_tasks)

        # Aggregate results
        total_targets_confirmed = sum(result.get('targets_confirmed', 0) for result in survey_results)
        total_area_surveyed = sum(result.get('area_surveyed', 0) for result in survey_results)

        return {
            'targets_confirmed': total_targets_confirmed,
            'area_surveyed_hectares': total_area_surveyed,
            'survey_efficiency': total_targets_confirmed / max(total_area_surveyed, 1)
        }

    async def _survey_assigned_area(self, drone_id: str, state: PollinationState) -> Dict[str, Any]:
        """Survey assigned area for a single drone."""

        assignment = self._get_drone_assignment(drone_id)
        if not assignment:
            return {'targets_confirmed': 0, 'area_surveyed': 0}

        targets_confirmed = 0
        area_surveyed = 0.5  # hectares per drone

        # Simulate survey of assigned targets
        for target in assignment.assigned_targets:
            # Simulate flight to target and confirmation
            await asyncio.sleep(0.1)  # Survey time per target

            # Confirm target validity (simulate computer vision detection)
            confirmation_probability = 0.9  # 90% confirmation rate
            if np.random.random() < confirmation_probability:
                targets_confirmed += 1
                target.last_visit_time = time.time()

        return {
            'targets_confirmed': targets_confirmed,
            'area_surveyed': area_surveyed
        }

    async def _phase_pollination_execution(self) -> Dict[str, Any]:
        """Phase 3: Execute main pollination operations."""

        # Main coordination loop
        max_iterations = 100
        successful_pollinations = 0

        for iteration in range(max_iterations):
            # Get observations for all active drones
            observations = []
            active_drones = []

            for drone_id, state in self.drone_states.items():
                if state.battery_percent > 20 and state.action != PollinationAction.RETURN_FOR_RECHARGE:
                    obs = self._create_observation(drone_id, state)
                    observations.append(obs.to_tensor())
                    active_drones.append(drone_id)

            if not observations:
                break  # No active drones

            # Get coordinated actions from MAPPO
            obs_batch = torch.stack(observations)
            with torch.no_grad():
                actions = self.trainer.get_actions(obs_batch, deterministic=False)

            # Execute actions for each drone
            for i, drone_id in enumerate(active_drones):
                if i < len(actions):
                    action_idx = actions[i].item() % len(PollinationAction)
                    action = list(PollinationAction)[action_idx]

                    # Execute the action
                    success = await self._execute_drone_action(drone_id, action)
                    if success and action == PollinationAction.TRANSFER_POLLEN:
                        successful_pollinations += 1

            # Update performance metrics
            self.performance_metrics['successful_pollinations'] = successful_pollinations

            # Check completion criteria
            if self._check_mission_completion():
                break

            # Brief pause between coordination steps
            await asyncio.sleep(0.1)

        return {
            'successful_pollinations': successful_pollinations,
            'coordination_iterations': iteration + 1,
            'final_efficiency': self._calculate_efficiency()
        }

    async def _execute_drone_action(self, drone_id: str, action: PollinationAction) -> bool:
        """Execute a specific action for a drone."""

        state = self.drone_states[drone_id]
        state.action = action

        # Simulate action execution
        if action == PollinationAction.NAVIGATE_TO_TARGET:
            await self._navigate_to_target(drone_id)
        elif action == PollinationAction.COLLECT_POLLEN:
            await self._collect_pollen(drone_id)
        elif action == PollinationAction.TRANSFER_POLLEN:
            return await self._transfer_pollen(drone_id)
        elif action == PollinationAction.HOVER_AND_POLLINATE:
            return await self._hover_and_pollinate(drone_id)
        elif action == PollinationAction.RETURN_FOR_RECHARGE:
            await self._return_for_recharge(drone_id)

        # Update flight time and battery
        state.flight_time_minutes += 0.1  # Each action takes ~6 seconds
        state.battery_percent = max(0, state.battery_percent - 0.5)  # Battery consumption

        return True

    async def _navigate_to_target(self, drone_id: str):
        """Navigate drone to its current target."""
        state = self.drone_states[drone_id]

        if state.current_target:
            # Simulate navigation with collision avoidance
            target_pos = state.current_target.location_3d
            current_pos = state.position_3d

            # Use IntelleSwarm collision avoidance
            obs = torch.randn(32)  # Mock observation
            messages = torch.randn(2, 8)  # Messages from nearby drones

            brain = self.drone_brains[drone_id]
            with torch.no_grad():
                action, internal_state = brain.step(obs, messages)

            # Update position (simplified)
            new_pos = (
                current_pos[0] + (target_pos[0] - current_pos[0]) * 0.1,
                current_pos[1] + (target_pos[1] - current_pos[1]) * 0.1,
                target_pos[2]  # Match target altitude
            )
            state.position_3d = new_pos

        await asyncio.sleep(0.05)  # Navigation time

    async def _collect_pollen(self, drone_id: str):
        """Collect pollen from current target."""
        state = self.drone_states[drone_id]

        if state.current_target and state.current_target.flower_detection.pollen_source_quality > 0.6:
            # Simulate pollen collection
            pollen_collected = state.current_target.flower_detection.pollen_source_quality * 2.0
            state.pollen_load = min(10.0, state.pollen_load + pollen_collected)

        await asyncio.sleep(0.1)  # Collection time

    async def _transfer_pollen(self, drone_id: str) -> bool:
        """Transfer pollen to current target flower."""
        state = self.drone_states[drone_id]

        if (state.current_target and
            state.pollen_load > 0.1 and
            state.current_target.flower_detection.pollination_status.value == "ready"):

            # Simulate pollen transfer
            transfer_amount = min(state.pollen_load, state.current_target.estimated_pollen_load)
            success_probability = 0.85  # 85% success rate

            if np.random.random() < success_probability:
                state.pollen_load -= transfer_amount
                state.current_target.successful_pollination = True
                state.current_target.visit_count += 1
                state.targets_completed += 1
                state.last_pollination_success = True

                self.performance_metrics['pollen_transfer_events'] += 1
                await asyncio.sleep(0.15)  # Transfer time
                return True

        state.last_pollination_success = False
        await asyncio.sleep(0.05)
        return False

    async def _hover_and_pollinate(self, drone_id: str) -> bool:
        """Hover at target and perform pollination action."""
        # Combine collection and transfer in one action
        await self._collect_pollen(drone_id)
        return await self._transfer_pollen(drone_id)

    async def _return_for_recharge(self, drone_id: str):
        """Return drone for battery recharge."""
        state = self.drone_states[drone_id]

        # Simulate return to base
        state.position_3d = (37.7749, -122.4194, 0)  # Landing position
        await asyncio.sleep(0.2)  # Return flight time

        # Simulate recharging
        state.battery_percent = 100.0
        state.flight_time_minutes = 0.0

    async def _phase_cross_pollination(self) -> Dict[str, Any]:
        """Phase 4: Execute cross-pollination coordination."""

        cross_pollination_events = 0

        # Identify cross-pollination opportunities
        for drone_id, state in self.drone_states.items():
            assignment = self._get_drone_assignment(drone_id)
            if assignment and assignment.cross_pollination_pairs:

                for source_target, dest_target in assignment.cross_pollination_pairs:
                    # Coordinate pollen transfer between specific flowers
                    success = await self._execute_cross_pollination(
                        drone_id, source_target, dest_target
                    )
                    if success:
                        cross_pollination_events += 1

        return {
            'cross_pollination_events': cross_pollination_events,
            'cross_pollination_efficiency': cross_pollination_events / max(1, len(self.drone_states))
        }

    async def _execute_cross_pollination(self,
                                       drone_id: str,
                                       source: PollinationTarget,
                                       destination: PollinationTarget) -> bool:
        """Execute cross-pollination between two specific targets."""

        state = self.drone_states[drone_id]

        # Navigate to source
        state.current_target = source
        await self._navigate_to_target(drone_id)
        await self._collect_pollen(drone_id)

        # Navigate to destination
        state.current_target = destination
        await self._navigate_to_target(drone_id)
        success = await self._transfer_pollen(drone_id)

        return success

    async def _phase_mission_completion(self) -> Dict[str, Any]:
        """Phase 5: Complete mission and return all drones."""

        return_tasks = []

        for drone_id, state in self.drone_states.items():
            task = self._return_drone_to_base(drone_id)
            return_tasks.append(task)

        await asyncio.gather(*return_tasks)

        # Calculate final metrics
        total_flight_time = sum(state.flight_time_minutes for state in self.drone_states.values())
        mission_duration = (time.time() - self.mission_start_time) / 60.0  # minutes

        return {
            'mission_duration_minutes': mission_duration,
            'total_flight_time_minutes': total_flight_time,
            'all_drones_returned': True,
            'mission_success': self._evaluate_mission_success()
        }

    async def _return_drone_to_base(self, drone_id: str):
        """Return a single drone to base."""
        state = self.drone_states[drone_id]
        state.action = PollinationAction.RETURN_FOR_RECHARGE

        # Simulate return flight
        await asyncio.sleep(0.3)
        state.position_3d = (37.7749, -122.4194, 0)  # Landing position

    def _initialize_drone_states(self, assignments: List[DroneAssignment]):
        """Initialize drone states from mission assignments."""

        for assignment in assignments:
            initial_target = assignment.assigned_targets[0] if assignment.assigned_targets else None

            state = PollinationState(
                drone_id=assignment.drone_id,
                position_3d=(37.7749, -122.4194, 0),  # Starting position
                battery_percent=100.0,
                pollen_load=0.0,
                current_target=initial_target,
                action=PollinationAction.NAVIGATE_TO_TARGET,
                targets_completed=0,
                flight_time_minutes=0.0,
                last_pollination_success=False,
                environmental_conditions={
                    'temperature': 20.0,
                    'humidity': 60.0,
                    'wind_speed': 2.0,
                    'weather_optimal': True
                }
            )

            self.drone_states[assignment.drone_id] = state

    def _create_observation(self, drone_id: str, state: PollinationState) -> PollinationObservation:
        """Create observation for reinforcement learning."""

        # Find nearby drones
        nearby_drones = []
        for other_id, other_state in self.drone_states.items():
            if other_id != drone_id:
                distance = self._calculate_distance(state.position_3d, other_state.position_3d)
                if distance <= self.communication_range:
                    nearby_drones.append(other_state)

        # Get visible targets
        assignment = self._get_drone_assignment(drone_id)
        visible_targets = assignment.assigned_targets if assignment else []

        return PollinationObservation(
            state=state,
            nearby_drones=nearby_drones[:3],  # Max 3 nearby drones
            visible_targets=visible_targets[:5],  # Max 5 visible targets
            environmental_data=state.environmental_conditions
        )

    def _get_drone_assignment(self, drone_id: str) -> Optional[DroneAssignment]:
        """Get mission assignment for a specific drone."""
        if not self.active_mission:
            return None

        for assignment in self.active_mission.drone_assignments:
            if assignment.drone_id == drone_id:
                return assignment
        return None

    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate 3D distance between two positions."""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    def _check_mission_completion(self) -> bool:
        """Check if mission completion criteria are met."""
        if not self.active_mission:
            return True

        # Check success criteria
        criteria = self.active_mission.success_criteria
        targets_completed = sum(state.targets_completed for state in self.drone_states.values())
        total_targets = len(self.active_mission.total_targets)

        completion_rate = targets_completed / max(total_targets, 1) * 100
        min_completion_rate = criteria.get('minimum_targets_completed_percent', 80.0)

        return completion_rate >= min_completion_rate

    def _calculate_efficiency(self) -> float:
        """Calculate current mission efficiency."""
        if not self.drone_states:
            return 0.0

        total_targets = sum(state.targets_completed for state in self.drone_states.values())
        total_flight_time = sum(state.flight_time_minutes for state in self.drone_states.values())

        if total_flight_time == 0:
            return 0.0

        return total_targets / total_flight_time  # Targets per minute

    def _evaluate_mission_success(self) -> bool:
        """Evaluate overall mission success."""
        if not self.active_mission:
            return False

        criteria = self.active_mission.success_criteria

        # Check multiple success criteria
        targets_completed = sum(state.targets_completed for state in self.drone_states.values())
        total_targets = len(self.active_mission.total_targets)
        completion_rate = targets_completed / max(total_targets, 1) * 100

        successful_pollinations = self.performance_metrics['successful_pollinations']
        success_rate = successful_pollinations / max(targets_completed, 1) * 100

        min_completion = criteria.get('minimum_targets_completed_percent', 80.0)
        min_efficiency = criteria.get('minimum_pollen_transfer_efficiency', 0.75) * 100

        return completion_rate >= min_completion and success_rate >= min_efficiency

    async def _emergency_mission_abort(self):
        """Execute emergency mission abort protocols."""
        logger.warning("Executing emergency mission abort")

        # Land all drones immediately
        for drone_id, state in self.drone_states.items():
            state.action = PollinationAction.EMERGENCY_LAND
            state.position_3d = (state.position_3d[0], state.position_3d[1], 0)

        await asyncio.sleep(1.0)  # Emergency landing time

    def _generate_mission_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive mission report."""

        targets_completed = sum(state.targets_completed for state in self.drone_states.values())
        total_targets = len(self.active_mission.total_targets) if self.active_mission else 1
        success_rate = (self.performance_metrics['successful_pollinations'] / max(targets_completed, 1)) * 100

        return {
            'mission_id': self.active_mission.mission_id if self.active_mission else 'unknown',
            'targets_completed': targets_completed,
            'total_targets': total_targets,
            'completion_percentage': (targets_completed / max(total_targets, 1)) * 100,  # Fixed: prevent division by zero
            'success_rate': success_rate,
            'successful_pollinations': self.performance_metrics['successful_pollinations'],
            'pollen_transfer_events': self.performance_metrics['pollen_transfer_events'],
            'total_flight_time_minutes': sum(state.flight_time_minutes for state in self.drone_states.values()),
            'mission_duration_minutes': (time.time() - self.mission_start_time) / 60.0,
            'coordination_efficiency': self._calculate_efficiency(),
            'drone_performance': {
                drone_id: {
                    'targets_completed': state.targets_completed,
                    'flight_time': state.flight_time_minutes,
                    'final_battery': state.battery_percent
                }
                for drone_id, state in self.drone_states.items()
            },
            'phase_results': results,
            'mission_success': self._evaluate_mission_success()
        }