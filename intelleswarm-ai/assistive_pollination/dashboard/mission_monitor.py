"""
Mission Monitor for Assistive Pollination System.

Real-time monitoring and analysis of individual drone missions,
providing detailed tracking of pollination progress, drone performance,
and environmental conditions during active missions.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np

# Import IntelleSwarm components
import sys
sys.path.append('../..')

# Import mission and coordination components
from mission.agricultural_planner import MissionPlan, DroneAssignment, PollinationTarget
from coordination.pollination_swarm import PollinationSwarm, PollinationState, PollinationAction
from models.flower_detector import FlowerDetection, PollinationStatus

logger = logging.getLogger(__name__)


class MissionPhase(Enum):
    """Mission execution phases."""
    INITIALIZING = "initializing"
    DEPLOYMENT = "deployment"
    FIELD_SURVEY = "field_survey"
    ACTIVE_POLLINATION = "active_pollination"
    CROSS_POLLINATION = "cross_pollination"
    MISSION_COMPLETION = "mission_completion"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass
class DronePerformanceMetrics:
    """Performance metrics for individual drone."""
    drone_id: str
    total_flight_time_minutes: float
    targets_approached: int
    targets_completed: int
    successful_pollinations: int
    pollen_collected_grams: float
    pollen_transferred_grams: float
    battery_consumption_percent: float
    average_speed_ms: float
    collision_avoidance_activations: int
    communication_quality_score: float
    position_accuracy_meters: float
    last_heartbeat: datetime


@dataclass
class MissionEnvironmentalData:
    """Environmental conditions during mission."""
    timestamp: datetime
    temperature_celsius: float
    humidity_percent: float
    wind_speed_ms: float
    wind_direction_degrees: float
    light_intensity_lux: float
    weather_conditions: str
    precipitation_mm: float
    air_pressure_hpa: float
    optimal_for_pollination: bool


@dataclass
class PollinationEvent:
    """Individual pollination event record."""
    event_id: str
    drone_id: str
    timestamp: datetime
    source_flower_id: Optional[str]
    target_flower_id: str
    flower_species: str
    pollen_amount_mg: float
    success_probability: float
    actual_success: bool
    environmental_conditions: MissionEnvironmentalData
    gps_coordinates: Tuple[float, float, float]
    image_captured: bool
    quality_score: float


class MissionMonitor:
    """
    Real-time mission monitoring and analysis system.

    Provides comprehensive tracking of drone missions including:
    - Real-time drone performance monitoring
    - Environmental condition tracking
    - Pollination event logging and analysis
    - Mission efficiency calculations
    - Anomaly detection and alerts
    - Historical performance comparison
    """

    def __init__(self):
        # Mission tracking
        self.active_missions: Dict[str, MissionPlan] = {}
        self.mission_phases: Dict[str, MissionPhase] = {}
        self.mission_start_times: Dict[str, datetime] = {}
        self.mission_environmental_history: Dict[str, List[MissionEnvironmentalData]] = {}

        # Performance tracking
        self.drone_metrics: Dict[str, DronePerformanceMetrics] = {}
        self.pollination_events: Dict[str, List[PollinationEvent]] = {}
        self.mission_alerts: Dict[str, List[Dict[str, Any]]] = {}

        # Real-time monitoring state
        self.monitoring_active: Dict[str, bool] = {}
        self.update_intervals: Dict[str, float] = {}
        self.performance_thresholds = {
            'low_battery_percent': 20.0,
            'max_flight_time_minutes': 45.0,
            'min_success_rate_percent': 75.0,
            'max_wind_speed_ms': 8.0,
            'min_communication_quality': 0.7
        }

    async def start_mission_monitoring(self,
                                     mission_plan: MissionPlan,
                                     swarm_coordinator: PollinationSwarm,
                                     update_interval: float = 1.0) -> str:
        """
        Start comprehensive monitoring of a mission.

        Args:
            mission_plan: The mission to monitor
            swarm_coordinator: Active swarm coordinator
            update_interval: Monitoring update frequency in seconds

        Returns:
            Monitor session ID
        """
        mission_id = mission_plan.mission_id
        logger.info(f"Starting mission monitoring for {mission_id}")

        # Initialize mission tracking
        self.active_missions[mission_id] = mission_plan
        self.mission_phases[mission_id] = MissionPhase.INITIALIZING
        self.mission_start_times[mission_id] = datetime.now()
        self.mission_environmental_history[mission_id] = []
        self.pollination_events[mission_id] = []
        self.mission_alerts[mission_id] = []
        self.monitoring_active[mission_id] = True
        self.update_intervals[mission_id] = update_interval

        # Initialize drone metrics
        for assignment in mission_plan.drone_assignments:
            drone_id = assignment.drone_id
            self.drone_metrics[drone_id] = DronePerformanceMetrics(
                drone_id=drone_id,
                total_flight_time_minutes=0.0,
                targets_approached=0,
                targets_completed=0,
                successful_pollinations=0,
                pollen_collected_grams=0.0,
                pollen_transferred_grams=0.0,
                battery_consumption_percent=0.0,
                average_speed_ms=0.0,
                collision_avoidance_activations=0,
                communication_quality_score=1.0,
                position_accuracy_meters=0.0,
                last_heartbeat=datetime.now()
            )

        # Start background monitoring task
        monitor_task = asyncio.create_task(
            self._monitor_mission_loop(mission_id, swarm_coordinator)
        )

        logger.info(f"Mission monitoring started for {mission_id}")
        return f"monitor_{mission_id}"

    async def _monitor_mission_loop(self, mission_id: str, swarm_coordinator: PollinationSwarm):
        """Main monitoring loop for a mission."""
        try:
            while self.monitoring_active.get(mission_id, False):
                # Update mission phase
                await self._update_mission_phase(mission_id, swarm_coordinator)

                # Collect drone telemetry and performance data
                await self._collect_drone_telemetry(mission_id, swarm_coordinator)

                # Record environmental conditions
                await self._record_environmental_conditions(mission_id)

                # Analyze performance and detect anomalies
                await self._analyze_mission_performance(mission_id)

                # Check for alerts and warnings
                await self._check_mission_alerts(mission_id)

                # Wait for next update
                await asyncio.sleep(self.update_intervals[mission_id])

        except Exception as e:
            logger.error(f"Mission monitoring error for {mission_id}: {e}")
            await self._generate_alert(mission_id, "MONITORING_ERROR", str(e), "HIGH")

        finally:
            logger.info(f"Mission monitoring ended for {mission_id}")

    async def _update_mission_phase(self, mission_id: str, swarm_coordinator: PollinationSwarm):
        """Update current mission phase based on swarm state."""
        # Analyze swarm state to determine current phase
        if mission_id not in swarm_coordinator.drone_states:
            return

        drone_states = swarm_coordinator.drone_states
        active_drones = [s for s in drone_states.values() if s.battery_percent > 10]

        if not active_drones:
            self.mission_phases[mission_id] = MissionPhase.COMPLETED
            return

        # Determine phase from drone actions
        actions = [drone.action for drone in active_drones]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        dominant_action = max(action_counts.items(), key=lambda x: x[1])[0]

        # Map actions to phases
        action_to_phase = {
            PollinationAction.NAVIGATE_TO_TARGET: MissionPhase.DEPLOYMENT,
            PollinationAction.COLLECT_POLLEN: MissionPhase.ACTIVE_POLLINATION,
            PollinationAction.TRANSFER_POLLEN: MissionPhase.ACTIVE_POLLINATION,
            PollinationAction.HOVER_AND_POLLINATE: MissionPhase.ACTIVE_POLLINATION,
            PollinationAction.RETURN_FOR_RECHARGE: MissionPhase.MISSION_COMPLETION,
            PollinationAction.FORMATION_ADJUST: MissionPhase.DEPLOYMENT
        }

        new_phase = action_to_phase.get(dominant_action, MissionPhase.ACTIVE_POLLINATION)

        if new_phase != self.mission_phases[mission_id]:
            logger.info(f"Mission {mission_id} phase transition: "
                       f"{self.mission_phases[mission_id].value} -> {new_phase.value}")
            self.mission_phases[mission_id] = new_phase

    async def _collect_drone_telemetry(self, mission_id: str, swarm_coordinator: PollinationSwarm):
        """Collect and analyze drone telemetry data."""
        for drone_id, state in swarm_coordinator.drone_states.items():
            if drone_id not in self.drone_metrics:
                continue

            metrics = self.drone_metrics[drone_id]

            # Update basic metrics
            metrics.total_flight_time_minutes = state.flight_time_minutes
            metrics.targets_completed = state.targets_completed
            metrics.battery_consumption_percent = 100.0 - state.battery_percent
            metrics.last_heartbeat = datetime.now()

            # Calculate derived metrics
            if state.flight_time_minutes > 0:
                distance_covered = self._estimate_distance_covered(drone_id, mission_id)
                metrics.average_speed_ms = distance_covered / (state.flight_time_minutes * 60)

            # Track pollination events
            if (state.last_pollination_success and
                state.action == PollinationAction.TRANSFER_POLLEN):
                await self._record_pollination_event(mission_id, drone_id, state)

            # Update success rates
            if state.targets_completed > 0:
                metrics.successful_pollinations = int(
                    state.targets_completed * 0.85  # Assumed success rate
                )

    async def _record_environmental_conditions(self, mission_id: str):
        """Record current environmental conditions."""
        # Simulate environmental sensor data (in production, this would come from real sensors)
        environmental_data = MissionEnvironmentalData(
            timestamp=datetime.now(),
            temperature_celsius=20.0 + np.random.normal(0, 2),
            humidity_percent=60.0 + np.random.normal(0, 5),
            wind_speed_ms=3.0 + np.random.uniform(-1, 2),
            wind_direction_degrees=np.random.uniform(0, 360),
            light_intensity_lux=50000 + np.random.normal(0, 5000),
            weather_conditions="partly_cloudy",
            precipitation_mm=0.0,
            air_pressure_hpa=1013.25 + np.random.normal(0, 2),
            optimal_for_pollination=True
        )

        # Check if conditions are optimal
        environmental_data.optimal_for_pollination = (
            15 <= environmental_data.temperature_celsius <= 27 and
            environmental_data.humidity_percent <= 80 and
            environmental_data.wind_speed_ms <= 8 and
            environmental_data.precipitation_mm == 0
        )

        self.mission_environmental_history[mission_id].append(environmental_data)

        # Alert if conditions become non-optimal
        if not environmental_data.optimal_for_pollination:
            await self._generate_alert(
                mission_id,
                "NON_OPTIMAL_CONDITIONS",
                f"Environmental conditions not optimal for pollination: "
                f"T={environmental_data.temperature_celsius:.1f}°C, "
                f"Wind={environmental_data.wind_speed_ms:.1f}m/s",
                "MEDIUM"
            )

    async def _record_pollination_event(self,
                                      mission_id: str,
                                      drone_id: str,
                                      state: PollinationState):
        """Record a pollination event."""
        if not state.current_target:
            return

        # Get latest environmental data
        env_data = (self.mission_environmental_history[mission_id][-1]
                   if self.mission_environmental_history[mission_id]
                   else None)

        event = PollinationEvent(
            event_id=f"{mission_id}_{drone_id}_{int(time.time() * 1000)}",
            drone_id=drone_id,
            timestamp=datetime.now(),
            source_flower_id=None,  # Would be tracked in production
            target_flower_id=f"flower_{state.current_target.target_id}",
            flower_species=state.current_target.flower_detection.species_prediction,
            pollen_amount_mg=state.pollen_load * 100,  # Convert to mg
            success_probability=0.85,
            actual_success=state.last_pollination_success,
            environmental_conditions=env_data,
            gps_coordinates=state.position_3d,
            image_captured=True,
            quality_score=state.current_target.flower_detection.confidence_score
        )

        self.pollination_events[mission_id].append(event)

    async def _analyze_mission_performance(self, mission_id: str):
        """Analyze overall mission performance and efficiency."""
        if mission_id not in self.active_missions:
            return

        mission_plan = self.active_missions[mission_id]
        start_time = self.mission_start_times[mission_id]
        elapsed_time = (datetime.now() - start_time).total_seconds() / 60.0  # minutes

        # Calculate mission-wide metrics
        total_targets_completed = sum(
            metrics.targets_completed for metrics in self.drone_metrics.values()
        )
        total_successful_pollinations = sum(
            metrics.successful_pollinations for metrics in self.drone_metrics.values()
        )

        # Performance analysis
        expected_completion_rate = self._calculate_expected_completion_rate(
            mission_plan, elapsed_time
        )
        actual_completion_rate = (total_targets_completed /
                                len(mission_plan.total_targets)) * 100

        performance_ratio = actual_completion_rate / max(expected_completion_rate, 1)

        # Generate performance insights
        if performance_ratio < 0.7:
            await self._generate_alert(
                mission_id,
                "LOW_PERFORMANCE",
                f"Mission performance below expected: {actual_completion_rate:.1f}% vs "
                f"expected {expected_completion_rate:.1f}%",
                "MEDIUM"
            )

    async def _check_mission_alerts(self, mission_id: str):
        """Check for various alert conditions."""
        # Check drone-specific alerts
        for drone_id, metrics in self.drone_metrics.items():
            # Low battery alert
            if metrics.battery_consumption_percent > 80:
                await self._generate_alert(
                    mission_id,
                    "LOW_BATTERY",
                    f"Drone {drone_id} battery low: "
                    f"{100 - metrics.battery_consumption_percent:.0f}%",
                    "HIGH"
                )

            # Communication quality alert
            if metrics.communication_quality_score < self.performance_thresholds['min_communication_quality']:
                await self._generate_alert(
                    mission_id,
                    "POOR_COMMUNICATION",
                    f"Drone {drone_id} communication quality poor: "
                    f"{metrics.communication_quality_score:.2f}",
                    "MEDIUM"
                )

            # Extended flight time alert
            if metrics.total_flight_time_minutes > self.performance_thresholds['max_flight_time_minutes']:
                await self._generate_alert(
                    mission_id,
                    "EXTENDED_FLIGHT",
                    f"Drone {drone_id} extended flight time: "
                    f"{metrics.total_flight_time_minutes:.1f} minutes",
                    "MEDIUM"
                )

    async def _generate_alert(self,
                            mission_id: str,
                            alert_type: str,
                            message: str,
                            priority: str):
        """Generate and store mission alert."""
        alert = {
            'alert_id': f"alert_{mission_id}_{int(time.time() * 1000)}",
            'mission_id': mission_id,
            'alert_type': alert_type,
            'message': message,
            'priority': priority,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False,
            'resolved': False
        }

        self.mission_alerts[mission_id].append(alert)
        logger.warning(f"Mission alert [{priority}] {alert_type}: {message}")

    def _calculate_expected_completion_rate(self,
                                          mission_plan: MissionPlan,
                                          elapsed_minutes: float) -> float:
        """Calculate expected mission completion rate based on elapsed time."""
        # Estimate mission duration based on targets and drone count
        total_targets = len(mission_plan.total_targets)
        num_drones = len(mission_plan.drone_assignments)

        # Estimated time per target (minutes) considering coordination overhead
        time_per_target = 2.5 + (0.5 * (12 - num_drones))  # More drones = better efficiency
        estimated_duration = (total_targets / num_drones) * time_per_target

        return min(100.0, (elapsed_minutes / estimated_duration) * 100)

    def _estimate_distance_covered(self, drone_id: str, mission_id: str) -> float:
        """Estimate total distance covered by a drone (simplified calculation)."""
        # In production, this would use actual GPS track data
        metrics = self.drone_metrics[drone_id]
        return metrics.targets_completed * 50.0  # Estimate 50m per target

    def get_mission_status_report(self, mission_id: str) -> Dict[str, Any]:
        """Generate comprehensive mission status report."""
        if mission_id not in self.active_missions:
            return {"error": f"Mission {mission_id} not found"}

        mission_plan = self.active_missions[mission_id]
        start_time = self.mission_start_times[mission_id]
        current_phase = self.mission_phases[mission_id]
        elapsed_time = (datetime.now() - start_time).total_seconds() / 60.0

        # Aggregate drone metrics
        drone_summaries = {}
        total_targets_completed = 0
        total_successful_pollinations = 0

        for drone_id, metrics in self.drone_metrics.items():
            total_targets_completed += metrics.targets_completed
            total_successful_pollinations += metrics.successful_pollinations

            drone_summaries[drone_id] = {
                'targets_completed': metrics.targets_completed,
                'flight_time_minutes': metrics.total_flight_time_minutes,
                'battery_remaining_percent': 100 - metrics.battery_consumption_percent,
                'successful_pollinations': metrics.successful_pollinations,
                'performance_score': self._calculate_drone_performance_score(metrics)
            }

        # Environmental summary
        recent_env_data = (self.mission_environmental_history[mission_id][-10:]
                          if self.mission_environmental_history[mission_id]
                          else [])

        avg_temp = np.mean([e.temperature_celsius for e in recent_env_data]) if recent_env_data else 0
        avg_wind = np.mean([e.wind_speed_ms for e in recent_env_data]) if recent_env_data else 0
        optimal_conditions_percent = (
            sum(1 for e in recent_env_data if e.optimal_for_pollination) /
            max(len(recent_env_data), 1) * 100
        )

        # Recent alerts (last 10)
        recent_alerts = self.mission_alerts[mission_id][-10:]

        return {
            'mission_id': mission_id,
            'mission_status': {
                'current_phase': current_phase.value,
                'elapsed_time_minutes': elapsed_time,
                'completion_percentage': (total_targets_completed / max(len(mission_plan.total_targets), 1)) * 100,  # Fixed: prevent division by zero
                'estimated_remaining_minutes': self._estimate_remaining_time(mission_id)
            },
            'performance_metrics': {
                'total_targets_completed': total_targets_completed,
                'total_successful_pollinations': total_successful_pollinations,
                'success_rate_percent': (total_successful_pollinations / max(total_targets_completed, 1)) * 100,
                'efficiency_targets_per_minute': total_targets_completed / max(elapsed_time, 1),
                'total_pollination_events': len(self.pollination_events[mission_id])
            },
            'drone_performance': drone_summaries,
            'environmental_conditions': {
                'average_temperature_celsius': avg_temp,
                'average_wind_speed_ms': avg_wind,
                'optimal_conditions_percent': optimal_conditions_percent,
                'current_conditions': recent_env_data[-1].__dict__ if recent_env_data else None
            },
            'recent_alerts': [
                {
                    'alert_type': alert['alert_type'],
                    'message': alert['message'],
                    'priority': alert['priority'],
                    'timestamp': alert['timestamp']
                }
                for alert in recent_alerts
            ],
            'mission_plan_summary': {
                'field_area_hectares': getattr(mission_plan, 'field_area_hectares', 0),
                'total_targets': len(mission_plan.total_targets),
                'drones_assigned': len(mission_plan.drone_assignments),
                'crop_species': getattr(mission_plan, 'crop_species', 'unknown')
            }
        }

    def _calculate_drone_performance_score(self, metrics: DronePerformanceMetrics) -> float:
        """Calculate overall performance score for a drone (0-100)."""
        # Weighted scoring based on various metrics
        scores = []

        # Target completion efficiency
        if metrics.total_flight_time_minutes > 0:
            target_efficiency = metrics.targets_completed / metrics.total_flight_time_minutes
            scores.append(min(100, target_efficiency * 50))  # Scale to 0-100

        # Battery efficiency
        if metrics.battery_consumption_percent > 0:
            battery_efficiency = metrics.targets_completed / metrics.battery_consumption_percent
            scores.append(min(100, battery_efficiency * 20))

        # Communication quality
        scores.append(metrics.communication_quality_score * 100)

        # Success rate
        if metrics.targets_completed > 0:
            success_rate = metrics.successful_pollinations / metrics.targets_completed
            scores.append(success_rate * 100)

        return np.mean(scores) if scores else 0.0

    def _estimate_remaining_time(self, mission_id: str) -> float:
        """Estimate remaining mission time in minutes."""
        if mission_id not in self.active_missions:
            return 0.0

        mission_plan = self.active_missions[mission_id]
        total_targets = len(mission_plan.total_targets)
        completed_targets = sum(
            metrics.targets_completed for metrics in self.drone_metrics.values()
        )

        remaining_targets = total_targets - completed_targets
        if remaining_targets <= 0:
            return 0.0

        # Calculate average completion rate
        elapsed_time = (datetime.now() -
                       self.mission_start_times[mission_id]).total_seconds() / 60.0

        if completed_targets > 0 and elapsed_time > 0:
            completion_rate = completed_targets / elapsed_time  # targets per minute
            return remaining_targets / completion_rate

        return 60.0  # Default estimate

    def get_pollination_events_summary(self, mission_id: str) -> Dict[str, Any]:
        """Get summary of pollination events for a mission."""
        if mission_id not in self.pollination_events:
            return {"error": f"No pollination events found for mission {mission_id}"}

        events = self.pollination_events[mission_id]

        if not events:
            return {
                'mission_id': mission_id,
                'total_events': 0,
                'successful_events': 0,
                'success_rate_percent': 0.0
            }

        successful_events = [e for e in events if e.actual_success]
        total_pollen_transferred = sum(e.pollen_amount_mg for e in successful_events)

        # Species breakdown
        species_counts = {}
        for event in events:
            species = event.flower_species
            if species not in species_counts:
                species_counts[species] = {'total': 0, 'successful': 0}
            species_counts[species]['total'] += 1
            if event.actual_success:
                species_counts[species]['successful'] += 1

        # Time distribution
        event_times = [e.timestamp.hour for e in events]
        hourly_distribution = {}
        for hour in range(24):
            hourly_distribution[hour] = event_times.count(hour)

        return {
            'mission_id': mission_id,
            'total_events': len(events),
            'successful_events': len(successful_events),
            'success_rate_percent': (len(successful_events) / len(events)) * 100,
            'total_pollen_transferred_mg': total_pollen_transferred,
            'average_pollen_per_event_mg': np.mean([e.pollen_amount_mg for e in events]),
            'species_breakdown': species_counts,
            'hourly_distribution': hourly_distribution,
            'quality_metrics': {
                'average_quality_score': np.mean([e.quality_score for e in events]),
                'events_with_images': sum(1 for e in events if e.image_captured)
            }
        }

    async def stop_mission_monitoring(self, mission_id: str):
        """Stop monitoring a specific mission."""
        if mission_id in self.monitoring_active:
            self.monitoring_active[mission_id] = False
            logger.info(f"Stopped monitoring mission {mission_id}")

    async def stop_all_monitoring(self):
        """Stop monitoring all active missions."""
        for mission_id in list(self.monitoring_active.keys()):
            await self.stop_mission_monitoring(mission_id)


# Example usage and testing
async def example_mission_monitoring():
    """Example of how to use the MissionMonitor."""
    from ..mission.agricultural_planner import AgriculturalMissionPlanner
    from ..coordination.pollination_swarm import PollinationSwarm

    # Initialize components
    monitor = MissionMonitor()
    planner = AgriculturalMissionPlanner()
    swarm = PollinationSwarm(max_drones=6)

    # Create a sample mission
    mission_plan = await planner.plan_mission(
        field_bounds={'lat_range': (37.77, 37.78), 'lon_range': (-122.43, -122.42)},
        crop_species="apple",
        num_drones=6,
        environmental_conditions={'temperature': 22, 'humidity': 65, 'wind_speed': 3}
    )

    # Start monitoring
    monitor_id = await monitor.start_mission_monitoring(mission_plan, swarm, update_interval=0.5)

    # Simulate mission execution for 30 seconds
    await asyncio.sleep(30)

    # Get mission status
    status_report = monitor.get_mission_status_report(mission_plan.mission_id)
    print(json.dumps(status_report, indent=2, default=str))

    # Get pollination events summary
    events_summary = monitor.get_pollination_events_summary(mission_plan.mission_id)
    print(json.dumps(events_summary, indent=2, default=str))

    # Stop monitoring
    await monitor.stop_mission_monitoring(mission_plan.mission_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_mission_monitoring())