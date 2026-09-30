"""
Farmer Dashboard for Assistive Pollination System.

Web-based interface for farmers to monitor and control drone pollination missions.
Provides real-time telemetry, field visualization, and mission management capabilities.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Web framework imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Import IntelleSwarm components
import sys
sys.path.append('../..')
from genai_framework.api.websocket_manager import ConnectionManager
from genai_framework.api.models import TelemetryPayload

# Import pollination system components
from mission.agricultural_planner import AgriculturalMissionPlanner, MissionPlan
from coordination.pollination_swarm import PollinationSwarm, PollinationState
from models.flower_detector import FlowerDetector, FlowerDetection
from models.crop_classifier import CropSpeciesClassifier

logger = logging.getLogger(__name__)


@dataclass
class FieldConfiguration:
    """Field configuration for pollination mission."""
    field_id: str
    field_name: str
    latitude_bounds: tuple[float, float]  # (min_lat, max_lat)
    longitude_bounds: tuple[float, float]  # (min_lon, max_lon)
    crop_species: str
    field_area_hectares: float
    estimated_flower_count: int
    pollination_requirements: Dict[str, Any]
    environmental_sensors: List[str]


class DashboardAPI(BaseModel):
    """API models for dashboard communication."""

    class MissionRequest(BaseModel):
        field_id: str
        num_drones: int = Field(ge=1, le=12, description="Number of drones (1-12)")
        priority_areas: Optional[List[Dict[str, float]]] = None
        weather_threshold: Optional[Dict[str, float]] = None
        mission_duration_hours: float = Field(ge=0.5, le=8.0, description="Mission duration (0.5-8 hours)")

    class MissionStatus(BaseModel):
        mission_id: str
        status: str
        progress_percentage: float
        drones_active: int
        targets_completed: int
        estimated_completion: datetime
        performance_metrics: Dict[str, float]

    class DroneStatusUpdate(BaseModel):
        drone_id: str
        position: List[float]  # [lat, lon, alt]
        battery_percent: float
        action: str
        targets_completed: int
        pollen_load: float
        last_update: datetime


class FarmerDashboard:
    """
    Main dashboard interface for farmers to manage drone pollination missions.

    Provides:
    - Real-time mission monitoring
    - Field visualization with 3D mapping
    - Drone telemetry and status tracking
    - Mission planning and control
    - Performance analytics and reporting
    """

    def __init__(self, port: int = 8080):
        self.port = port
        self.app = FastAPI(
            title="AgriSwarm Dashboard",
            description="Assistive Pollination Drone Management System",
            version="1.0.0"
        )

        # Configure CORS for web interface
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # WebSocket connection manager
        self.connection_manager = ConnectionManager()

        # System components
        self.mission_planner = AgriculturalMissionPlanner()
        self.swarm_coordinator = PollinationSwarm(max_drones=12)
        self.flower_detector = FlowerDetector(num_species=50)
        self.crop_classifier = CropSpeciesClassifier(num_species=50)

        # Dashboard state
        self.registered_fields: Dict[str, FieldConfiguration] = {}
        self.active_missions: Dict[str, MissionPlan] = {}
        self.mission_status: Dict[str, Dict[str, Any]] = {}
        self.live_telemetry: Dict[str, Dict[str, Any]] = {}

        # Setup API routes
        self._setup_routes()

        # Initialize with example field
        self._setup_example_field()

    def _setup_routes(self):
        """Setup all API routes for the dashboard."""

        # Static files for web interface
        static_path = Path(__file__).parent / "static"
        if static_path.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve main dashboard HTML interface."""
            return await self._render_dashboard_html()

        @self.app.get("/api/fields")
        async def get_fields():
            """Get all registered fields."""
            return {
                "fields": [asdict(field) for field in self.registered_fields.values()],
                "total_fields": len(self.registered_fields)
            }

        @self.app.post("/api/fields")
        async def register_field(field: Dict[str, Any]):
            """Register a new field for pollination missions."""
            try:
                field_config = FieldConfiguration(**field)
                self.registered_fields[field_config.field_id] = field_config

                logger.info(f"Registered field: {field_config.field_name}")
                return {"status": "success", "field_id": field_config.field_id}

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid field configuration: {str(e)}"
                )

        @self.app.post("/api/missions")
        async def create_mission(request: DashboardAPI.MissionRequest):
            """Create and start a new pollination mission."""
            try:
                # Validate field exists
                if request.field_id not in self.registered_fields:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Field {request.field_id} not found"
                    )

                field_config = self.registered_fields[request.field_id]

                # Create mission plan
                mission_plan = await self.mission_planner.plan_mission(
                    field_bounds={
                        'lat_range': field_config.latitude_bounds,
                        'lon_range': field_config.longitude_bounds
                    },
                    crop_species=field_config.crop_species,
                    num_drones=request.num_drones,
                    environmental_conditions={
                        'temperature': 22.0,
                        'humidity': 65.0,
                        'wind_speed': 3.0,
                        'weather_optimal': True
                    }
                )

                # Store mission
                self.active_missions[mission_plan.mission_id] = mission_plan
                self.mission_status[mission_plan.mission_id] = {
                    'status': 'starting',
                    'progress': 0.0,
                    'start_time': datetime.now(),
                    'field_id': request.field_id
                }

                # Start mission execution (background task)
                asyncio.create_task(self._execute_mission_background(mission_plan))

                return {
                    "mission_id": mission_plan.mission_id,
                    "status": "created",
                    "estimated_duration_hours": request.mission_duration_hours,
                    "drones_assigned": request.num_drones
                }

            except Exception as e:
                logger.error(f"Mission creation error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Mission creation failed: {str(e)}"
                )

        @self.app.get("/api/missions")
        async def get_missions():
            """Get all missions and their status."""
            mission_summaries = []

            for mission_id, status in self.mission_status.items():
                mission_plan = self.active_missions.get(mission_id)
                if mission_plan:
                    mission_summaries.append({
                        'mission_id': mission_id,
                        'field_id': status['field_id'],
                        'status': status['status'],
                        'progress_percentage': status['progress'],
                        'drones_assigned': len(mission_plan.drone_assignments),
                        'start_time': status['start_time'].isoformat(),
                        'targets_total': len(mission_plan.total_targets),
                        'targets_completed': status.get('targets_completed', 0)
                    })

            return {"missions": mission_summaries}

        @self.app.get("/api/missions/{mission_id}")
        async def get_mission_details(mission_id: str):
            """Get detailed information about a specific mission."""
            if mission_id not in self.active_missions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Mission {mission_id} not found"
                )

            mission_plan = self.active_missions[mission_id]
            status = self.mission_status[mission_id]

            return {
                'mission_plan': asdict(mission_plan),
                'current_status': status,
                'drone_telemetry': self._get_mission_telemetry(mission_id)
            }

        @self.app.delete("/api/missions/{mission_id}")
        async def abort_mission(mission_id: str):
            """Abort an active mission."""
            if mission_id not in self.active_missions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Mission {mission_id} not found"
                )

            # Update status to aborted
            self.mission_status[mission_id]['status'] = 'aborted'
            self.mission_status[mission_id]['abort_time'] = datetime.now()

            # Broadcast abort command to all connected clients
            await self.connection_manager.broadcast({
                'type': 'mission_abort',
                'mission_id': mission_id,
                'timestamp': datetime.now().isoformat()
            })

            return {"status": "mission_aborted", "mission_id": mission_id}

        @self.app.get("/api/telemetry/live")
        async def get_live_telemetry():
            """Get current live telemetry for all active drones."""
            return {
                "telemetry": self.live_telemetry,
                "timestamp": datetime.now().isoformat(),
                "active_drones": len(self.live_telemetry)
            }

        @self.app.get("/api/analytics/field/{field_id}")
        async def get_field_analytics(field_id: str):
            """Get analytics and performance data for a specific field."""
            if field_id not in self.registered_fields:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Field {field_id} not found"
                )

            # Generate field analytics
            analytics = await self._generate_field_analytics(field_id)
            return analytics

        @self.app.websocket("/ws/dashboard")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time dashboard updates."""
            await self.connection_manager.connect(websocket)

            try:
                # Send initial dashboard state
                initial_data = {
                    'type': 'dashboard_init',
                    'fields': [asdict(f) for f in self.registered_fields.values()],
                    'active_missions': len(self.active_missions),
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send_json(initial_data)

                # Keep connection alive and handle incoming messages
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    # Handle dashboard commands
                    await self._handle_websocket_command(websocket, message)

            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connection_manager.disconnect(websocket)

    async def _execute_mission_background(self, mission_plan: MissionPlan):
        """Execute mission in background and update status."""
        try:
            logger.info(f"Starting background execution for mission {mission_plan.mission_id}")

            # Update status to active
            self.mission_status[mission_plan.mission_id]['status'] = 'active'

            # Execute mission using swarm coordinator
            mission_results = await self.swarm_coordinator.execute_pollination_mission(mission_plan)

            # Update final status
            final_status = 'completed' if mission_results.get('mission_success') else 'failed'
            self.mission_status[mission_plan.mission_id].update({
                'status': final_status,
                'progress': 100.0,
                'completion_time': datetime.now(),
                'results': mission_results
            })

            # Broadcast completion
            await self.connection_manager.broadcast({
                'type': 'mission_completed',
                'mission_id': mission_plan.mission_id,
                'results': mission_results,
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"Mission {mission_plan.mission_id} completed with status: {final_status}")

        except Exception as e:
            logger.error(f"Mission execution error: {e}")
            self.mission_status[mission_plan.mission_id].update({
                'status': 'error',
                'error_message': str(e),
                'error_time': datetime.now()
            })

    async def update_drone_telemetry(self, telemetry: TelemetryPayload):
        """Update drone telemetry and broadcast to dashboard clients."""
        self.live_telemetry[telemetry.drone_id] = {
            'drone_id': telemetry.drone_id,
            'position': telemetry.position,
            'velocity': telemetry.velocity,
            'battery': telemetry.battery,
            'timestamp': telemetry.timestamp,
            'last_update': datetime.now().isoformat()
        }

        # Broadcast telemetry update
        await self.connection_manager.broadcast({
            'type': 'telemetry_update',
            'drone_id': telemetry.drone_id,
            'data': self.live_telemetry[telemetry.drone_id]
        })

    def _get_mission_telemetry(self, mission_id: str) -> Dict[str, Any]:
        """Get telemetry data for all drones in a specific mission."""
        mission_plan = self.active_missions.get(mission_id)
        if not mission_plan:
            return {}

        mission_telemetry = {}
        for assignment in mission_plan.drone_assignments:
            drone_id = assignment.drone_id
            if drone_id in self.live_telemetry:
                mission_telemetry[drone_id] = self.live_telemetry[drone_id]

        return mission_telemetry

    async def _generate_field_analytics(self, field_id: str) -> Dict[str, Any]:
        """Generate comprehensive analytics for a field."""
        field_config = self.registered_fields[field_id]

        # Calculate field metrics
        completed_missions = [
            m for m in self.mission_status.values()
            if m.get('field_id') == field_id and m['status'] == 'completed'
        ]

        total_missions = len([
            m for m in self.mission_status.values()
            if m.get('field_id') == field_id
        ])

        success_rate = len(completed_missions) / max(total_missions, 1) * 100

        # Aggregate performance metrics
        total_targets_completed = 0
        total_pollination_events = 0
        average_efficiency = 0.0

        for status in completed_missions:
            results = status.get('results', {})
            total_targets_completed += results.get('targets_completed', 0)
            total_pollination_events += results.get('successful_pollinations', 0)
            average_efficiency += results.get('coordination_efficiency', 0.0)

        if completed_missions:
            average_efficiency /= len(completed_missions)

        return {
            'field_id': field_id,
            'field_name': field_config.field_name,
            'area_hectares': field_config.field_area_hectares,
            'crop_species': field_config.crop_species,
            'missions': {
                'total_missions': total_missions,
                'completed_missions': len(completed_missions),
                'success_rate_percent': success_rate
            },
            'performance': {
                'total_targets_completed': total_targets_completed,
                'total_pollination_events': total_pollination_events,
                'average_efficiency': average_efficiency,
                'targets_per_hectare': total_targets_completed / field_config.field_area_hectares
            },
            'environmental_data': {
                'optimal_conditions_percent': 85.0,
                'average_temperature': 22.5,
                'average_humidity': 62.0,
                'weather_delays': 2
            },
            'recommendations': await self._generate_field_recommendations(field_config, completed_missions)
        }

    async def _generate_field_recommendations(self,
                                           field_config: FieldConfiguration,
                                           completed_missions: List[Dict]) -> List[str]:
        """Generate recommendations for field management."""
        recommendations = []

        if len(completed_missions) < 3:
            recommendations.append("Consider running more pollination missions for better crop yield optimization")

        # Analyze pollination efficiency
        if completed_missions:
            avg_success_rate = sum(
                m.get('results', {}).get('successful_pollinations', 0) /
                max(m.get('results', {}).get('targets_completed', 1), 1)
                for m in completed_missions
            ) / len(completed_missions) * 100

            if avg_success_rate < 75:
                recommendations.append("Consider optimizing drone pollination parameters for higher success rate")

        # Field-specific recommendations
        if field_config.crop_species in ['apple', 'cherry', 'almond']:
            recommendations.append("Schedule missions during peak flowering period (early morning) for optimal pollen viability")

        if field_config.field_area_hectares > 5:
            recommendations.append("Consider increasing drone count for larger field coverage")

        return recommendations

    async def _handle_websocket_command(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming WebSocket commands from dashboard clients."""
        command_type = message.get('type')

        if command_type == 'subscribe_telemetry':
            # Subscribe to specific drone telemetry
            drone_id = message.get('drone_id')
            if drone_id and drone_id in self.live_telemetry:
                await websocket.send_json({
                    'type': 'telemetry_data',
                    'drone_id': drone_id,
                    'data': self.live_telemetry[drone_id]
                })

        elif command_type == 'request_mission_update':
            # Send current mission status
            mission_id = message.get('mission_id')
            if mission_id and mission_id in self.mission_status:
                await websocket.send_json({
                    'type': 'mission_status',
                    'mission_id': mission_id,
                    'status': self.mission_status[mission_id]
                })

        elif command_type == 'emergency_stop':
            # Emergency stop all missions
            for mission_id in list(self.active_missions.keys()):
                self.mission_status[mission_id]['status'] = 'emergency_stopped'

            await self.connection_manager.broadcast({
                'type': 'emergency_stop',
                'timestamp': datetime.now().isoformat()
            })

    def _setup_example_field(self):
        """Setup an example field for demonstration."""
        example_field = FieldConfiguration(
            field_id="FIELD_001",
            field_name="Sunrise Orchard - Apple Block A",
            latitude_bounds=(37.7700, 37.7800),
            longitude_bounds=(-122.4300, -122.4200),
            crop_species="apple",
            field_area_hectares=2.5,
            estimated_flower_count=12500,
            pollination_requirements={
                'cross_pollination_needed': True,
                'optimal_pollen_transfer_time': 'early_morning',
                'wind_speed_threshold': 10.0,
                'temperature_range': [15, 25]
            },
            environmental_sensors=['temperature', 'humidity', 'wind_speed', 'light_intensity']
        )

        self.registered_fields[example_field.field_id] = example_field
        logger.info("Setup example field: Sunrise Orchard")

    async def _render_dashboard_html(self) -> str:
        """Render the main dashboard HTML interface."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgriSwarm Dashboard - Assistive Pollination System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', roboto, sans-serif;
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               min-height: 100vh; color: #333; }
        .dashboard { display: grid; grid-template-columns: 300px 1fr; height: 100vh; }

        .sidebar { background: rgba(255,255,255,0.95); padding: 20px; overflow-y: auto;
                   border-right: 1px solid #e0e0e0; }
        .sidebar h2 { margin-bottom: 20px; color: #2c5530; font-size: 18px; }
        .sidebar .section { margin-bottom: 30px; }
        .sidebar .field-item, .sidebar .mission-item {
            background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px;
            border: 1px solid #e9ecef; cursor: pointer; transition: all 0.3s; }
        .sidebar .field-item:hover, .sidebar .mission-item:hover {
            background: #e3f2fd; border-color: #2196f3; }
        .sidebar .status { display: inline-block; padding: 3px 8px; border-radius: 12px;
                          font-size: 11px; font-weight: 600; text-transform: uppercase; }
        .status.active { background: #4caf50; color: white; }
        .status.completed { background: #2196f3; color: white; }
        .status.error { background: #f44336; color: white; }

        .main-content { padding: 20px; overflow-y: auto; }
        .header { background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px;
                  margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header h1 { color: #2c5530; margin-bottom: 10px; }
        .header .controls button { background: #4caf50; color: white; border: none;
                                  padding: 10px 20px; border-radius: 5px; margin-right: 10px;
                                  cursor: pointer; font-size: 14px; transition: all 0.3s; }
        .header .controls button:hover { background: #45a049; transform: translateY(-1px); }
        .header .controls button.abort { background: #f44336; }
        .header .controls button.abort:hover { background: #da190b; }

        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: rgba(255,255,255,0.9); border-radius: 10px; padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .card h3 { margin-bottom: 15px; color: #2c5530; }

        .map-container { height: 400px; background: #f0f0f0; border-radius: 8px;
                        display: flex; align-items: center; justify-content: center;
                        color: #666; font-size: 16px; }

        .telemetry-list { max-height: 350px; overflow-y: auto; }
        .drone-item { background: #f8f9fa; padding: 10px; border-radius: 5px;
                     margin-bottom: 8px; border-left: 4px solid #4caf50; }
        .drone-item .drone-id { font-weight: 600; color: #2c5530; }
        .drone-item .metrics { font-size: 12px; color: #666; margin-top: 5px; }

        .analytics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                         gap: 15px; margin-top: 15px; }
        .metric-card { background: linear-gradient(135deg, #667eea, #764ba2); color: white;
                      padding: 15px; border-radius: 8px; text-align: center; }
        .metric-card .value { font-size: 28px; font-weight: 700; }
        .metric-card .label { font-size: 12px; opacity: 0.9; }

        .mission-form { background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px;
                       margin-top: 20px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: 600; color: #2c5530; }
        .form-group input, .form-group select { width: 100%; padding: 8px; border: 1px solid #ddd;
                                               border-radius: 4px; font-size: 14px; }

        .connection-status { position: fixed; top: 20px; right: 20px; padding: 8px 16px;
                           border-radius: 20px; font-size: 12px; font-weight: 600; }
        .connection-status.connected { background: #4caf50; color: white; }
        .connection-status.disconnected { background: #f44336; color: white; }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="sidebar">
            <h2>🌱 Fields</h2>
            <div class="section" id="fields-section">
                <!-- Fields will be populated by JavaScript -->
            </div>

            <h2>🚁 Active Missions</h2>
            <div class="section" id="missions-section">
                <!-- Missions will be populated by JavaScript -->
            </div>
        </div>

        <div class="main-content">
            <div class="header">
                <h1>🌾 AgriSwarm Dashboard</h1>
                <p>Assistive Pollination System - Real-time Monitoring & Control</p>
                <div class="controls">
                    <button onclick="createNewMission()">🚀 New Mission</button>
                    <button onclick="refreshData()">🔄 Refresh</button>
                    <button class="abort" onclick="emergencyStop()">🛑 Emergency Stop</button>
                </div>
            </div>

            <div class="dashboard-grid">
                <div class="card">
                    <h3>🗺️ Field Visualization</h3>
                    <div class="map-container">
                        3D Field Map Loading...<br>
                        <small>Real-time drone positions and pollination targets</small>
                    </div>
                </div>

                <div class="card">
                    <h3>📡 Live Drone Telemetry</h3>
                    <div class="telemetry-list" id="telemetry-list">
                        <!-- Telemetry will be updated by WebSocket -->
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📊 Mission Analytics</h3>
                <div class="analytics-grid">
                    <div class="metric-card">
                        <div class="value" id="total-missions">0</div>
                        <div class="label">Total Missions</div>
                    </div>
                    <div class="metric-card">
                        <div class="value" id="active-drones">0</div>
                        <div class="label">Active Drones</div>
                    </div>
                    <div class="metric-card">
                        <div class="value" id="targets-completed">0</div>
                        <div class="label">Targets Completed</div>
                    </div>
                    <div class="metric-card">
                        <div class="value" id="success-rate">0%</div>
                        <div class="label">Success Rate</div>
                    </div>
                </div>
            </div>

            <div class="mission-form" id="mission-form" style="display: none;">
                <h3>🎯 Create New Mission</h3>
                <div class="form-group">
                    <label>Select Field:</label>
                    <select id="field-select">
                        <option value="">Choose a field...</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Number of Drones:</label>
                    <input type="number" id="drone-count" min="1" max="12" value="4">
                </div>
                <div class="form-group">
                    <label>Mission Duration (hours):</label>
                    <input type="number" id="mission-duration" min="0.5" max="8" step="0.5" value="2">
                </div>
                <button onclick="submitMission()">🚀 Start Mission</button>
                <button onclick="cancelMissionForm()">❌ Cancel</button>
            </div>
        </div>
    </div>

    <div class="connection-status" id="connection-status">Connecting...</div>

    <script>
        // Dashboard JavaScript
        let websocket = null;
        let dashboardData = {
            fields: [],
            missions: [],
            telemetry: {}
        };

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
            loadInitialData();
            setInterval(updateUI, 1000); // Update UI every second
        });

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + window.location.host + '/ws/dashboard';

            websocket = new WebSocket(wsUrl);

            websocket.onopen = function() {
                console.log('WebSocket connected');
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').className = 'connection-status connected';
            };

            websocket.onclose = function() {
                console.log('WebSocket disconnected');
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').className = 'connection-status disconnected';

                // Reconnect after 3 seconds
                setTimeout(connectWebSocket, 3000);
            };

            websocket.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            };
        }

        function handleWebSocketMessage(message) {
            switch(message.type) {
                case 'dashboard_init':
                    dashboardData.fields = message.fields;
                    updateFieldsList();
                    break;

                case 'telemetry_update':
                    dashboardData.telemetry[message.drone_id] = message.data;
                    updateTelemetryDisplay();
                    break;

                case 'mission_completed':
                case 'mission_abort':
                case 'emergency_stop':
                    loadMissionsData();
                    break;
            }
        }

        async function loadInitialData() {
            try {
                // Load fields
                const fieldsResponse = await fetch('/api/fields');
                const fieldsData = await fieldsResponse.json();
                dashboardData.fields = fieldsData.fields;
                updateFieldsList();

                // Load missions
                await loadMissionsData();

                // Load telemetry
                const telemetryResponse = await fetch('/api/telemetry/live');
                const telemetryData = await telemetryResponse.json();
                dashboardData.telemetry = telemetryData.telemetry;
                updateTelemetryDisplay();

            } catch (error) {
                console.error('Error loading initial data:', error);
            }
        }

        async function loadMissionsData() {
            try {
                const response = await fetch('/api/missions');
                const data = await response.json();
                dashboardData.missions = data.missions;
                updateMissionsList();
                updateAnalytics();
            } catch (error) {
                console.error('Error loading missions:', error);
            }
        }

        function updateFieldsList() {
            const fieldsSection = document.getElementById('fields-section');
            fieldsSection.innerHTML = '';

            dashboardData.fields.forEach(field => {
                const fieldElement = document.createElement('div');
                fieldElement.className = 'field-item';
                fieldElement.innerHTML =
                    '<strong>' + field.field_name + '</strong><br>' +
                    '<small>' + field.crop_species + ' • ' + field.field_area_hectares + ' ha</small>';

                fieldElement.onclick = () => selectField(field.field_id);
                fieldsSection.appendChild(fieldElement);
            });

            // Update field select options
            const fieldSelect = document.getElementById('field-select');
            fieldSelect.innerHTML = '<option value="">Choose a field...</option>';
            dashboardData.fields.forEach(field => {
                const option = document.createElement('option');
                option.value = field.field_id;
                option.textContent = field.field_name;
                fieldSelect.appendChild(option);
            });
        }

        function updateMissionsList() {
            const missionsSection = document.getElementById('missions-section');
            missionsSection.innerHTML = '';

            dashboardData.missions.forEach(mission => {
                const missionElement = document.createElement('div');
                missionElement.className = 'mission-item';
                missionElement.innerHTML =
                    '<strong>Mission ' + mission.mission_id.substring(0, 8) + '</strong><br>' +
                    '<span class="status ' + mission.status + '">' + mission.status + '</span><br>' +
                    '<small>' + mission.progress_percentage.toFixed(1) + '% • ' +
                    mission.drones_assigned + ' drones</small>';

                missionElement.onclick = () => selectMission(mission.mission_id);
                missionsSection.appendChild(missionElement);
            });
        }

        function updateTelemetryDisplay() {
            const telemetryList = document.getElementById('telemetry-list');
            telemetryList.innerHTML = '';

            Object.values(dashboardData.telemetry).forEach(drone => {
                const droneElement = document.createElement('div');
                droneElement.className = 'drone-item';
                droneElement.innerHTML =
                    '<div class="drone-id">' + drone.drone_id + '</div>' +
                    '<div class="metrics">' +
                    '📍 ' + drone.position[0].toFixed(4) + ', ' + drone.position[1].toFixed(4) + ' • ' +
                    '🔋 ' + drone.battery.toFixed(0) + '% • ' +
                    '⚡ ' + (drone.velocity || [0,0,0])[2].toFixed(1) + ' m/s' +
                    '</div>';

                telemetryList.appendChild(droneElement);
            });
        }

        function updateAnalytics() {
            const totalMissions = dashboardData.missions.length;
            const activeDrones = Object.keys(dashboardData.telemetry).length;
            const completedMissions = dashboardData.missions.filter(m => m.status === 'completed');
            const totalTargets = dashboardData.missions.reduce((sum, m) => sum + m.targets_completed, 0);
            const successRate = completedMissions.length / Math.max(totalMissions, 1) * 100;

            document.getElementById('total-missions').textContent = totalMissions;
            document.getElementById('active-drones').textContent = activeDrones;
            document.getElementById('targets-completed').textContent = totalTargets;
            document.getElementById('success-rate').textContent = successRate.toFixed(0) + '%';
        }

        function updateUI() {
            // Update timestamps and other dynamic content
            const now = new Date();
            // Add any periodic UI updates here
        }

        // UI Event Handlers
        function createNewMission() {
            document.getElementById('mission-form').style.display = 'block';
        }

        function cancelMissionForm() {
            document.getElementById('mission-form').style.display = 'none';
        }

        async function submitMission() {
            const fieldId = document.getElementById('field-select').value;
            const droneCount = parseInt(document.getElementById('drone-count').value);
            const duration = parseFloat(document.getElementById('mission-duration').value);

            if (!fieldId) {
                alert('Please select a field');
                return;
            }

            try {
                const response = await fetch('/api/missions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        field_id: fieldId,
                        num_drones: droneCount,
                        mission_duration_hours: duration
                    })
                });

                const result = await response.json();
                if (response.ok) {
                    alert('Mission created successfully!');
                    cancelMissionForm();
                    loadMissionsData();
                } else {
                    alert('Error creating mission: ' + result.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function emergencyStop() {
            if (confirm('Emergency stop all missions? This cannot be undone.')) {
                if (websocket) {
                    websocket.send(JSON.stringify({ type: 'emergency_stop' }));
                }
            }
        }

        function refreshData() {
            loadInitialData();
        }

        function selectField(fieldId) {
            console.log('Selected field:', fieldId);
            // Implement field selection logic
        }

        function selectMission(missionId) {
            console.log('Selected mission:', missionId);
            // Implement mission selection logic
        }
    </script>
</body>
</html>
        """

    async def start_server(self):
        """Start the dashboard server."""
        import uvicorn

        logger.info(f"Starting AgriSwarm Dashboard on port {self.port}")
        print(f"\n🌾 AgriSwarm Dashboard starting...")
        print(f"   📡 Dashboard URL: http://localhost:{self.port}")
        print(f"   🔗 WebSocket: ws://localhost:{self.port}/ws/dashboard")
        print(f"   📊 API Docs: http://localhost:{self.port}/docs")

        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )


# Example usage and testing
if __name__ == "__main__":
    dashboard = FarmerDashboard(port=8080)
    asyncio.run(dashboard.start_server())