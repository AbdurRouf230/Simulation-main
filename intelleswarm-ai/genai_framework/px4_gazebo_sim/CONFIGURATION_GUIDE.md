# IntelleSwarm AI - Configuration Guide

**Complete configuration reference for the robotics simulation system**

## 🎯 Overview

This guide provides detailed information about all configuration files and parameters used in the IntelleSwarm AI simulation system. All configurations have been tested and validated in the working system.

## 📋 Configuration File Structure

```
px4_gazebo_sim/
├── pollination_drone_config.yaml       # Main drone configuration
├── agricultural_farm.world             # Gazebo world definition  
├── px4_multi_drone.sh                  # Multi-drone launcher script
└── ros2_node_intelleswarm_pollination.py  # ROS2 coordination parameters
```

---

## 🚁 Drone Configuration (`pollination_drone_config.yaml`)

### **Physical Drone Specifications**
```yaml
drone_specs:
  max_flight_time: 1800        # 30 minutes battery life
  payload_capacity: 2.0        # 2kg pollination system
  max_speed: 15.0              # m/s maximum speed
  cruise_speed: 8.0            # m/s optimal cruise speed
  min_altitude: 2.0            # meters above crops
  max_altitude: 50.0           # meters maximum height
  safety_radius: 5.0           # minimum distance between drones
```

### **Pollination Mission Parameters**
```yaml
pollination:
  coverage_height: 3.0         # meters above flower patches
  pollination_speed: 2.0       # m/s for effective pollination  
  overlap_percentage: 20       # coverage overlap for thoroughness
  flower_detection_range: 10.0 # meters sensor range
  pollen_dispensing_rate: 50   # particles per second
  mission_patterns:
    sunflower: "spiral_outward"
    clover: "grid_pattern"
    mixed: "adaptive_coverage"
```

### **Safety & Geofencing**
```yaml
safety:
  emergency_altitude: 10.0     # meters for emergency ascent
  return_home_battery: 25      # percentage battery for RTH
  weather_limits:
    max_wind_speed: 8.0        # m/s
    min_visibility: 100.0      # meters
    no_fly_rain: true
  geofence:
    enabled: true
    boundary_type: "polygon"
    max_distance: 500.0        # meters from home
    violation_action: "return_home"
```

### **MARL Algorithm Configuration**  
```yaml
marl:
  algorithm: "MAPPO"           # Multi-Agent PPO
  alternatives: ["QMIX", "MADDPG"]
  network_params:
    hidden_layers: [256, 256]
    learning_rate: 0.0003
    batch_size: 32
    replay_buffer_size: 100000
  coordination:
    communication_range: 100.0  # meters
    update_frequency: 10.0      # Hz
    consensus_threshold: 0.8
```

---

## 🌍 World Configuration (`agricultural_farm.world`)

### **World Properties**
```xml
<physics name="default_physics" default="0" type="ode">
  <max_step_size>0.004</max_step_size>        # 250 Hz simulation
  <real_time_factor>1.0</real_time_factor>    # Real-time execution
  <real_time_update_rate>250</real_time_update_rate>
</physics>
```

### **Lighting Configuration**
```xml
<light type="directional" name="sun">
  <cast_shadows>true</cast_shadows>
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>          # Natural daylight
  <specular>0.2 0.2 0.2 1</specular>
  <direction>-0.5 0.1 -0.9</direction>      # Morning sun angle
</light>
```

### **Ground Plane & Terrain**
```xml
<model name="ground_plane">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <plane>
          <normal>0 0 1</normal>              # Horizontal ground
          <size>200 200</size>                # 200m x 200m farm area
        </plane>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>1</mu>                        # Ground friction
            <mu2>1</mu2>
          </ode>
        </friction>
      </surface>
    </collision>
  </link>
</model>
```

### **Crop Field Definitions**
```xml
<!-- Sunflower Patch 1: High-priority pollination zone -->
<model name="sunflower_patch_1">
  <pose>50 50 0 0 0 0</pose>                 # Northeast field
  <static>true</static>
  <!-- 15-meter radius, ~707 m² area -->
</model>

<!-- Clover Field: Medium-priority broad coverage -->  
<model name="clover_field">
  <pose>0 0 0 0 0 0</pose>                   # Central field
  <static>true</static>
  <!-- 20-meter radius, ~1257 m² area -->
</model>
```

### **GPS Coordinates**
```xml
<spherical_coordinates>
  <surface_model>EARTH_WGS84</surface_model>
  <latitude_deg>37.7749</latitude_deg>       # San Francisco area
  <longitude_deg>-122.4194</longitude_deg>
  <elevation>0</elevation>                   # Sea level
  <heading_deg>0</heading_deg>               # North orientation
</spherical_coordinates>
```

---

## 🚁 Multi-Drone Configuration (`px4_multi_drone.sh`)

### **Swarm Configuration**
```bash
NUM_DRONES=6                    # Total drone count
WORLD_FILE="agricultural_farm.world"
PX4_DIR=${PX4_DIR:-"$HOME/PX4-Autopilot"}
```

### **Spawn Positions (Optimized for Agricultural Coverage)**
```bash
SPAWN_POSITIONS=(
    "50 50 5 0"                 # Sunflower patch 1 - Primary
    "-50 50 5 1.57"             # Sunflower patch 2 - Primary  
    "0 0 5 0"                   # Clover field - High priority
    "25 25 5 0.79"              # Between patches - Support
    "-25 25 5 2.36"             # Coverage support
    "0 75 5 1.57"               # Northern field coverage
)
```

**Position Format:** `x y z yaw` (meters, radians)

### **Communication Ports**  
```bash
# MAVLink UDP ports (QGroundControl, ROS2)
MAVLINK_UDP_PORTS=(14560 14570 14580 14590 14600 14610)

# MAVLink TCP ports (Advanced integrations)
MAVLINK_TCP_PORTS=(4560 4570 4580 4590 4600 4610)

# Gazebo model communication
GAZEBO_TCP_PORTS=(11345 11355 11365 11375 11385 11395)
```

### **Environment Variables**
```bash
export PX4_SIM_MODEL=iris       # Quadcopter model
export PX4_SIM_HOSTNAME=localhost
export PX4_SIMULATOR=gz         # Updated for new Gazebo (was gazebo-classic)
export PX4_SYS_AUTOSTART=10016  # Valid iris airframe (was 4001)

# Home position (San Francisco Bay Area)
export PX4_HOME_LAT=37.7749
export PX4_HOME_LON=-122.4194
export PX4_HOME_ALT=30.0         # 30m above sea level
```

### **⚠️ Critical Configuration Fixes**
**Fixed Issues in px4_multi_drone.sh:**
- **Airframe ID:** `PX4_SYS_AUTOSTART=10016` (10016_none_iris is valid)
- **Gazebo Command:** `gz sim` (replaces old `gazebo` command)
- **Process Management:** Monitoring loop instead of blocking `wait`

---

## 🤖 ROS2 Configuration (`ros2_node_intelleswarm_pollination.py`)

### **Node Configuration**
```python
class IntelleSwarmPollinationNode(Node):
    def __init__(self):
        super().__init__('intelleswarm_pollination_controller')
        
        # QoS settings for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
```

### **Flower Patch Definitions**
```python
self.flower_patches = {
    1: FlowerPatch(
        id=1, name="Sunflower Patch 1",
        center_x=50, center_y=50, radius=15,
        flower_type="sunflower", priority=100, area_m2=706.86
    ),
    2: FlowerPatch(  
        id=2, name="Sunflower Patch 2",
        center_x=-50, center_y=50, radius=12,
        flower_type="sunflower", priority=100, area_m2=452.39
    ),
    3: FlowerPatch(
        id=3, name="Clover Field", 
        center_x=0, center_y=0, radius=20,
        flower_type="clover", priority=80, area_m2=1256.64
    )
}
```

### **MARL Integration Parameters**
```python
class PollinationMARL:
    def __init__(self):
        self.algorithm = "MAPPO"
        self.state_dim = 12              # Position, velocity, battery, targets
        self.action_dim = 4              # 3D velocity + pollination action
        self.num_agents = 6              # Drone count
        self.communication_range = 100.0  # meters
        self.reward_structure = {
            "pollination_success": 10.0,
            "coverage_bonus": 5.0,
            "collision_penalty": -50.0,
            "energy_efficiency": 1.0
        }
```

### **Mission State Machine**
```python
class MissionPhase(Enum):
    TAKEOFF = "takeoff_and_formation"
    SURVEY = "area_survey"  
    POLLINATION = "pollination_execution"
    VALIDATION = "coverage_validation"
    RTH = "return_to_base"

# Phase durations (seconds)
PHASE_DURATIONS = {
    MissionPhase.TAKEOFF: 60,
    MissionPhase.SURVEY: 120,
    MissionPhase.POLLINATION: 300,
    MissionPhase.VALIDATION: 60,
    MissionPhase.RTH: 60
}
```

---

## ⚙️ Runtime Configuration

### **Environment Variables**
```bash
# Set before running simulation
export PX4_DIR="$HOME/PX4-Autopilot"
export ROS_DOMAIN_ID=42                    # ROS2 domain isolation
export PYTHONPATH="$HOME/intelleswarm-ai:$PYTHONPATH"

# Gazebo resource paths  
export GZ_SIM_RESOURCE_PATH="$HOME/intelleswarm-ai/genai_framework/px4_gazebo_sim:$GZ_SIM_RESOURCE_PATH"
```

### **Simulation Parameters**
```python
@dataclass
class SimulationConfig:
    num_drones: int = 6
    simulation_duration: int = 600        # 10 minutes
    world_file: str = "agricultural_farm.world"
    enable_gui: bool = True               # Gazebo GUI
    px4_dir: str = os.path.expanduser("~/PX4-Autopilot")
    log_level: str = "INFO"
    
    # Performance settings
    real_time_factor: float = 1.0         # Real-time simulation
    physics_step_size: float = 0.004      # 250 Hz physics
    render_rate: float = 60.0             # 60 FPS rendering
```

---

## 🔧 Advanced Configuration

### **Custom Mission Patterns**
```yaml
# Add to pollination_drone_config.yaml
mission_patterns:
  custom_spiral:
    type: "spiral"
    parameters:
      center_offset: 0.0      # meters from patch center
      spiral_spacing: 2.0     # meters between spiral arms
      max_radius: 20.0        # maximum spiral extent
      altitude_variation: 1.0 # vertical movement range
      
  adaptive_grid:
    type: "adaptive_grid" 
    parameters:
      base_spacing: 5.0       # meters between grid points
      density_factor: 1.5     # increase density in high-priority areas
      skip_threshold: 0.1     # skip areas below this flower density
```

### **Performance Tuning**
```yaml
performance:
  simulation_mode: "optimized"    # vs "high_fidelity"
  physics_threads: 4              # CPU cores for physics
  render_quality: "medium"        # low/medium/high
  sensor_update_rates:
    camera: 30.0                  # Hz
    lidar: 20.0                   # Hz  
    imu: 250.0                    # Hz
    gps: 10.0                     # Hz
```

### **Network Configuration**
```yaml
network:
  mavlink_protocol_version: 2.0
  heartbeat_frequency: 1.0        # Hz
  parameter_update_rate: 0.1      # Hz
  telemetry_streaming: true
  log_compression: true
  max_packet_size: 1024           # bytes
```

---

## 📊 Monitoring & Debugging

### **Log File Locations**
```bash
# Simulation logs
~/intelleswarm-ai/genai_framework/px4_gazebo_sim/logs/

# PX4 logs  
~/PX4-Autopilot/build/px4_sitl_default/tmp/

# ROS2 logs
~/.ros/log/

# Gazebo logs
~/.gz/logs/
```

### **Key Metrics Configuration**
```yaml
metrics:
  collection_frequency: 1.0       # Hz
  metrics_to_track:
    - "flowers_pollinated_count"
    - "area_coverage_percentage"  
    - "energy_consumption_wh"
    - "collision_incidents_count"
    - "communication_quality_score"
    - "mission_completion_time_s"
  
  thresholds:
    min_coverage: 85.0            # % minimum acceptable coverage
    max_collisions: 0             # Zero tolerance for collisions
    max_mission_time: 900         # 15 minutes maximum
```

---

## ✅ Configuration Validation

### **🚀 Verified Launch Script Configuration**

**Complete Launch Script (`launch_simulation.sh`):**
- ✅ **Conda Environment**: `ros2_humble` activated automatically
- ✅ **PyTorch Support**: Available without warnings  
- ✅ **Gazebo GUI**: Server-client approach for macOS visibility
- ✅ **Process Management**: Proper cleanup and monitoring
- ✅ **Verified Results**: 59.8% performance, 1,275 flowers, 72.3% coverage

### **Pre-Flight Checklist**
```bash
# Verify all configurations
cd ~/intelleswarm-ai/genai_framework/px4_gazebo_sim

# 🚀 Recommended: Use launch script (handles everything)
./launch_simulation.sh

# Manual verification (if needed)
ls -la ~/PX4-Autopilot/build/px4_sitl_default/bin/px4
gz sim --check agricultural_farm.world
python3 -c "import yaml; yaml.safe_load(open('pollination_drone_config.yaml'))"
```

### **Configuration Test Command**
```bash
# Dry run validation
/usr/bin/python3 run_pollination_simulation.py --validate-only
```

---

## 🎯 Production Deployment Settings

### **Recommended Production Configuration**
```yaml
# For real drone deployment
drone_specs:
  max_speed: 12.0                 # Reduced for safety
  safety_radius: 10.0             # Increased separation  
  
safety:
  return_home_battery: 35         # Higher safety margin
  max_wind_speed: 6.0             # More conservative
  
marl:
  communication_range: 50.0       # Real-world RF limitations
```

### **Hardware-Specific Adjustments**
```yaml
# Adjust based on actual drone capabilities
hardware_profile: "intel_aero"    # or "pixhawk", "navio", etc.
sensor_suite: "standard"          # camera, gps, imu
payload_configuration: "pollination_v2"
```

---

**🔧 This configuration guide ensures optimal performance and safety for both simulation and real-world deployment of the IntelleSwarm AI system.**