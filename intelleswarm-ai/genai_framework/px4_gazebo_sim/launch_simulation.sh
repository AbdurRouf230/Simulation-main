#!/bin/bash
##############################################################################
# IntelleSwarm AI - Complete Simulation Launcher
#
# This script provides a complete launch solution for the pollination simulation
# with Gazebo GUI and PyTorch support.
#
# Features:
# - Activates conda environment for PyTorch support
# - Starts Gazebo server and GUI client for macOS compatibility
# - Runs simulation with all dependencies available
# - Handles cleanup and process management
#
# Author: IntelleSwarm AI Team
# Date: May 2026
##############################################################################

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONDA_ENV_NAME="ros2_humble"
CONDA_PATH="$HOME/miniconda3"
SIMULATION_SCRIPT="run_pollination_simulation.py"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Process tracking
GAZEBO_SERVER_PID=""
GAZEBO_CLIENT_PID=""
SIMULATION_PID=""

echo -e "${BLUE}🚀 IntelleSwarm AI - Complete Simulation Launcher${NC}"
echo "============================================================"
echo "📍 Directory: $SCRIPT_DIR"
echo "🐍 Conda Environment: $CONDA_ENV_NAME"
echo "🌍 Gazebo GUI: Enabled"
echo "🧠 PyTorch Support: Enabled"
echo "============================================================"

# Cleanup function
cleanup() {
    trap - SIGINT SIGTERM
    echo -e "\n${YELLOW}🛑 Shutting down simulation...${NC}"

    # Stop simulation
    if [ ! -z "$SIMULATION_PID" ]; then
        echo "   Stopping simulation..."
        kill $SIMULATION_PID 2>/dev/null || true
    fi

    # Stop all simulation processes
    pkill -f "run_pollination_simulation" 2>/dev/null || true
    pkill -f "ros2_node_intelleswarm" 2>/dev/null || true
    pkill -f "px4_multi_drone" 2>/dev/null || true
    pkill -x px4 2>/dev/null || true

    # Stop Gazebo
    if [ ! -z "$GAZEBO_CLIENT_PID" ]; then
        echo "   Stopping Gazebo GUI..."
        kill $GAZEBO_CLIENT_PID 2>/dev/null || true
    fi

    if [ ! -z "$GAZEBO_SERVER_PID" ]; then
        echo "   Stopping Gazebo server..."
        kill $GAZEBO_SERVER_PID 2>/dev/null || true
    fi

    pkill -x gz 2>/dev/null || true

    echo -e "${GREEN}✅ Cleanup complete${NC}"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Function to check if conda environment exists
check_conda_env() {
    if [ ! -d "$CONDA_PATH/envs/$CONDA_ENV_NAME" ]; then
        echo -e "${RED}❌ Error: Conda environment '$CONDA_ENV_NAME' not found${NC}"
        echo "   Expected path: $CONDA_PATH/envs/$CONDA_ENV_NAME"
        echo "   Please run the installation guide to set up ROS2 environment"
        exit 1
    fi
}

# Function to check if simulation files exist
check_simulation_files() {
    if [ ! -f "$SCRIPT_DIR/$SIMULATION_SCRIPT" ]; then
        echo -e "${RED}❌ Error: Simulation script not found${NC}"
        echo "   Expected: $SCRIPT_DIR/$SIMULATION_SCRIPT"
        exit 1
    fi

    if [ ! -f "$SCRIPT_DIR/agricultural_farm.world" ]; then
        echo -e "${RED}❌ Error: Gazebo world file not found${NC}"
        echo "   Expected: $SCRIPT_DIR/agricultural_farm.world"
        exit 1
    fi
}

# Validation
echo -e "${BLUE}🔍 Validating environment...${NC}"

check_conda_env
check_simulation_files

# Check Gazebo installation
if ! command -v gz &> /dev/null; then
    echo -e "${RED}❌ Error: Gazebo (gz) not found${NC}"
    echo "   Please install Gazebo Harmonic: brew install gz-harmonic"
    exit 1
fi

# Check PX4 installation
PX4_DIR=${PX4_DIR:-"$HOME/PX4-Autopilot"}
if [ ! -f "$PX4_DIR/build/px4_sitl_default/bin/px4" ]; then
    echo -e "${RED}❌ Error: PX4 executable not found${NC}"
    echo "   Expected: $PX4_DIR/build/px4_sitl_default/bin/px4"
    echo "   Please build PX4: cd $PX4_DIR && make px4_sitl_default"
    exit 1
fi

echo -e "${GREEN}✅ Environment validation complete${NC}"

# Clean up any existing processes
echo -e "${BLUE}🧹 Cleaning up existing processes...${NC}"
pkill -x gz 2>/dev/null || true
pkill -x px4 2>/dev/null || true
pkill -f "ros2_node_intelleswarm" 2>/dev/null || true
sleep 2

# Start Gazebo server
echo -e "${BLUE}🌍 Starting Gazebo server...${NC}"
cd "$SCRIPT_DIR"
/opt/homebrew/bin/gz sim agricultural_farm.world -s &
GAZEBO_SERVER_PID=$!
echo "   Server PID: $GAZEBO_SERVER_PID"

# Wait for server to initialize
echo "⏳ Waiting for Gazebo server to initialize..."
sleep 8

# Start Gazebo GUI client
echo -e "${BLUE}🖥️ Starting Gazebo GUI client...${NC}"
/opt/homebrew/bin/gz sim -g &
GAZEBO_CLIENT_PID=$!
echo "   GUI PID: $GAZEBO_CLIENT_PID"

# Wait for GUI to initialize
echo "⏳ Waiting for Gazebo GUI to initialize..."
sleep 5

echo -e "${GREEN}✅ Gazebo GUI should now be visible!${NC}"
echo "   Check your dock or use Cmd+Tab to find the Gazebo window"

# Activate conda environment and run simulation
echo -e "${BLUE}🤖 Starting simulation with PyTorch support...${NC}"

# Create activation script
TEMP_SCRIPT="/tmp/intelleswarm_simulation.sh"
cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
source $CONDA_PATH/bin/activate $CONDA_ENV_NAME
cd "$SCRIPT_DIR"
python $SIMULATION_SCRIPT
EOF

chmod +x "$TEMP_SCRIPT"

# Run simulation with conda environment
bash "$TEMP_SCRIPT" &
SIMULATION_PID=$!
echo "   Simulation PID: $SIMULATION_PID"

echo ""
echo -e "${GREEN}🎉 Simulation launched successfully!${NC}"
echo ""
echo "📊 You should see:"
echo "   ✅ Gazebo GUI with agricultural farm"
echo "   ✅ 6 drones appearing and taking off"
echo "   ✅ PyTorch available (no warnings)"
echo "   ✅ Mission progress through 5 phases"
echo ""
echo "🛑 Press Ctrl+C to stop the simulation"
echo ""

# Wait for simulation to complete
wait $SIMULATION_PID
SIMULATION_EXIT_CODE=$?

# Clean up temp script
rm -f "$TEMP_SCRIPT"

if [ $SIMULATION_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}🎉 Simulation completed successfully!${NC}"
else
    echo -e "${YELLOW}⚠️ Simulation exited with code: $SIMULATION_EXIT_CODE${NC}"
fi

# Cleanup
cleanup