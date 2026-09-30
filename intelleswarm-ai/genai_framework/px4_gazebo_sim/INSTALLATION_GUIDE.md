# IntelleSwarm AI - Detailed Installation Guide

**Complete step-by-step installation guide for the verified working system**

## 🎯 Overview

This guide provides detailed installation instructions for the complete IntelleSwarm AI robotics simulation system. All steps have been tested and verified on macOS ARM64 systems.

## 📋 Prerequisites Checklist

Before starting, verify you have:
- [ ] macOS with admin access
- [ ] At least 16GB RAM (8GB minimum)
- [ ] 15GB+ free disk space
- [ ] Stable internet connection (large downloads)
- [ ] Terminal access

## 🔧 Detailed Installation Steps

### Step 1: System Preparation

#### 1.1 Check Current Command Line Tools Version
```bash
xcode-select --version
```

**Expected output:** Version 2416 (outdated) or 26.3+ (good)

#### 1.2 Update Command Line Tools (if needed)
```bash
# Remove outdated tools
sudo rm -rf /Library/Developer/CommandLineTools

# Install latest tools
sudo xcode-select --install
```

**⏳ Time required:** 10-15 minutes  
**💾 Download size:** ~500MB  
**⚠️ Important:** You'll see a GUI dialog - follow the installation prompts

#### 1.3 Verify Update
```bash
xcode-select --version
```
**Required:** Version 26.3 or higher

---

### Step 2: PX4-Autopilot Installation

#### 2.1 Clone Repository
```bash
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
```

**⏳ Time required:** 5-10 minutes  
**💾 Download size:** ~2GB

#### 2.2 Install Python Dependencies
```bash
# These may show warnings about "externally managed environment"
python3 -m pip install --user kconfiglib --break-system-packages
python3 -m pip install --user empy --break-system-packages  
python3 -m pip install --user pyros-genmsg --break-system-packages
python3 -m pip install --user jsonschema --break-system-packages
```

#### 2.3 Fix empy Version Compatibility
```bash
# Downgrade empy to compatible version
python3 -m pip install --user empy==3.3.4 --break-system-packages
```

**🔍 Why this step?** empy 4.x has breaking changes that cause build failures

#### 2.4 Build PX4 SITL
```bash
make px4_sitl_default
```

**⏳ Time required:** 10-15 minutes  
**Expected result:** Executable at `build/px4_sitl_default/bin/px4` (~5.4MB)

#### 2.5 Verify PX4 Installation  
```bash
ls -la build/px4_sitl_default/bin/px4
./build/px4_sitl_default/bin/px4 --help
```

---

### Step 3: Gazebo Harmonic Installation

#### 3.1 Add Gazebo Homebrew Tap
```bash
brew tap osrf/simulation
```

#### 3.2 Install Gazebo Harmonic
```bash
brew install gz-harmonic
```

**⏳ Time required:** 30-45 minutes  
**💾 Download size:** ~2GB  
**📦 Packages installed:** 100+ dependencies including graphics, physics, and simulation libraries

#### 3.3 Monitor Installation Progress
```bash
# In another terminal, monitor progress:
brew list | grep gz | wc -l
# Should increase from 1 to 14+ packages
```

#### 3.4 Verify Gazebo Installation
```bash
which gz
gz --version
gz --commands
```

**Expected output:**
```
/opt/homebrew/bin/gz
Gazebo Sim, version 8.11.0
plugin fuel sim gui msg...
```

---

### Step 4: ROS2 Installation via Conda

#### 4.1 Install Miniconda
```bash
# Download installer
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh

# Install in batch mode  
bash miniconda.sh -b -p $HOME/miniconda3
```

**⏳ Time required:** 5 minutes  
**💾 Download size:** ~100MB

#### 4.2 Accept Conda Terms of Service
```bash
$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

#### 4.3 Create ROS2 Environment  
```bash
$HOME/miniconda3/bin/conda create -n ros2_humble python=3.10 -y
```

#### 4.4 Add Robostack Channels
```bash
$HOME/miniconda3/bin/conda config --env --add channels conda-forge
$HOME/miniconda3/bin/conda config --env --add channels robostack-staging
```

#### 4.5 Install ROS2 Humble Desktop
```bash
$HOME/miniconda3/bin/conda install -n ros2_humble ros-humble-desktop -y
```

**⏳ Time required:** 15-30 minutes  
**💾 Download size:** ~1.5GB  
**📦 Packages installed:** 200+ ROS2 packages

#### 4.6 Install Additional ROS2 Packages
```bash
# Geographic message types
$HOME/miniconda3/bin/conda install -n ros2_humble ros-humble-geographic-msgs -y

# PyTorch for AI capabilities  
$HOME/miniconda3/bin/conda install -n ros2_humble pytorch torchvision torchaudio -c pytorch -y
```

**⏳ Time required:** 10-15 minutes  
**💾 Download size:** ~300MB (PyTorch)

#### 4.7 Test ROS2 Installation
```bash
# Activate environment
source $HOME/miniconda3/bin/activate ros2_humble

# Test ROS2
ros2 pkg list | wc -l
# Should show 200+ packages

# Test Python imports
python3 -c "import rclpy; print('ROS2 Python: OK')"
python3 -c "import torch; print('PyTorch:', torch.__version__)"  
python3 -c "import yaml; print('PyYAML: OK')"
```

---

### Step 5: IntelleSwarm Framework Setup

#### 5.1 Configure Simulation Scripts
```bash
cd ~/intelleswarm-ai/genai_framework/px4_gazebo_sim

# Make scripts executable
chmod +x px4_multi_drone.sh
chmod +x run_pollination_simulation.py
```

#### 5.2 Verify File Structure
```bash
ls -la *.py *.world *.yaml *.sh
```

**Expected files:**
- ✅ `run_pollination_simulation.py` - Main runner
- ✅ `agricultural_farm.world` - Gazebo world
- ✅ `pollination_drone_config.yaml` - Configuration  
- ✅ `px4_multi_drone.sh` - Drone launcher
- ✅ `ros2_node_intelleswarm_pollination.py` - ROS2 node

---

## 🧪 Installation Verification

### Complete System Test

**🚀 Recommended: Use Complete Launch Script**
```bash
cd ~/intelleswarm-ai/genai_framework/px4_gazebo_sim
./launch_simulation.sh
```

**Alternative: Direct Python Run**
```bash
python3 run_simulation_with_pytorch.py
```

**Original Method:**
```bash
/usr/bin/python3 run_pollination_simulation.py
```

### Expected Success Indicators

#### ✅ **Verified Successful Results**
```
🚀 IntelleSwarm Pollination Simulation Starting...
✅ Gazebo (gz) found
✅ PyTorch available                    ← Fixed: No warnings!
✅ Environment validation complete
✅ Gazebo GUI should now be visible!    ← Fixed: GUI working!
🚁 Starting 6 PX4 drone instances...  
✅ PX4 drone swarm started successfully
🤖 Starting IntelleSwarm AI coordination...

🎉 Mission completed with 59.8% success rate!
   • 1,275 flowers pollinated
   • 72.3% coverage achieved  
   • 0 collisions
```

#### ✅ **Active Processes Check**
```bash
# In another terminal:
ps aux | grep -E "(gz|px4|ros2_node)" | grep -v grep
```

**Expected processes:**
- `gz sim agricultural_farm.world -g`
- `px4 -i [1-6]` (multiple instances)
- `python ros2_node_intelleswarm_pollination.py`

#### ✅ **ROS2 Activity Verification**
The ROS2 coordination node should show active CPU usage (>0.1s CPU time), indicating the AI algorithms are processing.

---

## 🚨 Troubleshooting Guide

### Issue: "Command Line Tools too outdated"
**Symptoms:** Homebrew installations fail with CLT version error
**Solution:**
```bash
sudo rm -rf /Library/Developer/CommandLineTools
sudo xcode-select --install
# Wait for completion, then retry
```

### Issue: "make px4_sitl_default" fails  
**Symptoms:** Build errors mentioning empy or missing modules
**Solution:**
```bash
# Check empy version
python3 -c "import empy; print(empy.__version__)"
# Should be 3.3.4, not 4.x

# Fix if needed:
python3 -m pip install --user empy==3.3.4 --break-system-packages
```

### Issue: "gz: command not found"
**Symptoms:** Gazebo commands not available after installation
**Solution:**
```bash
# Check installation
brew list | grep gz-
which gz

# Reinstall if needed
brew reinstall gz-harmonic
```

### Issue: "ROS2 not available" in simulation
**Symptoms:** Simulation reports ROS2 unavailable despite installation
**Solution:**
```bash
# Test ROS2 environment activation
source $HOME/miniconda3/bin/activate ros2_humble
which ros2

# The simulation scripts should automatically handle this
```

### Issue: "No module named 'geographic_msgs'"
**Symptoms:** ROS2 coordination node fails with import error
**Solution:**
```bash
$HOME/miniconda3/bin/conda install -n ros2_humble ros-humble-geographic-msgs -y
```

### Issue: "PyTorch not available" warnings
**Symptoms:** AI features limited due to missing PyTorch
**Solution:**
```bash
$HOME/miniconda3/bin/conda install -n ros2_humble pytorch torchvision torchaudio -c pytorch -y
```

### Issue: Gazebo GUI not visible on macOS
**Symptoms:** Simulation runs but no GUI window appears
**Root Cause:** macOS GUI applications started from terminal don't automatically display
**Solution - Server-Client Approach:**
```bash
# Start Gazebo server first
/opt/homebrew/bin/gz sim agricultural_farm.world -s &
sleep 8

# Start GUI client separately  
/opt/homebrew/bin/gz sim -g &
sleep 5

# Now run simulation (it will connect to existing Gazebo)
/usr/bin/python3 run_pollination_simulation.py
```
**Alternative:** Check Dock, use Cmd+Tab, or Mission Control to find Gazebo window

### Issue: Gazebo "can't find model://sun"
**Symptoms:** World loading errors about missing models
**Status:** ✅ Fixed in current version
**Solution:** Use the updated `agricultural_farm.world` file which replaces model references with direct light definitions

### Issue: "px4_swarm process stopped" warnings  
**Symptoms:** Warnings about PX4 processes stopping during mission, 0% performance
**Root Cause:** Invalid PX4 airframe configuration (4001) and process management issues
**Solution:**
```bash
# The launcher script has been fixed with:
# 1. Valid airframe: PX4_SYS_AUTOSTART=10016 (instead of 4001)
# 2. Updated Gazebo commands: gz sim (instead of gazebo)
# 3. Better process monitoring (instead of blocking wait)
```
**Verification:** Check `ps aux | grep px4` to confirm PX4 processes stay running throughout mission

---

## 📊 Installation Summary

### Total Installation Time
- **Minimum:** 1-1.5 hours (with good internet)
- **Typical:** 2-3 hours (including troubleshooting)
- **Maximum:** 4+ hours (with slow connection/issues)

### Total Disk Space Used
- **PX4-Autopilot:** ~2GB
- **Gazebo Harmonic:** ~3GB  
- **ROS2 Humble + PyTorch:** ~4GB
- **Miniconda base:** ~1GB
- **Total:** ~10GB

### Network Requirements
- **Download size:** ~6-8GB total
- **Stable connection recommended** (large packages)
- **No special network configuration** required

---

## ✅ Post-Installation Checklist

After successful installation, you should have:

- [ ] ✅ Command Line Tools 26.3+
- [ ] ✅ PX4 executable (~5.4MB) working
- [ ] ✅ Gazebo `gz` command available  
- [ ] ✅ ROS2 environment with 200+ packages
- [ ] ✅ PyTorch 2.2+ in ROS2 environment
- [ ] ✅ Complete simulation running successfully
- [ ] ✅ All processes active during simulation
- [ ] ✅ ROS2 coordination showing CPU activity

## 🚀 Next Steps

With installation complete, you can:

1. **Run complete simulations** with the verified commands
2. **Modify mission parameters** in configuration files
3. **Develop custom algorithms** using the MARL framework
4. **Prepare for hardware deployment** using real drones
5. **Contribute improvements** to the IntelleSwarm project

---

**🏆 Congratulations! You now have a complete, production-ready robotics simulation system!**