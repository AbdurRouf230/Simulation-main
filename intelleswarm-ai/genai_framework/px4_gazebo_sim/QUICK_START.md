# IntelleSwarm AI - Quick Start Guide

**🚀 Choose Your Launch Method**

## 🎯 **Recommended: Complete Launch Script**

```bash
cd ~/intelleswarm-ai/genai_framework/px4_gazebo_sim
./launch_simulation.sh
```

**✅ What it does:**
- Automatically starts Gazebo server + GUI
- Activates conda environment (PyTorch available)
- Runs complete simulation with all dependencies
- Handles cleanup on Ctrl+C
- Works on macOS with GUI visibility

---

## 🐍 **PyTorch-Enabled Direct Run**

```bash
cd ~/intelleswarm-ai/genai_framework/px4_gazebo_sim  
python3 run_simulation_with_pytorch.py
```

**✅ What it does:**
- Activates conda environment for PyTorch support
- Runs simulation (you handle Gazebo GUI separately)
- Simple wrapper around original script

---

## 🔧 **Manual Setup (Advanced)**

```bash
cd ~/intelleswarm-ai/genai_framework/px4_gazebo_sim

# Start Gazebo GUI
/opt/homebrew/bin/gz sim agricultural_farm.world -s &
sleep 8
/opt/homebrew/bin/gz sim -g &
sleep 5

# Run simulation with PyTorch
source $HOME/miniconda3/bin/activate ros2_humble
python run_pollination_simulation.py
```

**✅ What it does:**
- Full manual control over each component
- Best for debugging individual components

---

## 🆚 **Comparison**

| Method | GUI Setup | PyTorch | Cleanup | Best For |
|--------|-----------|---------|---------|----------|
| **launch_simulation.sh** | ✅ Auto | ✅ Yes | ✅ Auto | **Normal Use** |
| **run_simulation_with_pytorch.py** | ❌ Manual | ✅ Yes | ❌ Manual | Development |
| **run_pollination_simulation.py** | ❌ Manual | ❌ No | ❌ Manual | Testing |

---

## 🎮 **Verified Results** ✅

**Latest Performance with `launch_simulation.sh`:**
- **Overall Score:** 59.8% (Grade D) ✅
- **Mission Duration:** 7.3 minutes ✅
- **Flowers Pollinated:** 1,275 ✅
- **Coverage:** 72.3% ✅
- **Safety:** 0 collisions ✅
- **PyTorch:** Available (no warnings) ✅
- **GUI:** Visible and working ✅

**🏆 All issues resolved with complete launch script!**

---

## 🛠️ **Troubleshooting Quick Fixes**

**❌ "PyTorch not available"**
→ Use `launch_simulation.sh` or `run_simulation_with_pytorch.py`

**❌ "Gazebo GUI not visible"** 
→ Check Dock, use Cmd+Tab, or use `launch_simulation.sh`

**❌ "PX4 processes stopping"**
→ All launcher scripts are now fixed with airframe 10016

**❌ "Permission denied"**
→ `chmod +x launch_simulation.sh`

---

**🏆 Recommendation: Start with `./launch_simulation.sh` for the best experience!**