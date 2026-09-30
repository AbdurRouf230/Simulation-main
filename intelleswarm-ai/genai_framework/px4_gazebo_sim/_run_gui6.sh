#!/bin/bash
set -e
cd "/mnt/e/Multi Drone/Merge folder/intelleswarm-ai/genai_framework/px4_gazebo_sim"
source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
export DISPLAY=:0
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
python3 run_pollination_simulation.py --world arg_fruits_tree.sdf --num-drones 6 --duration 200 --px4-dir "$HOME/PX4-Main/PX4-Autopilot"
