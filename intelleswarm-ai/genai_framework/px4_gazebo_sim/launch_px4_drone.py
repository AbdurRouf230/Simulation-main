#!/usr/bin/env python3
"""
Individual PX4 Drone Instance Launcher

This script launches a single PX4 SITL instance with specific parameters
for the pollination simulation. Called by the main launch file.

Author: IntelleSwarm AI Team
Date: May 2026
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path


def launch_px4_drone(drone_id, spawn_x, spawn_y, spawn_z, spawn_yaw,
                    mavlink_port, px4_dir):
    """Launch a single PX4 SITL drone instance"""

    print(f"🚁 Launching Drone {drone_id}")
    print(f"   Position: ({spawn_x}, {spawn_y}, {spawn_z}) @ {spawn_yaw} rad")
    print(f"   MAVLink Port: {mavlink_port}")
    print(f"   PX4 Directory: {px4_dir}")

    # Validate PX4 directory
    px4_path = Path(px4_dir)
    if not px4_path.exists():
        print(f"❌ Error: PX4 directory not found: {px4_dir}")
        return False

    px4_executable = px4_path / "build" / "px4_sitl_default" / "bin" / "px4"
    if not px4_executable.exists():
        print(f"❌ Error: PX4 executable not found: {px4_executable}")
        print("   Please build PX4-Autopilot first: make px4_sitl gazebo-classic")
        return False

    # Set environment variables for this drone instance
    env = os.environ.copy()
    env.update({
        'PX4_SIM_MODEL': 'iris',
        'PX4_SIM_HOSTNAME': 'localhost',
        'PX4_SIMULATOR': 'gazebo-classic',
        'PX4_HOME_LAT': '37.7749',  # San Francisco coordinates
        'PX4_HOME_LON': '-122.4194',
        'PX4_HOME_ALT': '30.0',
        'PX4_SYS_AUTOSTART': '4001',  # Generic quadrotor
    })

    # Calculate port offsets
    tcp_port = 4560 + (drone_id - 1) * 10
    gazebo_tcp_port = 11345 + (drone_id - 1) * 10

    # PX4 startup arguments
    px4_args = [
        str(px4_executable),
        '-i', str(drone_id),
        '-s', str(px4_path / "ROMFS" / "px4fmu_common" / "init.d-posix" / "rcS"),
        '-t', str(px4_path / "test_data"),
        '-d', str(px4_path / "platforms" / "posix" / "rootfs"),
    ]

    print(f"   Starting PX4 SITL with args: {' '.join(px4_args)}")

    try:
        # Launch PX4 SITL
        process = subprocess.Popen(
            px4_args,
            env=env,
            cwd=px4_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"   ✅ Drone {drone_id} PX4 SITL started (PID: {process.pid})")

        # Monitor the process briefly to ensure it starts correctly
        time.sleep(3)

        if process.poll() is None:
            print(f"   🎯 Drone {drone_id} is running successfully")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"   ❌ Drone {drone_id} failed to start")
            print(f"   STDOUT: {stdout}")
            print(f"   STDERR: {stderr}")
            return False

    except Exception as e:
        print(f"   ❌ Error launching Drone {drone_id}: {e}")
        return False


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Launch individual PX4 drone for pollination simulation")

    parser.add_argument('--drone_id', type=int, required=True, help='Drone ID (1-6)')
    parser.add_argument('--spawn_x', type=float, required=True, help='Spawn X position')
    parser.add_argument('--spawn_y', type=float, required=True, help='Spawn Y position')
    parser.add_argument('--spawn_z', type=float, required=True, help='Spawn Z position')
    parser.add_argument('--spawn_yaw', type=float, required=True, help='Spawn yaw rotation (radians)')
    parser.add_argument('--mavlink_port', type=int, required=True, help='MAVLink UDP port')
    parser.add_argument('--px4_dir', type=str, required=True, help='Path to PX4-Autopilot directory')

    args = parser.parse_args()

    # Validate arguments
    if not (1 <= args.drone_id <= 10):
        print("❌ Error: drone_id must be between 1 and 10")
        return 1

    if not (14560 <= args.mavlink_port <= 14900):
        print("❌ Error: mavlink_port should be in range 14560-14900")
        return 1

    # Launch the drone
    success = launch_px4_drone(
        args.drone_id, args.spawn_x, args.spawn_y, args.spawn_z,
        args.spawn_yaw, args.mavlink_port, args.px4_dir
    )

    if success:
        print(f"🎉 Drone {args.drone_id} launched successfully")

        # Keep the process alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Shutting down Drone {args.drone_id}")
            return 0
    else:
        print(f"💥 Failed to launch Drone {args.drone_id}")
        return 1


if __name__ == '__main__':
    sys.exit(main())