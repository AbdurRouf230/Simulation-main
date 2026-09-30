#!/usr/bin/env python3
"""
IntelleSwarm AI - Pollination Simulation Launch File

This launch file orchestrates the complete pollination simulation:
1. Gazebo with agricultural world
2. Multiple PX4 SITL instances
3. IntelleSwarm pollination coordination node
4. Mission monitoring and control

Author: IntelleSwarm AI Team
Date: May 2026
"""

import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    RegisterEventHandler, LogInfo, TimerAction
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def get_package_share_directory(package_name):
    """Get package share directory (simplified for direct execution)"""
    return os.path.join(os.getcwd(), package_name)


def generate_launch_description():
    """Generate the complete launch description for pollination simulation"""

    # Get current directory (where this script is located)
    current_dir = Path(__file__).parent.absolute()
    world_file = current_dir / "agricultural_farm.world"
    config_file = current_dir / "pollination_drone_config.yaml"

    # Launch arguments
    declare_num_drones = DeclareLaunchArgument(
        'num_drones',
        default_value='6',
        description='Number of drones in the swarm'
    )

    declare_world_file = DeclareLaunchArgument(
        'world_file',
        default_value=str(world_file),
        description='Path to Gazebo world file'
    )

    declare_headless = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode'
    )

    declare_px4_dir = DeclareLaunchArgument(
        'px4_dir',
        default_value=os.path.expanduser('~/PX4-Autopilot'),
        description='Path to PX4-Autopilot directory'
    )

    declare_enable_ai = DeclareLaunchArgument(
        'enable_ai',
        default_value='true',
        description='Enable IntelleSwarm AI coordination'
    )

    # Gazebo launch
    gazebo_launch = ExecuteProcess(
        cmd=[
            'gazebo',
            '--verbose',
            '-s', 'libgazebo_ros_factory.so',
            '-s', 'libgazebo_ros_state.so',
            LaunchConfiguration('world_file')
        ],
        output='screen',
        name='gazebo'
    )

    # PX4 SITL drone instances
    drone_launches = []
    mavlink_ports = [14560, 14570, 14580, 14590, 14600, 14610]
    spawn_positions = [
        [50, 50, 5, 0],      # Sunflower patch 1
        [-50, 50, 5, 1.57],  # Sunflower patch 2
        [0, 0, 5, 0],        # Clover field
        [25, 25, 5, 0.79],   # Support drone 1
        [-25, 25, 5, 2.36],  # Support drone 2
        [0, 75, 5, 1.57],    # Support drone 3
    ]

    for i in range(6):  # 6 drones
        drone_id = i + 1
        mavlink_port = mavlink_ports[i]
        spawn_x, spawn_y, spawn_z, spawn_yaw = spawn_positions[i]

        # PX4 SITL instance
        px4_launch = ExecuteProcess(
            cmd=[
                'python3',
                str(current_dir / 'launch_px4_drone.py'),
                '--drone_id', str(drone_id),
                '--spawn_x', str(spawn_x),
                '--spawn_y', str(spawn_y),
                '--spawn_z', str(spawn_z),
                '--spawn_yaw', str(spawn_yaw),
                '--mavlink_port', str(mavlink_port),
                '--px4_dir', LaunchConfiguration('px4_dir')
            ],
            output='screen',
            name=f'px4_drone_{drone_id}'
        )

        # Add delay between drone launches
        delayed_px4_launch = TimerAction(
            period=i * 5.0,  # 5 second intervals
            actions=[px4_launch]
        )

        drone_launches.append(delayed_px4_launch)

    # IntelleSwarm Pollination Coordination Node
    intelleswarm_node = Node(
        package='intelleswarm_pollination',  # Would be package name
        executable='ros2_node_intelleswarm_pollination.py',
        name='intelleswarm_pollination_controller',
        parameters=[str(config_file)],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_ai'))
    )

    # Alternative: Direct execution for development
    intelleswarm_process = ExecuteProcess(
        cmd=[
            'python3',
            str(current_dir / 'ros2_node_intelleswarm_pollination.py')
        ],
        output='screen',
        name='intelleswarm_pollination',
        condition=IfCondition(LaunchConfiguration('enable_ai'))
    )

    # Mission Control Interface
    mission_control = Node(
        package='intelleswarm_pollination',
        executable='mission_control_interface.py',
        name='mission_control',
        output='screen'
    )

    # Performance Monitor
    performance_monitor = Node(
        package='intelleswarm_pollination',
        executable='performance_monitor.py',
        name='performance_monitor',
        output='screen'
    )

    # Log startup message
    startup_log = LogInfo(
        msg="🌻 IntelleSwarm Pollination Simulation Starting..."
    )

    # Delayed startup of AI coordination (wait for drones to initialize)
    delayed_ai_start = TimerAction(
        period=30.0,  # Wait 30 seconds for all drones to be ready
        actions=[
            LogInfo(msg="🤖 Starting IntelleSwarm AI Coordination..."),
            intelleswarm_process
        ]
    )

    # Build launch description
    ld = LaunchDescription([
        # Arguments
        declare_num_drones,
        declare_world_file,
        declare_headless,
        declare_px4_dir,
        declare_enable_ai,

        # Startup log
        startup_log,

        # Gazebo (starts first)
        gazebo_launch,

        # Delayed drone launches
        *drone_launches,

        # Delayed AI coordination start
        delayed_ai_start,

        # Mission monitoring (optional)
        # mission_control,
        # performance_monitor,
    ])

    return ld


if __name__ == '__main__':
    """Allow direct execution for testing"""
    from launch.launch_service import LaunchService

    # Simple direct execution
    print("🚀 IntelleSwarm Pollination Simulation")
    print("=====================================")
    print("This would normally be launched with:")
    print("ros2 launch intelleswarm_pollination launch_pollination_sim.py")
    print("")
    print("For direct testing, run components manually:")
    print("1. ./px4_multi_drone.sh")
    print("2. python3 ros2_node_intelleswarm_pollination.py")