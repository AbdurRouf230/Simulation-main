#!/usr/bin/env python3
"""
IntelleSwarm Mission Control Interface

Simple command-line interface for controlling the pollination simulation.
Provides easy commands for mission management and monitoring.

Author: IntelleSwarm AI Team
Date: May 2026
"""

import time
import threading
from typing import Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray


class MissionControlInterface(Node):
    """Interactive mission control interface"""

    def __init__(self):
        super().__init__("mission_control_interface")

        self.get_logger().info("🎮 IntelleSwarm Mission Control Interface")

        # Publishers for mission control
        self.mission_cmd_pub = self.create_publisher(
            String, "/mission/control", 10
        )

        # Subscribers for status monitoring
        self.mission_status_sub = self.create_subscription(
            String, "/mission/status", self._mission_status_callback, 10
        )

        self.flower_map_sub = self.create_subscription(
            Float32MultiArray, "/mission/flower_map", self._flower_map_callback, 10
        )

        # Mission state tracking
        self.mission_status = "Unknown"
        self.flower_patches = []
        self.performance_metrics = {}

        # Start interactive interface in separate thread
        self.interface_thread = threading.Thread(target=self._run_interface, daemon=True)
        self.interface_thread.start()

    def _mission_status_callback(self, msg: String):
        """Handle mission status updates"""
        self.mission_status = msg.data

        # Parse status message if it contains metrics
        if "|" in msg.data:
            parts = msg.data.split("|")
            for part in parts[1:]:  # Skip first part (status)
                if ":" in part:
                    key, value = part.split(":")
                    try:
                        self.performance_metrics[key.lower()] = float(value)
                    except ValueError:
                        pass

    def _flower_map_callback(self, msg: Float32MultiArray):
        """Handle flower map updates"""
        data = msg.data
        self.flower_patches = []

        # Parse flower patch data (5 values per patch)
        for i in range(0, len(data), 5):
            if i + 4 < len(data):
                patch = {
                    'x': data[i],
                    'y': data[i + 1],
                    'radius': data[i + 2],
                    'priority': data[i + 3],
                    'status': data[i + 4]
                }
                self.flower_patches.append(patch)

    def send_mission_command(self, command: str):
        """Send mission command"""
        msg = String()
        msg.data = command
        self.mission_cmd_pub.publish(msg)
        self.get_logger().info(f"📡 Sent command: {command}")

    def print_status(self):
        """Print current mission status"""
        print("\n" + "="*60)
        print("🌻 INTELLESWARM POLLINATION MISSION STATUS")
        print("="*60)
        print(f"Mission Status: {self.mission_status}")

        if self.performance_metrics:
            print("\n📊 Performance Metrics:")
            for key, value in self.performance_metrics.items():
                print(f"  • {key.title()}: {value:.2%}")

        if self.flower_patches:
            print(f"\n🌸 Flower Patches ({len(self.flower_patches)}):")
            for i, patch in enumerate(self.flower_patches, 1):
                status_bar = "█" * int(patch['status'] * 20)
                status_empty = "░" * (20 - int(patch['status'] * 20))
                print(f"  {i}. ({patch['x']:6.1f}, {patch['y']:6.1f}) "
                      f"[{status_bar}{status_empty}] {patch['status']:.1%}")

        print("="*60)

    def _run_interface(self):
        """Run the interactive command interface"""
        print("\n🎮 IntelleSwarm Mission Control Interface")
        print("=========================================")
        print("Available commands:")
        print("  start    - Start pollination mission")
        print("  pause    - Pause current mission")
        print("  resume   - Resume paused mission")
        print("  abort    - Abort mission and return home")
        print("  status   - Show mission status")
        print("  help     - Show this help message")
        print("  quit     - Exit mission control")
        print("=========================================")

        while rclpy.ok():
            try:
                command = input("\n🎯 Command: ").strip().lower()

                if command == "start":
                    self.send_mission_command("start_pollination")

                elif command == "pause":
                    self.send_mission_command("pause_mission")

                elif command == "resume":
                    self.send_mission_command("resume_mission")

                elif command == "abort":
                    confirm = input("⚠️  Are you sure you want to abort? (y/N): ").strip().lower()
                    if confirm == 'y':
                        self.send_mission_command("abort_mission")
                    else:
                        print("❌ Abort cancelled")

                elif command == "status":
                    self.print_status()

                elif command == "help":
                    print("\n📖 Available commands:")
                    print("  start    - Start pollination mission")
                    print("  pause    - Pause current mission")
                    print("  resume   - Resume paused mission")
                    print("  abort    - Abort mission and return home")
                    print("  status   - Show mission status")
                    print("  help     - Show this help message")
                    print("  quit     - Exit mission control")

                elif command == "quit" or command == "exit":
                    print("👋 Goodbye!")
                    break

                elif command == "":
                    # Empty command, just continue
                    continue

                else:
                    print(f"❓ Unknown command: {command}")
                    print("   Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n\n👋 Mission Control shutting down...")
                break
            except EOFError:
                print("\n\n👋 Mission Control shutting down...")
                break


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    try:
        mission_control = MissionControlInterface()

        # Show initial status
        time.sleep(2)
        mission_control.print_status()

        # Spin the node
        rclpy.spin(mission_control)

    except KeyboardInterrupt:
        print("\n🛑 Mission Control interrupted")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()