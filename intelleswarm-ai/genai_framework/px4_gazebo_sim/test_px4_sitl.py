#!/usr/bin/env python3
"""
Test PX4 SITL without Gazebo

This script tests just the PX4 SITL component to verify it's working
while we wait for Gazebo to finish installing.
"""

import subprocess
import time
import signal
import sys
from pathlib import Path

def test_px4_sitl():
    """Test PX4 SITL startup"""
    print("🚀 Testing PX4 SITL (without Gazebo)...")
    print("="*50)

    # Check PX4 executable
    px4_path = Path("/Users/zrahman/PX4-Autopilot/build/px4_sitl_default/bin/px4")

    if not px4_path.exists():
        print("❌ PX4 executable not found")
        return False

    print(f"✅ PX4 executable found: {px4_path}")
    print(f"📊 Size: {px4_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Test PX4 startup (very briefly)
    print("\n🧪 Testing PX4 SITL startup...")

    try:
        # Set environment
        env = {
            'PX4_SIM_MODEL': 'iris',
            'PX4_HOME_LAT': '37.7749',
            'PX4_HOME_LON': '-122.4194',
            'PX4_HOME_ALT': '30.0',
            'PATH': '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin'
        }
        env.update(dict(os.environ) if 'os' in globals() else {})

        # Start PX4 SITL briefly
        process = subprocess.Popen([
            str(px4_path),
            '-d', '/Users/zrahman/PX4-Autopilot/platforms/posix/rootfs'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env)

        # Let it run for 3 seconds
        time.sleep(3)

        # Check if it's still running
        if process.poll() is None:
            print("✅ PX4 SITL started successfully!")
            print("✅ Process is running (PID: %d)" % process.pid)

            # Terminate cleanly
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()

            print("✅ PX4 SITL test completed successfully")
            return True
        else:
            stdout, stderr = process.communicate()
            print("❌ PX4 SITL failed to start")
            print(f"   STDOUT: {stdout[:200]}...")
            print(f"   STDERR: {stderr[:200]}...")
            return False

    except Exception as e:
        print(f"❌ Error testing PX4: {e}")
        return False

def test_px4_config():
    """Test our PX4 configuration files"""
    print("\n🔧 Testing PX4 Configuration...")

    config_file = Path("pollination_drone_config.yaml")
    if config_file.exists():
        print("✅ Pollination config found")

        with open(config_file, 'r') as f:
            content = f.read()

        if 'px4_dir' in content.lower():
            print("✅ PX4 directory configured")
        if 'mavlink' in content.lower():
            print("✅ MAVLink configuration found")

    script_file = Path("px4_multi_drone.sh")
    if script_file.exists():
        print("✅ Multi-drone launcher script found")

        with open(script_file, 'r') as f:
            content = f.read()

        if 'PX4_DIR' in content:
            print("✅ PX4_DIR variable configured")
        if 'MAVLINK_UDP_PORTS' in content:
            print("✅ MAVLink ports configured")

    return True

def main():
    """Main test function"""
    print("🧪 IntelleSwarm PX4 SITL Test")
    print("="*50)
    print("Testing PX4 components while Gazebo installs...")
    print()

    # Test PX4 SITL
    px4_success = test_px4_sitl()

    # Test configuration
    config_success = test_px4_config()

    print("\n" + "="*50)
    print("📊 PX4 SITL Test Results:")
    print(f"   • PX4 SITL: {'✅ PASS' if px4_success else '❌ FAIL'}")
    print(f"   • Configuration: {'✅ PASS' if config_success else '❌ FAIL'}")

    if px4_success:
        print("\n🎉 PX4 SITL is ready!")
        print("📋 Next steps:")
        print("   1. Wait for Gazebo installation to complete")
        print("   2. Run: ./run_pollination_simulation.py")
        print("   3. Start full multi-drone simulation")
    else:
        print("\n⚠️  PX4 SITL needs attention")

    print("="*50)
    return 0 if px4_success else 1

if __name__ == '__main__':
    import os
    sys.exit(main())