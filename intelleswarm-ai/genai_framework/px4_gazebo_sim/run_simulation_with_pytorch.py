#!/usr/bin/env python3
"""
IntelleSwarm AI - PyTorch-Enabled Simulation Wrapper

This wrapper ensures PyTorch is available by activating the conda environment
before running the simulation. Use this instead of run_pollination_simulation.py
when running from system terminal.

Usage:
    python3 run_simulation_with_pytorch.py

Author: IntelleSwarm AI Team
Date: May 2026
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run simulation with PyTorch support via conda environment."""

    print("🐍 IntelleSwarm AI - PyTorch-Enabled Simulation Wrapper")
    print("=" * 60)

    # Configuration
    script_dir = Path(__file__).parent
    conda_path = Path.home() / "miniconda3"
    conda_env = "ros2_humble"
    simulation_script = script_dir / "run_pollination_simulation.py"

    # Check if conda environment exists
    env_path = conda_path / "envs" / conda_env
    if not env_path.exists():
        print(f"❌ Error: Conda environment '{conda_env}' not found")
        print(f"   Expected path: {env_path}")
        print("   Please run the installation guide to set up ROS2 environment")
        sys.exit(1)

    # Check if simulation script exists
    if not simulation_script.exists():
        print(f"❌ Error: Simulation script not found")
        print(f"   Expected: {simulation_script}")
        sys.exit(1)

    print(f"✅ Found conda environment: {env_path}")
    print(f"✅ Found simulation script: {simulation_script}")
    print()
    print("🚀 Running simulation with PyTorch support...")
    print()

    # Create command to activate conda and run simulation
    activation_script = conda_path / "bin" / "activate"

    # Build command
    cmd = [
        "bash", "-c",
        f"source {activation_script} {conda_env} && cd {script_dir} && python {simulation_script}"
    ]

    # Run simulation with conda environment
    try:
        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            print("\n🎉 Simulation completed successfully!")
        else:
            print(f"\n⚠️ Simulation exited with code: {result.returncode}")

        return result.returncode

    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running simulation: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)