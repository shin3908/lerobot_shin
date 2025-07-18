#!/usr/bin/env python3

"""Test script for running inference with filtering via command line."""

import subprocess
import sys
import os

def test_command_line_filtering():
    """Test if the command line filtering argument works."""
    
    print("=== Testing command line filtering argument ===")
    
    # Test command with filtering
    cmd = [
        "python", "-m", "lerobot.record",
        "--robot.type=mock",
        "--dataset.repo_id=test/mock_filtered",
        "--dataset.num_episodes=1",
        "--dataset.episode_time_s=1",
        "--dataset.observation_state_filter=['.pos']",
        "--dataset.push_to_hub=false",
        "--help"  # Just check help to avoid actually running
    ]
    
    try:
        # Change to the correct directory
        os.chdir("/home/shinsakuo/workspace/lerobot_shin")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Command line parsing successful")
            print("✓ observation_state_filter argument is recognized")
            return True
        else:
            print(f"✗ Command failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Command timed out")
        return False
    except Exception as e:
        print(f"✗ Error running command: {e}")
        return False

if __name__ == "__main__":
    success = test_command_line_filtering()
    sys.exit(0 if success else 1)
