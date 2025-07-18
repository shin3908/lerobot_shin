#!/usr/bin/env python3

"""Test script for inference with observation state filtering."""

import sys
import numpy as np
import torch

# Mock robot observation data
mock_observation = {
    "joint1.pos": 1.0,
    "joint1.vel": 2.0,
    "joint2.pos": 3.0,
    "joint2.vel": 4.0,
    "joint3.pos": 5.0,
    "joint3.vel": 6.0,
    "front": torch.zeros((3, 480, 640), dtype=torch.uint8),
    "top": torch.zeros((3, 480, 640), dtype=torch.uint8),
}

# Mock robot features (original, before filtering)
mock_robot_motor_features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["joint1.pos", "joint1.vel", "joint2.pos", "joint2.vel", "joint3.pos", "joint3.vel"]
    }
}

# Mock dataset features (after filtering to position only)
mock_dataset_features = {
    "observation.state": {
        "dtype": "float32", 
        "shape": (3,),  # Only position features
        "names": ["joint1.pos", "joint2.pos", "joint3.pos"]
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"]
    },
    "observation.images.top": {
        "dtype": "video", 
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"]
    },
    "action": {
        "dtype": "float32",
        "shape": (3,),
        "names": None
    }
}

def test_inference_filtering():
    """Test inference with observation state filtering."""
    
    # Test imports
    try:
        from lerobot.common.datasets.utils import (
            build_dataset_frame, 
            apply_observation_state_filter_to_frame
        )
        print("✓ Successfully imported required functions")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test 1: build_dataset_frame without filtering
    print("\n=== Test 1: build_dataset_frame without filtering ===")
    try:
        observation_frame = build_dataset_frame(
            mock_dataset_features, mock_observation, prefix="observation"
        )
        print(f"Observation frame keys: {list(observation_frame.keys())}")
        print(f"observation.state shape: {observation_frame['observation.state'].shape}")
        print(f"observation.state: {observation_frame['observation.state']}")
        print("✓ build_dataset_frame works without filtering")
    except Exception as e:
        print(f"✗ build_dataset_frame failed: {e}")
        return False
    
    # Test 2: apply_observation_state_filter_to_frame
    print("\n=== Test 2: apply_observation_state_filter_to_frame ===")
    try:
        # Create observation frame with all state values (full unfiltered state)
        full_observation_frame = {
            "observation.state": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            "observation.images.front": np.zeros((3, 480, 640), dtype=np.uint8),
            "observation.images.top": np.zeros((3, 480, 640), dtype=np.uint8),
        }
        
        original_state_names = ["joint1.pos", "joint1.vel", "joint2.pos", "joint2.vel", "joint3.pos", "joint3.vel"]
        observation_state_filter = [".pos"]
        
        filtered_frame = apply_observation_state_filter_to_frame(
            full_observation_frame.copy(),  # Make a copy to avoid modifying original
            original_state_names, 
            observation_state_filter
        )
        
        print(f"Original state shape: {full_observation_frame['observation.state'].shape}")
        print(f"Original state: {full_observation_frame['observation.state']}")
        print(f"Filtered state shape: {filtered_frame['observation.state'].shape}")
        print(f"Filtered state: {filtered_frame['observation.state']}")
        
        # Verify filtering worked correctly (should keep positions: indices 0, 2, 4)
        expected = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        if np.array_equal(filtered_frame['observation.state'], expected):
            print("✓ apply_observation_state_filter_to_frame works correctly")
        else:
            print(f"✗ Filtering result incorrect. Expected: {expected}, Got: {filtered_frame['observation.state']}")
            return False
            
    except Exception as e:
        print(f"✗ apply_observation_state_filter_to_frame failed: {e}")
        return False
    
    # Test 3: No filtering case
    print("\n=== Test 3: No filtering case ===")
    try:
        filtered_frame_none = apply_observation_state_filter_to_frame(
            full_observation_frame.copy(),
            original_state_names,
            None  # No filtering
        )
        
        if np.array_equal(filtered_frame_none['observation.state'], full_observation_frame['observation.state']):
            print("✓ No filtering case works correctly")
        else:
            print("✗ No filtering case failed")
            return False
            
    except Exception as e:
        print(f"✗ No filtering case failed: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    return True

if __name__ == "__main__":
    success = test_inference_filtering()
    sys.exit(0 if success else 1)
