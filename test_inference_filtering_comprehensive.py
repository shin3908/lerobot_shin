#!/usr/bin/env python3

"""Comprehensive test script for inference filtering functionality."""

import sys
import numpy as np
import torch

def test_inference_filtering_comprehensive():
    """Comprehensive test of inference filtering functionality."""
    
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
    
    # Test basic functionality
    print("\n=== Test 1: Basic filtering functionality ===")
    try:
        observation_frame = {
            "observation.state": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            "observation.images.front": np.zeros((3, 480, 640), dtype=np.uint8),
        }
        
        original_state_names = ["joint1.pos", "joint1.vel", "joint2.pos", "joint2.vel", "joint3.pos", "joint3.vel"]
        observation_state_filter = [".pos"]
        
        filtered_frame = apply_observation_state_filter_to_frame(
            observation_frame.copy(), original_state_names, observation_state_filter
        )
        
        expected = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        if np.array_equal(filtered_frame['observation.state'], expected):
            print("✓ Basic filtering works correctly")
        else:
            print(f"✗ Basic filtering failed. Expected: {expected}, Got: {filtered_frame['observation.state']}")
            return False
            
    except Exception as e:
        print(f"✗ Basic filtering failed: {e}")
        return False
    
    # Test multiple filters
    print("\n=== Test 2: Multiple filter suffixes ===")
    try:
        observation_frame = {
            "observation.state": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        }
        
        original_state_names = ["joint1.pos", "joint1.vel", "joint2.pos", "joint2.vel", "joint3.pos", "joint3.vel"]
        observation_state_filter = [".pos", ".vel"]
        
        filtered_frame = apply_observation_state_filter_to_frame(
            observation_frame.copy(), original_state_names, observation_state_filter
        )
        
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        if np.array_equal(filtered_frame['observation.state'], expected):
            print("✓ Multiple filter suffixes work correctly")
        else:
            print(f"✗ Multiple filter suffixes failed. Expected: {expected}, Got: {filtered_frame['observation.state']}")
            return False
            
    except Exception as e:
        print(f"✗ Multiple filter suffixes failed: {e}")
        return False
    
    # Test no matching features error
    print("\n=== Test 3: No matching features error handling ===")
    try:
        observation_frame = {
            "observation.state": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        }
        
        original_state_names = ["joint1.pos", "joint1.vel", "joint2.pos", "joint2.vel", "joint3.pos", "joint3.vel"]
        observation_state_filter = [".nonexistent"]
        
        try:
            filtered_frame = apply_observation_state_filter_to_frame(
                observation_frame.copy(), original_state_names, observation_state_filter
            )
            print("✗ Should have raised ValueError for no matching features")
            return False
        except ValueError as e:
            if "No features match the observation_state_filter" in str(e):
                print("✓ No matching features error handled correctly")
            else:
                print(f"✗ Wrong error message: {e}")
                return False
            
    except Exception as e:
        print(f"✗ No matching features error handling failed: {e}")
        return False
    
    # Test dimension mismatch error
    print("\n=== Test 4: Dimension mismatch error handling ===")
    try:
        observation_frame = {
            "observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float32),  # Wrong size
        }
        
        original_state_names = ["joint1.pos", "joint1.vel", "joint2.pos", "joint2.vel", "joint3.pos", "joint3.vel"]
        observation_state_filter = [".pos"]
        
        try:
            filtered_frame = apply_observation_state_filter_to_frame(
                observation_frame.copy(), original_state_names, observation_state_filter
            )
            print("✗ Should have raised ValueError for dimension mismatch")
            return False
        except ValueError as e:
            if "Mismatch between original state names" in str(e):
                print("✓ Dimension mismatch error handled correctly")
            else:
                print(f"✗ Wrong error message: {e}")
                return False
            
    except Exception as e:
        print(f"✗ Dimension mismatch error handling failed: {e}")
        return False
    
    # Test no observation.state key
    print("\n=== Test 5: No observation.state key ===")
    try:
        observation_frame = {
            "observation.images.front": np.zeros((3, 480, 640), dtype=np.uint8),
        }
        
        original_state_names = ["joint1.pos", "joint1.vel"]
        observation_state_filter = [".pos"]
        
        filtered_frame = apply_observation_state_filter_to_frame(
            observation_frame.copy(), original_state_names, observation_state_filter
        )
        
        # Should return unchanged frame
        if "observation.state" not in filtered_frame:
            print("✓ No observation.state key handled correctly")
        else:
            print("✗ Should not have added observation.state key")
            return False
            
    except Exception as e:
        print(f"✗ No observation.state key handling failed: {e}")
        return False
    
    # Test None filter
    print("\n=== Test 6: None filter ===")
    try:
        observation_frame = {
            "observation.state": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        }
        
        original_state_names = ["joint1.pos", "joint1.vel", "joint2.pos", "joint2.vel", "joint3.pos", "joint3.vel"]
        observation_state_filter = None
        
        filtered_frame = apply_observation_state_filter_to_frame(
            observation_frame.copy(), original_state_names, observation_state_filter
        )
        
        if np.array_equal(filtered_frame['observation.state'], observation_frame['observation.state']):
            print("✓ None filter handled correctly")
        else:
            print("✗ None filter changed the observation")
            return False
            
    except Exception as e:
        print(f"✗ None filter handling failed: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    return True

if __name__ == "__main__":
    success = test_inference_filtering_comprehensive()
    sys.exit(0 if success else 1)
