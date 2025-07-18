#!/usr/bin/env python3

"""Integration test for inference filtering functionality."""

import sys
import os
import tempfile
import json
import numpy as np
import torch
from pathlib import Path

def test_integration():
    """Integration test for the full inference filtering pipeline."""
    
    print("=== Integration Test: Inference Filtering Pipeline ===")
    
    try:
        # Test imports
        from lerobot.common.datasets.utils import (
            dataset_to_policy_features,
            apply_observation_state_filter_to_frame
        )
        from lerobot.common.policies.factory import make_policy
        from lerobot.configs.types import PolicyFeature
        
        print("✓ Successfully imported required functions")
        
        # Mock dataset metadata
        mock_features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joint1.pos", "joint1.vel", "joint2.pos", "joint2.vel", "joint3.pos", "joint3.vel"]
            },
            "observation.images.front": {
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
        
        # Test 1: dataset_to_policy_features without filtering
        print("\n--- Test 1: dataset_to_policy_features without filtering ---")
        policy_features = dataset_to_policy_features(mock_features)
        print(f"Policy features keys: {list(policy_features.keys())}")
        
        if "observation.state" in policy_features:
            obs_state_feature = policy_features["observation.state"]
            print(f"observation.state shape: {obs_state_feature.shape}")
            print(f"observation.state feature: {obs_state_feature}")
            
            if obs_state_feature.shape == (6,):
                print("✓ Unfiltered features have correct shape")
            else:
                print(f"✗ Unexpected shape: {obs_state_feature.shape}")
                return False
        else:
            print("✗ observation.state not found in policy features")
            return False
        
        # Test 2: dataset_to_policy_features with filtering
        print("\n--- Test 2: dataset_to_policy_features with filtering ---")
        filtered_policy_features = dataset_to_policy_features(mock_features, [".pos"])
        
        if "observation.state" in filtered_policy_features:
            filtered_obs_state_feature = filtered_policy_features["observation.state"]
            print(f"Filtered observation.state shape: {filtered_obs_state_feature.shape}")
            print(f"Filtered observation.state feature: {filtered_obs_state_feature}")
            
            if filtered_obs_state_feature.shape == (3,):
                print("✓ Filtered features have correct shape")
            else:
                print(f"✗ Unexpected filtered shape: {filtered_obs_state_feature.shape}")
                return False
        else:
            print("✗ observation.state not found in filtered policy features")
            return False
        
        # Test 3: Inference-time filtering
        print("\n--- Test 3: Inference-time observation filtering ---")
        
        # Simulate robot observation (full state)
        robot_observation = {
            "observation.state": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            "observation.images.front": np.zeros((3, 480, 640), dtype=np.uint8),
        }
        
        # Apply inference-time filtering
        original_state_names = ["joint1.pos", "joint1.vel", "joint2.pos", "joint2.vel", "joint3.pos", "joint3.vel"]
        observation_state_filter = [".pos"]
        
        filtered_observation = apply_observation_state_filter_to_frame(
            robot_observation.copy(),
            original_state_names,
            observation_state_filter
        )
        
        expected_filtered_state = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        if np.array_equal(filtered_observation["observation.state"], expected_filtered_state):
            print("✓ Inference-time filtering works correctly")
            print(f"Original state: {robot_observation['observation.state']}")
            print(f"Filtered state: {filtered_observation['observation.state']}")
        else:
            print(f"✗ Inference-time filtering failed")
            print(f"Expected: {expected_filtered_state}")
            print(f"Got: {filtered_observation['observation.state']}")
            return False
        
        # Test 4: Shape consistency between training and inference
        print("\n--- Test 4: Shape consistency between training and inference ---")
        
        # Training features (from dataset_to_policy_features)
        training_shape = filtered_policy_features["observation.state"].shape
        
        # Inference features (from apply_observation_state_filter_to_frame)
        inference_shape = filtered_observation["observation.state"].shape
        
        if training_shape == inference_shape:
            print(f"✓ Shape consistency maintained: {training_shape} == {inference_shape}")
        else:
            print(f"✗ Shape inconsistency: training={training_shape}, inference={inference_shape}")
            return False
        
        # Test 5: Multi-filter scenario
        print("\n--- Test 5: Multi-filter scenario ---")
        
        multi_filter = [".pos", ".vel"]
        multi_filtered_policy_features = dataset_to_policy_features(mock_features, multi_filter)
        multi_filtered_observation = apply_observation_state_filter_to_frame(
            robot_observation.copy(),
            original_state_names,
            multi_filter
        )
        
        training_multi_shape = multi_filtered_policy_features["observation.state"].shape
        inference_multi_shape = multi_filtered_observation["observation.state"].shape
        
        if training_multi_shape == inference_multi_shape == (6,):
            print(f"✓ Multi-filter shape consistency: {training_multi_shape} == {inference_multi_shape}")
        else:
            print(f"✗ Multi-filter shape inconsistency: training={training_multi_shape}, inference={inference_multi_shape}")
            return False
        
        # Test 6: No filter scenario
        print("\n--- Test 6: No filter scenario ---")
        
        no_filter_policy_features = dataset_to_policy_features(mock_features, None)
        no_filter_observation = apply_observation_state_filter_to_frame(
            robot_observation.copy(),
            original_state_names,
            None
        )
        
        training_no_filter_shape = no_filter_policy_features["observation.state"].shape
        inference_no_filter_shape = no_filter_observation["observation.state"].shape
        
        if training_no_filter_shape == inference_no_filter_shape == (6,):
            print(f"✓ No filter shape consistency: {training_no_filter_shape} == {inference_no_filter_shape}")
        else:
            print(f"✗ No filter shape inconsistency: training={training_no_filter_shape}, inference={inference_no_filter_shape}")
            return False
        
        print("\n=== All integration tests passed! ===")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
