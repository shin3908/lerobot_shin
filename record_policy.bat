python lerobot/scripts/control_robot.py ^
  --robot.type=koch ^
  --control.type=record ^
  --control.single_task="Evaluate Policy for Grasp a block (different position) and put it in the hole with some shapes." ^
  --control.fps=30 ^
  --control.repo_id=shin1107/eval_act_koch_move_block_with_some_shapes_different_position ^
  --control.warmup_time_s=5 ^
  --control.episode_time_s=30 ^
  --control.reset_time_s=20 ^
  --control.num_episodes=8 ^
  --control.push_to_hub=true ^
  --control.policy.path=trainedmodel/titan/act_koch_move_block_with_some_shapes/pretrained_model
  REM --control.policy.path=outputs/train/koch_move_block_with_some_shapes/checkpoints/last/pretrained_model
