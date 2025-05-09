python lerobot/scripts/control_robot.py ^
  --robot.type=koch ^
  --control.type=record ^
  --control.single_task="Evaluate Policy for default position task Resnet18" ^
  --control.fps=30 ^
  --control.repo_id=shin1107/eval_koch_defaultpos_resnet18 ^
  --control.warmup_time_s=5 ^
  --control.episode_time_s=30 ^
  --control.reset_time_s=20 ^
  --control.num_episodes=10 ^
  --control.push_to_hub=true ^
  --control.policy.path=trainedmodel\sirius\koch_defaultpos_resnet18\pretrained_model
