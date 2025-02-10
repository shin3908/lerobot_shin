python lerobot/scripts/train.py ^
  --dataset.repo_id=shin1107/koch_move_block_with_some_shapes ^
  --policy.type=act ^
  --output_dir=outputs/train/koch_move_block_with_some_shapes ^
  --job_name=act_koch_move_block_with_some_shapes ^
  --device=cuda ^
  --wandb.enable=true