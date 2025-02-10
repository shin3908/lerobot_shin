python lerobot/scripts/train.py ^
  --dataset.repo_id=shin1107/koch_train_block ^
  --policy.type=act ^
  --output_dir=outputs/train/act_koch_train_block2 ^
  --job_name=act_koch_train_block ^
  --device=cuda ^
  --wandb.enable=true