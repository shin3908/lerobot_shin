python lerobot/scripts/train.py \
  --dataset.repo_id=shin1107/koch_move_block_with_some_shapes \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_move_block_with_some_shapes \
  --job_name=act_koch_train_move_block_with_some_shapes_sirius \
  --device=cuda \
  --wandb.enable=true

※ そのままbashを実行するとエラーが発生するので、ターミナルで実行すること
