python lerobot/scripts/train.py ^
  --dataset.repo_id=shin1107/koch_move_block_with_some_shapes ^
  --policy.type=act ^
  --output_dir=outputs/train/koch_default_resnet18_not_pretrained ^
  --job_name=act_koch_default_resnet50_not_pretrained ^
  --device=cuda ^
  --wandb.enable=true ^
  --policy.vision_backbone="resnet50" ^
  --policy.pretrained_backbone_weights=None ^