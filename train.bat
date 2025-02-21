python lerobot/scripts/train.py ^
  --dataset.repo_id=shin1107/koch_move_block_with_some_positions ^
  --policy.type=act ^
  --output_dir=outputs/train/koch_defaultpos_resnet18 ^
  --job_name=act_koch_defaultpos_resnet50 ^
  --device=cuda ^
  --wandb.enable=true ^
  --policy.vision_backbone="resnet50" ^
  --policy.pretrained_backbone_weights="ResNet50_Weights.IMAGENET1K_V2" ^