python lerobot/scripts/train.py \
  --dataset.repo_id=shin1107/koch_move_block_with_some_positions \
  --policy.type=act \
  --output_dir=outputs/train/koch_defaultpos_resnet18 \
  --job_name=act_koch_defaultpos_resnet18 \
  --device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_pos \
  --policy.vision_backbone="resnet18" \
  --policy.pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1"

※ そのままbashを実行するとエラーが発生するので、ターミナルで実行すること
※ export CUDA_VISIBLE_DEVICES=1
※ resnet18: "ResNet18_Weights.IMAGENET1K_V1", resnet50: "ResNet50_Weights.IMAGENET1K_V2"