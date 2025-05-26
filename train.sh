python lerobot/scripts/train.py \
  --dataset.repo_id=shin1107/koch_base \
  --policy.type=act \
  --output_dir=outputs/train/koch_base_resnet18 \
  --job_name=act_koch_base_resnet18 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_base \
  --policy.vision_backbone="resnet18" \
  --policy.pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1"

※ そのままbashを実行するとエラーが発生するので、ターミナルで実行すること
※ export CUDA_VISIBLE_DEVICES=1
※ resnet18: "ResNet18_Weights.IMAGENET1K_V1", resnet50: "ResNet50_Weights.IMAGENET1K_V2"

python lerobot/scripts/train.py \
  --dataset.repo_id=shin1107/koch_base \
  --policy.type=act \
  --output_dir=outputs/train/koch_base_resnet50 \
  --job_name=act_koch_base_resnet50 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_base \
  --policy.vision_backbone="resnet50" \
  --policy.pretrained_backbone_weights="ResNet50_Weights.IMAGENET1K_V2"


※ pi0 ファインチューニング用
python lerobot/scripts/train.py \
  --dataset.repo_id=shin1107/koch_base_episodes \
  --policy.type=pi0 \
  --output_dir=outputs/train/koch_base_pi0 \
  --job_name=act_koch_base_pi0 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobotpi_base

※ pi0fast ファインチューニング用
  python lerobot/scripts/train.py \
  --dataset.repo_id=shin1107/koch_base_episodes \
  --policy.type=pi0fast \
  --output_dir=outputs/train/koch_base_pi0fast \
  --job_name=act_koch_base_pi0fast \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobotpi_base