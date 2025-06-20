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
※ export TMPDIR=/data3/shinsakuo/tmp
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
  --policy.path=lerobot/pi0 \
  --output_dir=data3/train/koch_base_pi0_pretrained \
  --job_name=act_koch_base_pi0 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobotpi_base_pretrained

※ pi0fast ファインチューニング用
python lerobot/scripts/train.py \
  --dataset.repo_id=shin1107/koch_base_episodes \
  --policy.path=lerobot/pi0fast_base \
  --output_dir=data3/train/koch_base_pi0fast_pretrained \
  --job_name=act_koch_base_pi0fast \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobotpi_base_pretrained

# smolvlaの学習
python lerobot/scripts/train.py \
  --dataset.repo_id=shin1107/koch_base_episodes \
  --policy.path=lerobot/smolvla_base \
  --output_dir=data3/train/koch_base_smolvla_pretrained \
  --job_name=act_koch_base_smolvla \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobotsmolvla_base_pretrained