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

※ smolvlaの学習 cuda指定が必須
export CUDA_VISIBLE_DEVICES=1
python lerobot/scripts/train.py \
  --dataset.repo_id=shin1107/koch_new2 \
  --policy.path=lerobot/smolvla_base \
  --output_dir=data3/train/new/koch_base_smolvla_pretrained \
  --job_name=act_koch_base_smolvla \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobotsmolvla_base_pretrained

------------------------------------------------------------------------------
※ 以下new環境での学習
※ observation_state_filterを使用するには、YAMLファイルを使用してください:
※   --config_path=configs/smolvla_pos_only.yaml (位置データのみ)
※   --config_path=configs/smolvla_pos_current.yaml (位置+電流データ)
※ または、コマンドラインから直接は機能しない可能性があります
※ record.pyでの評価時は、"[OBSERVATION FILTER]"のログでフィルタリング状況が確認できる
※ smolvla
※ バッチサイズに注意！！
※  --policy.max_state_dim=12 --policy.path=data3/smolvla_base_dim12 
export CUDA_VISIBLE_DEVICES=0

python lerobot/scripts/train.py \
  --config_path=configs/smolvla_fb_all.yaml \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=shin1107/koch_new_fb \
  --batch_size=64 \
  --steps=100000 \
  --output_dir=data3/train/new/koch_base_smolvla_fb_all_2 \
  --job_name=act_koch_base_smolvla_fb_all_2 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_policy


python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=shin1107/koch_new_fb \
  --batch_size=64 \
  --steps=100000 \
  --output_dir=data3/train/new/koch_base_smolvla_wofb_3 \
  --job_name=act_koch_base_smolvla_wofb_3 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_policy

※ pi0 with position only
export CUDA_VISIBLE_DEVICES=1

python lerobot/scripts/train.py \
  --policy.path=lerobot/pi0 \
  --dataset.repo_id=shin1107/koch_new_fb \
  --batch_size=8 \
  --steps=100000 \
  --output_dir=data3/train/new/koch_base_pi0_fb_all \
  --job_name=act_koch_base_pi0_fb_all \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_policy

python lerobot/scripts/train.py \
  --policy.path=lerobot/pi0 \
  --dataset.repo_id=shin1107/koch_new_fb \
  --batch_size=8 \
  --steps=100000 \
  --output_dir=data3/train/new/koch_base_pi0_wofb_3 \
  --job_name=act_koch_base_pi0_wofb_3 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_policy
