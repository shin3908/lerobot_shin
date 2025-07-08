#!/bin/bash

# LeRobot Training Script with Advanced Observation Filtering and Dimension Compression
# This script demonstrates how to train SmolVLA and Pi0 policies with flexible observation filtering
# and intelligent dimension compression for handling different observation dimensions.

# ==============================================================================
# IMPORTANT NOTES:
# 1. CLI arguments can only accept single values, not lists
# 2. For multi-value observation_state_filter, use YAML config files
# 3. The new compression features help when using pretrained models with different dimensions
# ==============================================================================

# Example 1: Basic training with position-only filtering (CLI approach)
echo "=== Example 1: Basic training with position-only filtering ==="
echo "python lerobot/scripts/train.py \\"
echo "  --policy.type=smolvla \\"
echo "  --policy.observation_state_filter=pos \\"
echo "  --policy.dimension_compression_method=auto \\"
echo "  --dataset.repo_id=shin1107/koch_new_fb \\"
echo "  --steps=200000 \\"
echo "  --batch_size=64"
echo

# Example 2: Training with advanced PCA compression (config file approach)
echo "=== Example 2: Training with advanced PCA compression ==="
echo "python lerobot/scripts/train.py \\"
echo "  --config-path=configs/smolvla_pca_compression.yaml"
echo

# Example 3: Fine-tuning pretrained model with dimension compression
echo "=== Example 3: Fine-tuning pretrained model with dimension compression ==="
echo "python lerobot/scripts/train.py \\"
echo "  --policy.path=lerobot/smolvla_base \\"
echo "  --policy.dimension_compression_method=position_selection \\"
echo "  --policy.enable_dimension_compression=true \\"
echo "  --dataset.repo_id=shin1107/koch_new_fb \\"
echo "  --steps=200000 \\"
echo "  --batch_size=64"
echo

# Example 4: Pi0 training with advanced compression
echo "=== Example 4: Pi0 training with advanced compression ==="
echo "python lerobot/scripts/train.py \\"
echo "  --config-path=configs/pi0_advanced_compression.yaml"
echo

# ==============================================================================
# OBSERVATION FILTERING OPTIONS:
# ==============================================================================
echo "=== Observation Filtering Options ==="
echo "1. observation_state_filter: Specify which observation features to include"
echo "   - Single value (CLI): --policy.observation_state_filter=pos"
echo "   - Multiple values (YAML): observation_state_filter: ['pos', 'current']"
echo "   - Feature matching supports:"
echo "     * Exact names: 'shoulder_pan.pos'"
echo "     * Suffixes: 'pos' (matches all *.pos features)"
echo "     * Prefixes: 'shoulder_pan' (matches shoulder_pan.* features)"
echo

# ==============================================================================
# DIMENSION COMPRESSION OPTIONS:
# ==============================================================================
echo "=== Dimension Compression Options ==="
echo "2. dimension_compression_method: How to compress dimensions when loading pretrained models"
echo "   - 'auto': Automatically select best method based on dimensions"
echo "   - 'position_selection': Use position features for 12D->6D compression"
echo "   - 'weighted_groups': Use weighted averaging for moderate compression"
echo "   - 'pca': Use Principal Component Analysis for maximum information preservation (requires sklearn)"
echo "   - 'uniform_sampling': Use uniform sampling as fallback"
echo
echo "3. enable_dimension_compression: Enable/disable dimension compression (default: true)"
echo

# ==============================================================================
# CONFIGURATION FILE EXAMPLES:
# ==============================================================================
echo "=== Available Configuration Files ==="
echo "- configs/smolvla_pos_only.yaml: SmolVLA with position-only filtering"
echo "- configs/smolvla_pos_current.yaml: SmolVLA with position and current filtering"
echo "- configs/smolvla_advanced_compression.yaml: SmolVLA with advanced compression"
echo "- configs/smolvla_pca_compression.yaml: SmolVLA with PCA compression"
echo "- configs/pi0_pos_only.yaml: Pi0 with position-only filtering"
echo "- configs/pi0_advanced_compression.yaml: Pi0 with advanced compression"
echo

# ==============================================================================
# ADVANCED USAGE:
# ==============================================================================
echo "=== Advanced Usage ==="
echo "1. Install scikit-learn for PCA compression:"
echo "   pip install scikit-learn"
echo
echo "2. Check compression methods available:"
echo "   python -c \"from lerobot.common.policies.smolvla.modeling_smolvla import get_compression_info; print(get_compression_info(12, 6))\""
echo
echo "3. Monitor compression during training:"
echo "   The system will log compression operations and methods used"
echo

# ==============================================================================
# TROUBLESHOOTING:
# ==============================================================================
echo "=== Troubleshooting ==="
echo "1. If you see dimension mismatch errors:"
echo "   - Check that your observation_state_filter matches your dataset features"
echo "   - Ensure enable_dimension_compression=true when loading pretrained models"
echo
echo "2. If PCA compression is not available:"
echo "   - Install scikit-learn: pip install scikit-learn"
echo "   - Or use alternative compression methods like 'position_selection'"
echo
echo "3. For debugging observation filtering:"
echo "   - Check training logs for '[OBSERVATION FILTER]' messages"
echo "   - Verify dataset feature names in the dataset info.json"
echo

echo "=== Training script ready! ==="
echo "Choose one of the examples above or create your own configuration."

# ==============================================================================
# ORIGINAL TRAINING COMMANDS (PRESERVED FOR REFERENCE):
# ==============================================================================
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
export CUDA_VISIBLE_DEVICES=0
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=shin1107/koch_new_fb \
  --batch_size=64 \
  --steps=100000 \
  --output_dir=data3/train/new/koch_base_smolvla_fb_all \
  --job_name=act_koch_base_smolvla_fb_all \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_policy

python lerobot/scripts/train.py \
  --config_path=configs/smolvla_pos_only.yaml \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=shin1107/koch_new_fb \
  --batch_size=64 \
  --steps=100000 \
  --output_dir=data3/train/new/koch_base_smolvla_wofb \
  --job_name=act_koch_base_smolvla_wofb \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_policy

※ pi0 with position only
CUDA_VISIBLE_DEVICES=1 
python lerobot/scripts/train.py \
  --config_path=configs/pi0_pos_only.yaml \
  --policy.path=lerobot/pi0 \
  --dataset.repo_id=shin1107/koch_new_fb \
  --batch_size=8 \
  --steps=100000 \
  --output_dir=data3/train/new/koch_base_pi0_pos \
  --job_name=act_koch_base_pi0_pos \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_policy

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

