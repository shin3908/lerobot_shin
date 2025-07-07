# LeRobot Training Configuration Files

このディレクトリには、LeRobotのトレーニングスクリプトで使用するYAML設定ファイルが含まれています。これらの設定ファイルを使用することで、CLIパラメータを使わずに`--config-path`のみで訓練を実行できます。

## 使用方法

### 基本的な実行方法
```bash
python lerobot/scripts/train.py --config-path=configs/<config_file>.yaml
```

### GPU指定
```bash
export CUDA_VISIBLE_DEVICES=0  # Linux/Mac
set CUDA_VISIBLE_DEVICES=0     # Windows
```

## 利用可能な設定ファイル

### SmolVLA設定
- **`smolvla_complete_all_features.yaml`**: 全観測特徴を使用するSmolVLA
- **`smolvla_complete_pos_only.yaml`**: 位置データのみを使用するSmolVLA（推奨）
- **`smolvla_from_scratch.yaml`**: ゼロから訓練するSmolVLA
- **`smolvla_advanced_compression.yaml`**: 高度な圧縮設定
- **`smolvla_pca_compression.yaml`**: PCA圧縮設定
- **`smolvla_pos_only.yaml`**: 位置データのみ（基本設定）
- **`smolvla_pos_current.yaml`**: 位置+電流データ

### Pi0設定
- **`pi0_complete_pos_only.yaml`**: 位置データのみを使用するPi0
- **`pi0_original_episodes.yaml`**: 元のエピソードデータセットでのPi0ファインチューニング
- **`pi0_advanced_compression.yaml`**: 高度な圧縮設定
- **`pi0_pos_only.yaml`**: 位置データのみ（基本設定）

### ACT設定
- **`act_resnet18_complete.yaml`**: ResNet18バックボーンを使用するACT
- **`act_resnet50_complete.yaml`**: ResNet50バックボーンを使用するACT

### Pi0Fast設定
- **`pi0fast_original_episodes.yaml`**: Pi0Fastファインチューニング

## 推奨設定

### 新しいデータセットでの訓練
```bash
# SmolVLA（位置データのみ、最も安定）
python lerobot/scripts/train.py --config-path=configs/smolvla_complete_pos_only.yaml

# Pi0（位置データのみ）
python lerobot/scripts/train.py --config-path=configs/pi0_complete_pos_only.yaml
```

### 事前訓練モデルのファインチューニング
```bash
# SmolVLA（自動次元圧縮）
python lerobot/scripts/train.py --config-path=configs/smolvla_complete_all_features.yaml

# Pi0（自動次元圧縮）
python lerobot/scripts/train.py --config-path=configs/pi0_original_episodes.yaml
```

## 設定のカスタマイズ

YAML ファイルを直接編集することで、訓練パラメータをカスタマイズできます：

```yaml
# 訓練パラメータ
batch_size: 64
steps: 100000

# 出力設定
output_dir: "your/custom/path"
job_name: "your_job_name"

# 観測フィルタリング
policy:
  observation_state_filter: ["pos"]  # 位置データのみ
  # observation_state_filter: ["pos", "current"]  # 位置+電流データ

# 次元圧縮設定
policy:
  dimension_compression_method: "auto"  # auto, position_selection, pca, weighted_groups
  enable_dimension_compression: true
```

## 次元圧縮オプション

- **`auto`**: 次元比率に基づいて自動選択（推奨）
- **`position_selection`**: 位置特徴のみを選択（12D→6D最適化）
- **`weighted_groups`**: 重み付き平均による圧縮
- **`pca`**: 主成分分析による最大情報保持（scikit-learn必要）
- **`uniform_sampling`**: 均一サンプリング（フォールバック）

## 環境設定

### Linux/Mac
```bash
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/data3/shinsakuo/tmp  # 必要に応じて
```

### Windows
```cmd
set CUDA_VISIBLE_DEVICES=0
set TMPDIR=C:\data3\shinsakuo\tmp
```

## 簡単実行スクリプト

- **`train_config_only.sh`**: Linux/Mac用実行スクリプト
- **`train_config_only.bat`**: Windows用実行スクリプト

これらのスクリプトを実行すると、利用可能な設定ファイルと使用方法が表示されます。

## トラブルシューティング

### 次元不一致エラー
- `enable_dimension_compression: true`を設定
- 適切な`dimension_compression_method`を選択

### PCA圧縮が利用できない
```bash
pip install scikit-learn
```

### 観測フィルタリングの問題
- データセットのinfo.jsonで特徴名を確認
- 訓練ログで`[OBSERVATION FILTER]`メッセージを確認

## ログの確認

訓練中に以下のようなログが表示され、圧縮と観測フィルタリングの状況を確認できます：

```
[standardise_state_dict] Compressing buffer_observation_state.mean from 12D to 6D
[standardise_state_dict] Using compression method: position_selection
[OBSERVATION FILTER] Applied filtering: ['pos'] -> 6 features selected from 12
```
