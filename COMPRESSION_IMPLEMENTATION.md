# 12次元観測データの効果的な6次元圧縮機能の実装

本実装では、LeRobotのSmolVLAおよびPi0ポリシーにおいて、12次元のKoch観測データを6次元のSO100形式に効果的に圧縮する機能を追加しました。

## 実装された機能

### 1. 高度な圧縮メソッド

#### 自動選択 (`auto`)
- 次元比率に基づいて最適な圧縮方法を自動選択
- 12D→6D の場合は位置特徴選択を使用
- 中程度の圧縮では重み付き平均を使用
- 大きな圧縮比ではPCAを使用

#### 位置特徴選択 (`position_selection`)
- 12次元データから位置データ（偶数インデックス）のみを選択
- Kochロボットの関節位置情報を最大限保持
- インデックス [0, 2, 4, 6, 8, 10] を使用

#### 重み付きグループ化 (`weighted_groups`)
- 複数の次元を重み付き平均で圧縮
- より重要な特徴に高い重みを付与
- 中程度の圧縮比に最適

#### PCA圧縮 (`pca`)
- 主成分分析を使用して最大分散を保持
- 最も情報量の多い成分を選択
- scikit-learnが必要

#### 均一サンプリング (`uniform_sampling`)
- 均等間隔でサンプリング
- フォールバック方法として使用

### 2. 設定オプション

#### SmolVLA設定
```yaml
policy:
  dimension_compression_method: "auto"  # 圧縮方法
  enable_dimension_compression: true    # 圧縮有効化
  observation_state_filter: ["pos"]     # 観測フィルタ
```

#### Pi0設定
```yaml
policy:
  dimension_compression_method: "auto"
  enable_dimension_compression: true
  observation_state_filter: ["pos"]
```

### 3. 使用可能な設定ファイル

- `configs/smolvla_advanced_compression.yaml`: 高度な圧縮設定
- `configs/smolvla_pca_compression.yaml`: PCA圧縮設定
- `configs/pi0_advanced_compression.yaml`: Pi0用高度な圧縮設定
- `configs/smolvla_pos_only.yaml`: 位置データのみ
- `configs/smolvla_pos_current.yaml`: 位置+電流データ

### 4. ログ出力と診断

#### 圧縮操作ログ
```
[standardise_state_dict] Compressing buffer_observation_state.mean from 12D to 6D
[standardise_state_dict] Using compression method: position_selection
[standardise_state_dict] Applied 2 compression operations:
  - buffer_observation_state.mean: torch.Size([12]) -> torch.Size([6]) using position_selection
  - buffer_observation_state.std: torch.Size([12]) -> torch.Size([6]) using position_selection
```

#### 圧縮方法情報
```python
from lerobot.common.policies.smolvla.modeling_smolvla import get_compression_info
info = get_compression_info(12, 6)
print(info)  # {'method': 'position_selection', 'description': 'Koch to SO100: selecting position features'}
```

### 5. 使用例

#### 基本的な訓練（位置データのみ）
```bash
python lerobot/scripts/train.py \
  --policy.type=smolvla \
  --policy.observation_state_filter=pos \
  --policy.dimension_compression_method=auto \
  --dataset.repo_id=shin1107/koch_new_fb \
  --steps=200000 \
  --batch_size=64
```

#### 事前訓練モデルのファインチューニング
```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --policy.dimension_compression_method=position_selection \
  --policy.enable_dimension_compression=true \
  --dataset.repo_id=shin1107/koch_new_fb \
  --steps=200000 \
  --batch_size=64
```

#### PCA圧縮を使用した高度な設定
```bash
python lerobot/scripts/train.py \
  --config-path=configs/smolvla_pca_compression.yaml
```

### 6. トラブルシューティング

#### 次元不一致エラー
- `observation_state_filter`がデータセット特徴と一致することを確認
- 事前訓練モデル使用時は`enable_dimension_compression=true`を設定

#### PCA圧縮が利用できない
- scikit-learnをインストール: `pip install scikit-learn`
- または他の圧縮方法を使用: `position_selection`、`weighted_groups`

#### 観測フィルタリングのデバッグ
- 訓練ログで`[OBSERVATION FILTER]`メッセージを確認
- データセットのinfo.jsonで特徴名を確認

### 7. 技術的詳細

#### 圧縮アルゴリズム
1. **位置特徴選択**: 最も重要な関節位置情報を保持
2. **重み付き平均**: 複数特徴を重要度に応じて統合
3. **PCA**: 最大分散を保持する主成分を選択
4. **均一サンプリング**: 安全なフォールバック

#### 後方互換性
- フィルタが設定されていない場合は全特徴を使用
- 特徴名が不明な場合は圧縮をスキップ
- 既存のワークフローを破壊しない設計

## 利点

1. **情報保持**: 12次元から6次元への圧縮で最大限の情報を保持
2. **柔軟性**: 複数の圧縮方法から選択可能
3. **自動化**: 次元に基づいて最適な方法を自動選択
4. **互換性**: 既存のSO100事前訓練モデルとの互換性を確保
5. **診断**: 詳細なログで圧縮過程を可視化

この実装により、Kochロボットの12次元観測データをSO100形式の6次元に効果的に圧縮し、事前訓練モデルを効率的に活用できるようになりました。
