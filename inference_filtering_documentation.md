# 推論時の観測状態フィルタリング機能

この機能により、学習時と同様に推論時にも観測状態の特徴量をフィルタリングできます。

## 使用方法

### 1. コマンドライン引数による指定

```bash
python -m lerobot.record \
  --robot.type=koch_follower \
  --policy.path=path/to/model \
  --dataset.observation_state_filter=['.pos'] \
  # その他のオプション...
```

### 2. フィルタリングオプション

- `['.pos']`: 位置データのみを使用
- `['.pos', '.vel']`: 位置と速度データを使用
- `None` または指定なし: フィルタリングなし（全データを使用）

### 3. record_policy_with_filter.bat の使用

新しいバッチファイル `record_policy_with_filter.bat` を使用して、対話的にフィルタリング設定を選択できます：

```
Enter FILTER (1 for pos only, 2 for pos+vel, 3 for no filter):
```

- `1`: 位置のみ (`['.pos']`)
- `2`: 位置と速度 (`['.pos', '.vel']`)
- `3`: フィルタリングなし

## 実装詳細

### 1. 学習時のフィルタリング

学習時に `dataset_to_policy_features` 関数でフィルタリングが適用され、ポリシーは特定の特徴量のみで学習されます。

### 2. 推論時のフィルタリング

推論時は以下の手順でフィルタリングが適用されます：

1. ロボットから全ての観測データを取得
2. `build_dataset_frame` でデータセット形式に変換
3. `apply_observation_state_filter_to_frame` でフィルタリングを適用
4. フィルタリングされたデータでポリシーの推論を実行

### 3. 対応するファイル

- `lerobot/record.py`: 推論実行時のメインスクリプト
- `lerobot/common/policies/factory.py`: ポリシー作成時のフィルタリング適用
- `lerobot/common/datasets/utils.py`: フィルタリング関数の実装

## 使用例

### 位置のみでの推論

```bash
python -m lerobot.record \
  --robot.type=koch_follower \
  --robot.port=COM3 \
  --robot.id=follower \
  --dataset.repo_id=test/eval_pos_only \
  --dataset.num_episodes=10 \
  --policy.path=trainedmodel/models/koch_base_pi0/020000/pretrained_model \
  --dataset.observation_state_filter=['.pos']
```

### 注意事項

1. **フィルタリング一致**: 推論時のフィルタリングは、学習時に使用したフィルタリングと一致させる必要があります。

2. **特徴量の互換性**: ポリシーが期待する特徴量の数と形状が、フィルタリング後のデータと一致することを確認してください。

3. **性能**: フィルタリングは推論時の各フレームに適用されるため、処理速度に若干の影響があります。

## テスト

推論時フィルタリング機能のテストは `test_inference_filtering.py` で実行できます：

```bash
python test_inference_filtering.py
```

これにより、フィルタリング関数が正しく動作することを確認できます。
