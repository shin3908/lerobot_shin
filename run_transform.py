# run_transform.py

import os
from datasets import load_dataset, DatasetDict
import json
import numpy as np

# ステップ1で作成した変換関数をインポート
# my_transform.pyと同一ディレクトリにこのスクリプトを配置してください
from my_transform import my_transform

# 既存のデータセットのパス
source_path = "C:/Users/harry/workspace/lerobot_shin/dataset/koch_new_fb"

# 新しいデータセットを保存するパス
target_path = "C:/Users/harry/workspace/lerobot_shin/dataset/koch_new_fb_6"

print("データセットを読み込み中...")
# datasets.load_datasetを使ってlerobot形式のデータセットを読み込み
dataset = load_dataset(source_path, split="train")

print("変換関数を適用中...")
# my_transform関数をmap()で適用
# remove_columnsをNoneにすると、新しい列を追加する代わりに既存の列を更新できる場合があります
# 念のため、元の列はそのまま残し、新しい列を作成する方が安全かもしれません
# 以下では、元のobservation.stateを新しいものに置き換える処理を行います
transformed_dataset = dataset.map(
    my_transform,
    batched=True,
    batch_size=1000,
    remove_columns=[], # 列を削除しない設定
    desc="Applying transform to observation.state..."
)

# 新しいデータセットを保存
print("新しいデータセットを保存中...")
transformed_dataset.save_to_disk(target_path)

# ----------------------------------------------------
# 3. info.jsonを更新する (手動)
# ----------------------------------------------------
print("info.jsonを更新中...")

# 元のinfo.jsonを読み込み
with open(os.path.join(source_path, "meta", "info.json"), 'r') as f:
    info_data = json.load(f)

# observation.stateのfeaturesを更新
state_features = info_data["features"]["observation.state"]
indices_to_remove = [1, 3, 5, 7, 9, 11]
new_names = [name for i, name in enumerate(state_features["names"]) if i not in indices_to_remove]
new_shape = [len(new_names)]

state_features["names"] = new_names
state_features["shape"] = new_shape

# 新しいinfo.jsonを保存
new_info_path = os.path.join(target_path, "meta", "info.json")
os.makedirs(os.path.dirname(new_info_path), exist_ok=True)
with open(new_info_path, 'w') as f:
    json.dump(info_data, f, indent=4)

print("info.jsonの更新が完了しました。")

# ----------------------------------------------------
# 4. episode_stats.jsonlを再計算・更新する (手動)
# ----------------------------------------------------
print("episode_stats.jsonlを再計算中...")

# エピソードごとの統計情報を再計算する
episode_stats = []
total_episodes = info_data["total_episodes"]

for i in range(total_episodes):
    episode_data = transformed_dataset.filter(lambda example: example["episode_index"] == i)
    
    # observation.stateの統計情報を計算
    states = episode_data["observation.state"]
    states_array = np.array(states)
    
    stats_entry = {
        "episode_index": i,
        "stats": {
            "observation.state": {
                "min": states_array.min(axis=0).tolist(),
                "max": states_array.max(axis=0).tolist(),
                "mean": states_array.mean(axis=0).tolist(),
                "std": states_array.std(axis=0).tolist(),
                "count": [len(states)]
            },
            # actionやtimestampなどの統計情報は、元データをコピーするか、必要に応じて再計算
            # ここでは簡略化のため、observation.stateのみを扱います
        }
    }
    episode_stats.append(stats_entry)

# 新しいepisode_stats.jsonlを保存
new_stats_path = os.path.join(target_path, "meta", "episode_stats.jsonl")
with open(new_stats_path, 'w') as f:
    for entry in episode_stats:
        f.write(json.dumps(entry) + "\n")

print("episode_stats.jsonlの再計算が完了しました。")

print("すべての変換が完了しました。")