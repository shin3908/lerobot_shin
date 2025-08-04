import os
import shutil
from datasets import load_dataset
import json
import numpy as np
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# 1. 変換関数を定義
# --------------------------------------------------------------------------

def my_transform(batch):
    """
    observation.stateから特定のインデックスを削除する変換関数。
    """
    indices_to_remove = [1, 3, 5, 7, 9, 11]

    new_states = []
    for state in batch["observation.state"]:
        state_array = np.array(state)
        new_state_array = np.delete(state_array, indices_to_remove)
        new_states.append(new_state_array.tolist())
    
    batch["observation.state"] = new_states
    
    return batch

# --------------------------------------------------------------------------
# 2. メインスクリプト
# --------------------------------------------------------------------------

# データセットのパスを設定
source_path = "C:/Users/harry/workspace/lerobot_shin/dataset/koch_new_fb"
target_path = "C:/Users/harry/workspace/lerobot_shin/dataset/koch_new_fb_transformed"

# ターゲットディレクトリをクリーンアップ
if os.path.exists(target_path):
    print(f"既存のターゲットディレクトリを削除: {target_path}")
    shutil.rmtree(target_path)

# 新しいデータセットのディレクトリ構造を作成
os.makedirs(os.path.join(target_path, "meta"), exist_ok=True)
os.makedirs(os.path.join(target_path, "videos"), exist_ok=True)
os.makedirs(os.path.join(target_path, "data", "chunk-000"), exist_ok=True)

print("データセットを読み込み中...")
dataset = load_dataset(source_path, split="train")

# --------------------------------------------------------------------------
# 3. データの変換のみ
# --------------------------------------------------------------------------

print("変換関数を適用中...")
transformed_dataset = dataset.map(
    my_transform,
    batched=True,
    batch_size=1000,
    desc="Applying transform to observation.state..."
)

# --------------------------------------------------------------------------
# 4. 変換後のデータを個別のParquetファイルとして保存
# --------------------------------------------------------------------------

print("変換後のデータを個別のParquetファイルとして保存中...")
unique_episodes = sorted(transformed_dataset.unique("episode_index"))
for episode_index in unique_episodes:
    print(f"  - エピソード {episode_index} を保存中...")
    
    episode_data = transformed_dataset.filter(lambda example: example["episode_index"] == episode_index)
    
    if len(episode_data) > 0:
        table = episode_data.data.table
        save_path = os.path.join(target_path, "data", "chunk-000", f"episode_{episode_index:06d}.parquet")
        pq.write_table(table, save_path)
    else:
        print(f"警告: エピソード {episode_index} にはデータがありませんでした。スキップします。")

# --------------------------------------------------------------------------
# 5. その他の関連ファイルをコピー
# --------------------------------------------------------------------------

print("関連ファイルをコピー中...")
source_videos_path = os.path.join(source_path, "videos")
target_videos_path = os.path.join(target_path, "videos")
shutil.copytree(source_videos_path, target_videos_path, dirs_exist_ok=True)

source_meta_path = os.path.join(source_path, "meta")
target_meta_path = os.path.join(target_path, "meta")
# そのままコピーするmetaファイルリスト
for filename in ["episodes.jsonl", "tasks.jsonl"]:
    source_file = os.path.join(source_meta_path, filename)
    if os.path.exists(source_file):
        shutil.copy(source_file, target_meta_path)
        print(f"  - コピー完了: {filename}")

# --------------------------------------------------------------------------
# 6. info.jsonとepisodes_stats.jsonlを更新
# --------------------------------------------------------------------------

print("info.jsonとepisodes_stats.jsonlを更新中...")

source_info_path_jsonl = os.path.join(source_path, "meta", "info.jsonl")
source_info_path_json = os.path.join(source_path, "meta", "info.json")

if os.path.exists(source_info_path_jsonl):
    with open(source_info_path_jsonl, 'r') as f:
        info_data = json.load(f)
elif os.path.exists(source_info_path_json):
    with open(source_info_path_json, 'r') as f:
        info_data = json.load(f)
else:
    info_data = {}

state_features = info_data.get("features", {}).get("observation.state", {})
indices_to_remove = [1, 3, 5, 7, 9, 11]

if "names" in state_features and "shape" in state_features:
    new_names = [name for i, name in enumerate(state_features["names"]) if i not in indices_to_remove]
    new_shape = [len(new_names)]
    state_features["names"] = new_names
    state_features["shape"] = new_shape
    info_data["features"]["observation.state"] = state_features

new_info_path = os.path.join(target_path, "meta", "info.json")
with open(new_info_path, 'w') as f:
    json.dump(info_data, f, indent=4)
print("  - info.jsonの更新完了")

episode_stats_list = []
unique_episodes = sorted(transformed_dataset.unique("episode_index"))
print(f"再計算対象エピソード数: {len(unique_episodes)}")

for i in unique_episodes:
    episode_data = transformed_dataset.filter(lambda example: example["episode_index"] == i)
    
    if len(episode_data) > 0:
        states_array = np.array(episode_data["observation.state"])
        action_array = np.array(episode_data["action"])
        
        if states_array.ndim > 1 and states_array.shape[0] > 0:
            stats_entry = {
                "episode_index": i,
                "stats": {
                    "action": {
                        "min": action_array.min(axis=0).tolist(),
                        "max": action_array.max(axis=0).tolist(),
                        "mean": action_array.mean(axis=0).tolist(),
                        "std": action_array.std(axis=0).tolist(),
                        "count": [len(action_array)]
                    },
                    "observation.state": {
                        "min": states_array.min(axis=0).tolist(),
                        "max": states_array.max(axis=0).tolist(),
                        "mean": states_array.mean(axis=0).tolist(),
                        "std": states_array.std(axis=0).tolist(),
                        "count": [len(states_array)]
                    }
                }
            }
            episode_stats_list.append(stats_entry)
        else:
            print(f"警告: エピソード {i} のデータが空または無効です。")

new_stats_path = os.path.join(target_path, "meta", "episodes_stats.jsonl")
with open(new_stats_path, 'w') as f:
    for entry in episode_stats_list:
        f.write(json.dumps(entry) + "\n")
print("  - episodes_stats.jsonlの更新完了")

print("すべての処理が完了しました。新しいデータセットは以下の場所に保存されました:")
print(target_path)