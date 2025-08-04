import numpy as np
from datasets import Dataset
from typing import List

def my_transform(dataset: Dataset, episode_indices: List[int]) -> Dataset:
    """
    データセットから特定の 'observation.state' の特徴量を削除します。
    """
    
    # 削除したい特徴量のインデックス
    # info.jsonlのfeatures.observation.state.namesを見ると、
    # 'shoulder_pan.pos', 'shoulder_pan.current', ... と並んでいることがわかります。
    # 削除したいのは以下の項目です。
    # shoulder_pan.current (インデックス1)
    # shoulder_lift.current (インデックス3)
    # elbow_flex.current (インデックス5)
    # wrist_flex.current (インデックス7)
    # wrist_roll.current (インデックス9)
    # gripper.current (インデックス11)
    
    # 削除するインデックス
    indices_to_remove = [1, 3, 5, 7, 9, 11]

    def _transform_fn(batch):
        # バッチ内の各エピソードデータを処理
        new_states = []
        for state in batch["observation.state"]:
            # NumPy配列に変換して、指定されたインデックスを削除
            state_array = np.array(state)
            new_state_array = np.delete(state_array, indices_to_remove)
            new_states.append(new_state_array.tolist())
        
        batch["observation.state"] = new_states
        return batch

    # 新しいデータセットを作成
    new_dataset = dataset.map(
        _transform_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Removing observation.state features..."
    )

    return new_dataset