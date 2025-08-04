# my_transform.py

import numpy as np
from typing import List

def my_transform(batch):
    """
    データセットのバッチから特定の 'observation.state' の特徴量を削除します。
    """
    
    indices_to_remove = [1, 3, 5, 7, 9, 11]

    new_states = []
    for state in batch["observation.state"]:
        state_array = np.array(state)
        new_state_array = np.delete(state_array, indices_to_remove)
        new_states.append(new_state_array.tolist())
    
    batch["observation.state"] = new_states
    
    # info.jsonlとepisode_stats.jsonlを更新するために必要な情報
    # datasets.map()にはこれらのファイルを更新する機能がないため、手動で処理する必要があります
    
    return batch