import os
import shutil
import json
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import traceback
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import glob

# ==========================================================================
# 設定項目 - ここを変更して実行してください
# ==========================================================================

# データセットパス設定
SOURCE_PATH = "C:/Users/harry/workspace/lerobot_shin/dataset/koch_new_fb"
TARGET_PATH = "C:/Users/harry/workspace/lerobot_shin/dataset/koch_new_fb_6d3"

# フィルタ設定
INDICES_TO_REMOVE = [1, 3, 5, 7, 9, 11]  # observation.stateから削除するインデックス
TARGET_COLUMN = "observation.state"  # 変換対象カラム

# 処理設定
MAX_WORKERS = 8  # 最大並列処理数（CPUコア数より少なく設定）
CHUNK_NAME = "chunk-000"  # 処理対象チャンク名

# コピーするmetaファイルリスト
META_FILES_TO_COPY = ["episodes.jsonl", "tasks.jsonl"]

# 画像統計計算用の設定
IMAGE_COLUMNS = ["observation.images.front", "observation.images.top"]  # 統計計算対象の画像カラム

# ==========================================================================

def my_transform_array(state_array):
    """
    単一の observation.state 配列から特定のインデックスを削除する変換関数。
    既に変換済みの場合は何もしない。
    """
    try:
        state_array = np.array(state_array)
        
        # 配列のサイズをチェック
        if len(state_array) <= len(INDICES_TO_REMOVE):
            # 既に変換済みまたは元々サイズが小さい場合は何もしない
            return state_array.tolist()
        
        # 元のサイズから変換後のサイズを計算
        expected_original_size = max(INDICES_TO_REMOVE) + 1
        expected_transformed_size = expected_original_size - len(INDICES_TO_REMOVE)
        
        if len(state_array) == expected_original_size:
            # 変換を実行
            new_state_array = np.delete(state_array, INDICES_TO_REMOVE)
            return new_state_array.tolist()
        elif len(state_array) == expected_transformed_size:
            # 既に変換済みの場合
            return state_array.tolist()
        else:
            # 予期しないサイズの場合は警告を出して元のままを返す
            print(f"警告: 予期しない{TARGET_COLUMN}配列サイズ: {len(state_array)}")
            return state_array.tolist()
            
    except Exception as e:
        print(f"警告: {TARGET_COLUMN}変換でエラーが発生しました: {e}")
        return state_array.tolist() if hasattr(state_array, 'tolist') else state_array

def filter_stats_array(stats_array, indices_to_remove):
    """統計配列から指定されたインデックスを削除"""
    try:
        filtered_array = [val for i, val in enumerate(stats_array) if i not in indices_to_remove]
        return filtered_array
    except Exception as e:
        print(f"統計配列フィルタリングでエラー: {e}")
        return stats_array

def process_single_parquet_file(args):
    """単一のparquetファイルを処理（並列処理用）"""
    input_file_path, output_file_path = args
    
    try:
        # Parquetファイルを読み込み
        table = pq.read_table(input_file_path)
        df = table.to_pandas()
        
        if len(df) == 0:
            return {
                'file': os.path.basename(input_file_path),
                'status': 'empty',
                'count': 0
            }
        
        # 指定されたカラムを変換（サイズチェック付き）
        if TARGET_COLUMN in df.columns:
            # まず最初の行でサイズを確認
            first_state = df[TARGET_COLUMN].iloc[0]
            state_size = len(first_state) if hasattr(first_state, '__len__') else 0
            
            expected_original_size = max(INDICES_TO_REMOVE) + 1
            expected_transformed_size = expected_original_size - len(INDICES_TO_REMOVE)
            
            if state_size == expected_transformed_size:
                # 既に変換済みと判断してファイルコピー
                shutil.copy2(input_file_path, output_file_path)
                return {
                    'file': os.path.basename(input_file_path),
                    'status': 'already_transformed',
                    'count': len(df)
                }
            elif state_size == expected_original_size:
                # 変換を実行
                transformed_states = []
                for state in df[TARGET_COLUMN]:
                    transformed_state = my_transform_array(state)
                    transformed_states.append(transformed_state)
                
                df[TARGET_COLUMN] = transformed_states
            else:
                print(f"警告: ファイル {os.path.basename(input_file_path)}の{TARGET_COLUMN}次元数が予期しない値です: {state_size}")
                return {
                    'file': os.path.basename(input_file_path),
                    'status': 'unexpected_size',
                    'count': len(df),
                    'size': state_size
                }
        
        # 変換後のデータを新しいParquetファイルとして保存
        new_table = pa.Table.from_pandas(df)
        pq.write_table(new_table, output_file_path)
        
        return {
            'file': os.path.basename(input_file_path),
            'status': 'success',
            'count': len(df)
        }
        
    except Exception as e:
        return {
            'file': os.path.basename(input_file_path),
            'status': 'error',
            'error': str(e),
            'count': 0
        }

def update_info_json(source_path, target_path):
    """info.jsonを更新"""
    try:
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

        # 指定されたカラムの情報を更新
        if "features" in info_data and TARGET_COLUMN in info_data["features"]:
            column_features = info_data["features"][TARGET_COLUMN]
            
            if "names" in column_features and "shape" in column_features:
                # インデックスを削除して新しいnamesとshapeを作成
                new_names = [name for i, name in enumerate(column_features["names"]) if i not in INDICES_TO_REMOVE]
                new_shape = [len(new_names)]
                
                column_features["names"] = new_names
                column_features["shape"] = new_shape
                info_data["features"][TARGET_COLUMN] = column_features

        new_info_path = os.path.join(target_path, "meta", "info.json")
        with open(new_info_path, 'w') as f:
            json.dump(info_data, f, indent=4)
        
        print("  - info.jsonの更新完了")

    except Exception as e:
        print(f"エラー: info.json更新中に問題が発生しました: {e}")
        traceback.print_exc()

def copy_and_filter_episodes_stats(source_path, target_path):
    """episodes_stats.jsonlをコピーしてフィルタリング"""
    try:
        source_stats_path = os.path.join(source_path, "meta", "episodes_stats.jsonl")
        target_stats_path = os.path.join(target_path, "meta", "episodes_stats.jsonl")
        
        if not os.path.exists(source_stats_path):
            print(f"警告: ソースの統計ファイルが見つかりません: {source_stats_path}")
            return
        
        print("元の統計情報をコピーしてフィルタリング中...")
        
        with open(source_stats_path, 'r') as source_file, open(target_stats_path, 'w') as target_file:
            for line in source_file:
                try:
                    entry = json.loads(line.strip())
                    
                    # TARGET_COLUMNの統計をフィルタリング
                    if "stats" in entry and TARGET_COLUMN in entry["stats"]:
                        target_stats = entry["stats"][TARGET_COLUMN]
                        
                        # 各統計値をフィルタリング
                        if "min" in target_stats:
                            target_stats["min"] = filter_stats_array(target_stats["min"], INDICES_TO_REMOVE)
                        if "max" in target_stats:
                            target_stats["max"] = filter_stats_array(target_stats["max"], INDICES_TO_REMOVE)
                        if "mean" in target_stats:
                            target_stats["mean"] = filter_stats_array(target_stats["mean"], INDICES_TO_REMOVE)
                        if "std" in target_stats:
                            target_stats["std"] = filter_stats_array(target_stats["std"], INDICES_TO_REMOVE)
                        
                        entry["stats"][TARGET_COLUMN] = target_stats
                    
                    # フィルタリング済みエントリを書き込み
                    target_file.write(json.dumps(entry) + "\n")
                    
                except json.JSONDecodeError as e:
                    print(f"JSONデコードエラー: {e}")
                    continue
                except Exception as e:
                    print(f"エントリ処理エラー: {e}")
                    continue
        
        print("  - episodes_stats.jsonlのコピー・フィルタリング完了")
        
    except Exception as e:
        print(f"エラー: episodes_stats.jsonl処理中に問題が発生しました: {e}")
        traceback.print_exc()

# --------------------------------------------------------------------------
# メインスクリプト
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # パスを設定
    source_data_path = os.path.join(SOURCE_PATH, "data", CHUNK_NAME)
    target_data_path = os.path.join(TARGET_PATH, "data", CHUNK_NAME)

    print("=" * 60)
    print("データセット変換処理を開始します")
    print(f"ソース: {SOURCE_PATH}")
    print(f"ターゲット: {TARGET_PATH}")
    print(f"変換対象カラム: {TARGET_COLUMN}")
    print(f"削除インデックス: {INDICES_TO_REMOVE}")
    print(f"画像統計対象カラム: {IMAGE_COLUMNS}")
    print("=" * 60)

    # ターゲットディレクトリをクリーンアップ
    if os.path.exists(TARGET_PATH):
        print(f"既存のターゲットディレクトリを削除: {TARGET_PATH}")
        shutil.rmtree(TARGET_PATH)

    # 新しいデータセットのディレクトリ構造を作成
    os.makedirs(os.path.join(TARGET_PATH, "meta"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_PATH, "videos"), exist_ok=True)
    os.makedirs(target_data_path, exist_ok=True)

    # parquetファイル一覧を取得
    print("parquetファイル一覧を取得中...")
    parquet_files = glob.glob(os.path.join(source_data_path, "*.parquet"))
    parquet_files.sort()
    print(f"処理対象ファイル数: {len(parquet_files)}")

    if len(parquet_files) == 0:
        print("エラー: parquetファイルが見つかりません")
        exit(1)

    # 並列処理のパラメータ
    max_workers = min(mp.cpu_count(), len(parquet_files), MAX_WORKERS)
    print(f"並列処理数: {max_workers}")

    # 並列処理でparquetファイルを処理
    print("並列処理でparquetファイルを処理中...")
    
    # 処理タスクを準備
    tasks = []
    for input_file in parquet_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(target_data_path, filename)
        tasks.append((input_file, output_file))
    
    successful_files = 0
    failed_files = 0
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 並列処理実行
            results = list(executor.map(process_single_parquet_file, tasks))
            
            # 結果を処理
            for result in results:
                if result['status'] == 'success':
                    successful_files += 1
                    if successful_files % 10 == 0:
                        print(f"    進捗: {successful_files}/{len(parquet_files)} ファイル完了")
                elif result['status'] == 'already_transformed':
                    print(f"スキップ: ファイル {result['file']} は既に変換済み")
                    successful_files += 1
                elif result['status'] == 'empty':
                    print(f"警告: ファイル {result['file']} は空でした。")
                    failed_files += 1
                elif result['status'] == 'unexpected_size':
                    print(f"警告: ファイル {result['file']} の次元数が予期しない値: {result['size']}")
                    failed_files += 1
                else:  # error
                    print(f"エラー: ファイル {result['file']} の処理中に問題が発生: {result.get('error', '不明なエラー')}")
                    failed_files += 1
                    
    except Exception as e:
        print(f"並列処理中にエラーが発生しました: {e}")
        traceback.print_exc()

    print(f"Parquetファイル保存完了: 成功 {successful_files}個, 失敗 {failed_files}個")

    # その他の関連ファイルをコピー
    print("関連ファイルをコピー中...")
    try:
        # videosディレクトリをコピー
        source_videos_path = os.path.join(SOURCE_PATH, "videos")
        target_videos_path = os.path.join(TARGET_PATH, "videos")
        if os.path.exists(source_videos_path):
            shutil.copytree(source_videos_path, target_videos_path, dirs_exist_ok=True)
            print("  - videosディレクトリのコピー完了")
        
        # metaファイルをコピー
        source_meta_path = os.path.join(SOURCE_PATH, "meta")
        target_meta_path = os.path.join(TARGET_PATH, "meta")
        for filename in META_FILES_TO_COPY:
            source_file = os.path.join(source_meta_path, filename)
            if os.path.exists(source_file):
                shutil.copy(source_file, target_meta_path)
                print(f"  - コピー完了: {filename}")
                
    except Exception as e:
        print(f"エラー: ファイルコピー中に問題が発生しました: {e}")
        traceback.print_exc()

    # info.jsonを更新
    print("info.jsonを更新中...")
    update_info_json(SOURCE_PATH, TARGET_PATH)

    # episodes_stats.jsonlをコピーしてフィルタリング
    copy_and_filter_episodes_stats(SOURCE_PATH, TARGET_PATH)

    print("=" * 60)
    print("すべての処理が完了しました。")
    print(f"新しいデータセットの場所: {TARGET_PATH}")
    print(f"成功したファイル数: {successful_files}")
    print(f"失敗したファイル数: {failed_files}")
    print("=" * 60)