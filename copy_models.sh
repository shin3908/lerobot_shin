#!/bin/bash

# このスクリプトは、trainフォルダ内の各サブフォルダにあるpretrained_modelフォルダをtrainedmodel/siriusにコピーします。
# 元のディレクトリ
SRC_DIR="/home/shinsakuo/workspace/lerobot_shin/outputs/train"
# コピー先のベースディレクトリ
DST_DIR="/home/shinsakuo/workspace/lerobot_shin/trainedmodel/sirius"

# trainフォルダ内の各サブフォルダを処理
for train_subdir in "$SRC_DIR"/*; do
    # サブフォルダがディレクトリか確認
    if [ -d "$train_subdir" ]; then
        # フォルダ名を取得
        folder_name=$(basename "$train_subdir")
        
        # コピー元のパス
        SRC_PATH="$train_subdir/checkpoints/last/pretrained_model"
        
        # コピー先のパス
        DST_PATH="$DST_DIR/$folder_name/pretrained_model"
        
        # コピー元が存在するか確認
        if [ -d "$SRC_PATH" ]; then
            # 事前にディレクトリを作成
            mkdir -p "$DST_PATH"

            # rsyncを使用してコピー（既存ファイルは無視）
            rsync -av --ignore-existing "$SRC_PATH/" "$DST_PATH/"
            echo "Copied $SRC_PATH to $DST_PATH"
        else
            echo "Skipped $folder_name (pretrained_model not found)"
        fi
    fi
done
