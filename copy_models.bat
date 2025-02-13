@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: サーバー情報
set "SERVER_USER=shinsakuo"
set "SERVER_ADDRESS=192.168.34.12"
set "SRC_DIR=/home/shinsakuo/workspace/lerobot_shin/outputs/train"
set "DST_DIR=C:\Users\harry\workspace\lerobot_shin\trainedmodel\sirius"

:: サーバー上のフォルダリストを取得
for /f %%F in ('ssh %SERVER_USER%@%SERVER_ADDRESS% "ls -1 %SRC_DIR%"') do (
    set "FOLDER_NAME=%%F"
    set "SRC_PATH=%SRC_DIR%/%%F/checkpoints/last/pretrained_model"
    set "DST_PATH=%DST_DIR%\%%F\pretrained_model"

    :: すでに存在する場合はスキップ
    if exist "!DST_PATH!" (
        echo スキップ: !FOLDER_NAME! はすでに存在します。
    ) else (
        :: ローカルのフォルダを作成（エラー防止）
        mkdir "!DST_PATH!"

        :: SCPでコピー
        echo コピー中: !FOLDER_NAME!
        scp -r %SERVER_USER%@%SERVER_ADDRESS%:"!SRC_PATH!" "!DST_PATH!"

        echo コピー完了: !FOLDER_NAME!
    )
)

echo すべてのデータをコピーしました！
pause
