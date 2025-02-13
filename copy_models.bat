@echo off
set SERVER_USER=shinsakuo
set SERVER_ADDRESS=192.168.34.12
set SRC_DIR=/home/shinsakuo/workspace/lerobot_shin/outputs/train
set DST_DIR=C:\Users\harry\workspace\lerobot_shin\trainedmodel\sirius

:: 作業ディレクトリを一時的に変更
cd /d %DST_DIR%

:: SCPでファイルを取得
for /D %%F in (%SRC_DIR%\*) do (
    set "FOLDER_NAME=%%~nxF"
    set "SRC_PATH=%SRC_DIR%/%%F/checkpoints/last/pretrained_model"
    set "DST_PATH=%DST_DIR%\%%~nxF\pretrained_model"

    :: フォルダ作成
    mkdir "%DST_PATH%" 2>nul

    :: SCPでコピー
    scp -r %SERVER_USER%@%SERVER_ADDRESS%:%SRC_PATH% "%DST_PATH%"
)

echo 完了しました！
pause
