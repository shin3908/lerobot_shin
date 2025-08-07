@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: サーバー情報
set "SERVER_USER=shinsakuo"
set "SERVER_ADDRESS=192.168.34.12"
set "SRC_BASE_DIR=C:\Users\harry\workspace\lerobot_shin\dataset"
set "DST_BASE_DIR=/home/shinsakuo/workspace/lerobot_shin/data3/train/dataset"

echo === コピー開始 ===
echo コピー元: %SRC_BASE_DIR%
echo コピー先: %SERVER_USER%@%SERVER_ADDRESS%:%DST_BASE_DIR%

:: フォルダ列挙
for /d %%F in ("%SRC_BASE_DIR%\*") do (
    set "FOLDER_PATH=%%F"
    set "FOLDER_NAME=%%~nxF"

    rem 「setlocal enabledelayedexpansion」を使っているので、中で遅延展開
    for /f "delims=" %%A in ("!FOLDER_NAME!") do set "FOLDER_NAME=%%A"

    if not "!FOLDER_NAME!"=="" (
        echo(
        echo **************************************************
        echo 処理中のフォルダ: !FOLDER_NAME!
        echo **************************************************

        echo   サーバー上に !FOLDER_NAME! が存在するか確認します
        ssh %SERVER_USER%@%SERVER_ADDRESS% "test -d '%DST_BASE_DIR%/!FOLDER_NAME!'"
        if errorlevel 1 (
            echo     コピー中 - パスワード入力が必要な場合があります...
            scp -r "!FOLDER_PATH!" %SERVER_USER%@%SERVER_ADDRESS%:"%DST_BASE_DIR%/"
            if errorlevel 1 (
                echo       ■■■ エラー: scp に失敗しました ■■■
            ) else (
                echo       コピー完了
            )
        ) else (
            echo     スキップ: サーバーにすでに存在します
        )
    )
)

echo(
echo ==================================
echo すべての処理が完了しました！
echo ==================================
pause
