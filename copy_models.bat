@echo off
:: UTF-8モードに設定
chcp 65001 >nul
setlocal enabledelayedexpansion

:: サーバー情報
set "SERVER_USER=shinsakuo"
set "SERVER_ADDRESS=192.168.34.12"
set "SRC_BASE_DIR=/home/shinsakuo/workspace/lerobot_shin/data3/train/new"
set "DST_DIR=C:\Users\harry\workspace\lerobot_shin\trainedmodel\models"
set "TEMP_FILE=_temp_folder_list.txt"

echo スクリプト開始...
echo サーバー上のフォルダリストを取得中...

:: サーバー上のフォルダリストを取得 (パスワード入力が必要です)
for /f "delims=" %%F in ('ssh %SERVER_USER%@%SERVER_ADDRESS% "ls -1 %SRC_BASE_DIR%"') do (
    set "FOLDER_NAME=%%F"
    for /f "delims=" %%A in ("!FOLDER_NAME!") do set "FOLDER_NAME=%%A"

    if not "!FOLDER_NAME!"=="" (
        set "CHECKPOINTS_DIR=%SRC_BASE_DIR%/!FOLDER_NAME!/checkpoints"
        echo(
        echo **************************************************
        echo 処理中のモデルフォルダ: !FOLDER_NAME!
        echo **************************************************

        echo   '!CHECKPOINTS_DIR!' の 'last' 以外のフォルダを確認します - パスワード入力
        :: ssh側で grep -v '^last$' を使い、'last' を除外し、CR除去
        ssh %SERVER_USER%@%SERVER_ADDRESS% "ls -1 !CHECKPOINTS_DIR! 2>/dev/null | grep -v '^last$' | tr -d '\r'" > "!TEMP_FILE!" 2>&1

        if exist "!TEMP_FILE!" (
            echo   サーバーからの応答開始
            type "!TEMP_FILE!"
            echo   サーバーからの応答終了

            set "processed=0"
            :: 一時ファイルを読み込んで処理
            for /f "usebackq delims=" %%N in ("!TEMP_FILE!") do (
                set "CHECKPOINT_FOLDER=%%N"
                :: CR除去 (念のため)
                for /f "delims=" %%B in ("!CHECKPOINT_FOLDER!") do set "CHECKPOINT_FOLDER=%%B"

                if not "!CHECKPOINT_FOLDER!"=="" (
                    echo     [処理] フォルダ: !CHECKPOINT_FOLDER!
                    set "processed=1"
                    set "SRC_PATH=!CHECKPOINTS_DIR!/!CHECKPOINT_FOLDER!/pretrained_model"
                    set "DST_PATH=%DST_DIR%\!FOLDER_NAME!\!CHECKPOINT_FOLDER!\pretrained_model"

                    :: pretrained_model が存在するかをサーバーで確認 (パスワード入力)
                    ssh %SERVER_USER%@%SERVER_ADDRESS% "test -d !SRC_PATH!"
                    if not errorlevel 1 (
                        if exist "!DST_PATH!\" (
                            echo       スキップ: すでに存在します
                        ) else (
                            mkdir "!DST_PATH!"
                            if not errorlevel 1 (
                                echo       コピー中 - パスワード入力...
                                scp -r %SERVER_USER%@%SERVER_ADDRESS%:"!SRC_PATH!/." "!DST_PATH!"
                                if errorlevel 1 (
                                    echo       ■■■ エラー: scp 失敗 ■■■
                                ) else (
                                    echo       コピー完了
                                )
                            ) else (
                                echo       ■■■ エラー: mkdir 失敗 ■■■
                            )
                        )
                    ) else (
                        echo       [警告] '!SRC_PATH!' が存在しません - スキップします
                    )
                )
            )

            if !processed!==0 (
                echo   [情報] 対象となるフォルダが見つかりませんでした
            )

            del "!TEMP_FILE!" 2>nul
        ) else (
            echo   [情報] '!CHECKPOINTS_DIR!' のリスト取得に失敗 - ssh失敗の可能性
        )
    )
)

echo(
echo ==================================
echo すべての処理が完了しました！
echo ==================================
pause