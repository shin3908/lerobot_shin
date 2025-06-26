REM === 引数チェック ===
IF "%~1"=="" (
    echo [ERROR] ステップ数 (例: 100000) を引数として指定してください。
    exit /b 1
)
IF "%~2"=="" (
    echo [ERROR] 形状 (circle / octagonal / cross / square) を引数として指定してください。
    exit /b 1
)

REM === 変数定義 ===
set STEP=%~1
set SHAPE=%~2

REM === SHAPEに応じたタスク文設定 ===
set TASK=""
IF "%SHAPE%"=="circle" (
    set TASK=put the cylinder block into the corresponding SHAPE
) ELSE IF "%SHAPE%"=="octagonal" (
    set TASK=put the octagonal pillar block into the corresponding SHAPE
) ELSE IF "%SHAPE%"=="cross" (
    set TASK=put the cross columnar block into the corresponding SHAPE
) ELSE IF "%SHAPE%"=="square" (
    set TASK=put the square block into the corresponding SHAPE
) ELSE (
    echo [ERROR] 無効な形状が指定されました: %SHAPE%
    exit /b 1
)

REM === GPU設定 ===
set CUDA_VISIBLE_DEVICES=0

REM === 実行コマンド ===
python -m lerobot.record ^
  --robot.type=koch_follower ^
  --robot.port=COM3 ^
  --robot.id=follower ^
  --robot.cameras="{\"front\": {\"type\": \"opencv\", \"index_or_path\": 0, \"width\": 640, \"height\": 480, \"fps\": 30}, \"top\": {\"type\": \"opencv\", \"index_or_path\": 1, \"width\": 640, \"height\": 480, \"fps\": 30}}" ^
  --dataset.single_task="%TASK%" ^
  --dataset.repo_id=shin1107/eval_koch_base_smolvla_pretrained_%STEP%_%SHAPE% ^
  --dataset.episode_time_s=30 ^
  --dataset.num_episodes=20 ^
  --teleop.type=koch_leader ^
  --teleop.port=COM4 ^
  --teleop.id=leader ^
  --policy.path=trainedmodel/models/koch_base_smolvla_pretrained/%STEP%/pretrained_model
