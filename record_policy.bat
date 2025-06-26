chcp 65001 >nul  

set /p MODEL=Enter MODEL (pi0 / smolvla):
set /p STEP=Enter STEP *e4 (02 / 04 / 06 / 08 / 10): 
set /p SHAPE=Enter SHAPE (circle / octagonal / cross / square):

IF "%STEP%"=="" (
    echo [ERROR] STEP is required.
    exit /b 1
)
IF "%SHAPE%"=="" (
    echo [ERROR] SHAPE is required.
    exit /b 1
)

set TASK=""
IF "%SHAPE%"=="circle" (
    set TASK=put the cylinder block into the corresponding hole
) ELSE IF "%SHAPE%"=="octagonal" (
    set TASK=put the octagonal pillar block into the corresponding hole
) ELSE IF "%SHAPE%"=="cross" (
    set TASK=put the cross columnar block into the corresponding hole
) ELSE IF "%SHAPE%"=="square" (
    set TASK=put the square block into the corresponding hole
) ELSE (
    echo [ERROR] Invalid shape specified: %SHAPE%
    exit /b 1
)

set CUDA_VISIBLE_DEVICES=0

python -m lerobot.record ^
  --robot.type=koch_follower ^
  --robot.port=COM3 ^
  --robot.id=follower ^
  --robot.cameras="{\"front\": {\"type\": \"opencv\", \"index_or_path\": 0, \"width\": 640, \"height\": 480, \"fps\": 30}, \"top\": {\"type\": \"opencv\", \"index_or_path\": 1, \"width\": 640, \"height\": 480, \"fps\": 30}}" ^
  --dataset.single_task="%TASK%" ^
  --dataset.repo_id=shin1107/eval_koch_base_%MODEL%_pretrained_%STEP%0000_%SHAPE% ^
  --dataset.episode_time_s=30 ^
  --dataset.num_episodes=20 ^
  --teleop.type=koch_leader ^
  --teleop.port=COM4 ^
  --teleop.id=leader ^
  --dataset.push_to_hub=false ^
  --policy.path=trainedmodel/models/koch_base_%MODEL%_pretrained/%STEP%0000/pretrained_model ^
  --display_data=true
