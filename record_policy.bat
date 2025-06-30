chcp 65001 >nul  

set /p MODEL=Enter MODEL (pi0 / smolvla):
set /p STEP=Enter STEP *1e4 (02 / 04 / 06 / 08 / 10): 
set /p HOLE=Enter HOLE_positon (1 / 2 / 12 / 13 / ...):
set /p EPISODES=Enter number of EPISODES (20 / 40 / 60 / 80):

IF "%STEP%"=="" (
    echo [ERROR] STEP is required.
    exit /b 1
)

IF "%MODEL%"=="" (
    echo [ERROR] MODEL is required.
    exit /b 1
)

IF "%HOLE%"=="" (
    echo [ERROR] HOLE is required.
    exit /b 1
)

IF "%EPISODES%"=="" (
    echo [ERROR] EPISODES is required.
    exit /b 1
)

set CUDA_VISIBLE_DEVICES=0

python -m lerobot.record ^
  --robot.type=koch_follower ^
  --robot.port=COM3 ^
  --robot.id=follower ^
  --robot.cameras="{\"front\": {\"type\": \"opencv\", \"index_or_path\": 0, \"width\": 640, \"height\": 480, \"fps\": 30}, \"top\": {\"type\": \"opencv\", \"index_or_path\": 1, \"width\": 640, \"height\": 480, \"fps\": 30}}" ^
  --dataset.repo_id=shin1107/eval_koch_base_%MODEL%_%STEP%0000_Pos%HOLE% ^
  --dataset.num_episodes=%EPISODES% ^
  --teleop.type=koch_leader ^
  --teleop.port=COM4 ^
  --teleop.id=leader ^
  --dataset.push_to_hub=false ^
  --policy.path=trainedmodel/models/koch_base_%MODEL%/%STEP%0000/pretrained_model ^
  --display_data=true
