@echo off
REM Simplified Training Script for Windows - Config File Only Approach
REM All parameters are defined in YAML files, no CLI arguments needed

echo === LeRobot Training with Configuration Files Only ===
echo This script uses only --config-path, no other CLI arguments needed.
echo.

REM ==============================================================================
REM AVAILABLE CONFIGURATIONS:
REM ==============================================================================
echo === Available Training Configurations ===
echo.

echo 1. SmolVLA Configurations:
echo    - smolvla_complete_all_features.yaml: SmolVLA with all observation features
echo    - smolvla_complete_pos_only.yaml: SmolVLA with position-only filtering
echo    - smolvla_from_scratch.yaml: SmolVLA training from scratch
echo.

echo 2. Pi0 Configurations:
echo    - pi0_complete_pos_only.yaml: Pi0 with position-only filtering
echo    - pi0_original_episodes.yaml: Pi0 fine-tuning on original episodes dataset
echo.

echo 3. ACT Configurations:
echo    - act_resnet18_complete.yaml: ACT with ResNet18 backbone
echo    - act_resnet50_complete.yaml: ACT with ResNet50 backbone
echo.

echo 4. Pi0Fast Configuration:
echo    - pi0fast_original_episodes.yaml: Pi0Fast fine-tuning
echo.

REM ==============================================================================
REM USAGE EXAMPLES FOR WINDOWS:
REM ==============================================================================
echo === Usage Examples for Windows ===
echo.

echo 1. SmolVLA with all features (recommended for new datasets):
echo    set CUDA_VISIBLE_DEVICES=0
echo    python lerobot/scripts/train.py --config-path=configs/smolvla_complete_all_features.yaml
echo.

echo 2. SmolVLA with position-only filtering:
echo    set CUDA_VISIBLE_DEVICES=0
echo    python lerobot/scripts/train.py --config-path=configs/smolvla_complete_pos_only.yaml
echo.

echo 3. Pi0 with position-only filtering:
echo    set CUDA_VISIBLE_DEVICES=1
echo    python lerobot/scripts/train.py --config-path=configs/pi0_complete_pos_only.yaml
echo.

echo 4. ACT with ResNet18:
echo    set CUDA_VISIBLE_DEVICES=1
echo    set TMPDIR=/data3/shinsakuo/tmp
echo    python lerobot/scripts/train.py --config-path=configs/act_resnet18_complete.yaml
echo.

REM ==============================================================================
REM DIRECT EXECUTION COMMANDS FOR WINDOWS:
REM ==============================================================================
echo === Ready-to-Run Commands for Windows ===
echo.

echo REM SmolVLA with all features
echo set CUDA_VISIBLE_DEVICES=0 ^&^& python lerobot/scripts/train.py --config-path=configs/smolvla_complete_all_features.yaml
echo.

echo REM SmolVLA with position filtering
echo set CUDA_VISIBLE_DEVICES=0 ^&^& python lerobot/scripts/train.py --config-path=configs/smolvla_complete_pos_only.yaml
echo.

echo REM Pi0 with position filtering
echo set CUDA_VISIBLE_DEVICES=1 ^&^& python lerobot/scripts/train.py --config-path=configs/pi0_complete_pos_only.yaml
echo.

REM ==============================================================================
REM CONFIGURATION CUSTOMIZATION:
REM ==============================================================================
echo === Configuration Customization ===
echo.

echo To modify training parameters, edit the YAML files directly:
echo - Batch size: batch_size: 64
echo - Training steps: steps: 100000
echo - Output directory: output_dir: "your/path/here"
echo - Compression method: dimension_compression_method: "pca"
echo - Observation filtering: observation_state_filter: ["pos", "current"]
echo.

echo === All CLI arguments replaced with config files! ===
echo Choose a configuration above and run with --config-path only.

pause
