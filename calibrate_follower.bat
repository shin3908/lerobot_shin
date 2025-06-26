REM     --robot.calibration_dir=calibration/new

python -m lerobot.calibrate ^
    --robot.type=koch_follower ^
    --robot.port=COM3 ^
    --robot.id=follower ^
    --robot.use_degrees=True
