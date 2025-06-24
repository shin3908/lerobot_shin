python -m lerobot.teleoperate ^
    --robot.type=koch_follower ^
    --robot.port=COM3 ^
    --robot.id=follower ^
    --robot.cameras="{\"front\": {\"type\": \"opencv\", \"index_or_path\": 0, \"width\": 640, \"height\": 480, \"fps\": 30}, \"top\": {\"type\": \"opencv\", \"index_or_path\": 1, \"width\": 640, \"height\": 480, \"fps\": 30}}" ^
    --teleop.type=koch_leader ^
    --teleop.port=COM4 ^
    --teleop.id=leader ^
    --display_data=true