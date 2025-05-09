import cv2
import time
import threading

def test_camera(cam_id, backend=cv2.CAP_DSHOW, num_frames=10):
    print(f"\n=== Starting Camera {cam_id} ===")
    cap = cv2.VideoCapture(cam_id, backend)

    if not cap.isOpened():
        print(f"Camera {cam_id} could not be opened.")
        return

    for i in range(num_frames):
        start = time.time()
        ret, frame = cap.read()
        end = time.time()
        duration = (end - start) * 1000  # ms

        status = "Success" if ret else "Fail"
        print(f"Camera {cam_id} - Frame {i}: {status} - Time: {duration:.2f} ms")

    cap.release()
    print(f"=== Finished Camera {cam_id} ===")

# 実行部（同時並列処理）
if __name__ == "__main__":
    thread0 = threading.Thread(target=test_camera, args=(0,))
    thread1 = threading.Thread(target=test_camera, args=(1,))

    thread0.start()
    thread1.start()

    thread0.join()
    thread1.join()
