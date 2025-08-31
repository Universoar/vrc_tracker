# src/pose_with_hands.py
from ultralytics import YOLO
import mediapipe as mp
import cv2
import numpy as np

# YOLOv8 Pose
pose_model = YOLO("yolov8n-pose.pt")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(static_image_mode=False,
                             max_num_hands=2,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def get_full_skeleton(frame):
    results = pose_model.predict(frame, verbose=False)
    skeletons = []

    h, w, _ = frame.shape

    for r in results:
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        keypoints = r.keypoints.xy[0].cpu().numpy()  # (17,2)
        hands_kp = []

        for wrist_idx in [9, 10]:  # 左手/右手
            x, y = keypoints[wrist_idx]

            # 正方形裁剪
            size = 300  # 可以调节
            cx, cy = int(x), int(y)
            x1 = max(cx - size//2, 0)
            y1 = max(cy - size//2, 0)
            x2 = min(cx + size//2, w)
            y2 = min(cy + size//2, h)
            hand_crop = frame[y1:y2, x1:x2]

            # 转 RGB
            hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            hand_results = hands_model.process(hand_rgb)

            hand_points = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        # 按正方形映射回原图
                        hx = int(lm.x * (x2 - x1)) + x1
                        hy = int(lm.y * (y2 - y1)) + y1
                        hand_points.append((hx, hy))
            hands_kp.append(hand_points)

        skeletons.append({
            "body": keypoints,
            "left_hand": hands_kp[0],
            "right_hand": hands_kp[1]
        })

    return skeletons
