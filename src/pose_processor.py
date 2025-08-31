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
    """
    返回每个人的骨骼关键点和手部关键点
    """
    results = pose_model.predict(frame, verbose=False)
    skeletons = []

    for r in results:
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        keypoints = r.keypoints.xy[0].cpu().numpy()  # shape (17,2)
        
        # 手部关键点
        hands_kp = []
        for wrist_idx in [9, 10]:  # 左手/右手手腕
            x, y = keypoints[wrist_idx]
            h, w, _ = frame.shape
            # 裁剪手区域（简单固定大小，可调）
            x1, y1 = max(int(x)-50,0), max(int(y)-50,0)
            x2, y2 = min(int(x)+50,w), min(int(y)+50,h)
            hand_crop = frame[y1:y2, x1:x2]
            hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            hand_results = hands_model.process(hand_rgb)
            
            hand_points = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        # 转回原图坐标
                        hx = int(lm.x*(x2-x1)) + x1
                        hy = int(lm.y*(y2-y1)) + y1
                        hand_points.append((hx, hy))
            hands_kp.append(hand_points)
        
        skeletons.append({
            "body": keypoints,
            "hands": hands_kp
        })
    return skeletons
