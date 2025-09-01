# src/pose_with_hands.py
from ultralytics import YOLO
import mediapipe as mp
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque

# YOLOv8 Pose
pose_model = YOLO("yolov8n-pose.pt")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

class PointKalman:
    def __init__(self, history_len=1):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0

        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000.
        self.kf.R *= 5.
        self.kf.Q *= 0.01

        # 历史队列，用于滑动平均
        self.history = deque(maxlen=history_len)

    def update(self, x, y):
        z = np.array([x, y])
        self.kf.predict()
        self.kf.update(z)
        sx, sy = self.kf.x[0], self.kf.x[1]
        
        # 记录历史
        self.history.append((sx, sy))
        # 返回历史平均
        hx = np.mean([p[0] for p in self.history])
        hy = np.mean([p[1] for p in self.history])
        return hx, hy
    
# 初始化每个点的卡尔曼
def create_kalman_filters(num_points):
    return [PointKalman() for _ in range(num_points)]

# 应用到关键点
def kalman_smooth(points, kalman_list):
    smoothed = []
    for (x, y), kf in zip(points, kalman_list):
        sx, sy = kf.update(x, y)
        smoothed.append((sx, sy))
    return np.array(smoothed)

body_kf = create_kalman_filters(17)
left_hand_kf = create_kalman_filters(21)
right_hand_kf = create_kalman_filters(21)

def get_full_skeleton(frame):
    results = pose_model.predict(frame, verbose=False)
    skeletons = []

    h, w, _ = frame.shape

    for r in results:
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        keypoints = r.keypoints.xy[0].cpu().numpy()  # (17,2)
        keypoints = kalman_smooth(keypoints, body_kf)
        hands_kp = []

        for wrist_idx in [9, 10]:  # 左手/右手
            x, y = keypoints[wrist_idx]

            # 正方形裁剪
            size = 300  # 可以调节
            cx, cy = int(x), int(y)
            x1 = max(cx - size // 2, 0)
            y1 = max(cy - size // 2, 0)
            x2 = min(cx + size // 2, w)
            y2 = min(cy + size // 2, h)
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
            
            try:
                smoothed_left = kalman_smooth(hands_kp[0], left_hand_kf)
                smoothed_right = kalman_smooth(hands_kp[1], right_hand_kf)
                hands_kp = [smoothed_left, smoothed_right]
            except Exception:
                pass    

        skeletons.append(
            {"body": keypoints, "left_hand": hands_kp[0], "right_hand": hands_kp[1]}
        )

    return skeletons
