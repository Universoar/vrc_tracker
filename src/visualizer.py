# src/visualizer.py
import cv2
import numpy as np

# COCO keypoints 连线关系 (17关键点)
COCO_SKELETON = [
    (0, 1), (0, 2),         # 鼻子 → 左眼/右眼
    (1, 3), (2, 4),         # 左眼→左耳, 右眼→右耳
    (0, 5), (0, 6),         # 鼻子→左右肩膀
    (5, 7), (7, 9),         # 左肩→左肘→左手腕
    (6, 8), (8,10),         # 右肩→右肘→右手腕
    (5,11), (6,12),         # 左肩→左髋, 右肩→右髋
    (11,13), (13,15),       # 左髋→左膝→左踝
    (12,14), (14,16),       # 右髋→右膝→右踝
    (11,12)                 # 左髋→右髋
]

def draw_skeleton(frame, keypoints_list=None):
    """
    左右分屏显示
    左边: 只绘制骨骼关键点和连线
    右边: 原始图像
    keypoints_list: list of np.ndarray (N,2)
    """
    # 左图：复制原图，用于绘制骨骼
    left = frame.copy()
    right = frame.copy()

    if keypoints_list is not None:
        for keypoints in keypoints_list:
            # 绘制关键点
            for x, y in keypoints:
                cv2.circle(left, (int(x), int(y)), 5, (0, 255, 0), -1)
            # 绘制连线
            for i, j in COCO_SKELETON:
                if i < len(keypoints) and j < len(keypoints):
                    x1, y1 = keypoints[i]
                    x2, y2 = keypoints[j]
                    cv2.line(left, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # 确保左右图尺寸一致
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))

    combined = cv2.hconcat([left, right])
    cv2.imshow("Skeleton (Left) | Original (Right)", combined)
    if cv2.waitKey(1) & 0xFF == 27:
        return False
    return True
