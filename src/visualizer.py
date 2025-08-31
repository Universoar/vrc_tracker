import cv2
import numpy as np

def draw_skeleton(frame, keypoints=None):
    """
    左右分屏显示
    左边：绘制骨骼关键点
    右边：原始图像
    frame: 原始帧 (np.ndarray)
    keypoints: np.ndarray 或列表，形状 (N,2)
    """
    # 左图：复制原图，用于绘制骨骼
    left = frame.copy()
    if keypoints is not None:
        for x, y in keypoints:
            cv2.circle(left, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # 右图：原始帧
    right = frame.copy()
    
    # 确保左右图尺寸一致
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    
    # 水平拼接
    combined = cv2.hconcat([left, right])
    
    cv2.imshow("Skeleton (Left) | Original (Right)", combined)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
        return False
    return True