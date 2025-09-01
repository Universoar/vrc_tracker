# src/visualizer.py
import cv2
import numpy as np

# COCO skeleton
COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
]

def draw_full_skeleton(frame, skeletons, timestamp=None, fps=None):
    left = np.zeros_like(frame)  # 黑底
    right = frame.copy()

    if timestamp is not None:
        cv2.putText(
            right,
            f"Timestamp: {timestamp:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    if fps is not None:
        cv2.putText(
            right,
            f"fps: {fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    for sk in skeletons:
        body = sk["body"]
        left_hand = sk["left_hand"]
        right_hand = sk["right_hand"]

        # 画人体骨架
        body_int = [tuple(map(int, p)) for p in body]
        for i, j in COCO_SKELETON:
            if i < len(body) and j < len(body):
                cv2.line(
                    left,
                    body_int[i],
                    body_int[j],
                    (0, 255, 0),
                    2,
                )
        for x, y in body_int:
            cv2.circle(left, (x, y), 5, (0, 255, 0), -1)

        # 画手部骨架
        for hand_points in [left_hand, right_hand]:
            hand_int = [tuple(map(int, p)) for p in hand_points]
            for idx in range(len(hand_int) - 1):
                cv2.line(left, hand_int[idx], hand_int[idx + 1], (0, 255, 255), 1)
            for x, y in hand_int:
                cv2.circle(left, (x, y), 3, (0, 255, 255), -1)

    combined = cv2.hconcat([left, right])
    cv2.imshow("Skeleton+Hands (Left) | Original (Right)", combined)
    if cv2.waitKey(1) & 0xFF == 27:
        return False
    return True
