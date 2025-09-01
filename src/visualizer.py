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


def draw_full_skeleton(frame, skeletons, dataset=None):
    left = np.zeros_like(frame)  # 黑底
    right = frame.copy()

    # {
    #     fps: xx,
    #     xx: xx,
    # }
    if dataset is not None:
        hpos = 30
        for name, value in dataset.items():
            cv2.putText(
                right,
                f"{name}: {value}",
                (10, hpos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            hpos += 30

    for sk in skeletons:
        body = sk["body"]
        # left_hand = sk["left_hand"]
        # right_hand = sk["right_hand"]

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
        # for hand_points in [left_hand, right_hand]:
        #     hand_int = [tuple(map(int, p)) for p in hand_points]
        #     for idx in range(len(hand_int) - 1):
        #         cv2.line(left, hand_int[idx], hand_int[idx + 1], (0, 255, 255), 1)
        #     for x, y in hand_int:
        #         cv2.circle(left, (x, y), 3, (0, 255, 255), -1)

    return cv2.hconcat([left, right])


class PanelVisualizer:
    def __init__(self):
        self.panel = None
        self.row_heights = []
        self.width = None
        self.num_sources = 0

    def draw(self, datas):
        if self.panel is None:
            self.num_sources = len(datas)
            if self.num_sources == 0:
                return
            first_frame = draw_full_skeleton(
                datas[0]["frame"],
                datas[0].get("skeletons", [])
            )
            self.height, self.width = first_frame.shape[:2]
            self.row_heights = [self.height] * self.num_sources
            self.panel = np.zeros((self.height * self.num_sources, self.width, 3), dtype=np.uint8)

        for i, data in enumerate(datas):
            if i >= self.num_sources:
                break

            frame = data["frame"]
            timestamp = data.get("timestamp", None)
            fps = data.get("fps", None)
            skeletons = data.get("skeletons", [])

            drawn_frame = draw_full_skeleton(frame, skeletons, {"timestamp": f"{timestamp:.2f}", "fps": f"{fps:.2f}", "size": f"{frame.shape[1]}x{frame.shape[0]}"})

            drawn_frame = cv2.resize(drawn_frame, (self.width, self.row_heights[i]))

            start_y = sum(self.row_heights[:i])
            end_y = start_y + self.row_heights[i]
            self.panel[start_y:end_y, :, :] = drawn_frame

        cv2.imshow("Skeleton Panel", self.panel)