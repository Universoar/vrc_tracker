from pythonosc import udp_client
import numpy as np


class SkeletonSender:
    def __init__(self, ip="127.0.0.1", port=9000):
        self.client = udp_client.SimpleUDPClient(ip, port)

    def send_skeleton(self, skeleton):
        body = skeleton["body"]
        left_hand = skeleton["left_hand"]
        right_hand = skeleton["right_hand"]

        head_point = body[0]  # nose/top of head
        user_height_m = 1.7
        pixel_height = abs(body[0][1] - body[8][1])  # 头顶到骨盆
        scale = user_height_m / pixel_height

        def to_3d_rel(point):
            x, y = point
            hx, hy = head_point
            return [(x - hx) * scale, (y - hy) * scale, 0.0]

        # 发送头部 tracker
        self.client.send_message("/tracking/trackers/head/position", [0.0, 0.0, 0.0])
        self.client.send_message("/tracking/trackers/head/rotation", [0.0, 0.0, 0.0])

        # Tracker 映射：Hip、Chest、Feet、Knees、Elbows
        trackers = {
            1: [0, 8],  # Hip
            2: [0, 1],  # Chest
            3: [11],  # Left Foot
            4: [14],  # Right Foot
            5: [10],  # Left Knee
            6: [13],  # Right Knee
            7: [6],  # Left Elbow
            8: [3],  # Right Elbow
        }

        for i, idxs in trackers.items():
            points = [to_3d_rel(body[idx]) for idx in idxs]
            pos = np.mean(points, axis=0).tolist()
            rot = [0.0, 0.0, 0.0]
            self.client.send_message(f"/tracking/trackers/{i}/position", pos)
            self.client.send_message(f"/tracking/trackers/{i}/rotation", rot)
