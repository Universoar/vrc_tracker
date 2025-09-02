from pythonosc import udp_client
import numpy as np


class SkeletonSender:
    def __init__(self, ip="127.0.0.1", port=9000):
        self.client = udp_client.SimpleUDPClient(ip, port)

    def send_skeleton(self, skeleton):
        head_pos = skeleton[0].tolist()
        head_rot = [0.0, 0.0, 0.0]
        self.client.send_message("/tracking/trackers/head/position", head_pos)
        self.client.send_message("/tracking/trackers/head/rotation", head_rot)

        trackers = {
            1: [0, 8],  # Hip -> nose & pelvis
            2: [0, 1],  # Chest -> nose & neck
            3: [11],    # Left Foot
            4: [14],    # Right Foot
            5: [10],    # Left Knee
            6: [13],    # Right Knee
            7: [6],     # Left Elbow
            8: [3],     # Right Elbow
        }

        for i, idxs in trackers.items():
            points = [skeleton[idx] for idx in idxs]
            pos = np.mean(points, axis=0).tolist()
            rot = [0.0, 0.0, 0.0]
            self.client.send_message(f"/tracking/trackers/{i}/position", pos)
            self.client.send_message(f"/tracking/trackers/{i}/rotation", rot)
            
    def test_send(self, data):
        """
        data: np.array 或 list，形状 (17,3)
        直接发送原始骨架数据，展平成一维列表
        """
        # 确保是列表
        if hasattr(data, "tolist"):
            data_list = data.tolist()
        else:
            data_list = [list(p) for p in data]

        # 展平
        flat = [x for p in data_list for x in p]

        # 直接发送到 /skeleton/full
        self.client.send_message("/skeleton/full", flat)
        print(f"Sent skeleton, {len(flat)} floats")