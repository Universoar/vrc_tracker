# main.py
import asyncio
import cv2

from src.video_stream import recv_frame, recv_frame_from_camera
from src.visualizer import PanelVisualizer
from src.pose_processor import get_full_skeleton
from src.config import WS_URI1, WS_URI2, osc_ip, osc_port
import src.kalman as kalman
from src.osc_sender import SkeletonSender
from src.cal_real_pose import triangulate

skeleton_sender = SkeletonSender(ip=osc_ip, port=osc_port)

fps = 8
body_kf1 = kalman.create_kalman_filters(17)
body_kf2 = kalman.create_kalman_filters(17)

async def ws1_worker(latest_frames):
    async for frame in recv_frame(WS_URI1, 90):
        frame["skeletons"] = get_full_skeleton(frame["frame"], body_kf1)
        latest_frames["ws1"] = frame

async def ws2_worker(latest_frames):
    async for frame in recv_frame(WS_URI2, 270):
        frame["skeletons"] = get_full_skeleton(frame["frame"], body_kf2)
        latest_frames["ws2"] = frame

async def main():
    vis = PanelVisualizer()
    latest_frames = {"ws1": None, "ws2": None}

    ws1_task = asyncio.create_task(ws1_worker(latest_frames))
    ws2_task = asyncio.create_task(ws2_worker(latest_frames))

    draw_interval = 1 / fps

    try:
        while True:
            await asyncio.sleep(draw_interval)

            if latest_frames["ws1"] is None or latest_frames["ws2"] is None:
                continue

            vis.draw([latest_frames["ws1"], latest_frames["ws2"]])

            ws1_skeletons = latest_frames['ws1'].get('skeletons', [])
            ws2_skeletons = latest_frames['ws2'].get('skeletons', [])
            if not ws1_skeletons or not ws2_skeletons:
                continue

            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            result = triangulate(
                ws1_skeletons[0]['body'], 
                ws2_skeletons[0]['body'],
                user_height=1.8
            )
            
            skeleton_sender.send_skeleton(result)

    finally:
        ws1_task.cancel()
        ws2_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())