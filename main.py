# main.py
import asyncio
from src.ws_client import recv_frame
from src.visualizer import draw_skeleton
from src.pose_processor import get_keypoints
from src.config import WS_URI

async def main():
    async for frame in recv_frame(WS_URI):
        keypoints_list = get_keypoints(frame)
        if not draw_skeleton(frame, keypoints_list):
            break

if __name__ == "__main__":
    asyncio.run(main())