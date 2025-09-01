# main.py
import asyncio
from src.video_stream import recv_frame,recv_frame_from_camera
from src.visualizer import draw_full_skeleton
from src.pose_processor import get_full_skeleton
from src.osc_sender import SkeletonSender
from src.config import WS_URI, vrchat_ip, vrchat_port


async def main():
    sender = SkeletonSender(ip=vrchat_ip, port=vrchat_port)

    async for frame in recv_frame_from_camera(0):
        skeletons = get_full_skeleton(frame["frame"])
        if not draw_full_skeleton(frame["frame"], skeletons, frame["timestamp"], frame["fps"]):
            break

        # for sk in skeletons:
        #     sender.send_skeleton(sk)


if __name__ == "__main__":
    asyncio.run(main())
