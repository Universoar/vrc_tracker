# main.py
import asyncio

from src.ws_client import recv_frame
from src.visualizer import draw_skeleton

from src.config import WS_URI

async def main():
    async for frame in recv_frame(WS_URI):
        # 暂时没有骨骼点，先只显示原始视频
        if not draw_skeleton(frame):
            break

if __name__ == "__main__":
    asyncio.run(main())