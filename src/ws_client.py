import asyncio
import websockets
import cv2
import numpy as np


async def recv_frame(uri):
    """
    异步生成器：只处理二进制图片帧
    """
    async with websockets.connect(uri) as ws:
        while True:
            try:
                data = await ws.recv()

                # 只处理 bytes 类型
                if not isinstance(data, (bytes, bytearray)):
                    continue  # 忽略字符串或其他类型

                img_array = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame

            except Exception as e:
                print("Error receiving frame:", e)
                await asyncio.sleep(0.01)
