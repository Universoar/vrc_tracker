import asyncio
import websockets
import time

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

async def recv_frame_from_camera(camera_index=0):
    """
    异步生成器：从本地摄像头读取帧
    返回:
        {
            "data": frame,        # 图像帧 (numpy.ndarray)
            "timestamp": float    # Unix 时间戳（秒）
        }
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    else:
        print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("FPS:", cap.get(cv2.CAP_PROP_FPS))
        print("FourCC:", cap.get(cv2.CAP_PROP_FOURCC))
        print("Backend:", cap.getBackendName())
                

    loop = asyncio.get_running_loop()
    prev_time = time.time()
    
    while True:
        try:
            # 因为cv2.VideoCapture.read是阻塞的，我们用线程池异步化
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                await asyncio.sleep(0.01)
                continue

            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = curr_time

            yield {
                "frame": frame,
                "timestamp": curr_time,
                "fps": fps
            }

        except Exception as e:
            print("Error reading camera frame:", e)
            await asyncio.sleep(0.01)

    cap.release()