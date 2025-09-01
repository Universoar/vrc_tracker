import asyncio
import websockets
import time

import cv2
import numpy as np

def rotate_frame(frame, angle: int):
    angle = angle % 360
    if angle == 0:
        return frame
    elif angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError("angle must be 0, 90, 180, 270")

async def recv_frame(uri, rotation=0):
    async with websockets.connect(uri) as ws:
        prev_time = time.time()

        while True:
            try:
                data = await ws.recv()

                if not isinstance(data, (bytes, bytearray)):
                    continue

                img_array = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                curr_time = time.time()
                dt = curr_time - prev_time
                fps = 1.0 / dt if dt > 0 else 0.0
                prev_time = curr_time

                yield {
                    "frame": rotate_frame(frame, rotation),
                    "timestamp": curr_time,
                    "fps": fps
                }

            except Exception as e:
                print("Error receiving frame:", e)
                await asyncio.sleep(0.01)

async def recv_frame_from_camera(camera_index=0, target_fps=30):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    loop = asyncio.get_running_loop()
    frame_interval = 1.0 / target_fps
    prev_time = time.time()

    while True:
        start_time = time.time()

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

        elapsed = time.time() - start_time
        await asyncio.sleep(max(0, frame_interval - elapsed))