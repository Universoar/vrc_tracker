# config.py
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 获取配置
camera_ip = os.getenv("camera_ip")
camera_port = os.getenv("camera_port")

WS_URI = f"ws://{camera_ip}:{camera_port}/"

vrchat_ip = os.getenv("vrchat_ip")
vrchat_port = os.getenv("vrchat_port")
