# config.py
from dotenv import load_dotenv
import os

load_dotenv()

camera_ip1 = os.getenv("camera_ip1")
camera_ip2 = os.getenv("camera_ip2")
camera_port = os.getenv("camera_port")

WS_URI1 = f"ws://{camera_ip1}:{camera_port}/"
WS_URI2 = f"ws://{camera_ip2}:{camera_port}/"

vrchat_ip = os.getenv("vrchat_ip")
vrchat_port = os.getenv("vrchat_port")
