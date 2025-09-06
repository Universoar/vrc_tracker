import numpy as np
import math

CAM1_POS = np.array([0.0, 0.0, 1.2])
CAM2_POS = np.array([0.0, 0.13, 1.2])

CAM_ROT = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype=float)
CAM1_ROT = CAM_ROT
CAM2_ROT = CAM_ROT

IMG_W, IMG_H = 480, 640
FOV_X = np.deg2rad(42.74)
FOV_Y = np.deg2rad(65.34)
BASELINE = np.linalg.norm(CAM2_POS - CAM1_POS)

f_x = (IMG_W / 2.0) / math.tan(FOV_X / 2.0)
f_y = (IMG_H / 2.0) / math.tan(FOV_Y / 2.0)
