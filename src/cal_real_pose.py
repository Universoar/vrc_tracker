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

# 图像宽高 (像素)
IMG_W, IMG_H = 480, 640
FOV_X = np.deg2rad(42.74)
FOV_Y = np.deg2rad(65.34)

# 相机基线 (米)
BASELINE = np.linalg.norm(CAM2_POS - CAM1_POS)

# 焦距 (像素)
f_x = (IMG_W / 2.0) / math.tan(FOV_X / 2.0)
f_y = (IMG_H / 2.0) / math.tan(FOV_Y / 2.0)

# 根据双目关键点计算3D坐标 (X, Y, Z)
# ws1_body, ws2_body: (N x 2) array 相机坐标
# mode: "l-r" or "r-l" 视差方向
# 返回: (N x 3) array [X, Y, Z] (米)
def compute_3d_points(ws1_body, ws2_body, mode="r-l"):
    u_l = ws1_body[:, 0]
    v_l = ws1_body[:, 1]
    u_r = ws2_body[:, 0]

    if mode == "l-r":
        d = u_l - u_r
    else:
        d = u_r - u_l

    d[np.abs(d) < 1e-6] = np.nan  # 避免除零

    # 深度数组
    Z = (f_x * BASELINE) / d

    # 主点 (图像中心)
    c_x, c_y = IMG_W / 2.0, IMG_H / 2.0

    # 反投影
    X = (u_l - c_x) * Z / f_x
    Y = (v_l - c_y) * Z / f_y

    points_3d = np.stack([X, Y, Z], axis=1)
    return points_3d