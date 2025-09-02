import numpy as np
import math

# 你的相机参数（示例）
CAM1_POS = np.array([0.0, 0.0, 1.2])
CAM2_POS = np.array([0.0, 0.13, 1.2])

# 你给的 CAM_ROT（假定是 cam -> world）
CAM_ROT = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype=float)
CAM1_ROT = CAM_ROT
CAM2_ROT = CAM_ROT

IMG_W, IMG_H = 480, 640       # 你上面写的是反过来，注意这里 W=宽 H=高
FOV_X = np.deg2rad(42.74)
FOV_Y = np.deg2rad(65.34)
BASELINE = np.linalg.norm(CAM2_POS - CAM1_POS)  # d

# 焦距（像素）
f_x = (IMG_W / 2.0) / math.tan(FOV_X / 2.0)
f_y = (IMG_H / 2.0) / math.tan(FOV_Y / 2.0)

def pixel_to_angle(u_c, v_c):
    """把以中心为原点的像素偏移转为角度（弧度）"""
    phi = math.atan2(u_c, f_x)  # 水平角
    psi = math.atan2(v_c, f_y)  # 垂直角
    return phi, psi

def triangulate_point(u1, v1, u2, v2):
    """
    输入：两个相机的像素（以图像中心为原点的偏移 u_c, v_c）
    输出：相机1坐标系下的点 P_cam1 = [X_forward, Y_right, Z_up]
    """
    # 角度（相对于各自光轴）
    phi_L, psi_L = pixel_to_angle(u1, v1)
    phi_R, psi_R = pixel_to_angle(u2, v2)

    tanL = math.tan(phi_L)
    tanR = math.tan(phi_R)
    denom = tanL - tanR
    if abs(denom) < 1e-9:
        return None  # 不稳定或平行

    # 深度（相机前向 X_cam）
    X = BASELINE / denom

    # 横向 Y_cam：用左相机视线方程 Y = X * tan(phi_L) - d/2
    Y = X * tanL - (BASELINE / 2.0)

    # 垂直 Z_cam：用 left 的垂直角
    Z = X * math.tan(psi_L)

    return np.array([X, Y, Z], dtype=float)

def cam_to_world(cam_point, cam_rot, cam_pos):
    """把相机坐标系下点转换到世界坐标系"""
    return cam_rot @ cam_point + cam_pos

def triangulate_pixels_to_world(kps1, kps2):
    """
    kps1, kps2: list of (u, v) 像素坐标，原点为图像左上
    返回: N x 3 数组，坐标顺序为 (x_world, depth_along_cam_forward, z_world)
    """
    # 先把像素转到以中心为原点
    pts_world = []
    half_w = IMG_W / 2.0
    half_h = IMG_H / 2.0

    for (u1_pix, v1_pix), (u2_pix, v2_pix) in zip(kps1, kps2):
        # 中心化（以中心为0）
        u1 = u1_pix - half_w
        v1 = half_h - v1_pix  # 注意像素向下为增，转换成相机上为向上
        u2 = u2_pix - half_w
        v2 = half_h - v2_pix

        cam_pt = triangulate_point(u1, v1, u2, v2)
        if cam_pt is None:
            pts_world.append([np.nan, np.nan, np.nan])
            continue

        # 把相机1坐标系下点变换到世界坐标（使用 cam1 的 R 和 pos）
        world_pt = cam_to_world(cam_pt, CAM1_ROT, CAM1_POS)  # world vector

        # 按你的约定输出： x 不变（取 world 的 X）， depth = 沿相机前向的距离 (world的X方向),
        # z = world 的 Z 分量
        # 这里 world 的分量含义依赖于 CAM_ROT 定义；通常 world_pt = [Xw, Yw, Zw]
        x_world = world_pt[0]            # 横向/前向（取决于你的坐标）
        depth = cam_pt[0]                # 深度沿相机前向（X_cam）
        z_world = world_pt[2]            # 世界高度

        pts_world.append([x_world, depth, z_world])

    return np.array(pts_world, dtype=float)