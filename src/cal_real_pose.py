import numpy as np

CAM1_POS = np.array([0.0, 0.0, 1.2])
CAM2_POS = np.array([0.0, 0.13, 1.2])

CAM_ROT = np.array([
    [0, 1, 0],   
    [0, 0, 1], 
    [1, 0, 0]    
])
CAM1_ROT = CAM_ROT
CAM2_ROT = CAM_ROT


IMG_W, IMG_H = 640, 480
FOV_X = np.deg2rad(60)
FOV_Y = np.deg2rad(45)

def pixel_to_ray(x, y, img_w, img_h, cam_rot):
    """像素坐标 -> 世界坐标系下射线方向"""
    X = (x - img_w / 2) / (img_w / 2) * np.tan(FOV_X / 2)
    Y = -(y - img_h / 2) / (img_h / 2) * np.tan(FOV_Y / 2)
    ray_cam = np.array([X, Y, 1.0])
    ray_world = cam_rot @ ray_cam
    return ray_world / np.linalg.norm(ray_world)

def closest_point_between_rays(p1, d1, p2, d2):
    """求两条射线的最近点"""
    v12 = p2 - p1
    d1d1 = np.dot(d1, d1)
    d2d2 = np.dot(d2, d2)
    d1d2 = np.dot(d1, d2)
    denom = d1d1 * d2d2 - d1d2 * d1d2
    if abs(denom) < 1e-6:
        return None
    t1 = (np.dot(v12, d1) * d2d2 - np.dot(v12, d2) * d1d2) / denom
    t2 = (np.dot(v12, d1) * d1d2 - np.dot(v12, d2) * d1d1) / denom
    c1 = p1 + t1 * d1
    c2 = p2 + t2 * d2
    return (c1 + c2) / 2

def triangulate(keypoints1, keypoints2, user_height=1.3):
    """
    输入两个摄像头17个关键点集合 [(x,y), ...] 
    输出世界坐标系下17个关键点 [(x,y,z), ...]
    """
    points_3d = []
    for (x1, y1), (x2, y2) in zip(keypoints1, keypoints2):
        d1 = pixel_to_ray(x1, y1, IMG_W, IMG_H, CAM1_ROT)
        d2 = pixel_to_ray(x2, y2, IMG_W, IMG_H, CAM2_ROT)
        pt3d = closest_point_between_rays(CAM1_POS, d1, CAM2_POS, d2)
        points_3d.append(pt3d if pt3d is not None else np.array([np.nan]*3))
    points_3d = np.array(points_3d)

    # 缩放到真实身高（头顶到骨盆）
    head_idx, pelvis_idx = 0, 8  # nose/top -> pelvis
    head = points_3d[head_idx]
    pelvis = points_3d[pelvis_idx]
    if np.any(np.isnan(head)) or np.any(np.isnan(pelvis)):
        return points_3d
    current_height = np.linalg.norm(head - pelvis)
    if current_height > 0:
        scale = user_height / current_height
        points_3d *= scale

    return points_3d