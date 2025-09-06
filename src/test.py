import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pythonosc import udp_client, osc_server, dispatcher

COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
]

# 你的数据
data1 = np.array(
    [
        [198.37, 192.07],
        [204.14, 181.4],
        [186.37, 183.99],
        [212.22, 187.25],
        [168.56, 194.11],
        [246.06, 248.99],
        [145.83, 257.7],
        [306.08, 288.97],
        [104.44, 311.5],
        [348.14, 247.01],
        [55.285, 279.08],
        [235.18, 417.26],
        [173.92, 419.63],
        [236.45, 545.34],
        [173.45, 540.59],
        [242.07, 628.1],
        [173.78, 618.72],
    ]
)

data2 = np.array(
    [
        [273.61, 185.31],
        [280.97, 174.55],
        [260.42, 176.79],
        [291.31, 182],
        [242.37, 187.96],
        [324.09, 250.05],
        [215.47, 251.36],
        [377.27, 299.24],
        [166.57, 303.26],
        [421.65, 253.42],
        [124.68, 272.72],
        [303.63, 424.84],
        [236.73, 424.43],
        [305.18, 556.66],
        [234.71, 548.27],
        [312.36, 628.38],
        [235.24, 614.87],
    ]
)

points = np.array([
    [0.006924, -1.0598, 0.14023],
    [-0.0043299, -1.0378, 0.16216],
    [0.029146, -1.0767, 0.12333],
    [-0.019337, -1.0081, 0.19191],
    [0.060826, -1.0803, 0.11972],
    [-0.075089, -1.0218, 0.17822],
    [0.1108, -1.145, 0.05499],
    [-0.18566, -1.12, 0.079988],
    [0.21865, -1.2834, -0.08338],
    [-0.25623, -1.0846, 0.11536],
    [0.28105, -1.1491, 0.050942],
    [-0.055846, -1.1649, 0.035096],
    [0.071763, -1.2694, -0.069444],
    [-0.058281, -1.16, 0.039978],
    [0.076235, -1.3016, -0.10164],
    [-0.068829, -1.1345, 0.065538],
    [0.075069, -1.2973, -0.097282]
])

latest_points = np.zeros((17, 3), dtype=float)

def plot_skeleton(data, save_path=None):
    """
    绘制人体骨架图

    参数:
        data: numpy.ndarray，形状 (N, 2)，每行是一个点 (x, y)
        skeleton: list[tuple]，每个元组 (i, j) 表示要连接的点
        save_path: str，可选。如果指定，将图像保存到这个路径
    """
    # 分离x和y坐标
    x = data[:, 0]
    y = data[:, 1]

    # 创建画布
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c="blue", s=50, label="Points")  # 散点

    # 按骨架连接线段
    for i, j in COCO_SKELETON:
        if i < len(data) and j < len(data):
            plt.plot([x[i], x[j]], [y[i], y[j]], c="orange", lw=2)

    # 标注点编号
    for idx, (xi, yi) in enumerate(data):
        plt.text(xi + 3, yi + 3, str(idx), fontsize=9)

    # 设置图像属性
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("COCO Skeleton")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    # 保存或显示
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()

def plot_3d_skeleton(points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=50)

    # 绘制骨架连线
    for i, j in COCO_SKELETON:
        if i < len(points) and j < len(points):
            xs = [points[i, 0], points[j, 0]]
            ys = [points[i, 1], points[j, 1]]
            zs = [points[i, 2], points[j, 2]]
            ax.plot(xs, ys, zs, c='orange', linewidth=2)

    # 标注每个点
    for idx, (x, y, z) in enumerate(points):
        ax.text(x, y, z, str(idx), fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Skeleton Viewer')

    # 设置交互缩放
    ax.view_init(elev=20, azim=60)  # 初始视角
    plt.tight_layout()
    plt.show()

def setup_3d_plot():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    return fig, ax

def update_skeleton(ax, points):
    ax.cla()
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    ax.scatter(points[:,0], points[:,1], points[:,2], c='blue', s=50)
    for i,j in COCO_SKELETON:
        if i < len(points) and j < len(points):
            ax.plot([points[i,0], points[j,0]],
                    [points[i,1], points[j,1]],
                    [points[i,2], points[j,2]],
                    c='orange', lw=2)
    plt.draw()
    plt.pause(0.001)

# ---------------- OSC Handler ----------------
def skeleton_handler(address, *args):
    global latest_points
    arr = np.array(args)
    if arr.size != 17*3:
        print(f"Received data size {arr.size}, expected {17*3}")
        return
    latest_points = arr.reshape((17,3))

# ---------------- 接收 & 绘图 ----------------
def start_osc_3d_view(ip="0.0.0.0", port=9000, fps=8):
    global latest_points
    disp = dispatcher.Dispatcher()
    disp.map("/skeleton/full", skeleton_handler)

    server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"OSC UDP Server running on {ip}:{port}")

    fig, ax = setup_3d_plot()
    interval = 1/fps
    while True:
        update_skeleton(ax, latest_points)
        plt.pause(interval)

if __name__ == "__main__":
    plot_skeleton(data1,"data1_skeleton.png")
    plot_skeleton(data2,"data2_skeleton.png")
    # plot_3d_skeleton(points)
    # start_osc_3d_view("0.0.0.0", 9000)