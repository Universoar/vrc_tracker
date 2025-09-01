import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

class PointKalman:
    def __init__(self, history_len=3):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0

        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000.
        self.kf.R *= 2.
        self.kf.Q *= 0.1

        # 历史队列
        self.history = deque(maxlen=history_len)

    def update(self, x, y):
        z = np.array([x, y])
        self.kf.predict()
        self.kf.update(z)
        sx, sy = self.kf.x[0], self.kf.x[1]
        
        self.history.append((sx, sy))

        hx = np.mean([p[0] for p in self.history])
        hy = np.mean([p[1] for p in self.history])
        return hx, hy
    
def create_kalman_filters(num_points):
    return [PointKalman() for _ in range(num_points)]

def kalman_smooth(points, kalman_list):
    smoothed = []
    for (x, y), kf in zip(points, kalman_list):
        sx, sy = kf.update(x, y)
        smoothed.append((sx, sy))
    return np.array(smoothed)
