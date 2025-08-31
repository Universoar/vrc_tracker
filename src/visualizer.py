# src/visualizer.py
import cv2
import numpy as np

def draw_full_skeleton(frame, skeletons):
    left = np.zeros_like(frame)  # 黑底
    right = frame.copy()
    
    # COCO skeleton
    COCO_SKELETON = [
        (0, 1),(0,2),(1,3),(2,4),(0,5),(0,6),
        (5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
        (11,13),(13,15),(12,14),(14,16),(11,12)
    ]
    
    for sk in skeletons:
        body = sk["body"]
        left_hand = sk["left_hand"]
        right_hand = sk["right_hand"]
        
        # 画人体骨架
        for i,j in COCO_SKELETON:
            if i < len(body) and j < len(body):
                cv2.line(left, tuple(body[i].astype(int)), tuple(body[j].astype(int)), (0,255,0), 2)
        for x,y in body:
            cv2.circle(left, (int(x),int(y)), 5, (0,255,0), -1)
        
        # 画手部骨架
        for hand_points in [left_hand, right_hand]:
            for idx in range(len(hand_points)-1):
                cv2.line(left, hand_points[idx], hand_points[idx+1], (0,255,255), 1)
            for x,y in hand_points:
                cv2.circle(left, (x,y), 3, (0,255,255), -1)
    
    combined = cv2.hconcat([left, right])
    cv2.imshow("Skeleton+Hands (Left) | Original (Right)", combined)
    if cv2.waitKey(1) & 0xFF == 27:
        return False
    return True
