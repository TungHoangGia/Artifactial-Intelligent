from ultralytics import YOLO
from collections import deque
import cv2
import numpy as np
import time

fps = 20
hip_buffer = 10
drop_thresh = 200
angle_thresh = 45
ratio_thresh = 1.0
fall_window = 1.0
lying_window = 2.0
state = "STANDING"
fall_start = None
lying_start = None
lying_check = False
hip_history = deque(maxlen=hip_buffer)

model = YOLO("yolov8n-pose.pt")
cap   = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    result  = results[0]
    if result.keypoints is None:
        continue

    for i, kp_xy in enumerate(result.keypoints.xy):
        kp_conf   = result.keypoints.conf[i]
        keypoints = np.hstack([
            kp_xy.cpu().numpy(),
            kp_conf.cpu().numpy().reshape(-1,1)
        ])  
        
        left_hip, right_hip, nose = keypoints[11], keypoints[12], keypoints[0]
        if min(left_hip[2], right_hip[2], nose[2]) < 0.5:
            continue

        # Logic 1
        center_hip_y = (left_hip[1] + right_hip[1]) / 2
        hip_history.append(center_hip_y)

        drop_rate = 0
        if len(hip_history) == hip_buffer:
            total_time  = hip_history[0] - hip_history[-1]
            duration    = (hip_buffer - 1) / fps
            drop_rate = total_time / duration

        # Logic 2
        shoulder_y   = (keypoints[5][1] + keypoints[6][1]) / 2
        dx           = ((keypoints[5][0] + keypoints[6][0]) / 2) - ((left_hip[0] + right_hip[0]) / 2)
        dy           = shoulder_y - center_hip_y
        angle        = abs(np.degrees(np.arctan2(dy, dx)))

        # Logic 3
        xs = [p[0] for p in keypoints if p[2] > 0.5]
        ys = [p[1] for p in keypoints if p[2] > 0.5]
        if not xs or not ys:
            continue
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        ratio = width / height

        time_stamp = time.time()

        if state == "STANDING":
            if drop_rate > drop_thresh:
                state      = "POTENTIAL_FALL"
                fall_start = time_stamp
                lying_start = None
            elif angle < angle_thresh and ratio > ratio_thresh:
                if lying_start is None:
                    lying_start = time_stamp
                elif time_stamp - lying_start > lying_window and lying_check:
                    cv2.putText(frame, "LYING DOWN", (50,150),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 3)
            else:
                lying_start = None

        elif state == "POTENTIAL_FALL":
            if time_stamp - fall_start < fall_window:
                if angle < angle_thresh and ratio > ratio_thresh:
                    state = "FALL_CONFIRMED"
            else:
                state = "STANDING"

        elif state == "FALL_CONFIRMED":
            cv2.putText(frame, "FALL DETECTED", (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
            if angle > 60 or ratio < 0.7:
                state = "STANDING"
                fall_start = None

    annotated = result.plot()
    cv2.imshow("Fall Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
