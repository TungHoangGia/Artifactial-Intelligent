#Recommend Python Version: 3.10.0

import math
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp
from collections import deque

# Start webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

angle_history = deque(maxlen=10)
falling_frames = 0
fall_confirmed = False
FALL_CONFIRM_FRAMES = int(fps * 0.3)

# Initialize models
model = YOLO('sigma.pt')  # Replace with your own model path
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(shoulder_center, hip_center):
    dy = shoulder_center[1] - hip_center[1]
    dx = shoulder_center[0] - hip_center[0]
    angle = math.atan2(dy, dx)
    return abs(90 - np.degrees(angle))

def classify_posture(torso_angle, standing_threshold=10, lying_threshold=60):
    if torso_angle < standing_threshold:
        return "Standing"
    elif torso_angle > lying_threshold:
        return "Lying Down"
    else:
        return "Falling"

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        for result in results:
            for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) == 0:  # 'person' class
                    x1, y1, x2, y2 = map(int, bbox)
                    person_bbox = frame[y1:y2, x1:x2]

                    if person_bbox.size == 0:
                        continue

                    person_bbox_rgb = cv2.cvtColor(person_bbox, cv2.COLOR_BGR2RGB)
                    person_results = pose.process(person_bbox_rgb)

                    if not person_results.pose_landmarks:
                        continue

                    landmarks = person_results.pose_landmarks.landmark
                    visibility_scores = [lm.visibility for lm in landmarks]
                    if np.mean(visibility_scores) < 0.5:
                        continue

                    shoulders = [(landmarks[11].x * person_bbox.shape[1], landmarks[11].y * person_bbox.shape[0]),
                                 (landmarks[12].x * person_bbox.shape[1], landmarks[12].y * person_bbox.shape[0])]
                    hips = [(landmarks[23].x * person_bbox.shape[1], landmarks[23].y * person_bbox.shape[0]),
                            (landmarks[24].x * person_bbox.shape[1], landmarks[24].y * person_bbox.shape[0])]

                    shoulder_center = ((shoulders[0][0] + shoulders[1][0]) / 2, (shoulders[0][1] + shoulders[1][1]) / 2)
                    hip_center = ((hips[0][0] + hips[1][0]) / 2, (hips[0][1] + hips[1][1]) / 2)

                    torso_angle = calculate_angle(hip_center, shoulder_center)
                    angle_history.append(torso_angle)
                    avg_angle = sum(angle_history) / len(angle_history)
                    posture = classify_posture(avg_angle)

                    mp_drawing.draw_landmarks(person_bbox, person_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.putText(frame, posture, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    if posture == "Falling":
                        falling_frames += 1
                    else:
                        falling_frames = 0

                    if falling_frames >= FALL_CONFIRM_FRAMES:
                        fall_confirmed = True
                    if posture == "Standing":
                        fall_confirmed = False

                    frame[y1:y2, x1:x2] = person_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if fall_confirmed:
            cv2.putText(frame, "bro fell", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
