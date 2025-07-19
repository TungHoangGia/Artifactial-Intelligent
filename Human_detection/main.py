from ultralytics import YOLO
import cv2
import time

# Load model, you can use your own model, just make sure to put it near the main.py
model = YOLO('sigma.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

person_check = None
danger_check = False
danger_time = 0
zone_y = 600  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    check_line = False
    current_time = time.time()

    cv2.line(frame, (0, zone_y), (frame.shape[1], zone_y), (0, 0, 255), 3)
    cv2.putText(frame, "DANGER LINE", (10, zone_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if y1 <= zone_y <= y2:
                    check_line = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls_id]}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    if person_check is not None:
        elapsed = current_time - person_check
        cv2.putText(frame, f"Detected: {elapsed:.1f}s", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if check_line:
        if person_check is None:
            person_check = current_time
        elif current_time - person_check >= 0: # Set this bigger if you want it to wait atleast a customized amount of time between messages;
            danger_check = True
    else:
        person_check = None
        danger_check = False
        danger_time = 0

    if danger_check:
        if current_time - danger_time >= 5: # Send danger message every 5 second.
            print("Danger")
            danger_time = current_time

    cv2.imshow("so tuff", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
