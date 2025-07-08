from ultralytics import YOLO
import cv2
#pip install opencv-python
#pip install ultralytics

# Load model YOLOv8 pre-trained
model = YOLO('yolov8n.pt')

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Putting bounding box for Person only
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls_id] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8", frame)

    # Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
