import cv2
from ultralytics import solutions
#pip install opencv-python
#pip install ultralytics

cap = cv2.VideoCapture("example.mp4") #input the video you want here, make sure to put it next to your main.py in file explorer
assert cap.isOpened(), "Error reading video file"

region_points = [[528, 239], [40, 789], [1371, 772], [856, 233]] 

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = solutions.ObjectCounter(
    show=False,
    region=region_points,
    model="yolo11n.pt", #important line, switch to the suitable model on roboflow
    classes=[2],
    # tracker="botsort.yaml",
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    video_writer.write(results.plot_im)

    cv2.imshow("Object Counting", results.plot_im)
    if cv2.waitKey(1) & 0xFF == 27: #esc to escape
        print("Escape hit, closing...")
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()