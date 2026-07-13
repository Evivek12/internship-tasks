import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret: 
        break

    results = model(frame)
    frame = results[0].plot()

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()