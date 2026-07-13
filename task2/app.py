from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break

        # Run object detection
        results = model.predict(frame, verbose=False)

        # Draw detections
        annotated_frame = results[0].plot()

        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)

        if not ret:
            continue

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame +
               b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


import webbrowser

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5001")
    app.run(debug=True, port=5001)