import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, Response
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model("mask_detector.keras")

# -----------------------------
# Load Face Detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Open Webcam
# -----------------------------
camera = cv2.VideoCapture(0)

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face = cv2.resize(face, (224, 224))

            face = face.astype("float32")

            face = preprocess_input(face)

            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0][0]

            if prediction < 0.5:

                label = "Mask"

                color = (0, 255, 0)

            else:

                label = "No Mask"

                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route("/")
def home():

    return render_template("index.html")

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":

    app.run(debug=True)