import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load the trained model
model = load_model("digit_model.h5")

# ----------------------------
# Page Settings
# ----------------------------
st.set_page_config(
    page_title="AI Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.stApp{
    background-color:#0E1117;
}

h1{
    color:#00E5FF;
    text-align:center;
}

h3{
    color:white;
}

.stButton>button{
    background-color:#00E5FF;
    color:black;
    border-radius:10px;
    width:100%;
    height:45px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#00B8D4;
}

.sidebar .sidebar-content{
    background-color:#161B22;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🧠 AI Digit Recognizer")

st.sidebar.write("""
This project recognizes handwritten digits using
a CNN model trained on the MNIST dataset.

Technologies Used:

• Python

• TensorFlow

• OpenCV

• Streamlit
""")

# ----------------------------
# Title
# ----------------------------
st.title("✍️ AI-Based Handwritten Digit Recognition")

st.write("Draw a digit or upload an image and click Predict.")

st.divider()

# ----------------------------
# Select Input Method
# ----------------------------
option = st.radio(
    "Choose Input Method",
    ["Draw Digit", "Upload Image"],
    horizontal=True
)

image = None

# ----------------------------
# Drawing Canvas
# ----------------------------
if option == "Draw Digit":

    st.subheader("Draw Here")

    canvas = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas.image_data is not None:
        image = canvas.image_data

# ----------------------------
# Upload Image
# ----------------------------
else:

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png","jpg","jpeg"]
    )

    if uploaded_file is not None:

        img = Image.open(uploaded_file)

        st.image(img,width=250)

        image = np.array(img)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Digit"):

    if image is None:

        st.warning("Please draw or upload an image.")

    else:

        # Convert image to grayscale
        if len(image.shape)==3:
            image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_RGB2GRAY)

        # Resize
        image = cv2.resize(image,(28,28))

        # Invert colors
        image = cv2.bitwise_not(image)

        # Normalize
        image = image.astype("float32")/255.0

        # Reshape
        image = image.reshape(1,28,28,1)

        # Prediction
        prediction = model.predict(image)

        digit = np.argmax(prediction)

        confidence = np.max(prediction)

        st.success(f"Predicted Digit : {digit}")

        st.write(f"Confidence : {confidence*100:.2f}%")

        st.progress(float(confidence))

        if confidence > 0.95:
            st.success("Excellent Prediction!")

        elif confidence > 0.80:
            st.info("Good Prediction!")

        else:
            st.warning("Low Confidence. Try Again!")

st.divider()

st.markdown(
"""
### About This Project

This application uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits (0–9). Users can either draw a digit on the canvas or upload an image, and the AI predicts the digit along with its confidence score.
"""
)