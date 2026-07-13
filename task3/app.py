import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load trained model
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "disease_model.pkl")

st.set_page_config(page_title="AI Medical Diagnosis System", page_icon="🩺")

st.title("🩺 AI-Powered Medical Diagnosis System")

st.write("Enter the patient details below and click Predict.")

# User Inputs
age = st.number_input("Age", 1, 100, 30)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    80,
    250,
    120
)

chol = st.number_input(
    "Cholesterol",
    100,
    600,
    200
)

fbs = st.selectbox(
    "Fasting Blood Sugar >120 mg/dl",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    60,
    250,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "Old Peak",
    0.0,
    10.0,
    1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

if st.button("Predict"):

    sex_value = 1 if sex == "Male" else 0

    input_data = pd.DataFrame([[
        age,
        sex_value,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
        st.write("Please consult a medical professional for further evaluation.")
    else:
        st.success("✅ No Heart Disease Detected")
        st.write("The prediction indicates a lower likelihood of heart disease.")


    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
        st.write("Please consult a medical professional for further evaluation.")
    else:
        st.success("✅ No Heart Disease Detected")
        st.write("The prediction indicates a lower likelihood of heart disease.")
    