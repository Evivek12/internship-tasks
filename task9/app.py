# ==========================================
# AI-Based Sentiment Analysis for Social Media
# Streamlit Web Application
# ==========================================

import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (first time only)
nltk.download("stopwords")

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Text Preprocessing
# -----------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f2f7ff;
}

h1{
    text-align:center;
    color:#0B5394;
}

.result{
    font-size:25px;
    font-weight:bold;
}

textarea{
    font-size:18px;
}

.stButton>button{
    width:100%;
    height:50px;
    font-size:18px;
    background:#0B5394;
    color:white;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("😊 AI-Based Sentiment Analysis")
st.write("Analyze the sentiment of social media posts using Artificial Intelligence.")

st.markdown("---")

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "Enter a social media post:",
    height=180,
    placeholder="Example: I really enjoyed using this application..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        cleaned = clean_text(user_input)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        st.markdown("---")

        if prediction == "Positive":
            st.success("😊 Positive Sentiment")
            st.balloons()

        elif prediction == "Negative":
            st.error("😞 Negative Sentiment")

        else:
            st.info("😐 Neutral Sentiment")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Python, NLTK, Scikit-learn and Streamlit")