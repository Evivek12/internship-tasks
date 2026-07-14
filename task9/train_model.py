# ==============================
# AI-Based Sentiment Analysis
# Model Training Code
# ==============================

import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (Only first time)
nltk.download('stopwords')

# ---------------------------------------
# Load Dataset
# ---------------------------------------
data = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully!")
print(data.head())

# ---------------------------------------
# Text Preprocessing
# ---------------------------------------

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Split sentence into words
    words = text.split()

    # Remove stopwords and apply stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    # Join words
    return " ".join(words)

# Apply preprocessing
data["clean_text"] = data["text"].apply(clean_text)

print("\nPreprocessing Completed!\n")

# ---------------------------------------
# Convert Text into Numbers
# ---------------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["clean_text"])

y = data["sentiment"]

# ---------------------------------------
# Split Dataset
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# Train Model
# ---------------------------------------

model = LogisticRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ---------------------------------------
# Test Model
# ---------------------------------------

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy :", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, prediction))

# ---------------------------------------
# Save Model
# ---------------------------------------

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nFiles Saved Successfully!")

print("sentiment_model.pkl")
print("vectorizer.pkl")