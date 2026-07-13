import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Read intents
with open("intents.json", "r") as file:
    data = json.load(file)

questions = []
labels = []

# Collect training data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        questions.append(pattern)
        labels.append(intent["tag"])

# Convert text into numerical form
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(questions)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")