import json
import random
import joblib

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent


model = joblib.load(BASE_DIR / "model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


with open("intents.json","r") as file:
    intents=json.load(file)

def chatbot_response(message):

    message_vector=vectorizer.transform([message])

    tag=model.predict(message_vector)[0]

    for intent in intents["intents"]:
        if intent["tag"]==tag:
            return random.choice(intent["responses"])

    return "Sorry, I couldn't understand your question."
