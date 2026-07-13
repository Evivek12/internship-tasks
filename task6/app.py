from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)[0]

    if prediction == 1:
        result = "🟢 Real News"
    else:
        result = "🔴 Fake News"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)