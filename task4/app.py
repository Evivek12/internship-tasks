from flask import Flask, render_template, request
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Function to extract text from PDF
def extract_text(pdf_file):
    text = ""

    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page in pdf:
        text += page.get_text()

    return text


@app.route("/", methods=["GET", "POST"])
def home():

    score = None

    if request.method == "POST":

        job_description = request.form["job"]

        resume_file = request.files["resume"]

        resume_text = extract_text(resume_file)

        documents = [job_description, resume_text]

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(vectors)

        score = round(similarity[0][1] * 100, 2)

    return render_template("index.html", score=score)


if __name__ == "__main__":
    app.run(debug=True)