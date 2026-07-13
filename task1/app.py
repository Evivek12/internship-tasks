from flask import Flask,render_template,request,jsonify
from chatbot import chatbot_response

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get",methods=["POST"])
def get_response():

    data=request.get_json()

    message=data["message"]

    response=chatbot_response(message)

    return jsonify({"response":response})

import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open("http://127.0.0.1:5001")

if __name__ == "__main__":
    Timer(1, open_browser).start()   # Wait 1 second before opening
    app.run(debug=True, port=5001)

  