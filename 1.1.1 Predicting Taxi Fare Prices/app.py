# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model safely
model_path = os.path.join("models", "taxi_fare_model (1).joblib")
model = joblib.load(model_path)
if isinstance(model, dict):  # handle accidental dict saving
    model = model.get("model", None)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/notebooks")
def notebooks():
    return render_template("notebooks.html")

@app.route("/datasets")
def datasets():
    return render_template("datasets.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    prediction_result = None
    if request.method == "POST":
        try:
            distance_miles = float(request.form["distance_miles"])
            passenger_count = int(request.form["passenger_count"])
            hour_of_day = int(request.form["hour_of_day"])
            day_of_week = int(request.form["day_of_week"])
            month = int(request.form["month"])

            features = np.array([[distance_miles, passenger_count, hour_of_day, day_of_week, month]])
            pred = model.predict(features)[0]
            prediction_result = f"Estimated Taxi Fare: ${pred:.2f}"
        except Exception as e:
            prediction_result = f"Error: {str(e)}"
    return render_template("prediction.html", prediction_result=prediction_result)

@app.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
