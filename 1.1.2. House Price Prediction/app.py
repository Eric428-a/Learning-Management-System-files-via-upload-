# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd
import numpy as np
import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "supersecretkey_eric_2025"

# ---------- User / project config ----------
CONTACT_EMAIL = "ericmwaniki2004@gmail.com"
GITHUB_LINK = "https://github.com/Eric428-a"
LINKEDIN_LINK = "https://www.linkedin.com/in/eric-m-1325902b2/"
VIDEO_TUTORIAL_LINK = "https://youtu.be/Wqmtf9SA_kk"
DATASET_LINK = "https://www.kaggle.com/datasets/juhibhojani/house-price"
MODEL_FILENAME = "house_price_model.joblib"
MODEL_PATH = os.path.join("models", MODEL_FILENAME)

# ---------- Ensure static subfolders exist ----------
for folder in ["static/css", "static/js", "static/images", "static/slides", "static/charts", "models", "notebooks"]:
    os.makedirs(folder, exist_ok=True)

# ---------- Load model ----------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        app.logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        app.logger.error(f"Failed to load model: {e}")

# ---------- Helper functions ----------
def url_for_static(path):
    return url_for("static", filename=path)

def choose_random_house_image():
    img_dir = os.path.join("static", "images")
    try:
        imgs = [f for f in os.listdir(img_dir) if f.lower().startswith("house")]
        if not imgs:
            return None
        return url_for_static("images/" + random.choice(imgs))
    except:
        return None

def save_prediction_chart(prediction_value, chart_filename="prediction_chart.png"):
    try:
        avg_price = 180000
        values = [avg_price, prediction_value]
        labels = ["Average Price", "Predicted Price"]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, values)
        bars[0].set_alpha(0.6)
        bars[1].set_color("#2b8c2b")

        plt.ylabel("Price ($)")
        plt.title("Predicted vs Average House Price")

        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, val*1.01, f"${val:,.0f}", ha="center")

        chart_path = os.path.join("static/charts", chart_filename)
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        return url_for_static("charts/" + chart_filename)
    except:
        return None

# ---------- Context processor ----------
@app.context_processor
def inject_globals():
    return dict(contact_email=CONTACT_EMAIL, github_link=GITHUB_LINK, linkedin_link=LINKEDIN_LINK)

# ---------- Routes ----------
@app.route("/")
def index():
    slides = ["slides/slide1.jpg", "slides/slide2.jpg", "slides/slide3.jpg"]
    project_images = ["images/project1.jpg","images/project2.jpg","images/project3.jpg","images/project4.jpg","images/project5.jpg","images/project6.jpg"]
    return render_template("index.html", slides=slides, images=project_images)

@app.route("/notebooks")
def notebooks():
    notebook_dir = "notebooks"
    notebooks_list = [f for f in os.listdir(notebook_dir) if f.lower().endswith(".html")]
    return render_template("notebooks.html", notebooks=notebooks_list)

@app.route("/datasets")
def datasets():
    return render_template("datasets.html", dataset_link=DATASET_LINK)

@app.route("/prediction", methods=["GET","POST"])
def prediction():
    prediction_value = None
    chart_url = None
    house_image = None
    model_warning = None
    input_data = {}

    if model is None:
        model_warning = "Model not loaded. Predictions disabled."

    if request.method=="POST" and model:
        try:
            ms_subclass = request.form.get("MSSubClass", "")
            ms_zoning = request.form.get("MSZoning", "")
            lot_frontage = request.form.get("LotFrontage", "")
            lot_area = request.form.get("LotArea", "")
            street = request.form.get("Street", "")

            df = pd.DataFrame([{
                "MSSubClass": float(ms_subclass) if ms_subclass else np.nan,
                "MSZoning": ms_zoning if ms_zoning else np.nan,
                "LotFrontage": float(lot_frontage) if lot_frontage else np.nan,
                "LotArea": float(lot_area) if lot_area else np.nan,
                "Street": street if street else np.nan
            }])
            prediction_value = float(model.predict(df)[0])
            chart_url = save_prediction_chart(prediction_value)
            house_image = choose_random_house_image()
        except Exception as e:
            flash(f"Error: {e}", "danger")

    return render_template("prediction.html", prediction=prediction_value, chart_url=chart_url, house_image=house_image, model_warning=model_warning, input_data=input_data)

@app.route("/tutorial")
def tutorial():
    embed = VIDEO_TUTORIAL_LINK
    if "youtu.be" in VIDEO_TUTORIAL_LINK:
        vid = VIDEO_TUTORIAL_LINK.split("/")[-1].split("?")[0]
        embed = f"https://www.youtube.com/embed/{vid}"
    elif "watch?v=" in VIDEO_TUTORIAL_LINK:
        vid = VIDEO_TUTORIAL_LINK.split("watch?v=")[-1].split("&")[0]
        embed = f"https://www.youtube.com/embed/{vid}"
    return render_template("tutorial.html", video_embed=embed, video_raw=VIDEO_TUTORIAL_LINK)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET","POST"])
def contact():
    if request.method=="POST":
        name = request.form.get("name", "")
        email = request.form.get("email", "")
        message = request.form.get("message", "")
        flash("Message received! (placeholder)", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")

if __name__=="__main__":
    app.run(debug=True)
