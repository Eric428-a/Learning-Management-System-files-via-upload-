from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import json
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.json")

# --- Load artifacts ---
model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    expected_cols = json.load(f)

# --- FastAPI setup ---
app = FastAPI(title="Credit Scoring API")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict(payload: dict):
    """Payload must be: {"data": [ {col: val, ...}, {...} ]}"""
    rows = payload.get("data")
    if not rows:
        return JSONResponse({"error": "JSON must include key 'data' with records"}, status_code=400)

    df = pd.DataFrame(rows)
    # ensure all columns exist and in order
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    df = df[expected_cols]

    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {"predictions": preds.tolist(), "probabilities": probs.tolist()}
