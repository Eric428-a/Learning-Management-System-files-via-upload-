"""
FastAPI web service for Land Use Classification.
- Serves pages (index, prediction, datasets, notebooks, about, contact, tutorial)
- Serves embedded exported notebook HTML (renders inline, not just download)
- Loads model at startup (supports .joblib)
- Exposes programmatic /api/predict and UI upload /predict
- Designed for local testing with: uvicorn app:app --reload
"""

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import os
import traceback

from utils.prediction_helper import load_model_for_inference, predict_from_image_bytes

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LandUseLab - Land Use Classification")

# Mount static + templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load model at startup
try:
    MODEL, CLASS_NAMES, MODEL_META = load_model_for_inference(models_dir=str(MODELS_DIR))
except Exception as e:
    MODEL = None
    CLASS_NAMES = []
    MODEL_META = {"error": str(e)}
    print("Model load error:", e)
    traceback.print_exc()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request, "result": None, "error": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    UI-driven prediction. Accepts file upload and returns template with result.
    """
    try:
        contents = await file.read()
        pred_class, pred_score = predict_from_image_bytes(contents, MODEL, CLASS_NAMES)

        # save uploaded file for display
        save_path = UPLOAD_DIR / file.filename
        with open(save_path, "wb") as f:
            f.write(contents)

        result = {
            "class": pred_class,
            "score": float(pred_score),
            "filename": f"/static/uploads/{file.filename}"
        }
        return templates.TemplateResponse("prediction.html", {"request": request, "result": result, "error": None})
    except Exception as e:
        tb = traceback.format_exc()
        return templates.TemplateResponse(
            "prediction.html",
            {"request": request, "result": None, "error": f"{str(e)}\n{tb}"}
        )


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    """
    Programmatic JSON endpoint for inference.
    """
    try:
        contents = await file.read()
        pred_class, pred_score = predict_from_image_bytes(contents, MODEL, CLASS_NAMES)
        return {"predicted_class": pred_class, "score": float(pred_score)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# --- Embedded Notebook Routes ---
@app.get("/notebooks", response_class=HTMLResponse)
async def notebooks(request: Request):
    """
    Renders the exported Jupyter notebook directly inside the web UI.
    """
    nb_path = NOTEBOOKS_DIR / "2_6_1_Land_use_classification.html"
    notebook_html = ""
    if nb_path.exists():
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                notebook_html = f.read()
        except Exception as e:
            notebook_html = f"<p>Error reading notebook: {str(e)}</p>"
    else:
        notebook_html = "<p>Notebook file not found.</p>"

    return templates.TemplateResponse(
        "notebooks.html",
        {"request": request, "notebook_html": notebook_html}
    )


@app.get("/notebooks/exported")
def exported_notebook():
    nb_path = NOTEBOOKS_DIR / "2_6_1_Land_use_classification.html"
    if nb_path.exists():
        return FileResponse(nb_path, media_type="text/html", filename=nb_path.name)
    return RedirectResponse("/notebooks")


@app.get("/datasets", response_class=HTMLResponse)
async def datasets(request: Request):
    return templates.TemplateResponse("datasets.html", {"request": request})


@app.get("/tutorial", response_class=HTMLResponse)
async def tutorial(request: Request):
    return templates.TemplateResponse("tutorial.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    model_info = MODEL_META.copy() if isinstance(MODEL_META, dict) else {"info": str(MODEL_META)}

    if MODEL is not None:
        model_info.setdefault("type", getattr(MODEL, "name", "Joblib model"))
        model_info.setdefault("input_shape", "(64,64,3)")
        model_info.setdefault("output_classes", len(CLASS_NAMES))
        model_info.setdefault("file", "models/eurosat_cnn_model.joblib")
        model_file = MODELS_DIR / "eurosat_cnn_model.joblib"
        if model_file.exists():
            model_info.setdefault("size_mb", round(os.path.getsize(model_file)/(1024*1024), 2))

    return templates.TemplateResponse("about.html", {"request": request, "model_info": model_info})


@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


@app.get("/health")
def health():
    ok = True
    model_loaded = MODEL is not None
    return {"status": "ok" if ok else "error", "model_loaded": model_loaded}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
