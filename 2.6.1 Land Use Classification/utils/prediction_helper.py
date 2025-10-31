# utils/prediction_helper.py
"""
Model loader and prediction helper.
- Tries to load a Keras native model (.keras/.h5) first
- Falls back to joblib (.joblib)
- If joblib contains a Keras model object, we use it directly
- If joblib contains a scikit-learn style model, we handle it (expected to return a label)
- Provides predict_from_image_bytes(image_bytes, model, class_names)
"""

import os
from pathlib import Path
import joblib
import numpy as np
import traceback

# try import tensorflow lazily (only if needed)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

from .preprocessing import preprocess_image_bytes
from typing import Tuple, List, Any, Dict

DEFAULT_CLASS_NAMES = [
    "AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial",
    "Pasture","PermanentCrop","Residential","River","SeaLake"
]

def load_model_for_inference(models_dir: str = "models") -> Tuple[Any, List[str], Dict]:
    """
    Returns: (model_object, class_names_list, metadata_dict)
    - metadata contains helpful info for the /about page and diagnostics.
    """
    models_dir = Path(models_dir)
    keras_candidates = list(models_dir.glob("*.keras")) + list(models_dir.glob("*.h5"))
    joblib_candidate = models_dir / "eurosat_cnn_model.joblib"

    meta = {}
    model = None

    # prefer native TF/Keras models
    if keras_candidates and TF_AVAILABLE:
        keras_path = str(keras_candidates[0])
        try:
            model = tf.keras.models.load_model(keras_path)
            meta["source"] = keras_path
            meta["type"] = "keras_native"
            return model, DEFAULT_CLASS_NAMES, meta
        except Exception as e:
            meta["keras_load_error"] = str(e)
            # fallback to joblib area below

    # fallback: try joblib
    if joblib_candidate.exists():
        try:
            obj = joblib.load(str(joblib_candidate))
            # If this is a Keras model object (most likely), use it
            if TF_AVAILABLE and hasattr(obj, "predict") and hasattr(obj, "get_config"):
                model = obj
                meta["source"] = str(joblib_candidate)
                meta["type"] = "keras_object_joblib"
                return model, DEFAULT_CLASS_NAMES, meta
            # If it's an sklearn pipeline/classifier
            if hasattr(obj, "predict") and not TF_AVAILABLE:
                # we don't know expected input format; assume it takes flattened arrays or feature vectors.
                model = obj
                meta["source"] = str(joblib_candidate)
                meta["type"] = "sklearn_joblib"
                return model, DEFAULT_CLASS_NAMES, meta
            # If it's something else, return it but warn
            model = obj
            meta["source"] = str(joblib_candidate)
            meta["type"] = "unknown_joblib"
            return model, DEFAULT_CLASS_NAMES, meta
        except Exception as e:
            raise RuntimeError(f"Failed to load joblib model: {e}\n{traceback.format_exc()}")

    # no model found
    raise FileNotFoundError("No model found. Place eurosat_cnn_model.keras/.h5 or eurosat_cnn_model.joblib in the models/ dir.")

def _predict_with_keras(model, arr: np.ndarray):
    """
    Run model.predict and return probabilities.
    """
    # ensure dtype float32
    arr = arr.astype("float32")
    preds = model.predict(arr)
    # Sometimes keras returns nested list or 1D
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)
    return preds

def _predict_with_sklearn(model, arr: np.ndarray):
    """
    If the joblib object is sklearn-style, try to produce a prediction:
    - Flatten image and pass to model.predict_proba if available
    - If only predict (no proba), use predict and set score=1.0
    """
    flat = arr.reshape((arr.shape[0], -1))
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(flat)
        return probs
    else:
        preds = model.predict(flat)
        # convert to indices if labels are strings
        return np.stack([np.eye(len(DEFAULT_CLASS_NAMES))[preds]], axis=0) if np.issubdtype(preds.dtype, np.integer) else np.array(preds)

def predict_from_image_bytes(image_bytes: bytes, model, class_names: List[str]) -> Tuple[str, float]:
    """
    Unified prediction function:
      - preprocess bytes -> arr (1,64,64,3)
      - detect how to call model and return (label, score)
    """
    if model is None:
        raise RuntimeError("Model is not loaded.")
    arr = preprocess_image_bytes(image_bytes)
    # detect Keras-like by presence of 'predict' and 'get_config' or 'layers'
    try:
        if hasattr(model, "predict") and (TF_AVAILABLE and hasattr(model, "get_config") or hasattr(model, "layers")):
            preds = _predict_with_keras(model, arr)
        elif hasattr(model, "predict"):
            # sklearn-like
            preds = _predict_with_sklearn(model, arr)
        else:
            raise RuntimeError("Loaded model object is not callable or lacks predict().")
    except Exception as e:
        # bubble up error with context
        raise RuntimeError(f"Prediction failed: {e}")

    # normalize preds into probabilities vector and pick top
    preds = np.asarray(preds, dtype="float32")
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)
    # If preds look like class labels (strings), handle separately
    if preds.dtype.type is np.str_ or preds.dtype == object:
        # model returned labels directly
        label = str(preds[0])
        return label, 1.0

    # ensure sum > 0 and softmax if needed
    row = preds[0]
    if row.sum() <= 0:
        # fallback softmax
        exp = np.exp(row - np.max(row))
        probs = exp / exp.sum()
    else:
        probs = row / row.sum()

    top_idx = int(np.argmax(probs))
    score = float(probs[top_idx])
    label = class_names[top_idx] if top_idx < len(class_names) else str(top_idx)
    return label, score
