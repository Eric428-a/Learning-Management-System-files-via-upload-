# utils/preprocessing.py
"""
Image preprocessing utilities.
Handles reading bytes, resizing to model input, normalizing, and batch dimension.
"""

from PIL import Image, ImageOps
import numpy as np
import io

TARGET_SIZE = (64, 64)  # model input (width, height)

def preprocess_image_bytes(image_bytes: bytes):
    """
    Convert raw image bytes to a NumPy array ready for model.predict:
      - open bytes via PIL
      - convert to RGB
      - resize with bilinear interpolation
      - normalize to [0,1]
      - return shape (1, H, W, 3) dtype float32
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Some images may be smaller or have EXIF orientation tags
    img = ImageOps.exif_transpose(img)
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype="float32") / 255.0
    # ensure shape and dtype
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr
