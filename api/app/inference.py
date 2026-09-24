"""
Model loading and prediction.

The model is loaded once at startup (see main.py's lifespan) and kept in
memory, not reloaded per request. Resize/normalization are assumed to be
baked into the model itself as layers (see training/export.py), so this
module sends raw decoded pixels and never duplicates that preprocessing -
duplicating it is the classic source of train/serve mismatch.
"""

import json
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.schemas import CareInfo, ClassScore, PredictResponse
from app.knowledge import get_care_info

logger = logging.getLogger("croplens.inference")

TOP_K = 3

_model = None
_labels: list[str] = []


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes cannot be decoded as an image."""


def load_model() -> None:
    """Load the trained model and class labels into memory. Call once at
    startup. Raises if either file is missing - the app should fail to
    start rather than serve requests with no model."""
    global _model, _labels
    settings = get_settings()

    # Imported lazily so `python -m app.config` etc. don't require
    # TensorFlow, and so tests that stub this module stay fast.
    import tensorflow as tf

    model_path = Path(settings.model_path)
    labels_path = Path(settings.labels_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run the training pipeline (see training/) or check MODEL_PATH."
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found at {labels_path}.")

    logger.info("Loading model from %s", model_path)
    _model = tf.keras.models.load_model(model_path)

    with labels_path.open(encoding="utf-8") as f:
        _labels = json.load(f)

    if len(_labels) < 2:
        raise ValueError("labels.json must contain at least 2 classes.")

    logger.info("Model loaded. Classes: %s", _labels)


def is_loaded() -> bool:
    return _model is not None


def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw upload bytes into an RGB numpy array (H, W, 3), uint8.
    No resize here - that happens inside the model."""
    try:
        image = Image.open(BytesIO(file_bytes))
        image.load()  # force full decode now, so a truncated file raises here
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError("File is not a readable image.") from e

    image = image.convert("RGB")
    return np.array(image)


def predict(file_bytes: bytes) -> PredictResponse:
    """Decode an image, run the model, and attach care info.

    Raises InvalidImageError for bad input, RuntimeError if called before
    load_model().
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    settings = get_settings()
    image_array = _decode_image(file_bytes)
    batch = np.expand_dims(image_array, axis=0)  # (1, H, W, 3)

    raw_predictions = _model.predict(batch, verbose=0)[0]  # (num_classes,)

    # Defensive: apply softmax only if the model's output layer doesn't
    # already do so (outputs summing to ~1 are left as-is).
    if not np.isclose(raw_predictions.sum(), 1.0, atol=1e-3):
        exp = np.exp(raw_predictions - raw_predictions.max())
        raw_predictions = exp / exp.sum()

    top_indices = np.argsort(raw_predictions)[::-1][:TOP_K]
    top_k = [
        ClassScore(label=_labels[i], confidence=float(raw_predictions[i]))
        for i in top_indices
    ]

    best = top_k[0]
    low_confidence = best.confidence < settings.confidence_threshold

    care: CareInfo | None = None if low_confidence else get_care_info(best.label)

    return PredictResponse(
        label=best.label,
        confidence=best.confidence,
        low_confidence=low_confidence,
        top_k=top_k,
        care=care,
        model_version=settings.model_version,
    )
