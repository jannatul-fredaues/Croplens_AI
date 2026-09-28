"""
Tests for the FastAPI routes in app.main. Uses APP_ENV=test (set below,
before importing the app) so the lifespan handler skips loading the real
model - see the `if settings.app_env != "test"` check in app/main.py.

Prediction logic itself is unit-tested in test_inference.py; here we only
check routing, validation, and status codes, by monkeypatching
app.main.predict for the success/error-mapping cases.

Run: pytest tests/test_api.py
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("APP_ENV", "test")

from fastapi.testclient import TestClient  # noqa: E402

from app.inference import InvalidImageError  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import ClassScore, PredictResponse  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)


def _upload(filename: str, content_type: str):
    path = FIXTURES / filename
    return {"file": (filename, path.read_bytes(), content_type)}


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_version" in body
    assert "model_loaded" in body


def test_predict_rejects_wrong_content_type():
    response = client.post("/predict", files=_upload("sample_onion.jpg", "text/plain"))
    assert response.status_code == 415


def test_predict_rejects_empty_file():
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_predict_rejects_oversized_file(monkeypatch):
    from app.main import settings

    monkeypatch.setattr(settings, "max_upload_mb", 0)  # 0 MB -> anything trips the limit
    files = {"file": ("big.jpg", b"x" * 1024, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413


def test_predict_success(monkeypatch):
    fake_result = PredictResponse(
        label="onion",
        confidence=0.94,
        low_confidence=False,
        top_k=[
            ClassScore(label="onion", confidence=0.94),
            ClassScore(label="sweet_pea", confidence=0.04),
            ClassScore(label="black_cumin", confidence=0.02),
        ],
        care=None,
        model_version="v1",
    )
    monkeypatch.setattr("app.main.predict", lambda file_bytes: fake_result)

    response = client.post("/predict", files=_upload("sample_onion.jpg", "image/jpeg"))

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "onion"
    assert body["confidence"] == pytest.approx(0.94)
    assert len(body["top_k"]) == 3


def test_predict_invalid_image_returns_400(monkeypatch):
    def _raise(file_bytes):
        raise InvalidImageError("File is not a readable image.")

    monkeypatch.setattr("app.main.predict", _raise)

    response = client.post("/predict", files=_upload("sample_onion.jpg", "image/jpeg"))

    assert response.status_code == 400


def test_predict_model_not_ready_returns_503(monkeypatch):
    def _raise(file_bytes):
        raise RuntimeError("Model not loaded.")

    monkeypatch.setattr("app.main.predict", _raise)

    response = client.post("/predict", files=_upload("sample_onion.jpg", "image/jpeg"))

    assert response.status_code == 503


def test_cors_rejects_unlisted_origin():
    response = client.get("/health", headers={"Origin": "https://evil.example.com"})
    # The request itself still succeeds (CORS is enforced browser-side via
    # headers, not by blocking the request) - assert no matching
    # Access-Control-Allow-Origin is echoed back for a disallowed origin.
    assert response.headers.get("access-control-allow-origin") != "https://evil.example.com"
