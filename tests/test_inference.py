"""
Unit tests for app.inference. These stub the model directly (monkeypatch
inference._model / _labels) rather than loading a real .keras file, so
they run fast in CI and don't depend on models/v1/ existing yet.

Run: pytest tests/test_inference.py
"""

from pathlib import Path

import numpy as np
import pytest

from app import inference

FIXTURES = Path(__file__).parent / "fixtures"


class FakeModel:
    """Stands in for a loaded Keras model: .predict(batch, verbose=0)
    returns a fixed set of class probabilities, regardless of input."""

    def __init__(self, probs: list[float]):
        self._probs = np.array([probs], dtype="float32")  # shape (1, n_classes)

    def predict(self, batch, verbose=0):
        return self._probs


@pytest.fixture(autouse=True)
def reset_inference_state():
    """Ensure each test starts with a clean module state, and that a
    stub set in one test never leaks into another."""
    inference._model = None
    inference._labels = []
    yield
    inference._model = None
    inference._labels = []


def test_predict_before_load_raises():
    with pytest.raises(RuntimeError):
        inference.predict((FIXTURES / "sample_onion.jpg").read_bytes())


def test_decode_invalid_image_raises():
    inference._model = FakeModel([0.34, 0.33, 0.33])
    inference._labels = ["black_cumin", "sweet_pea", "onion"]

    with pytest.raises(inference.InvalidImageError):
        inference.predict((FIXTURES / "corrupt.jpg").read_bytes())


def test_predict_returns_top_class_and_sorted_top_k():
    inference._model = FakeModel([0.10, 0.85, 0.05])
    inference._labels = ["black_cumin", "sweet_pea", "onion"]

    result = inference.predict((FIXTURES / "sample_sweet_pea.jpg").read_bytes())

    assert result.label == "sweet_pea"
    assert result.confidence == pytest.approx(0.85, abs=1e-6)
    assert [c.label for c in result.top_k] == ["sweet_pea", "black_cumin", "onion"]
    assert result.model_version  # comes from settings, just check it's set


def test_low_confidence_flag_and_no_care_info():
    # Below the default 0.70 threshold (see app/config.py)
    inference._model = FakeModel([0.40, 0.35, 0.25])
    inference._labels = ["black_cumin", "sweet_pea", "onion"]

    result = inference.predict((FIXTURES / "sample_black_cumin.jpg").read_bytes())

    assert result.low_confidence is True
    # Care info is intentionally withheld when we're not confident,
    # even if crop_info.json has an entry for the top label.
    assert result.care is None


def test_high_confidence_flag():
    inference._model = FakeModel([0.05, 0.05, 0.90])
    inference._labels = ["black_cumin", "sweet_pea", "onion"]

    result = inference.predict((FIXTURES / "sample_onion.jpg").read_bytes())

    assert result.low_confidence is False


def test_top_k_capped_at_three(monkeypatch):
    monkeypatch.setattr(inference, "TOP_K", 3)
    inference._model = FakeModel([0.4, 0.3, 0.2, 0.1])
    inference._labels = ["a", "b", "c", "d"]

    result = inference.predict((FIXTURES / "sample_onion.jpg").read_bytes())

    assert len(result.top_k) == 3
