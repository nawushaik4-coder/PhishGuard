"""Prediction adapter for the full-URL TF-IDF model.

Maintains predict_features(dict) compatibility with the existing main.py.
"""
from __future__ import annotations

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "url_model.joblib"
MODEL_TYPE = "PhishGuard-full-url-char-tfidf-logistic-v1"
_ARTIFACT = None


def load_model():
    global _ARTIFACT
    if _ARTIFACT is not None:
        return _ARTIFACT
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Full-URL ML model is missing. From backend/, run: python -m ml.train_model"
        )
    # Load only an artifact that you trained locally. joblib files are pickle-based.
    artifact = joblib.load(MODEL_PATH)
    if artifact.get("model_type") != MODEL_TYPE:
        raise ValueError("Unexpected model artifact. Retrain using ml/train_model.py.")
    for key in ("vectorizer", "model", "threshold"):
        if key not in artifact:
            raise ValueError(f"Model artifact is missing '{key}'. Retrain the model.")
    _ARTIFACT = artifact
    return _ARTIFACT


def predict_features(feature_values: dict) -> dict:
    """Predict from extract_features(url)'s {url_text: ...} result."""
    if not isinstance(feature_values, dict) or not isinstance(feature_values.get("url_text"), str):
        raise ValueError("Expected URL text from ml.feature_extractor.extract_features().")
    url_text = feature_values["url_text"]
    artifact = load_model()
    vector = artifact["vectorizer"].transform([url_text])
    probability = float(artifact["model"].predict_proba(vector)[0, 1])
    label = int(probability >= float(artifact["threshold"]))
    return {
        "predicted_label": label,
        "verdict": "Potential phishing" if label == 1 else "Likely legitimate",
        "risk_score": round(probability * 100.0, 2),
        "decision_threshold": float(artifact["threshold"]),
        "note": (
            "Full-URL character n-gram model score is an estimate trained on urlset.csv. "
            "It is not a guarantee and should be validated against newer, independent URLs."
        ),
    }
