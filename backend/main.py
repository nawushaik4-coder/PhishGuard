
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from detection.url_analyzer import analyse_url
from database import save_scan, get_scan_history
from ml.feature_extractor import extract_features
from ml.predict import predict_features

app = FastAPI(title="PhishGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ScanRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


@app.get("/")
def home():
    return {"message": "Welcome to PhishGuard!"}


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/scan")
def scan_url(request: ScanRequest):
    # Existing rule-based detector.
    rule_result = analyse_url(request.url)

    # ML detector: only run when URL feature extraction succeeds.
    try:
        features = extract_features(request.url)
        ml_result = predict_features(features)
        ml_error = None
    except (ValueError, FileNotFoundError) as exc:
        ml_result = None
        ml_error = str(exc)

    # Keep the existing top-level fields for the current frontend
    # and preserve the existing MySQL history format.
    scan_id = save_scan(request.url, rule_result)

    return {
        "scan_id": scan_id,
        "url": request.url,
        **rule_result,
        "rule_based": rule_result,
        "ml_prediction": ml_result,
        "ml_error": ml_error,
        "notice": (
            "Rule-based and ML results are preliminary indicators, "
            "not proof that a URL is safe or malicious. The ML score "
            "has not been calibrated for real-world phishing risk."
        ),
    }


@app.get("/history")
def scan_history(limit: int = Query(default=20, ge=1, le=100)):
    records = get_scan_history(limit)

    return {
        "count": len(records),
        "history": records,
    }
