"""Train a full-URL character n-gram model for PhishGuard.

Run from backend/: python -m ml.train_model

This accuracy-focused variant preserves path/query text from the `domain`
column when present. It writes url_model.joblib and url_metrics.json so it
will not overwrite the previous PyTorch model or metrics.
"""
from __future__ import annotations

import html
import json
import random
import re
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from ml.feature_extractor import normalize_url_text

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_CANDIDATES = [
    DATA_DIR / "urlset.csv",
    BASE_DIR / "urlset.csv",
    DATA_DIR / "phishing_urls.csv",
    DATA_DIR / "phishing_url.csv",
]
MODEL_PATH = MODEL_DIR / "url_model.joblib"
METRICS_PATH = MODEL_DIR / "url_metrics.json"
MODEL_TYPE = "PhishGuard-full-url-char-tfidf-logistic-v1"


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def normalise_column_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def choose_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    normalized = {normalise_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalise_column_name(candidate)
        if key in normalized:
            return str(normalized[key])
    raise ValueError(
        f"Could not identify the {purpose} column. Columns found: {list(df.columns)}. "
        f"Expected one of {candidates}."
    )


def map_label(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return int(number) if number in (0.0, 1.0) else None
    text = str(value).strip().lower()
    if text in {"1", "phishing", "phish", "malicious", "malware", "bad", "fraud", "suspicious"}:
        return 1
    if text in {"0", "legitimate", "legit", "benign", "good", "safe", "clean", "non-phishing", "nonphishing"}:
        return 0
    return None


def host_from_normalized_text(text: str) -> str:
    # normalize_url_text removes the scheme; add one only for parsing.
    parsed = urlsplit("http://" + text)
    if not parsed.hostname:
        raise ValueError("URL has no valid hostname.")
    return parsed.hostname.lower().rstrip(".")


def main() -> None:
    seed_everything()
    data_path = next((path for path in DATA_CANDIDATES if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            f"Could not find urlset.csv. Expected {DATA_DIR / 'urlset.csv'}."
        )

    print(f"Loading dataset: {data_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(data_path, encoding="latin1", engine="python", on_bad_lines="skip")
    df.columns = [str(column).strip() for column in df.columns]

    url_col = choose_column(
        df,
        ["url", "urls", "phishing_url", "website", "web_site", "domain", "link", "uri"],
        "URL/domain",
    )
    label_col = choose_column(
        df,
        ["label", "labels", "status", "type", "result", "class", "target", "is_phishing"],
        "label",
    )
    print(f"URL/domain column: {url_col!r}; label column: {label_col!r}")

    records: list[tuple[str, str, int]] = []
    skipped = 0
    for raw_url, raw_label in zip(df[url_col], df[label_col]):
        label = map_label(raw_label)
        if pd.isna(raw_url) or label is None:
            skipped += 1
            continue
        try:
            # Dataset contains many values without a scheme. Parse those safely,
            # while retaining hostname + path + query for the model.
            url_text = normalize_url_text(str(raw_url), allow_bare_domain=True)
            host = host_from_normalized_text(url_text)
            records.append((url_text, host, int(label)))
        except (ValueError, TypeError):
            skipped += 1

    data = pd.DataFrame(records, columns=["url_text", "host", "label"])
    if data.empty:
        raise ValueError("No usable URL rows were found. Check the dataset columns.")

    # Avoid contradictory labels for the same host and prevent a hostname from
    # leaking between train/dev/test. This is stricter than random URL splitting.
    conflicts = set(data.groupby("host")["label"].nunique().loc[lambda s: s > 1].index)
    if conflicts:
        print(f"Removing {len(conflicts):,} hostnames with conflicting labels.")
        data = data[~data["host"].isin(conflicts)].copy()
    data = data.drop_duplicates(subset=["url_text"], keep="first").reset_index(drop=True)

    if data["label"].nunique() != 2:
        raise ValueError("Dataset must include both label classes 0 and 1.")

    print(f"Usable full URL records: {len(data):,}; skipped input rows: {skipped:,}")
    print(f"Unique hostnames: {data['host'].nunique():,}")
    print("ASSUMPTION: label 0=legitimate, label 1=phishing. Verify the dataset documentation.")

    # Split unique hosts first so URLs from one hostname cannot appear in both
    # training and evaluation partitions.
    host_table = data[["host", "label"]].drop_duplicates(subset=["host"]).reset_index(drop=True)
    train_hosts, temp_hosts = train_test_split(
        host_table["host"].to_numpy(),
        test_size=0.20,
        random_state=SEED,
        stratify=host_table["label"].to_numpy(),
    )
    temp_table = host_table[host_table["host"].isin(temp_hosts)]
    dev_hosts, test_hosts = train_test_split(
        temp_table["host"].to_numpy(),
        test_size=0.50,
        random_state=SEED,
        stratify=temp_table["label"].to_numpy(),
    )

    host_series = data["host"]
    train_mask = host_series.isin(train_hosts).to_numpy()
    dev_mask = host_series.isin(dev_hosts).to_numpy()
    test_mask = host_series.isin(test_hosts).to_numpy()
    train_rows = np.flatnonzero(train_mask)
    dev_rows = np.flatnonzero(dev_mask)
    test_rows = np.flatnonzero(test_mask)

    X_text = data["url_text"].to_numpy()
    y = data["label"].to_numpy(dtype=np.int64)
    print(f"Split sizes — train: {len(train_rows):,}; dev: {len(dev_rows):,}; test: {len(test_rows):,}")
    print("Hostname overlap between splits: 0 (grouped split)")
    print("Train label counts [0, 1]:", np.bincount(y[train_rows], minlength=2).tolist())

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    print("Vectorizing full URL text with character n-grams...")
    X_train = vectorizer.fit_transform(X_text[train_rows])
    X_dev = vectorizer.transform(X_text[dev_rows])
    X_test = vectorizer.transform(X_text[test_rows])
    print(f"TF-IDF matrix: {X_train.shape[0]:,} training rows × {X_train.shape[1]:,} features")

    model = LogisticRegression(C=4.0, max_iter=150, solver="liblinear", random_state=SEED)
    print("Training LogisticRegression classifier...")
    model.fit(X_train, y[train_rows])

    # Development metrics are informational. Test set is not used for selection.
    dev_pred = model.predict(X_dev)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_pred = (test_probabilities >= 0.5).astype(np.int64)

    accuracy = float(accuracy_score(y[test_rows], test_pred))
    precision = float(precision_score(y[test_rows], test_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y[test_rows], test_pred, pos_label=1, zero_division=0))
    phishing_f1 = float(f1_score(y[test_rows], test_pred, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y[test_rows], test_pred, average="macro", zero_division=0))
    matrix = confusion_matrix(y[test_rows], test_pred, labels=[0, 1])

    print(f"Dev accuracy at threshold 0.50: {accuracy_score(y[dev_rows], dev_pred):.4f}")
    print("\n========== HOSTNAME-HELD-OUT TEST RESULTS ==========")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Phishing precision: {precision:.4f}")
    print(f"Phishing recall: {recall:.4f}")
    print(f"Phishing F1: {phishing_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion matrix (actual rows; predicted columns; labels [0, 1]):\n", matrix)
    print(classification_report(y[test_rows], test_pred, labels=[0, 1], target_names=["Legitimate", "Phishing"], zero_division=0))

    artifact = {
        "model_type": MODEL_TYPE,
        "vectorizer": vectorizer,
        "model": model,
        "threshold": 0.5,
        "label_mapping": {"0": "legitimate (assumed; verify dataset docs)", "1": "phishing (assumed; verify dataset docs)"},
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)

    metrics = {
        "model_type": MODEL_TYPE,
        "dataset": data_path.name,
        "rows_used": int(len(data)),
        "unique_hosts": int(data["host"].nunique()),
        "features": "character TF-IDF n-grams (3–5) over normalized full URL text",
        "vectorizer_max_features": 120000,
        "classifier": "sklearn LogisticRegression(C=4.0, solver='liblinear')",
        "split_method": "Stratified host-group split; same hostname does not cross splits",
        "split_sizes_rows": {"train": int(len(train_rows)), "dev": int(len(dev_rows)), "test": int(len(test_rows))},
        "test_accuracy": round(accuracy, 6),
        "test_precision_phishing": round(precision, 6),
        "test_recall_phishing": round(recall, 6),
        "test_f1_phishing": round(phishing_f1, 6),
        "test_macro_f1": round(macro_f1, 6),
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "classification_report": classification_report(y[test_rows], test_pred, labels=[0, 1], target_names=["Legitimate", "Phishing"], output_dict=True, zero_division=0),
        "label_mapping_note": "Confirm in the dataset documentation that 0=legitimate and 1=phishing.",
        "caution": "Single-dataset evaluation; not a real-world guarantee. Use an external or future-dated dataset for a stronger final claim.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved live full-URL model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("Previous PyTorch artifacts were not overwritten.")


if __name__ == "__main__":
    main()
