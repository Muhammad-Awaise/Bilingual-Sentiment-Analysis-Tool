"""Load saved pipeline and run single / batch predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    LIMITATIONS_BLURB,
    MAX_BATCH_ROWS,
    MODEL_ARTIFACT_PATH,
    MODEL_NAME,
    SUPPORTED_LABELS,
)
from src.preprocessing import clean_text, is_effectively_empty


class PredictionError(ValueError):
    """Raised for invalid input or missing model artifacts."""


def load_model(artifact_path: str | Path | None = None) -> dict[str, Any]:
    """Load joblib payload containing the fitted sklearn Pipeline."""
    path = Path(artifact_path) if artifact_path else MODEL_ARTIFACT_PATH
    if not path.exists():
        raise PredictionError(
            f"Model artifact not found at {path}. "
            "Train first with: python -m src.train"
        )
    try:
        payload = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"Could not load model artifact: {exc}") from exc

    if "pipeline" not in payload:
        raise PredictionError("Artifact is missing the 'pipeline' key.")
    return payload


def _decision_score(pipeline, cleaned_texts: list[str]) -> list[float | None]:
    """
    Optional model score. Prefers max class probability from predict_proba
    (sigmoid-calibrated when the artifact used CalibratedClassifierCV).
    Falls back to absolute decision_function magnitude.
    """
    clf = pipeline.named_steps.get("clf")
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(cleaned_texts)
            return [float(np.max(row)) for row in proba]
        if hasattr(clf, "decision_function"):
            feats = pipeline.named_steps["tfidf"].transform(cleaned_texts)
            scores = clf.decision_function(feats)
            scores = np.atleast_1d(scores).astype(float)
            return [float(abs(s)) for s in scores]
    except Exception:
        return [None] * len(cleaned_texts)
    return [None] * len(cleaned_texts)


def _score_note(payload: dict[str, Any]) -> str:
    if payload.get("calibrated"):
        return (
            "Model score is the max sigmoid-calibrated class probability "
            "(approximate confidence, not a guarantee of correctness)."
        )
    return (
        "Model score is NOT a calibrated probability of correctness; "
        "it is a relative classifier signal."
    )


def predict_one(
    text: Any,
    artifact_path: str | Path | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict sentiment for a single review."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        raise PredictionError("Please enter a review. Empty or missing text is not allowed.")
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        raise PredictionError("Please enter a review. Empty or whitespace-only text is not allowed.")

    cleaned = clean_text(text)
    if not cleaned:
        raise PredictionError(
            "After cleaning, the text was empty. Try a longer review without only URLs or symbols."
        )

    payload = payload or load_model(artifact_path)
    pipeline = payload["pipeline"]
    model_name = payload.get("model_name", MODEL_NAME)

    pred = pipeline.predict([text])[0]  # preprocessor runs inside vectorizer
    if pred not in SUPPORTED_LABELS:
        # Guard against unexpected label encoding
        pred = LABEL_POSITIVE if str(pred) in {"1", "pos", LABEL_POSITIVE} else LABEL_NEGATIVE

    scores = _decision_score(pipeline, [text])
    return {
        "text": text,
        "cleaned_text": cleaned,
        "prediction": str(pred),
        "model_name": model_name,
        "model_score": scores[0],
        "score_note": _score_note(payload),
        "calibrated": bool(payload.get("calibrated", False)),
        "limitations": LIMITATIONS_BLURB,
    }


def predict_batch(
    texts: list[Any],
    artifact_path: str | Path | None = None,
    payload: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Predict for a list of texts; empty rows get a clear error marker."""
    if texts is None:
        raise PredictionError("No texts provided.")
    if len(texts) == 0:
        raise PredictionError("The text list is empty.")
    if len(texts) > MAX_BATCH_ROWS:
        raise PredictionError(
            f"Too many rows ({len(texts)}). Maximum allowed is {MAX_BATCH_ROWS}."
        )

    payload = payload or load_model(artifact_path)
    pipeline = payload["pipeline"]
    model_name = payload.get("model_name", MODEL_NAME)

    cleaned_list = []
    valid_mask = []
    for t in texts:
        if is_effectively_empty(t):
            cleaned_list.append("")
            valid_mask.append(False)
        else:
            cleaned_list.append(clean_text(t))
            valid_mask.append(True)

    predictions = [None] * len(texts)
    scores = [None] * len(texts)

    valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
    if valid_indices:
        valid_raw = [texts[i] if isinstance(texts[i], str) else str(texts[i]) for i in valid_indices]
        preds = pipeline.predict(valid_raw)
        sc = _decision_score(pipeline, valid_raw)
        for idx, p, s in zip(valid_indices, preds, sc):
            predictions[idx] = str(p)
            scores[idx] = s

    rows = []
    for i, t in enumerate(texts):
        rows.append(
            {
                "text": t if isinstance(t, str) else ("" if t is None else str(t)),
                "cleaned_text": cleaned_list[i],
                "prediction": predictions[i]
                if predictions[i] is not None
                else "invalid_or_empty",
                "model_score": scores[i],
                "model_name": model_name,
            }
        )
    return pd.DataFrame(rows)


def predict_dataframe(
    df: pd.DataFrame,
    text_column: str,
    artifact_path: str | Path | None = None,
    payload: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run batch prediction on a DataFrame column."""
    if df is None or df.empty:
        raise PredictionError("Uploaded table is empty.")
    if text_column not in df.columns:
        raise PredictionError(
            f"Column '{text_column}' not found. Available: {list(df.columns)}"
        )
    texts = df[text_column].tolist()
    result = predict_batch(texts, artifact_path=artifact_path, payload=payload)
    # Preserve original columns + predictions
    out = df.copy()
    out["cleaned_text"] = result["cleaned_text"].values
    out["prediction"] = result["prediction"].values
    out["model_score"] = result["model_score"].values
    out["model_name"] = result["model_name"].values
    return out


def sentiment_summary(predictions: pd.Series | list[str]) -> pd.DataFrame:
    """Aggregate counts and percentages for valid sentiment labels."""
    series = pd.Series(predictions)
    series = series[series.isin(SUPPORTED_LABELS)]
    if series.empty:
        return pd.DataFrame(columns=["sentiment", "count", "percentage"])
    counts = series.value_counts()
    total = int(counts.sum())
    rows = [
        {
            "sentiment": label,
            "count": int(counts.get(label, 0)),
            "percentage": round(100.0 * int(counts.get(label, 0)) / total, 2),
        }
        for label in (LABEL_NEGATIVE, LABEL_POSITIVE)
    ]
    return pd.DataFrame(rows)
