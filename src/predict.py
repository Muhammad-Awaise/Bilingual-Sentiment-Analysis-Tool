"""Load saved pipeline and run single / batch predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_ORDER,
    LABEL_POSITIVE,
    LIMITATIONS_BLURB,
    MAX_BATCH_ROWS,
    MODEL_ARTIFACT_PATH,
    MODEL_NAME,
    NEUTRAL_CONFIDENCE_MAX,
    NEUTRAL_MARGIN_MAX,
    NEUTRAL_MULTICLASS_FLOOR,
    SUPPORTED_LABELS,
    URDU_WEAK_SCORE_MAX,
)
from src.language_profile import detect_script_composition
from src.preprocessing import clean_text, contains_urdu_script, is_effectively_empty

# Hybrid post-hoc marker overrides — evaluated separately from the statistical model.
# Default True preserves existing behaviour; set False for pure ML predictions.
DEFAULT_USE_HYBRID_HEURISTICS = True


def _intensity_tie_break(cleaned: str) -> str | None:
    """If cleaned text has clear intensity markers, use them on near-ties."""
    pos_hits = cleaned.count("__pos_intense__")
    neg_hits = cleaned.count("__neg_intense__")
    neu_hits = cleaned.count("__neutral_cue__")
    if pos_hits > neg_hits and pos_hits > 0:
        return LABEL_POSITIVE
    if neg_hits > pos_hits and neg_hits > 0:
        return LABEL_NEGATIVE
    if neu_hits > 0 and pos_hits == 0 and neg_hits == 0:
        return LABEL_NEUTRAL
    return None


def _vectorizer_transform(pipeline, texts: list[str]):
    """Transform texts with word+char FeatureUnion or legacy tfidf step."""
    if "features" in pipeline.named_steps:
        return pipeline.named_steps["features"].transform(texts)
    if "tfidf" in pipeline.named_steps:
        return pipeline.named_steps["tfidf"].transform(texts)
    raise AttributeError("Pipeline has neither 'features' nor 'tfidf' step")



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
            feats = _vectorizer_transform(pipeline, cleaned_texts)
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


def _model_classes(pipeline) -> list[str]:
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        return []
    return [str(c) for c in classes]


def _class_proba_row(pipeline, cleaned_or_raw_text: str) -> list[float] | None:
    try:
        if hasattr(pipeline, "predict_proba"):
            row = pipeline.predict_proba([cleaned_or_raw_text])[0]
            return [float(x) for x in row]
    except Exception:
        return None
    return None


def _apply_neutral_if_uncertain(
    prediction: str,
    model_score: float | None,
    pipeline,
    proba_row: list[float] | None = None,
    cleaned_text: str | None = None,
) -> str:
    """
    Keep trained neutral predictions.

    - Multiclass: trust the classifier; on near-ties prefer intensity markers,
      else Neutral when the top score is weak.
    - Binary legacy: map very low max-probability to neutral.
    - Urdu-script safety: when the bag collapses to a near-prior score
      (~0.52–0.65) but preprocessing found clear polarity markers, trust the
      markers. This is the failure mode where every Nastaliq row gets the
      same negative prior while cleaned text still shows __pos_intense__.
    """
    pred = str(prediction)
    if pred == LABEL_NEUTRAL:
        return LABEL_NEUTRAL

    voted = _intensity_tie_break(cleaned_text) if cleaned_text else None
    # Nastaliq collapse guard: empty/OOV bags often score ~0.52–0.65 negative
    # while cleaned text still carries __pos_intense__ / __neg_intense__.
    # Prefer markers for ANY Urdu conflict under the weak-score ceiling, and
    # also when the score is the classic flat prior (~0.53) even on older apps.
    urdu_text = bool(cleaned_text and contains_urdu_script(cleaned_text))
    flat_prior = bool(
        model_score is not None and 0.50 <= float(model_score) <= 0.56
    )
    urdu_weak = bool(
        urdu_text
        and voted is not None
        and voted != pred
        and (
            flat_prior
            or (
                model_score is not None
                and float(model_score) < URDU_WEAK_SCORE_MAX
            )
        )
    )
    if urdu_weak:
        return voted if voted in SUPPORTED_LABELS else LABEL_NEUTRAL
    if urdu_text and voted is not None and voted != pred and flat_prior:
        return voted if voted in SUPPORTED_LABELS else LABEL_NEUTRAL

    classes = _model_classes(pipeline)
    if LABEL_NEUTRAL in classes:
        if proba_row and len(proba_row) >= 2:
            ordered = sorted(proba_row, reverse=True)
            margin = ordered[0] - ordered[1]
            if margin < NEUTRAL_MARGIN_MAX:
                if voted is not None:
                    return voted
                if ordered[0] < NEUTRAL_MULTICLASS_FLOOR:
                    return LABEL_NEUTRAL
        return pred if pred in SUPPORTED_LABELS else LABEL_NEUTRAL

    # Binary legacy: intensity markers before low-confidence → neutral
    if (
        voted is not None
        and voted != pred
        and model_score is not None
        and float(model_score) < URDU_WEAK_SCORE_MAX
    ):
        return voted
    if model_score is not None and float(model_score) < NEUTRAL_CONFIDENCE_MAX:
        return LABEL_NEUTRAL
    if pred in SUPPORTED_LABELS:
        return pred
    return LABEL_NEUTRAL


def predict_one(
    text: Any,
    artifact_path: str | Path | None = None,
    payload: dict[str, Any] | None = None,
    *,
    use_hybrid_heuristics: bool = DEFAULT_USE_HYBRID_HEURISTICS,
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
    composition = detect_script_composition(text)

    pred = str(pipeline.predict([text])[0])  # preprocessor runs inside vectorizer
    raw_model_pred = pred
    if pred not in SUPPORTED_LABELS:
        # Guard against unexpected label encoding on older artifacts
        if pred in {"1", "pos", LABEL_POSITIVE}:
            pred = LABEL_POSITIVE
        elif pred in {"0", "neg", LABEL_NEGATIVE}:
            pred = LABEL_NEGATIVE
        elif pred in {"neu", LABEL_NEUTRAL}:
            pred = LABEL_NEUTRAL
        else:
            pred = LABEL_NEUTRAL

    scores = _decision_score(pipeline, [text])
    proba_row = _class_proba_row(pipeline, text)
    if use_hybrid_heuristics:
        pred = _apply_neutral_if_uncertain(
            pred, scores[0], pipeline, proba_row, cleaned_text=cleaned
        )
    low_confidence = bool(
        scores[0] is not None and float(scores[0]) < NEUTRAL_MULTICLASS_FLOOR
    )
    return {
        "text": text,
        "cleaned_text": cleaned,
        "prediction": str(pred),
        "raw_model_prediction": str(raw_model_pred),
        "hybrid_heuristics_applied": bool(use_hybrid_heuristics),
        "model_name": model_name,
        "model_score": scores[0],
        "score_note": _score_note(payload),
        "calibrated": bool(payload.get("calibrated", False)),
        "language_composition": composition,
        "low_confidence": low_confidence,
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
        proba_rows: list[list[float] | None]
        try:
            if hasattr(pipeline, "predict_proba"):
                proba_rows = [
                    [float(x) for x in row]
                    for row in pipeline.predict_proba(valid_raw)
                ]
            else:
                proba_rows = [None] * len(valid_indices)
        except Exception:
            proba_rows = [None] * len(valid_indices)
        for idx, p, s, proba in zip(valid_indices, preds, sc, proba_rows):
            pred = str(p)
            if pred not in SUPPORTED_LABELS:
                if pred in {"1", "pos", LABEL_POSITIVE}:
                    pred = LABEL_POSITIVE
                elif pred in {"0", "neg", LABEL_NEGATIVE}:
                    pred = LABEL_NEGATIVE
                else:
                    pred = LABEL_NEUTRAL
            predictions[idx] = _apply_neutral_if_uncertain(
                pred,
                s,
                pipeline,
                proba,
                cleaned_text=cleaned_list[idx],
            )
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
    # If text_column is duplicated, take the first series
    text_values = df[text_column]
    if isinstance(text_values, pd.DataFrame):
        text_values = text_values.iloc[:, 0]
    texts = text_values.tolist()
    result = predict_batch(texts, artifact_path=artifact_path, payload=payload)

    out = df.copy()
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated(keep="last")]

    # Strip prior prediction/export artifacts so re-uploads never collide
    stale = {
        "row_number",
        "cleaned_text",
        "cleaned_review_text",
        "prediction",
        "predicted_sentiment",
        "model_score",
        "model_score_not_probability",
        "model_name",
        "score_note",
        "limitations",
    }
    drop = [c for c in out.columns if str(c).strip().lower() in stale]
    if drop:
        out = out.drop(columns=drop)

    # Assign fresh predictions (overwrite-safe; columns were removed above)
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
    rows = []
    for label in LABEL_ORDER:
        count = int(counts.get(label, 0))
        # Omit empty neutral so binary-only batches stay two-row summaries
        if label == LABEL_NEUTRAL and count == 0:
            continue
        rows.append(
            {
                "sentiment": label,
                "count": count,
                "percentage": round(100.0 * count / total, 2),
            }
        )
    return pd.DataFrame(rows)
