"""Tests for prediction helpers."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import SUPPORTED_LABELS
from src.predict import (
    PredictionError,
    predict_batch,
    predict_dataframe,
    predict_one,
    sentiment_summary,
)


def test_predict_one_valid(trained_artifact: Path):
    result = predict_one("great product love it", artifact_path=trained_artifact)
    assert result["prediction"] in SUPPORTED_LABELS
    assert result["cleaned_text"]
    assert "limitations" in result
    assert result["model_name"]


def test_predict_empty_raises(trained_artifact: Path):
    with pytest.raises(PredictionError):
        predict_one("   ", artifact_path=trained_artifact)


def test_predict_none_raises(trained_artifact: Path):
    with pytest.raises(PredictionError):
        predict_one(None, artifact_path=trained_artifact)


def test_prediction_in_label_set(trained_artifact: Path):
    for text in ["bohat acha", "bilkul bekaar", "excellent quality"]:
        pred = predict_one(text, artifact_path=trained_artifact)["prediction"]
        assert pred in SUPPORTED_LABELS


def test_consistent_same_model(trained_artifact: Path):
    text = "awful product broke immediately"
    a = predict_one(text, artifact_path=trained_artifact)["prediction"]
    b = predict_one(text, artifact_path=trained_artifact)["prediction"]
    assert a == b


def test_missing_artifact(tmp_path: Path):
    with pytest.raises(PredictionError, match="not found"):
        predict_one("hello", artifact_path=tmp_path / "nope.joblib")


def test_batch_and_summary(trained_artifact: Path):
    df = predict_batch(
        ["great product", "", "terrible quality"],
        artifact_path=trained_artifact,
    )
    assert len(df) == 3
    assert df.loc[1, "prediction"] == "invalid_or_empty"
    summary = sentiment_summary(df["prediction"])
    assert "sentiment" in summary.columns


def test_predict_dataframe(trained_artifact: Path):
    frame = pd.DataFrame({"review": ["amazing value", "worst purchase"]})
    out = predict_dataframe(frame, "review", artifact_path=trained_artifact)
    assert "prediction" in out.columns
    assert list(out["prediction"])[0] in SUPPORTED_LABELS


def test_predict_dataframe_missing_column(trained_artifact: Path):
    frame = pd.DataFrame({"other": ["x"]})
    with pytest.raises(PredictionError, match="not found"):
        predict_dataframe(frame, "review", artifact_path=trained_artifact)
