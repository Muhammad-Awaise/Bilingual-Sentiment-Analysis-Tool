"""Application-level tests (import, missing model, CSV handling)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from src.config import MODEL_ARTIFACT_PATH, SUPPORTED_LABELS
from src.predict import PredictionError, predict_dataframe, predict_one


def test_app_module_imports():
    mod = importlib.import_module("app")
    assert hasattr(mod, "main")
    assert hasattr(mod, "page_single")


def test_app_has_no_api_key_requirement():
    """Smoke: source should not reference paid API keys."""
    text = Path("app.py").read_text(encoding="utf-8").lower()
    for forbidden in ("openai", "api_key", "anthropic", "paid"):
        assert forbidden not in text


def test_empty_input_handled(trained_artifact: Path):
    with pytest.raises(PredictionError):
        predict_one("", artifact_path=trained_artifact)


def test_predict_with_trained_model(trained_artifact: Path):
    result = predict_one(
        "excellent quality highly recommended", artifact_path=trained_artifact
    )
    assert result["prediction"] in SUPPORTED_LABELS


def test_missing_model_message(tmp_path: Path):
    with pytest.raises(PredictionError, match="not found"):
        predict_one("hello", artifact_path=tmp_path / "missing.joblib")


def test_csv_upload_valid(trained_artifact: Path):
    df = pd.DataFrame({"review": ["great product love it", "terrible quality"]})
    out = predict_dataframe(df, "review", artifact_path=trained_artifact)
    assert "prediction" in out.columns
    assert len(out) == 2


def test_csv_missing_column(trained_artifact: Path):
    df = pd.DataFrame({"x": ["a"]})
    with pytest.raises(PredictionError, match="not found"):
        predict_dataframe(df, "review", artifact_path=trained_artifact)


def test_csv_empty(trained_artifact: Path):
    df = pd.DataFrame({"review": []})
    with pytest.raises(PredictionError, match="empty"):
        predict_dataframe(df, "review", artifact_path=trained_artifact)


def test_download_columns(trained_artifact: Path):
    df = pd.DataFrame({"review": ["amazing value for money"]})
    out = predict_dataframe(df, "review", artifact_path=trained_artifact)
    for col in ("prediction", "cleaned_text", "model_name"):
        assert col in out.columns


def test_user_input_not_executed(trained_artifact: Path):
    """Ensure malicious-looking text is treated as data, not code."""
    sneaky = "__import__('os').system('echo hacked')"
    result = predict_one(sneaky, artifact_path=trained_artifact)
    assert result["prediction"] in SUPPORTED_LABELS
    # Cleaning may keep tokens; the important property is we never exec() input.
    assert isinstance(result["cleaned_text"], str)
    assert "prediction" in result


def test_streamlit_apptest_optional():
    """
    If Streamlit AppTest is available and a project model exists, run a smoke start.
    Otherwise skip — core prediction paths are covered above without UI chrome.
    """
    pytest.importorskip("streamlit.testing.v1")
    if not MODEL_ARTIFACT_PATH.exists():
        pytest.skip("Project model artifact not trained yet")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception
