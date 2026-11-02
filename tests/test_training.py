"""Tests for training and artifact persistence."""

from pathlib import Path

import joblib
import pytest

from src.config import SUPPORTED_LABELS
from src.data_loader import DatasetError
from src.train import build_pipeline, majority_baseline_accuracy, train_and_save


def test_majority_baseline():
    assert majority_baseline_accuracy(["positive", "positive", "negative"]) == pytest.approx(
        2 / 3
    )


def test_train_save_load(synthetic_csv: Path, tmp_path: Path):
    artifact = tmp_path / "m.joblib"
    reports = tmp_path / "reports"
    meta = train_and_save(
        data_path=synthetic_csv,
        artifact_path=artifact,
        reports_dir=reports,
        write_processed=False,
    )
    assert artifact.exists()
    assert meta["train_size"] > 0
    assert meta["test_size"] > 0
    assert (reports / "train_meta.json").exists()
    assert (reports / "holdout_split.joblib").exists()

    payload = joblib.load(artifact)
    assert "pipeline" in payload
    assert payload.get("class_weight") == "balanced"
    # logreg uses native predict_proba (no CalibratedClassifierCV wrapper)
    assert "pipeline" in payload
    pipe = payload["pipeline"]
    assert "features" in pipe.named_steps
    pred = pipe.predict(["bohat acha product hai"])[0]
    assert pred in SUPPORTED_LABELS
    # Isolation: project reports must not be the write target
    assert Path(meta["reports_dir"]) == reports


def test_pipeline_not_fitted_before_train():
    pipe = build_pipeline()
    # Unfitted pipeline should raise on predict
    with pytest.raises(Exception):
        pipe.predict(["hello"])


def test_training_clear_error_on_missing(tmp_path: Path):
    with pytest.raises((DatasetError, SystemExit)):
        train_and_save(
            data_path=tmp_path / "missing.csv",
            reports_dir=tmp_path / "reports",
            write_processed=False,
        )


def test_train_does_not_pollute_project_reports(
    synthetic_csv: Path, tmp_path: Path, project_root: Path
):
    """Regression: fixtures must not overwrite reports/train_meta.json."""
    project_meta = project_root / "reports" / "train_meta.json"
    before = project_meta.read_text(encoding="utf-8") if project_meta.exists() else None
    train_and_save(
        data_path=synthetic_csv,
        artifact_path=tmp_path / "m.joblib",
        reports_dir=tmp_path / "reports",
        write_processed=False,
    )
    after = project_meta.read_text(encoding="utf-8") if project_meta.exists() else None
    assert before == after


def test_consistent_prediction_after_reload(trained_artifact: Path):
    payload = joblib.load(trained_artifact)
    pipe = payload["pipeline"]
    text = "terrible quality waste of money"
    a = pipe.predict([text])[0]
    payload2 = joblib.load(trained_artifact)
    b = payload2["pipeline"].predict([text])[0]
    assert a == b
