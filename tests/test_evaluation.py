"""Tests for evaluation metrics (never hardcoded)."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluate import (
    EvaluationError,
    compute_metrics,
    compute_pr_auc,
    per_language_metrics,
    plot_confusion_matrix,
)


def test_metrics_from_predictions():
    y_true = ["positive", "negative", "positive", "negative"]
    y_pred = ["positive", "negative", "negative", "negative"]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["n_samples"] == 4
    assert 0.0 <= metrics["accuracy"] <= 1.0
    expected_f1 = f1_score(
        y_true, y_pred, labels=["negative", "positive"], average="macro", zero_division=0
    )
    assert metrics["f1_macro"] == pytest.approx(expected_f1)


def test_confusion_matrix_shape():
    y_true = ["positive", "negative", "positive"]
    y_pred = ["positive", "positive", "negative"]
    metrics = compute_metrics(y_true, y_pred, labels=["negative", "positive"])
    cm = np.asarray(metrics["confusion_matrix"])
    assert cm.shape == (2, 2)


def test_imbalanced_missing_class_handled():
    y_true = ["positive", "positive", "positive"]
    y_pred = ["positive", "negative", "positive"]
    metrics = compute_metrics(y_true, y_pred)
    assert np.isfinite(metrics["f1_macro"])
    assert np.isfinite(metrics["precision_macro"])


def test_empty_raises():
    with pytest.raises(EvaluationError):
        compute_metrics([], [])


def test_plot_confusion_matrix(tmp_path):
    path = plot_confusion_matrix([[2, 1], [0, 3]], ["negative", "positive"], tmp_path / "cm.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_metrics_not_nan():
    metrics = compute_metrics(
        ["negative", "positive"],
        ["negative", "positive"],
    )
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert np.isfinite(metrics[key])


def test_pr_auc_positive():
    y_true = ["positive", "negative", "positive", "negative"]
    y_score = [0.9, 0.2, 0.8, 0.1]
    pr = compute_pr_auc(y_true, y_score)
    assert pr is not None
    assert 0.0 <= pr <= 1.0


def test_per_language_metrics():
    y_true = ["positive", "negative", "positive", "negative"]
    y_pred = ["positive", "negative", "negative", "negative"]
    langs = ["english", "english", "roman_urdu", "roman_urdu"]
    out = per_language_metrics(y_true, y_pred, langs)
    assert "english" in out and "roman_urdu" in out
    assert out["english"]["n_samples"] == 2
    assert 0.0 <= out["english"]["f1_macro"] <= 1.0
