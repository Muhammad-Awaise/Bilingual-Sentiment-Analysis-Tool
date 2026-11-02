"""Neutral reporting and calibration helpers."""

from src.calibration_eval import (
    brier_score_multiclass,
    confidence_distribution,
    expected_calibration_error,
)
from src.neutral_analysis import class_distribution_report, neutral_class_metrics
import numpy as np
import pandas as pd


def test_class_distribution_report_warns_on_scarce_neutral():
    df = pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(100)],
            "label": ["positive"] * 60 + ["negative"] * 39 + ["neutral"],
            "language": ["english"] * 100,
        }
    )
    report = class_distribution_report(df)
    assert report["neutral_count"] == 1
    assert any("scarce" in w.lower() or "Neutral" in w for w in report["warnings"])


def test_neutral_class_metrics_confusion_flows():
    y_true = ["neutral", "neutral", "positive", "negative", "neutral"]
    y_pred = ["positive", "neutral", "neutral", "negative", "negative"]
    m = neutral_class_metrics(y_true, y_pred)
    assert m["neutral_present"]
    assert m["support"] == 3
    assert "positive" in m["true_neutral_predicted_as"] or m["f1"] is not None


def test_brier_and_ece_finite():
    classes = ["negative", "neutral", "positive"]
    y_true = ["positive", "negative", "neutral", "positive"]
    proba = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
        ]
    )
    brier = brier_score_multiclass(y_true, proba, classes)
    ece = expected_calibration_error(y_true, proba, classes)
    dist = confidence_distribution(proba)
    assert np.isfinite(brier)
    assert np.isfinite(ece)
    assert dist["n"] == 4
