"""
Calibration evaluation for sigmoid-calibrated classifiers.

Uses predict_proba only — never presents raw decision scores as probabilities.
Confidence is treated as an approximate calibrated score, not certainty.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _proba_matrix(pipe, X) -> np.ndarray | None:
    try:
        if hasattr(pipe, "predict_proba"):
            return np.asarray(pipe.predict_proba(X), dtype=float)
    except Exception:
        return None
    return None


def brier_score_multiclass(y_true, proba: np.ndarray, classes: list[str]) -> float:
    """Mean squared error between one-hot labels and predicted probabilities."""
    class_to_i = {str(c): i for i, c in enumerate(classes)}
    y_true = [str(y) for y in y_true]
    n, k = proba.shape
    one_hot = np.zeros((n, k), dtype=float)
    for i, y in enumerate(y_true):
        if y in class_to_i:
            one_hot[i, class_to_i[y]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true,
    proba: np.ndarray,
    classes: list[str],
    *,
    n_bins: int = 10,
) -> float:
    """
    Top-label ECE: bin by max predicted probability vs correctness.

    Appropriate for multiclass when using the predicted class's confidence.
    """
    y_true = np.asarray([str(y) for y in y_true])
    classes = [str(c) for c in classes]
    pred_idx = np.argmax(proba, axis=1)
    conf = proba[np.arange(len(proba)), pred_idx]
    pred_labels = np.asarray([classes[i] for i in pred_idx])
    correct = (pred_labels == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf > b0) & (conf <= b1) if b0 > 0 else (conf >= b0) & (conf <= b1)
        if not np.any(mask):
            continue
        acc = float(np.mean(correct[mask]))
        avg_conf = float(np.mean(conf[mask]))
        ece += (np.sum(mask) / n) * abs(acc - avg_conf)
    return float(ece)


def reliability_diagram_data(
    y_true,
    proba: np.ndarray,
    classes: list[str],
    *,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Per-bin accuracy vs confidence for a reliability diagram."""
    y_true = np.asarray([str(y) for y in y_true])
    classes = [str(c) for c in classes]
    pred_idx = np.argmax(proba, axis=1)
    conf = proba[np.arange(len(proba)), pred_idx]
    pred_labels = np.asarray([classes[i] for i in pred_idx])
    correct = (pred_labels == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf > b0) & (conf <= b1) if b0 > 0 else (conf >= b0) & (conf <= b1)
        if not np.any(mask):
            rows.append(
                {
                    "bin_left": float(b0),
                    "bin_right": float(b1),
                    "count": 0,
                    "avg_confidence": float("nan"),
                    "accuracy": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "bin_left": float(b0),
                "bin_right": float(b1),
                "count": int(np.sum(mask)),
                "avg_confidence": float(np.mean(conf[mask])),
                "accuracy": float(np.mean(correct[mask])),
            }
        )
    return rows


def confidence_distribution(proba: np.ndarray) -> dict[str, Any]:
    conf = np.max(proba, axis=1)
    return {
        "mean": float(np.mean(conf)),
        "std": float(np.std(conf)),
        "min": float(np.min(conf)),
        "max": float(np.max(conf)),
        "p25": float(np.percentile(conf, 25)),
        "p50": float(np.percentile(conf, 50)),
        "p75": float(np.percentile(conf, 75)),
        "n": int(len(conf)),
    }


def evaluate_calibration(pipe, X, y_true) -> dict[str, Any]:
    """Full calibration block for reports. Returns notes if proba unavailable."""
    classes = [str(c) for c in getattr(pipe, "classes_", [])]
    proba = _proba_matrix(pipe, X)
    if proba is None or not classes:
        return {
            "available": False,
            "notes": (
                "predict_proba unavailable. Do not treat decision_function "
                "scores as probabilities or confidence."
            ),
        }
    return {
        "available": True,
        "brier_score": brier_score_multiclass(y_true, proba, classes),
        "expected_calibration_error": expected_calibration_error(
            y_true, proba, classes
        ),
        "reliability_diagram": reliability_diagram_data(y_true, proba, classes),
        "confidence_distribution": confidence_distribution(proba),
        "notes": (
            "Scores are sigmoid-calibrated class probabilities when the artifact "
            "used CalibratedClassifierCV. They are approximate confidence, not "
            "certainty about human sentiment."
        ),
    }
