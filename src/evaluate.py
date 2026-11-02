"""Reproducible evaluation: metrics written from real predictions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.calibration_eval import evaluate_calibration
from src.config import (
    CONFUSION_MATRIX_PATH,
    FIXTURES_DIR,
    HOLDOUT_SPLIT_PATH,
    LABEL_NEUTRAL,
    LABEL_ORDER,
    LABEL_POSITIVE,
    METRICS_JSON_PATH,
    METRICS_MD_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_COMPARISON_PATH,
    REPORTS_DIR,
)
from src.neutral_analysis import neutral_class_metrics


class EvaluationError(ValueError):
    """Raised when evaluation inputs are missing or invalid."""


def _finite_or_raise(name: str, value: float) -> float:
    if value is None or not np.isfinite(value):
        raise EvaluationError(f"Metric '{name}' is NaN or infinite: {value}")
    return float(value)


def compute_metrics(y_true, y_pred, labels: list[str] | None = None) -> dict[str, Any]:
    """Compute accuracy, precision, recall, macro F1, and confusion matrix."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise EvaluationError("y_true and y_pred length mismatch.")
    if not y_true:
        raise EvaluationError("Cannot evaluate empty prediction lists.")

    if labels is not None:
        label_list = list(labels)
    else:
        # Use LABEL_ORDER, but skip classes absent from both sides so a
        # binary-only batch is not diluted by an empty Neutral column.
        present = {str(x) for x in y_true} | {str(x) for x in y_pred}
        label_list = [lab for lab in LABEL_ORDER if lab in present]
        if not label_list:
            label_list = list(LABEL_ORDER)

    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    metrics = {
        "labels": label_list,
        "n_samples": len(y_true),
        "accuracy": _finite_or_raise("accuracy", accuracy_score(y_true, y_pred)),
        "precision_macro": _finite_or_raise(
            "precision_macro",
            precision_score(
                y_true, y_pred, labels=label_list, average="macro", zero_division=0
            ),
        ),
        "recall_macro": _finite_or_raise(
            "recall_macro",
            recall_score(
                y_true, y_pred, labels=label_list, average="macro", zero_division=0
            ),
        ),
        "f1_macro": _finite_or_raise(
            "f1_macro",
            f1_score(
                y_true, y_pred, labels=label_list, average="macro", zero_division=0
            ),
        ),
        "f1_weighted": _finite_or_raise(
            "f1_weighted",
            f1_score(
                y_true, y_pred, labels=label_list, average="weighted", zero_division=0
            ),
        ),
        "per_class": {
            lab: {
                "precision": _finite_or_raise(
                    f"precision_{lab}",
                    precision_score(
                        y_true,
                        y_pred,
                        labels=[lab],
                        average="macro",
                        zero_division=0,
                    ),
                ),
                "recall": _finite_or_raise(
                    f"recall_{lab}",
                    recall_score(
                        y_true,
                        y_pred,
                        labels=[lab],
                        average="macro",
                        zero_division=0,
                    ),
                ),
                "f1": _finite_or_raise(
                    f"f1_{lab}",
                    f1_score(
                        y_true,
                        y_pred,
                        labels=[lab],
                        average="macro",
                        zero_division=0,
                    ),
                ),
                "support": int(sum(1 for y in y_true if str(y) == lab)),
            }
            for lab in label_list
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_list,
            zero_division=0,
            output_dict=True,
        ),
    }
    return metrics


def per_source_metrics(
    y_true,
    y_pred,
    sources,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Macro metrics sliced by dataset source (only sources present)."""
    label_list = labels or list(LABEL_ORDER)
    y_true = list(y_true)
    y_pred = list(y_pred)
    sources = list(sources)
    if not (len(y_true) == len(y_pred) == len(sources)):
        raise EvaluationError("per-source inputs must have equal length.")

    out: dict[str, Any] = {}
    for src in sorted(set(str(x) for x in sources)):
        if src in {"", "nan", "none"}:
            continue
        idx = [i for i, s in enumerate(sources) if str(s) == src]
        if len(idx) < 2:
            continue
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        m = compute_metrics(yt, yp, labels=label_list)
        block = {
            "n_samples": m["n_samples"],
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
        }
        if LABEL_NEUTRAL in m.get("per_class", {}):
            block["neutral_f1"] = m["per_class"][LABEL_NEUTRAL]["f1"]
        out[src] = block
    return out


def evaluate_challenge_set(
    pipe,
    fixture_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Linguistic challenge-set evaluation.

    Probes are manually reviewed fixtures — predictions are NOT forced to match
    expected labels. Reports accuracy / errors for research transparency.
    """
    path = Path(fixture_path) if fixture_path else FIXTURES_DIR / "linguistic_examples.csv"
    if not path.exists():
        return {
            "available": False,
            "notes": f"Challenge fixture missing: {path}",
        }
    import pandas as pd

    df = pd.read_csv(path, encoding="utf-8")
    # Flexible column names
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = next(
        (
            c
            for c in ("label", "expected_sentiment", "gold_label", "sentiment")
            if c in df.columns
        ),
        None,
    )
    category_col = next(
        (c for c in ("category", "phenomenon", "challenge") if c in df.columns),
        None,
    )
    if label_col is None:
        return {"available": False, "notes": "No label column in challenge fixture."}

    texts = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(str).str.lower().tolist()
    y_pred = [str(p).lower() for p in pipe.predict(texts)]
    categories = (
        df[category_col].astype(str).tolist()
        if category_col
        else ["unspecified"] * len(df)
    )

    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    errors = [
        {
            "text": texts[i][:200],
            "gold": y_true[i],
            "predicted": y_pred[i],
            "category": categories[i],
        }
        for i in range(len(texts))
        if y_true[i] != y_pred[i]
    ]
    by_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == cat]
        if not idx:
            continue
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        cat_correct = sum(1 for a, b in zip(yt, yp) if a == b)
        entry: dict[str, Any] = {
            "n": len(idx),
            "correct": cat_correct,
            "incorrect": len(idx) - cat_correct,
            "accuracy": cat_correct / len(idx) if idx else 0.0,
        }
        if len(idx) >= 4 and len(set(yt)) >= 2:
            entry["f1_macro"] = float(
                f1_score(yt, yp, average="macro", zero_division=0)
            )
        by_cat[cat] = entry

    return {
        "available": True,
        "n_examples": len(texts),
        "correct": correct,
        "incorrect": len(texts) - correct,
        "accuracy": correct / len(texts) if texts else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if len(set(y_true)) >= 2
        else None,
        "by_category": by_cat,
        "error_examples": errors[:25],
        "notes": (
            "Challenge-set labels are linguistic probes for error analysis. "
            "The model is not forced to predict the expected label."
        ),
    }


def _positive_scores(pipe, X) -> np.ndarray | None:
    """Return positive-class scores for PR-AUC, or None if unavailable."""
    classes = list(getattr(pipe, "classes_", []))
    if LABEL_POSITIVE not in classes:
        return None
    pos_idx = classes.index(LABEL_POSITIVE)
    try:
        if hasattr(pipe, "predict_proba"):
            return np.asarray(pipe.predict_proba(X))[:, pos_idx]
        if hasattr(pipe, "decision_function"):
            scores = np.asarray(pipe.decision_function(X), dtype=float)
            return scores
    except Exception:
        return None
    return None


def compute_pr_auc(y_true, y_score) -> float | None:
    """Average precision (PR-AUC) for the positive class; None if not computable."""
    y_bin = np.asarray(
        [1 if str(y) == LABEL_POSITIVE else 0 for y in y_true], dtype=int
    )
    if len(np.unique(y_bin)) < 2:
        return None
    try:
        return _finite_or_raise(
            "pr_auc_positive",
            average_precision_score(y_bin, y_score),
        )
    except EvaluationError:
        return None


def per_language_metrics(
    y_true,
    y_pred,
    languages,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute macro-F1 / accuracy sliced by language tag."""
    label_list = labels or list(LABEL_ORDER)
    y_true = list(y_true)
    y_pred = list(y_pred)
    languages = list(languages)
    if not (len(y_true) == len(y_pred) == len(languages)):
        raise EvaluationError("per-language inputs must have equal length.")

    out: dict[str, Any] = {}
    for lang in sorted(set(str(x) for x in languages)):
        idx = [i for i, L in enumerate(languages) if str(L) == lang]
        if len(idx) < 2:
            continue
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        # Need both classes ideally; still report if one class present
        m = compute_metrics(yt, yp, labels=label_list)
        out[lang] = {
            "n_samples": m["n_samples"],
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
        }
    return out


def make_confusion_matrix_figure(
    cm: list[list[int]] | np.ndarray,
    labels: list[str],
):
    """
    Build a confusion-matrix figure that stays fully readable in Streamlit.

    Uses square cells, short tick labels, and generous margins so nothing
    is clipped at the bottom or sides.
    """
    matrix = np.asarray(cm, dtype=float)
    display_labels = [str(lab).title() for lab in labels]
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pct = 100.0 * matrix / row_sums

    # Compact landscape size — avoids an overly tall Streamlit image
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Count", fontsize=10)

    ticks = np.arange(len(display_labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.tick_params(axis="both", pad=4)
    ax.set_xlabel("Predicted label", fontsize=11, labelpad=8)
    ax.set_ylabel("True label", fontsize=11, labelpad=8)
    ax.set_title("Confusion matrix (held-out test set)", fontsize=12, pad=10)

    ax.set_xlim(-0.5, len(display_labels) - 0.5)
    ax.set_ylim(len(display_labels) - 0.5, -0.5)

    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = int(matrix[i, j])
            color = "white" if matrix[i, j] > thresh else "#102a43"
            ax.text(
                j,
                i,
                f"{count:,}\n({pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
                linespacing=1.25,
            )

    fig.tight_layout(pad=0.6)
    return fig


def confusion_matrix_png_bytes(
    cm: list[list[int]] | np.ndarray,
    labels: list[str],
) -> bytes:
    """PNG bytes with tight bounding box — safest way to show in Streamlit."""
    import io

    fig = make_confusion_matrix_figure(cm, labels)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    labels: list[str],
    output_path: str | Path | None = None,
) -> Path:
    """Save a confusion-matrix heatmap PNG."""
    out = Path(output_path) if output_path else CONFUSION_MATRIX_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(confusion_matrix_png_bytes(cm, labels))
    return out


def _metrics_for_artifact(
    artifact: Path,
    split_data: dict[str, Any],
) -> dict[str, Any]:
    payload = joblib.load(artifact)
    pipe = payload["pipeline"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, labels=list(LABEL_ORDER))
    metrics["model_name"] = payload.get("model_name", "unknown")
    metrics["majority_baseline_accuracy"] = float(
        split_data.get("majority_baseline_accuracy", 0.0)
    )
    metrics["train_size"] = int(payload.get("train_size", 0))
    metrics["test_size"] = int(payload.get("test_size", len(y_test)))
    metrics["class_counts_full"] = payload.get("class_counts_full", {})
    metrics["language_counts_full"] = payload.get("language_counts_full", {})
    metrics["source_counts_full"] = payload.get("source_counts_full", {})
    metrics["provenance_counts_full"] = payload.get("provenance_counts_full", {})
    metrics["data_path"] = payload.get("data_path", "")
    metrics["calibrated"] = bool(payload.get("calibrated", False))
    metrics["class_weight"] = payload.get("class_weight", None)
    metrics["tuned"] = bool(payload.get("tuned", False))
    metrics["tune_meta"] = payload.get("tune_meta")
    metrics["random_seed"] = payload.get("random_seed")
    metrics["delta_f1_vs_majority_baseline"] = float(
        metrics["f1_macro"] - metrics["majority_baseline_accuracy"]
    )
    metrics["neutral_analysis"] = neutral_class_metrics(y_test, y_pred)

    scores = _positive_scores(pipe, X_test)
    if scores is not None:
        pr = compute_pr_auc(y_test, scores)
        if pr is not None:
            metrics["pr_auc_positive"] = pr
            metrics["pr_auc_positive_note"] = (
                "One-vs-rest average precision for the positive class "
                "(positive=1, all other labels=0)."
            )

    lang_test = split_data.get("lang_test")
    if lang_test is not None and len(lang_test) == len(y_test):
        metrics["per_language"] = per_language_metrics(
            y_test, y_pred, lang_test, labels=list(LABEL_ORDER)
        )

    source_test = split_data.get("source_test")
    if source_test is not None and len(source_test) == len(y_test):
        metrics["per_source"] = per_source_metrics(
            y_test, y_pred, source_test, labels=list(LABEL_ORDER)
        )

    provenance_test = split_data.get("provenance_test")
    if provenance_test is not None and len(provenance_test) == len(y_test):
        # Reuse per_source helper for provenance slices (esp. synthetic vs real Urdu)
        metrics["per_provenance"] = per_source_metrics(
            y_test, y_pred, provenance_test, labels=list(LABEL_ORDER)
        )

    metrics["calibration"] = evaluate_calibration(pipe, X_test, y_test)
    metrics["challenge_set"] = evaluate_challenge_set(pipe)

    metrics["notes"] = (
        "Metrics are computed on a held-out stratified test set created during "
        "training. They are not hardcoded. Macro F1 and PR-AUC are preferred over "
        "accuracy when classes are imbalanced. Small curated subsets and synthetic "
        "Urdu templates can yield optimistic language-slice scores — check "
        "per_provenance. Calibration scores are approximate confidence, not certainty."
    )
    return metrics


def evaluate_artifact(
    artifact_path: str | Path | None = None,
    split_path: str | Path | None = None,
    *,
    reports_dir: str | Path | None = None,
    write_reports: bool = True,
) -> dict[str, Any]:
    """Load saved pipeline + holdout split; write metrics JSON/MD and CM plot."""
    reports = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    artifact = Path(artifact_path) if artifact_path else MODEL_ARTIFACT_PATH
    split = Path(split_path) if split_path else reports / HOLDOUT_SPLIT_PATH.name

    if not artifact.exists():
        raise EvaluationError(
            f"Model artifact not found: {artifact}. Run: python -m src.train"
        )
    if not split.exists():
        raise EvaluationError(
            f"Holdout split not found: {split}. Re-run training to create it."
        )

    split_data = joblib.load(split)
    metrics = _metrics_for_artifact(artifact, split_data)

    if write_reports:
        reports.mkdir(parents=True, exist_ok=True)
        cm_path = plot_confusion_matrix(
            metrics["confusion_matrix"],
            metrics["labels"],
            reports / CONFUSION_MATRIX_PATH.name,
        )
        metrics["confusion_matrix_path"] = str(cm_path)
        (reports / METRICS_JSON_PATH.name).write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (reports / METRICS_MD_PATH.name).write_text(
            _format_md(metrics), encoding="utf-8"
        )
    return metrics


def write_model_comparison(
    logreg_artifact: str | Path,
    linearsvc_artifact: str | Path,
    split_path: str | Path,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate LogReg vs LinearSVC on the same holdout; write comparison MD/JSON."""
    reports = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    split_data = joblib.load(split_path)
    logreg = _metrics_for_artifact(Path(logreg_artifact), split_data)
    svc = _metrics_for_artifact(Path(linearsvc_artifact), split_data)

    comparison = {
        "holdout_size": len(split_data["y_test"]),
        "majority_baseline_accuracy": float(
            split_data.get("majority_baseline_accuracy", 0.0)
        ),
        "logistic_regression": {
            "model_name": logreg.get("model_name"),
            "accuracy": logreg["accuracy"],
            "f1_macro": logreg["f1_macro"],
            "precision_macro": logreg["precision_macro"],
            "recall_macro": logreg["recall_macro"],
            "pr_auc_positive": logreg.get("pr_auc_positive"),
        },
        "linearsvc": {
            "model_name": svc.get("model_name"),
            "accuracy": svc["accuracy"],
            "f1_macro": svc["f1_macro"],
            "precision_macro": svc["precision_macro"],
            "recall_macro": svc["recall_macro"],
            "pr_auc_positive": svc.get("pr_auc_positive"),
        },
    }
    # Prefer higher macro F1; break ties with PR-AUC then accuracy
    lr_f1 = comparison["logistic_regression"]["f1_macro"]
    svc_f1 = comparison["linearsvc"]["f1_macro"]
    if lr_f1 > svc_f1:
        comparison["preferred"] = "logistic_regression"
    elif svc_f1 > lr_f1:
        comparison["preferred"] = "linearsvc"
    else:
        comparison["preferred"] = "tie"

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    (reports / MODEL_COMPARISON_PATH.name).write_text(
        _format_comparison_md(comparison), encoding="utf-8"
    )
    return comparison


def _format_comparison_md(comparison: dict[str, Any]) -> str:
    lr = comparison["logistic_regression"]
    svc = comparison["linearsvc"]

    def _row(name: str, block: dict[str, Any]) -> str:
        pr = block.get("pr_auc_positive")
        pr_s = f"{pr:.4f}" if isinstance(pr, (int, float)) else "n/a"
        return (
            f"| {name} | {block['accuracy']:.4f} | {block['precision_macro']:.4f} | "
            f"{block['recall_macro']:.4f} | {block['f1_macro']:.4f} | {pr_s} |"
        )

    lines = [
        "# Model Comparison (same holdout)",
        "",
        f"- **Holdout size:** {comparison['holdout_size']}",
        f"- **Majority baseline accuracy:** {comparison['majority_baseline_accuracy']:.4f}",
        f"- **Preferred (by macro F1):** {comparison['preferred']}",
        "",
        "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | PR-AUC (pos) |",
        "|-------|----------|-------------------|----------------|------------|--------------|",
        _row(str(lr.get("model_name", "LogReg")), lr),
        _row(str(svc.get("model_name", "LinearSVC")), svc),
        "",
        "Both models share the same train/test split and preprocessing.",
        "",
    ]
    return "\n".join(lines)


def _format_md(metrics: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- **Model:** {metrics.get('model_name')}",
        f"- **Data path:** `{metrics.get('data_path')}`",
        f"- **Train size:** {metrics.get('train_size')}",
        f"- **Test size:** {metrics.get('test_size')}",
        f"- **Class counts (full data):** {metrics.get('class_counts_full')}",
        f"- **Language counts (full data):** {metrics.get('language_counts_full')}",
        f"- **Class weight:** {metrics.get('class_weight')}",
        f"- **Calibrated:** {metrics.get('calibrated')}",
        f"- **Tuned:** {metrics.get('tuned')}",
        "",
        "## Metrics (held-out test set)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision (macro) | {metrics['precision_macro']:.4f} |",
        f"| Recall (macro) | {metrics['recall_macro']:.4f} |",
        f"| F1 (macro) | {metrics['f1_macro']:.4f} |",
        f"| F1 (weighted) | {metrics.get('f1_weighted', float('nan')):.4f} |",
        f"| Majority-class baseline accuracy | {metrics['majority_baseline_accuracy']:.4f} |",
        f"| Δ F1 vs majority baseline | {metrics.get('delta_f1_vs_majority_baseline', 0):.4f} |",
    ]
    if metrics.get("pr_auc_positive") is not None:
        lines.append(f"| PR-AUC (positive) | {metrics['pr_auc_positive']:.4f} |")
    lines.extend(
        [
            "",
            f"Confusion matrix saved to `{metrics.get('confusion_matrix_path')}`.",
            "",
        ]
    )

    per_class = metrics.get("per_class") or {}
    if per_class:
        lines.extend(
            [
                "## Per-class metrics",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|-------|-----------|--------|----|---------|",
            ]
        )
        for lab, block in per_class.items():
            lines.append(
                f"| {lab} | {block['precision']:.4f} | {block['recall']:.4f} | "
                f"{block['f1']:.4f} | {block['support']} |"
            )
        lines.append("")

    neu = metrics.get("neutral_analysis") or {}
    if neu.get("neutral_present"):
        lines.extend(
            [
                "## Neutral analysis",
                "",
                f"- Support: {neu.get('support')}",
                f"- Precision / Recall / F1: {neu.get('precision'):.4f} / "
                f"{neu.get('recall'):.4f} / {neu.get('f1'):.4f}",
                f"- Gold Neutral → other: {neu.get('true_neutral_predicted_as')}",
                f"- Other → predicted Neutral: {neu.get('predicted_neutral_but_true')}",
                "",
            ]
        )

    per_lang = metrics.get("per_language") or {}
    if per_lang:
        lines.extend(
            [
                "## Per-language metrics",
                "",
                "| Language | N | Accuracy | F1 (macro) |",
                "|----------|---|----------|------------|",
            ]
        )
        for lang, block in per_lang.items():
            lines.append(
                f"| {lang} | {block['n_samples']} | {block['accuracy']:.4f} | "
                f"{block['f1_macro']:.4f} |"
            )
        lines.append("")

    per_src = metrics.get("per_source") or {}
    if per_src:
        lines.extend(
            [
                "## Per-source metrics",
                "",
                "| Source | N | Accuracy | F1 (macro) |",
                "|--------|---|----------|------------|",
            ]
        )
        for src, block in per_src.items():
            lines.append(
                f"| {src} | {block['n_samples']} | {block['accuracy']:.4f} | "
                f"{block['f1_macro']:.4f} |"
            )
        lines.append("")

    cal = metrics.get("calibration") or {}
    if cal.get("available"):
        lines.extend(
            [
                "## Calibration",
                "",
                f"- Brier score: {cal.get('brier_score'):.4f}",
                f"- ECE: {cal.get('expected_calibration_error'):.4f}",
                f"- Mean confidence: {cal.get('confidence_distribution', {}).get('mean')}",
                f"- Note: {cal.get('notes')}",
                "",
            ]
        )

    challenge = metrics.get("challenge_set") or {}
    if challenge.get("available"):
        lines.extend(
            [
                "## Challenge set",
                "",
                f"- N: {challenge.get('n_examples')} | correct: {challenge.get('correct')} | "
                f"incorrect: {challenge.get('incorrect')} | "
                f"accuracy: {challenge.get('accuracy'):.4f}",
                f"- Note: {challenge.get('notes')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitations",
            "",
            metrics.get("notes", ""),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved sentiment model")
    parser.add_argument("--artifact", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--reports-dir", type=str, default=None)
    parser.add_argument(
        "--compare-svc",
        type=str,
        default=None,
        help="Optional LinearSVC artifact path for side-by-side comparison report",
    )
    args = parser.parse_args()
    try:
        metrics = evaluate_artifact(
            args.artifact, args.split, reports_dir=args.reports_dir
        )
    except EvaluationError as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc

    printable = {
        k: v for k, v in metrics.items() if k != "classification_report"
    }
    print(json.dumps(printable, indent=2))

    if args.compare_svc:
        split = Path(args.split) if args.split else (
            Path(args.reports_dir) if args.reports_dir else REPORTS_DIR
        ) / HOLDOUT_SPLIT_PATH.name
        comparison = write_model_comparison(
            logreg_artifact=args.artifact or MODEL_ARTIFACT_PATH,
            linearsvc_artifact=args.compare_svc,
            split_path=split,
            reports_dir=args.reports_dir,
        )
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
