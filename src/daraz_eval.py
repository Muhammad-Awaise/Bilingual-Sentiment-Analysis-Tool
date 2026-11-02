"""
External Daraz evaluation workflow.

Daraz rows must not be used for training, feature selection, threshold tuning,
or hyperparameter optimization when this module is used for external reporting.

If Daraz data is too small or contains incomplete references, limitations are
reported rather than fabricating metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import (
    EXTERNAL_DATA_DIR,
    LABEL_COLUMN,
    LANGUAGE_COLUMN,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TEXT_COLUMN,
)
from src.data_loader import DatasetError, infer_language, normalize_label
from src.evaluate import compute_metrics, per_language_metrics
from src.ingest import normalize_sentiment_label
from src.leakage import (
    ANNOTATION_STATUS_COLUMN,
    SOURCE_COLUMN,
    assert_no_partition_leakage,
    build_leakage_report,
    exact_raw_key,
)
from src.neutral_analysis import neutral_class_metrics
from src.predict import load_model

DARAZ_EXTERNAL_CSV = PROCESSED_DATA_DIR / "daraz_external_eval.csv"
DARAZ_EXTERNAL_MANIFEST = PROCESSED_DATA_DIR / "daraz_external_eval.manifest.json"
DARAZ_METRICS_JSON = REPORTS_DIR / "daraz_external_metrics.json"
DARAZ_METRICS_MD = REPORTS_DIR / "daraz_external_report.md"

# Sources that must be excluded from training when preparing an external Daraz test
DARAZ_SOURCE_TAGS = frozenset(
    {
        "local_daraz_reference",
        "daraz_fyp",
        "daraz_code_mixed",
        "daraz_external_eval",
        "daraz_batch_test",
    }
)


def _load_daraz_fyp() -> pd.DataFrame | None:
    path = EXTERNAL_DATA_DIR / "Daraz_FYP_Dataset.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    if "Review" not in df.columns or "Sentiment" not in df.columns:
        return None
    out = pd.DataFrame(
        {
            TEXT_COLUMN: df["Review"].astype(str),
            LABEL_COLUMN: df["Sentiment"].map(normalize_sentiment_label),
            SOURCE_COLUMN: "daraz_fyp",
            ANNOTATION_STATUS_COLUMN: "reference_labelled",
            "annotation_notes": "Labels from Daraz FYP reference export; not re-annotated in-repo.",
        }
    )
    return out.dropna(subset=[LABEL_COLUMN])


def _load_daraz_code_mixed() -> pd.DataFrame | None:
    path = EXTERNAL_DATA_DIR / "daraz_code_mixed_product_reviews.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    text_col = "Reviews" if "Reviews" in df.columns else "review"
    lab_col = "Sentiments" if "Sentiments" in df.columns else "sentiment"
    if text_col not in df.columns or lab_col not in df.columns:
        return None
    out = pd.DataFrame(
        {
            TEXT_COLUMN: df[text_col].astype(str),
            LABEL_COLUMN: df[lab_col].map(normalize_sentiment_label),
            SOURCE_COLUMN: "daraz_code_mixed",
            ANNOTATION_STATUS_COLUMN: "reference_labelled",
            "annotation_notes": (
                "Labels from code-mixed Daraz reference CSV; treat as "
                "external unless manually re-annotated."
            ),
        }
    )
    return out.dropna(subset=[LABEL_COLUMN])


def prepare_daraz_external_eval(
    *,
    include_neutral: bool = True,
    max_rows: int | None = None,
    out_path: Path | None = None,
    train_texts_to_exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build a Daraz-only external evaluation table with source + annotation status.

    Excludes any text that appears in ``train_texts_to_exclude`` (exact raw key).
    """
    parts: list[pd.DataFrame] = []
    for loader in (_load_daraz_fyp, _load_daraz_code_mixed):
        part = loader()
        if part is not None and not part.empty:
            parts.append(part)

    limitations: list[str] = []
    if not parts:
        limitations.append(
            "No Daraz raw CSVs found under data/raw/external/. "
            "Cannot build an external Daraz evaluation set."
        )
        empty = pd.DataFrame(
            columns=[
                TEXT_COLUMN,
                LABEL_COLUMN,
                LANGUAGE_COLUMN,
                SOURCE_COLUMN,
                ANNOTATION_STATUS_COLUMN,
            ]
        )
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_rows": 0,
            "limitations": limitations,
            "usable_for_external_eval": False,
        }
        return empty, manifest

    combined = pd.concat(parts, ignore_index=True)
    if not include_neutral:
        combined = combined[combined[LABEL_COLUMN] != "neutral"]

    # Drop empty / tiny reviews (reference noise)
    combined[TEXT_COLUMN] = combined[TEXT_COLUMN].astype(str).str.strip()
    combined = combined[combined[TEXT_COLUMN].str.len() >= 8]

    # Deduplicate
    combined["_key"] = combined[TEXT_COLUMN].map(exact_raw_key)
    combined = combined.drop_duplicates(subset=["_key"], keep="first")

    excluded = 0
    if train_texts_to_exclude:
        exclude_keys = {exact_raw_key(t) for t in train_texts_to_exclude}
        before = len(combined)
        combined = combined[~combined["_key"].isin(exclude_keys)]
        excluded = before - len(combined)

    combined = combined.drop(columns=["_key"]).reset_index(drop=True)
    combined[LANGUAGE_COLUMN] = combined[TEXT_COLUMN].map(infer_language)

    if max_rows is not None and len(combined) > max_rows:
        # Stratified sample for manageable external eval
        from sklearn.model_selection import train_test_split

        try:
            _, combined = train_test_split(
                combined,
                test_size=max_rows / len(combined),
                random_state=42,
                stratify=combined[LABEL_COLUMN],
            )
            combined = combined.reset_index(drop=True)
        except ValueError:
            combined = combined.sample(n=max_rows, random_state=42).reset_index(
                drop=True
            )

    # Quality checks
    if len(combined) < 100:
        limitations.append(
            f"Daraz external set is small (n={len(combined)}). "
            "Report metrics with caution; do not over-claim generalization."
        )
    # Detect placeholder / reference-like stubs
    short_frac = float((combined[TEXT_COLUMN].str.len() < 20).mean()) if len(combined) else 1.0
    if short_frac > 0.3:
        limitations.append(
            f"{100 * short_frac:.1f}% of Daraz rows are under 20 characters — "
            "may include incomplete references rather than full reviews."
        )

    out = Path(out_path) if out_path else DARAZ_EXTERNAL_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False, encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(out),
        "n_rows": int(len(combined)),
        "class_counts": combined[LABEL_COLUMN].value_counts().to_dict(),
        "language_counts": combined[LANGUAGE_COLUMN].value_counts().to_dict(),
        "source_counts": combined[SOURCE_COLUMN].value_counts().to_dict(),
        "annotation_status_counts": combined[ANNOTATION_STATUS_COLUMN]
        .value_counts()
        .to_dict(),
        "include_neutral": bool(include_neutral),
        "excluded_overlap_with_train": int(excluded),
        "limitations": limitations,
        "usable_for_external_eval": bool(len(combined) >= 50),
        "redistributable": False,
        "notes": (
            "External Daraz evaluation only. Do not train, tune thresholds, "
            "select features, or optimize hyperparameters on this set. "
            "Annotation status is 'reference_labelled' until manually reviewed."
        ),
    }
    DARAZ_EXTERNAL_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return combined, manifest


def exclude_daraz_from_training(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows whose source is Daraz-tagged."""
    if SOURCE_COLUMN not in df.columns:
        return df
    mask = ~df[SOURCE_COLUMN].astype(str).str.lower().isin(DARAZ_SOURCE_TAGS)
    # Also drop sources containing 'daraz'
    mask &= ~df[SOURCE_COLUMN].astype(str).str.lower().str.contains("daraz")
    return df.loc[mask].reset_index(drop=True)


def evaluate_on_daraz_external(
    artifact_path: str | Path | None = None,
    daraz_csv: str | Path | None = None,
    *,
    train_split_path: str | Path | None = None,
    internal_metrics: dict[str, Any] | None = None,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run the saved model on the external Daraz set and write a separate report.

    Verifies no text overlap with the training holdout's train texts when available.
    """
    reports = Path(reports_dir) if reports_dir else REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)

    csv_path = Path(daraz_csv) if daraz_csv else DARAZ_EXTERNAL_CSV
    if not csv_path.exists():
        # Try to prepare from raw files
        train_texts = None
        if train_split_path and Path(train_split_path).exists():
            split = joblib.load(train_split_path)
            train_texts = list(split.get("X_train") or [])
        df, manifest = prepare_daraz_external_eval(
            train_texts_to_exclude=train_texts
        )
        if not manifest.get("usable_for_external_eval"):
            result = {
                "usable": False,
                "limitations": manifest.get("limitations", []),
                "n_samples": 0,
                "notes": "Daraz external evaluation could not be constructed.",
            }
            (reports / DARAZ_METRICS_JSON.name).write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            return result
    else:
        df = pd.read_csv(csv_path, encoding="utf-8")
        manifest = {}
        if DARAZ_EXTERNAL_MANIFEST.exists():
            manifest = json.loads(DARAZ_EXTERNAL_MANIFEST.read_text(encoding="utf-8"))

    if df.empty or TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        result = {
            "usable": False,
            "limitations": ["Daraz eval CSV missing or empty."],
            "n_samples": 0,
        }
        (reports / DARAZ_METRICS_JSON.name).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    # Leakage check vs training texts
    leakage_info: dict[str, Any] = {}
    if train_split_path and Path(train_split_path).exists():
        split = joblib.load(train_split_path)
        train_df = pd.DataFrame({TEXT_COLUMN: split.get("X_train") or []})
        if not train_df.empty:
            try:
                leakage_info = assert_no_partition_leakage(train_df, df)
            except DatasetError as exc:
                # Remove overlapping rows rather than fabricating clean results
                train_keys = {exact_raw_key(t) for t in train_df[TEXT_COLUMN]}
                before = len(df)
                df = df[~df[TEXT_COLUMN].map(exact_raw_key).isin(train_keys)].reset_index(
                    drop=True
                )
                leakage_info = {
                    "removed_overlap_rows": before - len(df),
                    "warning": str(exc),
                }

    if len(df) < 50:
        result = {
            "usable": False,
            "n_samples": int(len(df)),
            "limitations": [
                "Fewer than 50 Daraz rows remain after overlap removal. "
                "Do not report this as a reliable external benchmark."
            ],
            "leakage": leakage_info,
            "manifest": manifest,
        }
        (reports / DARAZ_METRICS_JSON.name).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    payload = load_model(artifact_path)
    pipe = payload["pipeline"]
    X = df[TEXT_COLUMN].astype(str).tolist()
    y_true = df[LABEL_COLUMN].map(normalize_label).tolist()
    # Drop rows that failed label normalize
    keep = [i for i, y in enumerate(y_true) if y is not None]
    X = [X[i] for i in keep]
    y_true = [y_true[i] for i in keep]
    y_pred = [str(p) for p in pipe.predict(X)]

    metrics = compute_metrics(y_true, y_pred)
    metrics["neutral_analysis"] = neutral_class_metrics(y_true, y_pred)
    if LANGUAGE_COLUMN in df.columns:
        langs = df.iloc[keep][LANGUAGE_COLUMN].astype(str).tolist()
        metrics["per_language"] = per_language_metrics(y_true, y_pred, langs)

    delta: dict[str, Any] = {}
    if internal_metrics:
        for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
            if key in internal_metrics and key in metrics:
                delta[key] = float(metrics[key]) - float(internal_metrics[key])

    result = {
        "usable": True,
        "evaluation_type": "external_daraz",
        "n_samples": metrics["n_samples"],
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": metrics["labels"],
        "neutral_analysis": metrics["neutral_analysis"],
        "per_language": metrics.get("per_language"),
        "delta_vs_internal_holdout": delta,
        "leakage": leakage_info,
        "source_counts": (
            df[SOURCE_COLUMN].value_counts().to_dict()
            if SOURCE_COLUMN in df.columns
            else {}
        ),
        "annotation_status_counts": (
            df[ANNOTATION_STATUS_COLUMN].value_counts().to_dict()
            if ANNOTATION_STATUS_COLUMN in df.columns
            else {}
        ),
        "manifest": manifest,
        "limitations": list(manifest.get("limitations") or [])
        + [
            "Daraz labels are reference_labelled unless manually annotated.",
            "External Daraz metrics must not be compared as if they were "
            "in-distribution holdout scores.",
        ],
        "model_name": payload.get("model_name"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    (reports / DARAZ_METRICS_JSON.name).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (reports / DARAZ_METRICS_MD.name).write_text(
        _format_daraz_md(result), encoding="utf-8"
    )
    return result


def _format_daraz_md(result: dict[str, Any]) -> str:
    lines = [
        "# External Daraz Evaluation",
        "",
        f"- **Usable:** {result.get('usable')}",
        f"- **N:** {result.get('n_samples')}",
        f"- **Model:** {result.get('model_name')}",
        "",
    ]
    if not result.get("usable"):
        lines.append("## Limitations")
        for lim in result.get("limitations") or []:
            lines.append(f"- {lim}")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Metrics",
            "",
            f"| Accuracy | {result['accuracy']:.4f} |",
            f"| Macro precision | {result['precision_macro']:.4f} |",
            f"| Macro recall | {result['recall_macro']:.4f} |",
            f"| Macro F1 | {result['f1_macro']:.4f} |",
            "",
            "## Δ vs internal holdout",
            "",
            str(result.get("delta_vs_internal_holdout") or {}),
            "",
            "## Limitations",
            "",
        ]
    )
    for lim in result.get("limitations") or []:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)
