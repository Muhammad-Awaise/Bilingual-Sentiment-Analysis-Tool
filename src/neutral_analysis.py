"""
Neutral-class distribution reporting and error analysis.

Principles:
- Never manufacture Neutral labels from binary datasets
- Never auto-label "no sentiment words" as Neutral
- Report genuine Neutral scarcity honestly
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    LABEL_COLUMN,
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_ORDER,
    LABEL_POSITIVE,
    LANGUAGE_COLUMN,
)
from src.leakage import PROVENANCE_COLUMN, SOURCE_COLUMN


def class_distribution_report(df: pd.DataFrame) -> dict[str, Any]:
    """Detailed class / language / source / provenance distribution."""
    if df is None or df.empty:
        return {"n_rows": 0, "class_counts": {}, "class_fractions": {}}

    n = len(df)
    class_counts = {
        str(k): int(v)
        for k, v in df[LABEL_COLUMN].astype(str).value_counts().items()
    }
    class_fractions = {k: round(v / n, 6) for k, v in class_counts.items()}

    by_language: dict[str, dict[str, int]] = {}
    if LANGUAGE_COLUMN in df.columns:
        for lang, grp in df.groupby(df[LANGUAGE_COLUMN].astype(str)):
            by_language[str(lang)] = {
                str(k): int(v)
                for k, v in grp[LABEL_COLUMN].astype(str).value_counts().items()
            }

    by_source: dict[str, dict[str, int]] = {}
    if SOURCE_COLUMN in df.columns:
        for src, grp in df.groupby(df[SOURCE_COLUMN].astype(str)):
            by_source[str(src)] = {
                str(k): int(v)
                for k, v in grp[LABEL_COLUMN].astype(str).value_counts().items()
            }

    by_provenance: dict[str, dict[str, int]] = {}
    if PROVENANCE_COLUMN in df.columns:
        for prov, grp in df.groupby(df[PROVENANCE_COLUMN].astype(str)):
            by_provenance[str(prov)] = {
                str(k): int(v)
                for k, v in grp[LABEL_COLUMN].astype(str).value_counts().items()
            }

    neutral_mask = df[LABEL_COLUMN].astype(str) == LABEL_NEUTRAL
    neutral_n = int(neutral_mask.sum())
    neutral_language = {}
    if LANGUAGE_COLUMN in df.columns and neutral_n:
        neutral_language = {
            str(k): int(v)
            for k, v in df.loc[neutral_mask, LANGUAGE_COLUMN]
            .astype(str)
            .value_counts()
            .items()
        }
    neutral_source = {}
    if SOURCE_COLUMN in df.columns and neutral_n:
        neutral_source = {
            str(k): int(v)
            for k, v in df.loc[neutral_mask, SOURCE_COLUMN]
            .astype(str)
            .value_counts()
            .items()
        }

    return {
        "n_rows": n,
        "class_counts": class_counts,
        "class_fractions": class_fractions,
        "neutral_count": neutral_n,
        "neutral_fraction": round(neutral_n / n, 6) if n else 0.0,
        "neutral_by_language": neutral_language,
        "neutral_by_source": neutral_source,
        "by_language": by_language,
        "by_source": by_source,
        "by_provenance": by_provenance,
        "warnings": _distribution_warnings(class_counts, neutral_n, n),
    }


def _distribution_warnings(
    class_counts: dict[str, int], neutral_n: int, n: int
) -> list[str]:
    warnings: list[str] = []
    if n == 0:
        return ["Empty dataset."]
    if LABEL_NEUTRAL not in class_counts or neutral_n == 0:
        warnings.append(
            "No Neutral rows present. Do not manufacture Neutral from binary data."
        )
    elif neutral_n / n < 0.02:
        warnings.append(
            f"Neutral is scarce ({neutral_n}/{n} = {100 * neutral_n / n:.2f}%). "
            "Macro-F1 / Neutral F1 will be unstable. Prefer genuine Neutral "
            "collection over oversampling."
        )
    for lab in (LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_NEUTRAL):
        if lab not in class_counts:
            warnings.append(f"Class '{lab}' is absent from this dataset.")
    return warnings


def neutral_class_metrics(y_true, y_pred) -> dict[str, Any]:
    """Precision / recall / F1 for Neutral only, plus confusion slice."""
    y_true = [str(y) for y in y_true]
    y_pred = [str(y) for y in y_pred]
    labels = list(LABEL_ORDER)
    present = set(y_true) | set(y_pred)
    label_list = [lab for lab in labels if lab in present] or labels

    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    # Index of neutral in label_list
    if LABEL_NEUTRAL not in label_list:
        return {
            "neutral_present": False,
            "support": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "notes": "Neutral absent from gold and predictions.",
        }
    neu_i = label_list.index(LABEL_NEUTRAL)
    support = int(sum(1 for y in y_true if y == LABEL_NEUTRAL))
    prec = float(
        precision_score(
            y_true, y_pred, labels=[LABEL_NEUTRAL], average="macro", zero_division=0
        )
    )
    rec = float(
        recall_score(
            y_true, y_pred, labels=[LABEL_NEUTRAL], average="macro", zero_division=0
        )
    )
    f1 = float(
        f1_score(
            y_true, y_pred, labels=[LABEL_NEUTRAL], average="macro", zero_division=0
        )
    )

    # Confusion flows involving Neutral
    true_neu_pred_other = {
        label_list[j]: int(cm[neu_i, j])
        for j in range(len(label_list))
        if j != neu_i and cm[neu_i, j]
    }
    true_other_pred_neu = {
        label_list[i]: int(cm[i, neu_i])
        for i in range(len(label_list))
        if i != neu_i and cm[i, neu_i]
    }

    return {
        "neutral_present": True,
        "support": support,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "true_neutral_predicted_as": true_neu_pred_other,
        "predicted_neutral_but_true": true_other_pred_neu,
        "confusion_matrix_labels": label_list,
        "confusion_matrix": cm.tolist(),
        "notes": (
            "Neutral errors split into (a) gold Neutral predicted Pos/Neg and "
            "(b) gold Pos/Neg predicted Neutral. Do not treat low Neutral F1 as "
            "fixed by blind oversampling."
        ),
    }


def sampling_experiment_plan(
    class_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """
    Document controlled sampling experiments — does not execute them.

    Oversampling Neutral is listed as an *experiment*, not a default fix.
    """
    n_neu = int(class_counts.get(LABEL_NEUTRAL, 0))
    return [
        {
            "name": "baseline_balanced_weights",
            "description": "class_weight='balanced', no Neutral oversampling",
            "recommended_default": True,
        },
        {
            "name": "no_class_weight",
            "description": "class_weight=None — measure Neutral collapse under skew",
            "recommended_default": False,
        },
        {
            "name": "mild_neutral_upsample",
            "description": (
                f"Repeat Neutral train rows up to ~{min(2000, max(n_neu * 5, n_neu))} "
                "effective rows after split (train only)"
            ),
            "recommended_default": False,
            "caution": "Do not blindly oversample; check Neutral precision drop.",
        },
        {
            "name": "genuine_neutral_integration",
            "description": (
                "Add human-labelled Neutral from UCI Roman Urdu 3-class and "
                "manually annotated Daraz Neutral — never invent Neutral from binary"
            ),
            "recommended_default": True,
        },
    ]
