"""
Reproducible ablation / feature-engineering experiments.

Compares preprocessing and TF-IDF configurations on identical stratified splits.
Results are computed and saved — never hardcoded.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    CHAR_MAX_FEATURES,
    CHAR_NGRAM_RANGE,
    CV_FOLDS,
    LABEL_COLUMN,
    LANGUAGE_COLUMN,
    MAX_FEATURES,
    NGRAM_RANGE,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_SIZE,
    TEXT_COLUMN,
)
from src.evaluate import compute_metrics, per_language_metrics
from src.leakage import assert_no_partition_leakage
from src.neutral_analysis import neutral_class_metrics
from src.preprocessing import clean_text

ABLATION_RESULTS_JSON = REPORTS_DIR / "ablation_results.json"
ABLATION_RESULTS_MD = REPORTS_DIR / "ablation_results.md"


def _baseline_clean(text):
    """Minimal cleaning: lower + whitespace, no linguistic markers."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).strip().lower()
    return " ".join(s.split())


def _make_preprocessor(mode: str) -> Callable:
    """
    Return a clean_text-compatible callable for ablation modes.

    Modes toggle which parts of clean_text run by wrapping flags via
    environment-free closures around preprocessing helpers.
    """
    from src import preprocessing as prep

    if mode == "baseline_preproc":
        return _baseline_clean
    if mode == "nahi_only":
        def _fn(text):
            if text is None or (isinstance(text, float) and pd.isna(text)):
                return ""
            s = str(text).strip().lower()
            s = prep._URL_RE.sub(" ", s)
            s = prep._NON_TEXT_RE.sub(" ", s)
            s = prep._WHITESPACE_RE.sub(" ", s).strip()
            return prep._canonicalize_negation_spellings(s)

        return _fn
    if mode == "full_clean":
        return clean_text
    if mode == "no_contrast":
        def _fn(text):
            # full clean_text path but skip contrast emphasis
            if text is None or (isinstance(text, float) and pd.isna(text)):
                return ""
            if not isinstance(text, str):
                text = str(text)
            text = text.strip().lower()
            text = prep._URL_RE.sub(" ", text)
            text = prep._EMAIL_RE.sub(" ", text)
            text = prep._MENTION_RE.sub(" ", text)
            text = prep._HASHTAG_RE.sub(r"\1", text)
            text = prep._ARABIC_PUNCT_RE.sub(" ", text)
            text = prep._NON_TEXT_RE.sub(" ", text)
            text = prep._WHITESPACE_RE.sub(" ", text).strip()
            text = prep._canonicalize_negation_spellings(text)
            # skip contrast
            text = prep._emphasize_polarity_intensity(text)
            text = prep._emphasize_negation_scope(text)
            return prep._WHITESPACE_RE.sub(" ", text).strip()

        return _fn
    if mode == "contrast_marker_only":
        def _fn(text):
            if text is None or (isinstance(text, float) and pd.isna(text)):
                return ""
            if not isinstance(text, str):
                text = str(text)
            text = text.strip().lower()
            text = prep._URL_RE.sub(" ", text)
            text = prep._EMAIL_RE.sub(" ", text)
            text = prep._MENTION_RE.sub(" ", text)
            text = prep._HASHTAG_RE.sub(r"\1", text)
            text = prep._ARABIC_PUNCT_RE.sub(" ", text)
            text = prep._NON_TEXT_RE.sub(" ", text)
            text = prep._WHITESPACE_RE.sub(" ", text).strip()
            text = prep._canonicalize_negation_spellings(text)
            text = prep._emphasize_contrast_clause(text, repeat_verdict=False)
            text = prep._emphasize_polarity_intensity(text)
            text = prep._emphasize_negation_scope(text)
            return prep._WHITESPACE_RE.sub(" ", text).strip()

        return _fn
    if mode == "no_intensity":
        def _fn(text):
            if text is None or (isinstance(text, float) and pd.isna(text)):
                return ""
            if not isinstance(text, str):
                text = str(text)
            text = text.strip().lower()
            text = prep._URL_RE.sub(" ", text)
            text = prep._EMAIL_RE.sub(" ", text)
            text = prep._MENTION_RE.sub(" ", text)
            text = prep._HASHTAG_RE.sub(r"\1", text)
            text = prep._ARABIC_PUNCT_RE.sub(" ", text)
            text = prep._NON_TEXT_RE.sub(" ", text)
            text = prep._WHITESPACE_RE.sub(" ", text).strip()
            text = prep._canonicalize_negation_spellings(text)
            text = prep._emphasize_contrast_clause(text)
            text = prep._emphasize_negation_scope(text)
            return prep._WHITESPACE_RE.sub(" ", text).strip()

        return _fn
    return clean_text


FEATURE_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "word_tfidf_only",
        "features": "word",
        "preproc": "full_clean",
        "description": "Word TF-IDF (1–2 grams) only",
    },
    {
        "name": "char_tfidf_only",
        "features": "char",
        "preproc": "full_clean",
        "description": "Character TF-IDF (char_wb 3–5) only",
    },
    {
        "name": "word_char_tfidf",
        "features": "word+char",
        "preproc": "full_clean",
        "description": "Word + character TF-IDF (production baseline features)",
    },
    {
        "name": "word_char_negation",
        "features": "word+char",
        "preproc": "no_contrast",  # still has negation scope + intensity
        "description": "Word+char with negation features (no contrast)",
    },
    {
        "name": "word_char_contrast",
        "features": "word+char",
        "preproc": "no_intensity",
        "description": "Word+char with contrast features (no intensity markers)",
    },
    {
        "name": "word_char_negation_contrast",
        "features": "word+char",
        "preproc": "full_clean",
        "description": "Word+char with negation + contrast (full clean_text)",
    },
]

PREPROC_ABLATIONS: list[dict[str, Any]] = [
    {"name": "baseline_preproc", "preproc": "baseline_preproc", "features": "word+char"},
    {"name": "nahi_normalization", "preproc": "nahi_only", "features": "word+char"},
    {"name": "contrast_marker_only", "preproc": "contrast_marker_only", "features": "word+char"},
    {"name": "full_linguistic", "preproc": "full_clean", "features": "word+char"},
]


def build_ablation_pipeline(
    *,
    features: str = "word+char",
    preproc: str = "full_clean",
    calibrate: bool = True,
    C: float = 1.0,
) -> Pipeline:
    preprocessor = _make_preprocessor(preproc)
    transformers = []
    if features in {"word", "word+char"}:
        transformers.append(
            (
                "word",
                TfidfVectorizer(
                    preprocessor=preprocessor,
                    analyzer="word",
                    max_features=MAX_FEATURES,
                    ngram_range=NGRAM_RANGE,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            )
        )
    if features in {"char", "word+char"}:
        transformers.append(
            (
                "char",
                TfidfVectorizer(
                    preprocessor=preprocessor,
                    analyzer="char_wb",
                    ngram_range=CHAR_NGRAM_RANGE,
                    max_features=CHAR_MAX_FEATURES,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            )
        )
    if not transformers:
        raise ValueError(f"Unknown features mode: {features}")

    base = LinearSVC(
        C=C, max_iter=5000, random_state=RANDOM_SEED, class_weight="balanced"
    )
    clf: Any = (
        CalibratedClassifierCV(base, cv=CV_FOLDS, method="sigmoid", n_jobs=-1)
        if calibrate
        else base
    )
    if len(transformers) == 1:
        steps = [("features", transformers[0][1]), ("clf", clf)]
    else:
        steps = [("features", FeatureUnion(transformers)), ("clf", clf)]
    return Pipeline(steps)


def _lang_f1(per_lang: dict[str, Any], key: str) -> float | None:
    block = per_lang.get(key) or {}
    val = block.get("f1_macro")
    return float(val) if val is not None else None


def run_single_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    calibrate: bool = True,
) -> dict[str, Any]:
    X_train = train_df[TEXT_COLUMN].astype(str).tolist()
    y_train = train_df[LABEL_COLUMN].astype(str).tolist()
    X_test = test_df[TEXT_COLUMN].astype(str).tolist()
    y_test = test_df[LABEL_COLUMN].astype(str).tolist()
    lang_test = (
        test_df[LANGUAGE_COLUMN].astype(str).tolist()
        if LANGUAGE_COLUMN in test_df.columns
        else ["unknown"] * len(test_df)
    )

    pipe = build_ablation_pipeline(
        features=config["features"],
        preproc=config["preproc"],
        calibrate=calibrate,
    )
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0
    y_pred = [str(p) for p in pipe.predict(X_test)]
    metrics = compute_metrics(y_test, y_pred)
    neu = neutral_class_metrics(y_test, y_pred)
    per_lang = per_language_metrics(y_test, y_pred, lang_test)

    # Code-mixed F1: language tag code_mixed
    return {
        "configuration": config["name"],
        "description": config.get("description", ""),
        "features": config["features"],
        "preproc": config["preproc"],
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "neutral_f1": neu.get("f1"),
        "neutral_support": neu.get("support"),
        "english_f1": _lang_f1(per_lang, "english"),
        "roman_urdu_f1": _lang_f1(per_lang, "roman_urdu"),
        "code_mixed_f1": _lang_f1(per_lang, "code_mixed"),
        "urdu_f1": _lang_f1(per_lang, "urdu"),
        "training_time_seconds": round(train_seconds, 3),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


def run_ablation_suite(
    df: pd.DataFrame,
    *,
    configs: list[dict[str, Any]] | None = None,
    include_preproc_ablations: bool = True,
    calibrate: bool = True,
    reports_dir: str | Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """
    Run feature + preprocessing ablations on one fixed stratified split.

    If max_rows is set, subsample for CPU-friendly research sweeps.
    """
    working = df.copy()
    if max_rows is not None and len(working) > max_rows:
        # Stratified subsample
        try:
            working, _ = train_test_split(
                working,
                train_size=max_rows,
                random_state=RANDOM_SEED,
                stratify=working[LABEL_COLUMN],
            )
        except ValueError:
            working = working.sample(n=max_rows, random_state=RANDOM_SEED)
        working = working.reset_index(drop=True)

    train_df, test_df = train_test_split(
        working,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=working[LABEL_COLUMN],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    leakage = assert_no_partition_leakage(train_df, test_df)

    configs = list(configs or FEATURE_CONFIGS)
    if include_preproc_ablations:
        # Avoid duplicate names
        existing = {c["name"] for c in configs}
        for c in PREPROC_ABLATIONS:
            if c["name"] not in existing:
                configs.append(c)

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        rows.append(
            run_single_experiment(train_df, test_df, cfg, calibrate=calibrate)
        )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "n_full": int(len(df)),
        "n_used": int(len(working)),
        "subsampled": bool(max_rows is not None and len(df) > (max_rows or 0)),
        "leakage_report_summary": {
            "has_leakage": leakage.get("has_leakage"),
            "normalized_overlap_count": leakage.get("normalized_overlap_count"),
            "template_family_overlap_count": leakage.get(
                "template_family_overlap_count"
            ),
        },
        "results": rows,
        "notes": (
            "All metrics computed on the same holdout. Values are not hardcoded. "
            "Subsampled runs are for relative comparison only."
        ),
    }

    reports = Path(reports_dir) if reports_dir else REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    (reports / ABLATION_RESULTS_JSON.name).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (reports / ABLATION_RESULTS_MD.name).write_text(
        _format_ablation_md(payload), encoding="utf-8"
    )
    return payload


def _format_ablation_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Ablation Results",
        "",
        f"- Created: {payload.get('created_utc')}",
        f"- Random seed: {payload.get('random_seed')}",
        f"- Rows used: {payload.get('n_used')} / {payload.get('n_full')}",
        f"- Subsampled: {payload.get('subsampled')}",
        "",
        "| Configuration | Accuracy | Macro F1 | Neutral F1 | EN F1 | RU F1 | CM F1 | Urdu F1 | Train s |",
        "|---------------|----------|----------|------------|-------|-------|-------|---------|---------|",
    ]

    def _fmt(v):
        if v is None:
            return "n/a"
        return f"{float(v):.4f}"

    for r in payload.get("results") or []:
        lines.append(
            f"| {r['configuration']} | {_fmt(r.get('accuracy'))} | "
            f"{_fmt(r.get('f1_macro'))} | {_fmt(r.get('neutral_f1'))} | "
            f"{_fmt(r.get('english_f1'))} | {_fmt(r.get('roman_urdu_f1'))} | "
            f"{_fmt(r.get('code_mixed_f1'))} | {_fmt(r.get('urdu_f1'))} | "
            f"{r.get('training_time_seconds')} |"
        )
    lines.extend(["", payload.get("notes", ""), ""])
    return "\n".join(lines)
