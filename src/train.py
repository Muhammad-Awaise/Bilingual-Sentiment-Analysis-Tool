"""Train TF-IDF + Logistic Regression (and optional LinearSVC) sentiment models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    CHAR_MAX_FEATURES,
    CHAR_NGRAM_RANGE,
    CV_FOLDS,
    DEFAULT_RAW_CSV,
    EXPANDED_TRAIN_CSV,
    HOLDOUT_SPLIT_PATH,
    LABEL_COLUMN,
    LABEL_NEUTRAL,
    LANGUAGE_COLUMN,
    MAX_FEATURES,
    MODEL_ARTIFACT_PATH,
    MODEL_NAME,
    MODELS_DIR,
    NGRAM_RANGE,
    RANDOM_SEED,
    REPORTS_DIR,
    TEXT_COLUMN,
    TRAIN_META_PATH,
    TUNE_C_VALUES,
    TUNE_MAX_FEATURES,
    TUNE_NGRAM_RANGES,
    resolve_default_train_csv,
)
from src.data_loader import (
    DatasetError,
    assert_no_text_leakage,
    class_counts,
    load_reviews_csv,
    prepare_processed,
    stratified_split,
)
from src.daraz_eval import exclude_daraz_from_training
from src.leakage import (
    PROVENANCE_COLUMN,
    SOURCE_COLUMN,
    TEMPLATE_FAMILY_COLUMN,
    build_leakage_report,
    fingerprint_texts,
)
from src.neutral_analysis import class_distribution_report
from src.preprocessing import clean_text
from src.reproducibility import build_run_metadata, write_run_metadata
from src.urdu_augment import build_urdu_script_corpus


def _augment_with_curated_urdu_and_neutral(df):
    """
    When training on the expanded CSV, ensure Neutral + Urdu-script coverage.

    The expanded builder already embeds a large Urdu corpus; this path still
    merges curated Neutral/Urdu and the full synthetic set so older expanded
    CSVs (without Urdu) cannot silently collapse Nastaliq predictions.

    Unique rows only here — oversampling happens on the train split after
    stratified holdout to avoid train/test leakage.
    """
    import pandas as pd

    extras: list[pd.DataFrame] = []
    if DEFAULT_RAW_CSV.exists():
        try:
            curated = load_reviews_csv(DEFAULT_RAW_CSV)
            mask = (curated[LABEL_COLUMN] == LABEL_NEUTRAL) | (
                curated[LANGUAGE_COLUMN].astype(str).str.lower() == "urdu"
            )
            extra = curated.loc[mask]
            if not extra.empty:
                extras.append(extra)
        except DatasetError:
            pass

    extras.append(build_urdu_script_corpus())
    if not extras:
        return df

    merged = pd.concat([df, *extras], ignore_index=True)
    merged["_dedupe"] = merged[TEXT_COLUMN].astype(str).str.strip().str.lower()
    merged = merged.drop_duplicates(subset=["_dedupe"], keep="first")
    return merged.drop(columns=["_dedupe"]).reset_index(drop=True)


def _oversample_rare_train_rows(train_df, *, mild_neutral: bool = True):
    """
    Repeat Neutral / Urdu rows in *train only* so rare classes are learnable.

    Controlled: Neutral upsampling is capped mildly when genuine Neutral volume
    is already adequate. Never invents Neutral labels.
    """
    import pandas as pd

    if train_df is None or train_df.empty:
        return train_df

    neutral = train_df[train_df[LABEL_COLUMN] == LABEL_NEUTRAL]
    urdu = train_df[
        (train_df[LANGUAGE_COLUMN].astype(str).str.lower() == "urdu")
        & (train_df[LABEL_COLUMN] != LABEL_NEUTRAL)
    ]
    parts = [train_df]
    if not neutral.empty and mild_neutral:
        # Cap aggressively if Neutral already plentiful (e.g. after UCI Neutral merge)
        n_neu = len(neutral)
        if n_neu < 500:
            reps = max(1, min(20, 3000 // max(n_neu, 1)))
            parts.append(pd.concat([neutral] * reps, ignore_index=True))
        elif n_neu < 2000:
            reps = max(1, min(3, 2500 // max(n_neu, 1)))
            parts.append(pd.concat([neutral] * reps, ignore_index=True))
        # else: genuine Neutral volume sufficient — no blind oversample
    if not urdu.empty:
        # Prefer real/curated Urdu for denser features; still allow synthetic
        realish = urdu
        if PROVENANCE_COLUMN in urdu.columns:
            preferred = urdu[
                urdu[PROVENANCE_COLUMN]
                .astype(str)
                .isin({"real_urdu", "curated", "external_real"})
            ]
            if len(preferred) >= 20:
                realish = preferred
        reps = max(1, min(10, 8000 // max(len(realish), 1)))
        parts.append(pd.concat([realish] * reps, ignore_index=True))
    if len(parts) == 1:
        return train_df
    return pd.concat(parts, ignore_index=True)


def build_pipeline(
    classifier: str = "logreg",
    *,
    calibrate: bool = True,
    max_features: int = MAX_FEATURES,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    char_max_features: int = CHAR_MAX_FEATURES,
    char_ngram_range: tuple[int, int] = CHAR_NGRAM_RANGE,
    C: float = 1.0,
) -> Pipeline:
    """
    Build an unfitted sklearn Pipeline.

    Features = word TF-IDF ∪ character TF-IDF (char_wb). Word n-grams carry
    English / Roman Urdu polarity; character n-grams keep signal when
    Urdu-script word tokens are rare or out-of-vocabulary.

    Vectorizers are fit only on training data when pipeline.fit is called —
    never on the full dataset before splitting.

    LogisticRegression / LinearSVC use class_weight='balanced' for imbalance.
    When calibrate=True, the classifier is wrapped in CalibratedClassifierCV
    so predict_proba returns sigmoid-calibrated probabilities.
    """
    if classifier == "linearsvc":
        base: Any = LinearSVC(
            C=C,
            max_iter=5000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )
        name = MODEL_NAME  # "TF-IDF + LinearSVC"
    else:
        base = LogisticRegression(
            max_iter=2000,
            C=C,
            # lbfgs supports 3-class (pos/neu/neg); liblinear does not
            solver="lbfgs",
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )
        name = "TF-IDF + Logistic Regression"

    clf: Any
    is_calibrated = False
    if calibrate and classifier == "linearsvc":
        # LinearSVC needs calibration for predict_proba; parallelize folds.
        clf = CalibratedClassifierCV(
            base, cv=CV_FOLDS, method="sigmoid", n_jobs=-1
        )
        name = f"{name} (calibrated)"
        is_calibrated = True
    elif calibrate and classifier == "logreg":
        # LogisticRegression already exposes predict_proba — skip nested CV.
        clf = base
        name = f"{name} (proba)"
    else:
        clf = base

    word_tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="word",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        lowercase=False,  # already lowercased in clean_text
    )
    char_tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        max_features=char_max_features,
        sublinear_tf=True,
        lowercase=False,
    )
    pipe = Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        ("word", word_tfidf),
                        ("char", char_tfidf),
                    ]
                ),
            ),
            ("clf", clf),
        ]
    )
    setattr(pipe, "model_display_name", name)
    setattr(pipe, "is_calibrated", is_calibrated)
    return pipe


def majority_baseline_accuracy(y_true) -> float:
    """Accuracy of always predicting the most frequent class."""
    y = np.asarray(y_true)
    if len(y) == 0:
        return 0.0
    _values, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def _param_grid(calibrate: bool, classifier: str = "logreg") -> dict[str, list[Any]]:
    # Only LinearSVC nests under CalibratedClassifierCV when calibrate=True.
    nested = bool(calibrate) and classifier == "linearsvc"
    c_key = "clf__estimator__C" if nested else "clf__C"
    return {
        "features__word__max_features": list(TUNE_MAX_FEATURES),
        "features__word__ngram_range": list(TUNE_NGRAM_RANGES),
        c_key: list(TUNE_C_VALUES),
    }


def _fit_pipeline(
    pipe: Pipeline,
    X_train: list[str],
    y_train: list[str],
    *,
    tune: bool,
    calibrate: bool,
    classifier: str = "logreg",
) -> tuple[Pipeline, dict[str, Any] | None]:
    """Fit pipeline, optionally with stratified GridSearchCV on F1-macro."""
    if not tune:
        pipe.fit(X_train, y_train)
        return pipe, None

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(
        pipe,
        param_grid=_param_grid(calibrate, classifier=classifier),
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    setattr(
        best,
        "model_display_name",
        getattr(pipe, "model_display_name", MODEL_NAME) + " (tuned)",
    )
    setattr(best, "is_calibrated", bool(getattr(pipe, "is_calibrated", False)))
    tune_meta = {
        "best_params": search.best_params_,
        "best_cv_f1_macro": float(search.best_score_),
        "cv_folds": CV_FOLDS,
    }
    return best, tune_meta


def train_and_save(
    data_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    classifier: str = "logreg",
    compare_svc: bool = False,
    *,
    reports_dir: str | Path | None = None,
    calibrate: bool = True,
    tune: bool = False,
    write_processed: bool = False,
    exclude_daraz: bool = False,
) -> dict[str, Any]:
    """
    Load data, split, fit pipeline on train only, save artifact, return metadata.

    Full evaluation metrics are written by src.evaluate — this function returns
    split sizes and paths so callers can evaluate without leakage.

    reports_dir: where holdout_split.joblib and train_meta.json are written.
                 Defaults to project REPORTS_DIR. Tests should pass a temp dir
                 so they do not overwrite project reports.
    exclude_daraz: if True, drop Daraz-sourced rows before split so Daraz can
                   be evaluated externally.
    """
    resolved_data = Path(data_path) if data_path else resolve_default_train_csv()
    df = load_reviews_csv(resolved_data)
    # Expanded set is Latin-heavy; merge curated Neutral + Urdu-script rows.
    if resolved_data.resolve() == EXPANDED_TRAIN_CSV.resolve() or (
        data_path is None and EXPANDED_TRAIN_CSV.exists()
    ):
        df = _augment_with_curated_urdu_and_neutral(df)
    if exclude_daraz:
        df = exclude_daraz_from_training(df)
    if write_processed:
        prepare_processed(df)

    distribution = class_distribution_report(df)

    train_df, test_df = stratified_split(df)
    leakage_report = assert_no_text_leakage(train_df, test_df)
    # Persist full leakage report before fitting
    reports = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "leakage_report.json").write_text(
        json.dumps(
            leakage_report
            if isinstance(leakage_report, dict)
            else build_leakage_report(train_df, test_df),
            indent=2,
        ),
        encoding="utf-8",
    )

    train_df = _oversample_rare_train_rows(train_df)

    X_train = train_df[TEXT_COLUMN].tolist()
    y_train = train_df[LABEL_COLUMN].tolist()
    X_test = test_df[TEXT_COLUMN].tolist()
    y_test = test_df[LABEL_COLUMN].tolist()
    lang_train = train_df[LANGUAGE_COLUMN].astype(str).tolist()
    lang_test = test_df[LANGUAGE_COLUMN].astype(str).tolist()
    source_test = (
        test_df[SOURCE_COLUMN].astype(str).tolist()
        if SOURCE_COLUMN in test_df.columns
        else None
    )
    provenance_test = (
        test_df[PROVENANCE_COLUMN].astype(str).tolist()
        if PROVENANCE_COLUMN in test_df.columns
        else None
    )

    pipe = build_pipeline(classifier=classifier, calibrate=calibrate)
    pipe, tune_meta = _fit_pipeline(
        pipe,
        X_train,
        y_train,
        tune=tune,
        calibrate=calibrate,
        classifier=classifier,
    )

    artifact = Path(artifact_path) if artifact_path else MODEL_ARTIFACT_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "pipeline": pipe,
        "model_name": getattr(pipe, "model_display_name", MODEL_NAME),
        "labels": sorted(set(y_train)),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "class_counts_full": class_counts(df),
        "random_seed": RANDOM_SEED,
        "data_path": str(resolved_data),
        "calibrated": bool(getattr(pipe, "is_calibrated", False)),
        "class_weight": "balanced",
        "tuned": bool(tune),
        "tune_meta": tune_meta,
        "language_counts_full": (
            df[LANGUAGE_COLUMN].astype(str).value_counts().to_dict()
        ),
        "provenance_counts_full": (
            df[PROVENANCE_COLUMN].astype(str).value_counts().to_dict()
            if PROVENANCE_COLUMN in df.columns
            else {}
        ),
        "source_counts_full": (
            df[SOURCE_COLUMN].astype(str).value_counts().to_dict()
            if SOURCE_COLUMN in df.columns
            else {}
        ),
        "exclude_daraz": bool(exclude_daraz),
        "text_fingerprint_train": fingerprint_texts(X_train),
        "distribution_report": distribution,
        "preprocessing": {
            "clean_text": True,
            "nahi_normalization": True,
            "negation_scope": True,
            "contrast_clause_repeat": True,
            "intensity_markers": True,
        },
        "vectorizer": {
            "word_max_features": MAX_FEATURES,
            "word_ngram_range": list(NGRAM_RANGE),
            "char_max_features": CHAR_MAX_FEATURES,
            "char_ngram_range": list(CHAR_NGRAM_RANGE),
        },
        "classifier_params": {
            "type": classifier,
            "class_weight": "balanced",
        },
        "calibration": {
            "method": "sigmoid" if calibrate and classifier == "linearsvc" else (
                "native_predict_proba" if classifier == "logreg" else "none"
            ),
            "enabled": bool(calibrate),
        },
    }
    joblib.dump(payload, artifact)

    split_path = reports / HOLDOUT_SPLIT_PATH.name
    joblib.dump(
        {
            "X_test": X_test,
            "y_test": y_test,
            "X_train": X_train,
            "y_train": y_train,
            "lang_test": lang_test,
            "lang_train": lang_train,
            "source_test": source_test,
            "provenance_test": provenance_test,
            "majority_baseline_accuracy": majority_baseline_accuracy(y_test),
        },
        split_path,
    )

    run_meta = build_run_metadata(
        data_path=resolved_data,
        n_records=len(df),
        class_counts=class_counts(df),
        language_counts=payload["language_counts_full"],
        provenance_counts=payload["provenance_counts_full"],
        source_counts=payload["source_counts_full"],
        random_seed=RANDOM_SEED,
        preprocessing=payload["preprocessing"],
        vectorizer=payload["vectorizer"],
        classifier=payload["classifier_params"],
        calibration=payload["calibration"],
        artifact_path=artifact,
        extra={
            "artifact_path": str(artifact),
            "split_path": str(split_path),
            "model_name": payload["model_name"],
            "train_size": len(train_df),
            "test_size": len(test_df),
            "majority_baseline_accuracy_on_test": majority_baseline_accuracy(y_test),
            "calibrated": bool(getattr(pipe, "is_calibrated", False)),
            "tuned": bool(tune),
            "data_path": str(resolved_data),
            "reports_dir": str(reports),
            "exclude_daraz": bool(exclude_daraz),
            "distribution_warnings": distribution.get("warnings"),
            "tune_meta": tune_meta,
        },
    )
    write_run_metadata(run_meta, reports / TRAIN_META_PATH.name)

    meta: dict[str, Any] = dict(run_meta)
    if tune_meta:
        meta["tune_meta"] = tune_meta

    if compare_svc:
        alt_name = "logreg" if classifier == "linearsvc" else "linearsvc"
        alt_pipe = build_pipeline(classifier=alt_name, calibrate=calibrate)
        alt_pipe, alt_tune = _fit_pipeline(
            alt_pipe,
            X_train,
            y_train,
            tune=tune,
            calibrate=calibrate,
            classifier=alt_name,
        )
        alt_path = (
            MODELS_DIR / f"sentiment_pipeline_{alt_name}.joblib"
            if artifact.parent == MODELS_DIR
            else artifact.parent / f"sentiment_pipeline_{alt_name}.joblib"
        )
        joblib.dump(
            {
                **payload,
                "pipeline": alt_pipe,
                "model_name": getattr(
                    alt_pipe,
                    "model_display_name",
                    f"TF-IDF + {alt_name}",
                ),
                "tune_meta": alt_tune,
                "tuned": bool(tune),
            },
            alt_path,
        )
        meta["comparison_artifact"] = str(alt_path)
        if alt_name == "linearsvc":
            meta["linearsvc_artifact"] = str(alt_path)
        else:
            meta["logreg_artifact"] = str(alt_path)

    meta_path = reports / TRAIN_META_PATH.name
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["meta_path"] = str(meta_path)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bilingual sentiment model")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Path to labelled CSV "
            f"(default: {EXPANDED_TRAIN_CSV.name} if present, else curated raw CSV)"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output joblib path (default: models/sentiment_pipeline.joblib)",
    )
    parser.add_argument(
        "--classifier",
        choices=["logreg", "linearsvc"],
        default="linearsvc",
        help="Primary classifier (default: linearsvc; preferred on current holdout)",
    )
    parser.add_argument(
        "--compare-svc",
        action="store_true",
        help="Also train LinearSVC for comparison",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable CalibratedClassifierCV (not recommended for UI scores)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run stratified GridSearchCV over C / max_features / n-grams",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Directory for holdout split + train_meta (default: reports/)",
    )
    parser.add_argument(
        "--exclude-daraz",
        action="store_true",
        help=(
            "Drop Daraz-sourced rows before training so Daraz can be evaluated "
            "as a true external test set"
        ),
    )
    args = parser.parse_args()

    try:
        meta = train_and_save(
            data_path=args.data,
            artifact_path=args.out,
            classifier=args.classifier,
            compare_svc=args.compare_svc,
            reports_dir=args.reports_dir,
            calibrate=not args.no_calibrate,
            tune=args.tune,
            write_processed=True,
            exclude_daraz=args.exclude_daraz,
        )
    except DatasetError as exc:
        raise SystemExit(f"Training failed: {exc}") from exc

    print("Training complete.")
    print(json.dumps(meta, indent=2))

    if meta.get("comparison_artifact"):
        try:
            from src.evaluate import write_model_comparison

            primary = Path(meta["artifact_path"])
            alternate = Path(meta["comparison_artifact"])
            # Normalise argument order for the comparison helper
            if "linearsvc" in primary.name.lower() or "linearsvc" in str(
                meta.get("model_name", "")
            ).lower():
                logreg_path = alternate
                svc_path = primary
            else:
                logreg_path = primary
                svc_path = alternate

            comparison = write_model_comparison(
                logreg_artifact=logreg_path,
                linearsvc_artifact=svc_path,
                split_path=meta["split_path"],
                reports_dir=meta["reports_dir"],
            )
            print("Model comparison written:")
            print(json.dumps(comparison, indent=2))
        except Exception as exc:  # noqa: BLE001 — comparison is best-effort after train
            print(f"Warning: could not write model comparison: {exc}")


if __name__ == "__main__":
    main()
