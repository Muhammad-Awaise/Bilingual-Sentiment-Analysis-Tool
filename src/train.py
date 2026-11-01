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
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
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
from src.preprocessing import clean_text


def _augment_with_curated_urdu_and_neutral(df):
    """
    When training on the expanded binary CSV, also fold in curated rows that
    carry Neutral labels or Urdu-script text so those classes/features exist.

    Unique rows only here — oversampling happens on the train split after
    stratified holdout to avoid train/test leakage.
    """
    import pandas as pd

    if not DEFAULT_RAW_CSV.exists():
        return df
    try:
        curated = load_reviews_csv(DEFAULT_RAW_CSV)
    except DatasetError:
        return df

    mask = (curated[LABEL_COLUMN] == LABEL_NEUTRAL) | (
        curated[LANGUAGE_COLUMN].astype(str).str.lower() == "urdu"
    )
    extra = curated.loc[mask]
    if extra.empty:
        return df

    merged = pd.concat([df, extra], ignore_index=True)
    merged["_dedupe"] = merged[TEXT_COLUMN].astype(str).str.strip().str.lower()
    merged = merged.drop_duplicates(subset=["_dedupe"], keep="first")
    return merged.drop(columns=["_dedupe"]).reset_index(drop=True)


def _oversample_rare_train_rows(train_df):
    """Repeat Neutral / Urdu rows in *train only* so rare classes are learnable."""
    import pandas as pd

    if train_df is None or train_df.empty:
        return train_df

    neutral = train_df[train_df[LABEL_COLUMN] == LABEL_NEUTRAL]
    urdu = train_df[
        (train_df[LANGUAGE_COLUMN].astype(str).str.lower() == "urdu")
        & (train_df[LABEL_COLUMN] != LABEL_NEUTRAL)
    ]
    parts = [train_df]
    if not neutral.empty:
        parts.append(pd.concat([neutral] * 40, ignore_index=True))
    if not urdu.empty:
        parts.append(pd.concat([urdu] * 15, ignore_index=True))
    if len(parts) == 1:
        return train_df
    return pd.concat(parts, ignore_index=True)


def build_pipeline(
    classifier: str = "logreg",
    *,
    calibrate: bool = True,
    max_features: int = MAX_FEATURES,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    C: float = 1.0,
) -> Pipeline:
    """
    Build an unfitted sklearn Pipeline.

    TF-IDF is fit only on training data when pipeline.fit is called —
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
    if calibrate:
        clf = CalibratedClassifierCV(base, cv=CV_FOLDS, method="sigmoid")
        name = f"{name} (calibrated)"
    else:
        clf = base

    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                    lowercase=False,  # already lowercased in clean_text
                ),
            ),
            ("clf", clf),
        ]
    )
    setattr(pipe, "model_display_name", name)
    setattr(pipe, "is_calibrated", bool(calibrate))
    return pipe


def majority_baseline_accuracy(y_true) -> float:
    """Accuracy of always predicting the most frequent class."""
    y = np.asarray(y_true)
    if len(y) == 0:
        return 0.0
    _values, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def _param_grid(calibrate: bool) -> dict[str, list[Any]]:
    c_key = "clf__estimator__C" if calibrate else "clf__C"
    return {
        "tfidf__max_features": list(TUNE_MAX_FEATURES),
        "tfidf__ngram_range": list(TUNE_NGRAM_RANGES),
        c_key: list(TUNE_C_VALUES),
    }


def _fit_pipeline(
    pipe: Pipeline,
    X_train: list[str],
    y_train: list[str],
    *,
    tune: bool,
    calibrate: bool,
) -> tuple[Pipeline, dict[str, Any] | None]:
    """Fit pipeline, optionally with stratified GridSearchCV on F1-macro."""
    if not tune:
        pipe.fit(X_train, y_train)
        return pipe, None

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(
        pipe,
        param_grid=_param_grid(calibrate),
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
    setattr(best, "is_calibrated", bool(calibrate))
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
) -> dict[str, Any]:
    """
    Load data, split, fit pipeline on train only, save artifact, return metadata.

    Full evaluation metrics are written by src.evaluate — this function returns
    split sizes and paths so callers can evaluate without leakage.

    reports_dir: where holdout_split.joblib and train_meta.json are written.
                 Defaults to project REPORTS_DIR. Tests should pass a temp dir
                 so they do not overwrite project reports.
    """
    resolved_data = Path(data_path) if data_path else resolve_default_train_csv()
    df = load_reviews_csv(resolved_data)
    # Expanded set is binary Latin-heavy; merge curated Neutral + Urdu-script rows.
    if resolved_data.resolve() == EXPANDED_TRAIN_CSV.resolve() or (
        data_path is None and EXPANDED_TRAIN_CSV.exists()
    ):
        df = _augment_with_curated_urdu_and_neutral(df)
    if write_processed:
        prepare_processed(df)

    train_df, test_df = stratified_split(df)
    assert_no_text_leakage(train_df, test_df)
    train_df = _oversample_rare_train_rows(train_df)

    X_train = train_df[TEXT_COLUMN].tolist()
    y_train = train_df[LABEL_COLUMN].tolist()
    X_test = test_df[TEXT_COLUMN].tolist()
    y_test = test_df[LABEL_COLUMN].tolist()
    lang_train = train_df[LANGUAGE_COLUMN].astype(str).tolist()
    lang_test = test_df[LANGUAGE_COLUMN].astype(str).tolist()

    pipe = build_pipeline(classifier=classifier, calibrate=calibrate)
    pipe, tune_meta = _fit_pipeline(
        pipe, X_train, y_train, tune=tune, calibrate=calibrate
    )

    artifact = Path(artifact_path) if artifact_path else MODEL_ARTIFACT_PATH
    reports = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    payload = {
        "pipeline": pipe,
        "model_name": getattr(pipe, "model_display_name", MODEL_NAME),
        "labels": sorted(set(y_train)),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "class_counts_full": class_counts(df),
        "random_seed": RANDOM_SEED,
        "data_path": str(resolved_data),
        "calibrated": bool(calibrate),
        "class_weight": "balanced",
        "tuned": bool(tune),
        "tune_meta": tune_meta,
        "language_counts_full": (
            df[LANGUAGE_COLUMN].astype(str).value_counts().to_dict()
        ),
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
            "majority_baseline_accuracy": majority_baseline_accuracy(y_test),
        },
        split_path,
    )

    meta: dict[str, Any] = {
        "artifact_path": str(artifact),
        "split_path": str(split_path),
        "model_name": payload["model_name"],
        "train_size": len(train_df),
        "test_size": len(test_df),
        "class_counts": class_counts(df),
        "language_counts": payload["language_counts_full"],
        "majority_baseline_accuracy_on_test": majority_baseline_accuracy(y_test),
        "calibrated": bool(calibrate),
        "tuned": bool(tune),
        "data_path": str(resolved_data),
        "reports_dir": str(reports),
    }
    if tune_meta:
        meta["tune_meta"] = tune_meta

    if compare_svc:
        alt_name = "logreg" if classifier == "linearsvc" else "linearsvc"
        alt_pipe = build_pipeline(classifier=alt_name, calibrate=calibrate)
        alt_pipe, alt_tune = _fit_pipeline(
            alt_pipe, X_train, y_train, tune=tune, calibrate=calibrate
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
