"""Load, validate, and split labelled review datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_PROCESSED_CSV,
    DEFAULT_RAW_CSV,
    LABEL_COLUMN,
    LANGUAGE_COLUMN,
    RANDOM_SEED,
    SUPPORTED_LABELS,
    TEST_SIZE,
    TEXT_COLUMN,
)
from src.ingest import (
    IngestError,
    prepare_for_training,
    read_tabular_path,
)
from src.preprocessing import clean_text, contains_urdu_script


class DatasetError(ValueError):
    """Raised when a dataset is missing, empty, or invalid."""

# Lightweight cues for restoring language tags on expanded corpora
_ROMAN_URDU_CUES = frozenset(
    {
        "hai",
        "tha",
        "thi",
        "nahi",
        "nahin",
        "bohat",
        "acha",
        "achi",
        "bura",
        "kharab",
        "lekin",
        "magar",
        "mein",
        "main",
        "yeh",
        "ye",
        "bahut",
        "zabardast",
        "bekaar",
        "mat",
        "kya",
        "kaafi",
        "bilkul",
        "theek",
        "sahi",
        "gaya",
        "gayi",
        "hun",
        "hoon",
        "wala",
        "wali",
        "nahi",
        "nhi",
        "acha",
        "buhat",
        "boht",
        "delivery",  # often in code-mixed Daraz text; not used alone
    }
)
_ENGLISH_FUNCTION = frozenset(
    {
        "the",
        "and",
        "is",
        "was",
        "are",
        "were",
        "this",
        "that",
        "with",
        "for",
        "not",
        "very",
        "product",
        "quality",
        "good",
        "bad",
        "great",
        "but",
    }
)


def infer_language(text: Any) -> str:
    """
    Heuristic language tag for rows lacking an annotation.

    Returns english | roman_urdu | code_mixed | urdu | unknown.
    Conservative: prefers code_mixed when both cue sets fire.
    Urdu (Arabic) script is tagged as ``urdu``.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "unknown"
    raw = str(text).strip().lower()
    if not raw:
        return "unknown"
    if contains_urdu_script(raw):
        # Mixed Nastaliq + Latin still tagged urdu for slice reporting
        return "urdu"
    tokens = set(re.findall(r"[a-z']+", raw))
    if not tokens:
        return "unknown"
    # Exclude 'delivery' from RU cues for the pure-count check (too common in EN)
    ru_cues = _ROMAN_URDU_CUES - {"delivery"}
    ru_hits = len(tokens & ru_cues)
    en_hits = len(tokens & _ENGLISH_FUNCTION)
    if ru_hits >= 2 and en_hits >= 2:
        return "code_mixed"
    if ru_hits >= 2 and en_hits <= 1:
        return "roman_urdu"
    if en_hits >= 2 and ru_hits == 0:
        return "english"
    if ru_hits >= 1 and en_hits >= 1:
        return "code_mixed"
    if ru_hits >= 1:
        return "roman_urdu"
    if en_hits >= 1:
        return "english"
    return "unknown"


def normalize_label(value: Any) -> str | None:
    """Map common label spellings / star ratings to positive/negative/neutral."""
    from src.ingest import normalize_sentiment_label

    return normalize_sentiment_label(value)


def load_reviews_csv(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load a labelled table and normalize to text / label / language.

    Accepts CSV, TSV, Excel, JSON/JSONL. Column names may be org-specific
    (Review/Sentiment, review_body/star_rating, …) — they are mapped once
    via ``src.ingest``.
    """
    path = Path(path) if path else DEFAULT_RAW_CSV
    if not path.exists():
        raise DatasetError(
            f"Dataset file not found: {path}. "
            "Place a labelled reviews file in data/raw/ or pass a valid path."
        )

    try:
        raw_df, _meta = read_tabular_path(path)
        df = prepare_for_training(
            raw_df,
            infer_language_fn=infer_language,
        )
    except IngestError as exc:
        raise DatasetError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface as DatasetError
        raise DatasetError(f"Could not read dataset ({path}): {exc}") from exc

    if df.empty:
        raise DatasetError(f"Dataset is empty: {path}")

    return validate_reviews_dataframe(df)


def validate_reviews_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows; keep only supported labels; require non-empty text."""
    if df is None or len(df) == 0:
        raise DatasetError("Dataset has no rows.")

    working = df.copy()
    if TEXT_COLUMN not in working.columns or LABEL_COLUMN not in working.columns:
        raise DatasetError(
            f"Required columns missing. Need '{TEXT_COLUMN}' and '{LABEL_COLUMN}'."
        )

    working[LABEL_COLUMN] = working[LABEL_COLUMN].map(normalize_label)
    working = working.dropna(subset=[LABEL_COLUMN])
    working = working[working[LABEL_COLUMN].isin(SUPPORTED_LABELS)]

    working[TEXT_COLUMN] = working[TEXT_COLUMN].apply(
        lambda x: x if isinstance(x, str) else ("" if pd.isna(x) else str(x))
    )
    working = working[working[TEXT_COLUMN].str.strip() != ""]

    # Drop duplicate texts (case/whitespace-insensitive) to reduce leakage risk
    working["_dedupe_key"] = (
        working[TEXT_COLUMN].astype(str).str.strip().str.lower()
    )
    working = working.drop_duplicates(subset=["_dedupe_key"], keep="first")
    working = working.drop(columns=["_dedupe_key"]).reset_index(drop=True)

    if working.empty:
        raise DatasetError(
            "No valid rows remain after filtering empty text and unsupported labels. "
            f"Supported labels: {sorted(SUPPORTED_LABELS)}."
        )

    if working[LABEL_COLUMN].nunique() < 2:
        raise DatasetError(
            "Dataset must contain at least two sentiment classes "
            "(e.g. positive and negative) after cleaning."
        )

    return working


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return label → count mapping."""
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def prepare_processed(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add cleaned_text column and optionally write processed CSV."""
    processed = df.copy()
    processed["cleaned_text"] = processed[TEXT_COLUMN].map(clean_text)
    processed = processed[processed["cleaned_text"].str.strip() != ""].reset_index(
        drop=True
    )
    if processed.empty:
        raise DatasetError("All rows became empty after preprocessing.")

    out = Path(output_path) if output_path else DEFAULT_PROCESSED_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out, index=False, encoding="utf-8")
    return processed


def stratified_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on label. Raises if too small for stratification."""
    if len(df) < 4:
        raise DatasetError("Need at least 4 rows to create a train/test split.")

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[LABEL_COLUMN],
        )
    except ValueError as exc:
        raise DatasetError(f"Could not create stratified split: {exc}") from exc

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def assert_no_text_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Raise DatasetError if identical texts appear in both splits."""
    train_texts = set(train_df[TEXT_COLUMN].astype(str).str.strip().str.lower())
    test_texts = set(test_df[TEXT_COLUMN].astype(str).str.strip().str.lower())
    overlap = train_texts & test_texts
    if overlap:
        sample = list(overlap)[:3]
        raise DatasetError(
            f"Train/test leakage detected: {len(overlap)} shared texts. "
            f"Examples: {sample}"
        )
