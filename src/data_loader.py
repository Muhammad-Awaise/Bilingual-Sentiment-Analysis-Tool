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
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    LANGUAGE_COLUMN,
    RANDOM_SEED,
    SUPPORTED_LABELS,
    TEST_SIZE,
    TEXT_COLUMN,
)
from src.preprocessing import clean_text


class DatasetError(ValueError):
    """Raised when a dataset is missing, empty, or invalid."""


_LABEL_ALIASES = {
    "pos": LABEL_POSITIVE,
    "positive": LABEL_POSITIVE,
    "neg": LABEL_NEGATIVE,
    "negative": LABEL_NEGATIVE,
}

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

    Returns english | roman_urdu | code_mixed | unknown.
    Conservative: prefers code_mixed when both cue sets fire.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "unknown"
    raw = str(text).strip().lower()
    if not raw:
        return "unknown"
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
    """Map common label spellings to positive/negative; else None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    key = str(value).strip().lower()
    return _LABEL_ALIASES.get(key)


def load_reviews_csv(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load a labelled CSV. Accepts either our schema (text, label[, language])
    or common alternates (Review/Sentiment, Reviews/Sentiments).
    """
    path = Path(path) if path else DEFAULT_RAW_CSV
    if not path.exists():
        raise DatasetError(
            f"Dataset file not found: {path}. "
            "Place bilingual_product_reviews.csv in data/raw/ or pass a valid path."
        )

    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception as exc:  # noqa: BLE001 — surface as DatasetError
        raise DatasetError(f"Could not read CSV ({path}): {exc}") from exc

    if df.empty:
        raise DatasetError(f"Dataset is empty: {path}")

    df = _normalize_columns(df)
    return validate_reviews_dataframe(df)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    text_candidates = ("text", "review", "reviews", "comment", "comments")
    label_candidates = ("label", "sentiment", "sentiments", "class")

    text_col = next((cols[c] for c in text_candidates if c in cols), None)
    label_col = next((cols[c] for c in label_candidates if c in cols), None)

    if text_col is None or label_col is None:
        raise DatasetError(
            "CSV must contain a text column (text/review/reviews) and a "
            f"label column (label/sentiment). Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            TEXT_COLUMN: df[text_col],
            LABEL_COLUMN: df[label_col],
        }
    )
    lang_key = next((cols[c] for c in ("language", "lang") if c in cols), None)
    if lang_key is not None:
        out[LANGUAGE_COLUMN] = (
            df[lang_key]
            .map(lambda v: "unknown" if pd.isna(v) else str(v).strip().lower())
        )
        # Fill blank / missing language tags with a conservative heuristic
        blank = out[LANGUAGE_COLUMN].isin({"", "nan", "none", "unknown"})
        if blank.any():
            out.loc[blank, LANGUAGE_COLUMN] = out.loc[blank, TEXT_COLUMN].map(
                infer_language
            )
    else:
        out[LANGUAGE_COLUMN] = out[TEXT_COLUMN].map(infer_language)
    return out


def validate_reviews_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows; keep only supported binary labels; require non-empty text."""
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
            "Dataset must contain at least two classes (positive and negative) "
            "after cleaning."
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
