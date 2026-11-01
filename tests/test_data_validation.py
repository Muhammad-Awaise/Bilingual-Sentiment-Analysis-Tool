"""Tests for dataset loading and validation."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import LABEL_COLUMN, TEXT_COLUMN
from src.data_loader import (
    DatasetError,
    assert_no_text_leakage,
    class_counts,
    infer_language,
    load_reviews_csv,
    normalize_label,
    stratified_split,
    validate_reviews_dataframe,
)


def test_normalize_label_aliases():
    assert normalize_label("POS") == "positive"
    assert normalize_label("Negative") == "negative"
    assert normalize_label("neutral") is None


def test_required_columns(synthetic_csv: Path):
    df = load_reviews_csv(synthetic_csv)
    assert TEXT_COLUMN in df.columns
    assert LABEL_COLUMN in df.columns
    assert len(df) > 0


def test_missing_file():
    with pytest.raises(DatasetError, match="not found"):
        load_reviews_csv(Path("data/raw/does_not_exist.csv"))


def test_empty_dataset(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("text,label\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_reviews_csv(p)


def test_unsupported_labels_dropped(tmp_path: Path):
    p = tmp_path / "mixed.csv"
    p.write_text(
        "text,label\ngood,positive\nbad,negative\nmeh,neutral\n",
        encoding="utf-8",
    )
    df = load_reviews_csv(p)
    assert set(df[LABEL_COLUMN]) <= {"positive", "negative"}
    assert "neutral" not in set(df[LABEL_COLUMN])


def test_missing_text_handled(tmp_path: Path):
    p = tmp_path / "gaps.csv"
    p.write_text(
        "text,label\n,positive\nokay,negative\ngreat,positive\n",
        encoding="utf-8",
    )
    df = load_reviews_csv(p)
    assert len(df) == 2
    assert all(str(t).strip() for t in df[TEXT_COLUMN])


def test_class_counts(synthetic_csv: Path):
    df = load_reviews_csv(synthetic_csv)
    counts = class_counts(df)
    assert counts["positive"] > 0
    assert counts["negative"] > 0


def test_split_no_leakage(synthetic_csv: Path):
    df = load_reviews_csv(synthetic_csv)
    train_df, test_df = stratified_split(df)
    assert_no_text_leakage(train_df, test_df)


def test_validate_needs_two_classes():
    df = pd.DataFrame({"text": ["a", "b"], "label": ["positive", "positive"]})
    with pytest.raises(DatasetError, match="two classes"):
        validate_reviews_dataframe(df)


def test_infer_language_roman_urdu():
    assert infer_language("Product bohat acha hai lekin packing theek nahi") in {
        "roman_urdu",
        "code_mixed",
    }


def test_infer_language_english():
    assert infer_language("This product is very good and the quality is great") == "english"


def test_load_infers_language_when_missing(tmp_path: Path):
    p = tmp_path / "nolang.csv"
    p.write_text(
        "text,label\n"
        "great product quality is good,positive\n"
        "bohat bura product hai bilkul bekaar,negative\n",
        encoding="utf-8",
    )
    df = load_reviews_csv(p)
    assert "language" in df.columns
    assert set(df["language"]) <= {"english", "roman_urdu", "code_mixed", "unknown"}
