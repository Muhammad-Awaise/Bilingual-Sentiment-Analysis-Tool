"""Tests for flexible org-data ingest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import LABEL_COLUMN, LANGUAGE_COLUMN, TEXT_COLUMN
from src.data_loader import infer_language, load_reviews_csv, normalize_label
from src.ingest import (
    IngestError,
    detect_column_map,
    normalize_sentiment_label,
    prepare_for_prediction,
    prepare_for_training,
    read_tabular_bytes,
    read_tabular_path,
)


def test_label_aliases_and_stars():
    assert normalize_sentiment_label("POS") == "positive"
    assert normalize_sentiment_label("Negative") == "negative"
    assert normalize_sentiment_label("neu") == "neutral"
    assert normalize_sentiment_label(5) == "positive"
    assert normalize_sentiment_label(1) == "negative"
    assert normalize_sentiment_label(3) == "neutral"
    assert normalize_sentiment_label("4 stars") == "positive"
    assert normalize_sentiment_label("mixed") is None
    assert normalize_label("5") == "positive"


def test_detect_org_columns():
    df = pd.DataFrame(
        {
            "Review": ["great", "bad"],
            "Sentiment": ["Positive", "Negative"],
            "Lang": ["en", "en"],
        }
    )
    cmap = detect_column_map(df)
    assert cmap.text == "Review"
    assert cmap.label == "Sentiment"
    assert cmap.language == "Lang"


def test_prepare_training_normalizes_schema():
    df = pd.DataFrame(
        {
            "review_body": ["Excellent phone", "Terrible battery", "Just okay"],
            "star_rating": [5, 1, 3],
        }
    )
    out = prepare_for_training(df, infer_language_fn=infer_language)
    assert list(out.columns) == [TEXT_COLUMN, LABEL_COLUMN, LANGUAGE_COLUMN]
    assert set(out[LABEL_COLUMN]) == {"positive", "negative", "neutral"}


def test_prepare_prediction_keeps_org_columns():
    df = pd.DataFrame(
        {
            "store_id": [1, 2],
            "Comments": ["bohat acha", "bekaar"],
        }
    )
    kept, cmap, report = prepare_for_prediction(df)
    assert "store_id" in kept.columns
    assert cmap.text == "Comments"
    assert report.n_rows == 2
    assert report.text_column == "Comments"


def test_read_csv_encoding_and_json(tmp_path: Path):
    csv_path = tmp_path / "org.csv"
    csv_path.write_text(
        "Reviews,Sentiments\nquality zabardast,positive\nkharab item,negative\n",
        encoding="utf-8",
    )
    df, meta = read_tabular_path(csv_path)
    assert meta["format"] == "csv"
    assert "Reviews" in df.columns

    payload = [
        {"comment": "love it", "label": "pos"},
        {"comment": "hate it", "label": "neg"},
    ]
    raw = json.dumps(payload).encode("utf-8")
    jdf, jmeta = read_tabular_bytes(raw, filename="reviews.json")
    assert jmeta["format"] == "json"
    assert len(jdf) == 2


def test_read_excel_org_file(tmp_path: Path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "org.xlsx"
    frame = pd.DataFrame(
        {
            "Feedback": ["great product", "worst purchase"],
            "Rating": [5, 1],
        }
    )
    frame.to_excel(path, index=False)
    df, meta = read_tabular_path(path)
    assert meta["format"] == "excel"
    trained = prepare_for_training(df, infer_language_fn=infer_language)
    assert set(trained[LABEL_COLUMN]) == {"positive", "negative"}


def test_load_reviews_csv_accepts_review_sentiment(tmp_path: Path):
    path = tmp_path / "daraz_style.csv"
    path.write_text(
        "Reviews,Sentiments\n"
        "bohat acha product,positive\n"
        "bilkul bekaar,negative\n"
        "theek hai,neutral\n",
        encoding="utf-8",
    )
    df = load_reviews_csv(path)
    assert TEXT_COLUMN in df.columns
    assert LABEL_COLUMN in df.columns
    assert set(df[LABEL_COLUMN]) <= {"positive", "negative", "neutral"}


def test_training_requires_label_column():
    df = pd.DataFrame({"comment": ["great", "bad"]})
    with pytest.raises(IngestError, match="label"):
        prepare_for_training(df)
