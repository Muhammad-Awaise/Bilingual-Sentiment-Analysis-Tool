"""Tests for batch export formatting and ZIP package."""

from pathlib import Path

import pandas as pd

from src.export_results import (
    build_results_zip,
    dataframe_to_csv_bytes,
    format_predictions_csv,
    format_single_result_csv,
    format_summary_csv,
)
from src.predict import predict_dataframe, predict_one


def test_single_result_csv_excel_friendly(trained_artifact: Path):
    result = predict_one("great product love it", artifact_path=trained_artifact)
    formatted = format_single_result_csv(result)
    assert list(formatted.columns)[0] == "row_number"
    assert "predicted_sentiment" in formatted.columns
    assert "review" in formatted.columns
    raw = dataframe_to_csv_bytes(formatted)
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"predicted_sentiment" in raw


def test_format_predictions_has_clear_columns(trained_artifact: Path):
    df = pd.DataFrame({"review": ["great product", "terrible quality"]})
    result = predict_dataframe(df, "review", artifact_path=trained_artifact)
    formatted = format_predictions_csv(result, "review")
    assert "row_number" in formatted.columns
    assert "predicted_sentiment" in formatted.columns
    assert "model_score_not_probability" in formatted.columns
    assert list(formatted["row_number"]) == [1, 2]


def test_summary_has_total_row(trained_artifact: Path):
    df = pd.DataFrame({"review": ["great product", "terrible quality", "amazing"]})
    result = predict_dataframe(df, "review", artifact_path=trained_artifact)
    summary = format_summary_csv(result["prediction"])
    assert "TOTAL" in set(summary["sentiment"])
    assert int(summary.loc[summary["sentiment"] == "TOTAL", "count"].iloc[0]) == 3


def test_csv_bytes_are_excel_friendly():
    df = pd.DataFrame({"predicted_sentiment": ["positive"], "review": ["acha"]})
    raw = dataframe_to_csv_bytes(df)
    # UTF-8 BOM for Excel
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"predicted_sentiment" in raw


def test_zip_contains_csv_and_chart(trained_artifact: Path):
    import zipfile
    import io

    df = pd.DataFrame({"review": ["great product", "terrible quality"]})
    result = predict_dataframe(df, "review", artifact_path=trained_artifact)
    zip_bytes, meta = build_results_zip(result, "review")
    assert meta["chart_png"][:8] == b"\x89PNG\r\n\x1a\n"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
    assert "sentiment_predictions.csv" in names
    assert "sentiment_summary.csv" in names
    assert "sentiment_distribution.png" in names
    assert "README.txt" in names
