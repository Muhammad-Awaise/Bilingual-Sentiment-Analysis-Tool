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


def test_format_predictions_allows_existing_row_number(trained_artifact: Path):
    """Re-uploading a previous predictions CSV must not crash on row_number."""
    df = pd.DataFrame(
        {
            "row_number": [10, 20],
            "text": ["great product", "terrible quality"],
            "label": ["positive", "negative"],
        }
    )
    result = predict_dataframe(df, "text", artifact_path=trained_artifact)
    formatted = format_predictions_csv(result, "text")
    assert list(formatted["row_number"]) == [1, 2]
    assert formatted.columns.tolist().count("row_number") == 1


def test_format_predictions_reupload_full_export_schema(trained_artifact: Path):
    """Full prior-export schema (row_number + predicted_* + scores) must work."""
    df = pd.DataFrame(
        {
            "row_number": [1, 2],
            "text": ["great product", "terrible quality"],
            "label": ["positive", "negative"],
            "language": ["english", "english"],
            "cleaned_review_text": ["great", "terrible"],
            "predicted_sentiment": ["positive", "negative"],
            "model_score_not_probability": [0.9, 0.8],
            "model_name": ["old-model", "old-model"],
            "model_train_size": [100, 100],
            "urdu_train_rows": [10, 10],
        }
    )
    result = predict_dataframe(df, "text", artifact_path=trained_artifact)
    formatted = format_predictions_csv(result, "text")
    assert list(formatted["row_number"]) == [1, 2]
    assert formatted.columns.tolist().count("predicted_sentiment") == 1
    assert formatted.columns.tolist().count("model_score_not_probability") == 1
    assert isinstance(formatted["model_score_not_probability"], pd.Series)


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
