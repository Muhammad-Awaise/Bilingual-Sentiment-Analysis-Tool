"""Format batch prediction downloads (CSV + summary + chart)."""

from __future__ import annotations

import io
import zipfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.predict import sentiment_summary


def format_single_result_csv(result: dict[str, Any]) -> pd.DataFrame:
    """One-row Excel-friendly table for a single-review prediction."""
    score = result.get("model_score")
    if score is not None:
        try:
            score = round(float(score), 4)
        except (TypeError, ValueError):
            score = None
    return pd.DataFrame(
        [
            {
                "row_number": 1,
                "review": result.get("text", ""),
                "predicted_sentiment": result.get("prediction", ""),
                "model_score_not_probability": score,
                "cleaned_review_text": result.get("cleaned_text", ""),
                "model_name": result.get("model_name", ""),
                "score_note": result.get("score_note", ""),
                "limitations": result.get("limitations", ""),
            }
        ]
    )


def format_predictions_csv(result: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Build a clean, Excel-friendly results table.

    - Stable column order
    - Clear prediction column names
    - Rounded model scores
    - Leading row_number for easy lookup
    """
    out = result.copy()
    out.insert(0, "row_number", range(1, len(out) + 1))

    rename = {
        "prediction": "predicted_sentiment",
        "model_score": "model_score_not_probability",
        "cleaned_text": "cleaned_review_text",
        "model_name": "model_name",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    if "model_score_not_probability" in out.columns:
        out["model_score_not_probability"] = pd.to_numeric(
            out["model_score_not_probability"], errors="coerce"
        ).round(4)

    # Prefer a readable alias for the source text column when it is generic
    preferred_front = ["row_number"]
    if text_column in out.columns and text_column not in preferred_front:
        preferred_front.append(text_column)

    trailing = [
        c
        for c in (
            "predicted_sentiment",
            "model_score_not_probability",
            "cleaned_review_text",
            "model_name",
        )
        if c in out.columns
    ]
    middle = [c for c in out.columns if c not in preferred_front and c not in trailing]
    return out[preferred_front + middle + trailing]


def format_summary_csv(predictions: pd.Series | list[str]) -> pd.DataFrame:
    """Summary table with totals for download."""
    summary = sentiment_summary(predictions)
    if summary.empty:
        return summary
    total = int(summary["count"].sum())
    totals = pd.DataFrame(
        [{"sentiment": "TOTAL", "count": total, "percentage": 100.0}]
    )
    return pd.concat([summary, totals], ignore_index=True)


def make_distribution_figure(summary: pd.DataFrame):
    """
    Horizontal bar chart with count + percentage labels.

    Horizontal layout avoids clipped value labels above tall vertical bars.
    """
    plot_df = summary[summary["sentiment"] != "TOTAL"].copy()
    if plot_df.empty:
        plot_df = summary.copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    plot_df = plot_df.reset_index(drop=True)
    colors = []
    for s in plot_df["sentiment"]:
        key = str(s).lower()
        if key == "positive":
            colors.append("#2e7d32")
        elif key == "negative":
            colors.append("#c62828")
        else:
            colors.append("#546e7a")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    y_pos = range(len(plot_df))
    counts = plot_df["count"].astype(float).tolist()
    ax.barh(list(y_pos), counts, color=colors, height=0.55, edgecolor="white")

    max_count = max(counts) if counts else 1
    ax.set_xlim(0, max_count * 1.35 if max_count else 1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([str(s).title() for s in plot_df["sentiment"]])
    ax.invert_yaxis()
    ax.set_xlabel("Number of reviews")
    ax.set_title("Predicted sentiment distribution")

    for i, row in plot_df.iterrows():
        count = int(row["count"])
        pct = row["percentage"] if "percentage" in plot_df.columns else None
        label = f"{count} ({pct:.1f}%)" if pct is not None else str(count)
        ax.text(
            count + max_count * 0.03,
            i,
            label,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#212121",
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def render_distribution_chart_png(summary: pd.DataFrame) -> bytes:
    """Return a PNG of the sentiment distribution chart."""
    fig = make_distribution_figure(summary)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 with BOM so Excel on Windows opens accents/Roman Urdu correctly."""
    return df.to_csv(index=False, encoding="utf-8-sig", lineterminator="\r\n").encode(
        "utf-8-sig"
    )


def build_results_zip(
    result: pd.DataFrame,
    text_column: str,
) -> tuple[bytes, dict[str, Any]]:
    """
    Package predictions CSV + summary CSV + distribution PNG.

    Returns (zip_bytes, meta) where meta holds the individual file bytes for
    separate download buttons.
    """
    formatted = format_predictions_csv(result, text_column)
    summary = format_summary_csv(result["prediction"])
    chart_png = render_distribution_chart_png(summary)

    predictions_csv = dataframe_to_csv_bytes(formatted)
    summary_csv = dataframe_to_csv_bytes(summary)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sentiment_predictions.csv", predictions_csv)
        zf.writestr("sentiment_summary.csv", summary_csv)
        zf.writestr("sentiment_distribution.png", chart_png)
        zf.writestr(
            "README.txt",
            "\r\n".join(
                [
                    "Bilingual Sentiment Analyzer — batch export",
                    "",
                    "sentiment_predictions.csv  Row-level predictions (Excel-friendly UTF-8 BOM)",
                    "sentiment_summary.csv      Counts and percentages (+ TOTAL row)",
                    "sentiment_distribution.png Bar chart of predicted sentiments",
                    "",
                    "predicted_sentiment: positive or negative",
                    "model_score_not_probability: classifier signal, NOT calibrated confidence",
                    "",
                ]
            ),
        )
    meta = {
        "predictions_csv": predictions_csv,
        "summary_csv": summary_csv,
        "chart_png": chart_png,
        "formatted_df": formatted,
        "summary_df": summary,
    }
    return zip_buf.getvalue(), meta
