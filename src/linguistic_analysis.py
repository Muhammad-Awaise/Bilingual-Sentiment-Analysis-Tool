"""
Linguistic error-analysis helpers.

The fixtures document expected *linguistic* interpretations for interview /
error analysis. They are NOT forced targets the model must achieve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ERROR_ANALYSIS_PATH, FIXTURES_DIR, REPORTS_DIR
from src.predict import PredictionError, load_model, predict_one


LINGUISTIC_FIXTURE = FIXTURES_DIR / "linguistic_examples.csv"


def load_linguistic_examples(path: str | Path | None = None) -> pd.DataFrame:
    """Load manually documented linguistic probe examples."""
    p = Path(path) if path else LINGUISTIC_FIXTURE
    if not p.exists():
        raise FileNotFoundError(f"Linguistic fixture not found: {p}")
    df = pd.read_csv(p, encoding="utf-8")
    required = {"text", "category", "expected_sentiment", "notes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Linguistic fixture missing columns: {sorted(missing)}")
    return df


def run_error_analysis(
    artifact_path: str | Path | None = None,
    fixture_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Predict each linguistic example and write a markdown error report.

    Incorrect predictions are reported as model errors — we do not alter
    labels or force the classifier to pass.
    """
    examples = load_linguistic_examples(fixture_path)
    payload = load_model(artifact_path)

    rows: list[dict[str, Any]] = []
    for _, row in examples.iterrows():
        text = str(row["text"])
        expected = str(row["expected_sentiment"]).strip().lower()
        try:
            result = predict_one(text, payload=payload)
            predicted = result["prediction"]
            cleaned = result["cleaned_text"]
            error = None
        except PredictionError as exc:
            predicted = "error"
            cleaned = ""
            error = str(exc)

        match = predicted == expected
        rows.append(
            {
                "text": text,
                "category": row["category"],
                "expected_sentiment": expected,
                "predicted": predicted,
                "match": match,
                "cleaned_text": cleaned,
                "notes": row["notes"],
                "error": error,
            }
        )

    result_df = pd.DataFrame(rows)
    out = Path(output_path) if output_path else ERROR_ANALYSIS_PATH
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(_to_markdown(result_df), encoding="utf-8")
    return result_df


def _to_markdown(df: pd.DataFrame) -> str:
    n = len(df)
    correct = int(df["match"].sum()) if n else 0
    lines = [
        "# Linguistic Error Analysis",
        "",
        "These examples probe negation, contrast, code-switching, intensifiers,",
        "and informal Roman Urdu spelling. They test application behavior and",
        "provide a framework for qualitative error analysis.",
        "",
        "**They are not proof that the trained model classifies every sentence correctly.**",
        "",
        f"- Examples checked: {n}",
        f"- Matches expected linguistic label: {correct}/{n}",
        "",
        "| Category | Text | Expected | Predicted | Match | Notes |",
        "|----------|------|----------|-----------|-------|-------|",
    ]
    for _, r in df.iterrows():
        text = str(r["text"]).replace("|", "\\|")
        notes = str(r["notes"]).replace("|", "\\|")
        lines.append(
            f"| {r['category']} | {text} | {r['expected_sentiment']} | "
            f"{r['predicted']} | {r['match']} | {notes} |"
        )
    lines.extend(
        [
            "",
            "## How to read mismatches",
            "",
            "- Small training data and spelling variation often cause failures on negation scope.",
            "- Contrastive 'but' / 'lekin' clauses may leave residual positive lexical cues.",
            "- Annotation disagreement is possible even for humans on mixed sentiment.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    df = run_error_analysis()
    print(df[["category", "expected_sentiment", "predicted", "match"]].to_string(index=False))
