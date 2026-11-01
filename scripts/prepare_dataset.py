"""
Optional: prepare a binary dataset from local reference CSVs.

Does NOT claim redistribution rights. Writes to data/processed/ only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import LABEL_COLUMN, LANGUAGE_COLUMN, PROCESSED_DATA_DIR, ROOT_DIR, TEXT_COLUMN
from src.data_loader import normalize_label, validate_reviews_dataframe

REF2 = ROOT_DIR / "references" / "AIML-Bilingual-Sentiment-Analysis-of-Daraz-Reviews"


def _load_daraz_fyp() -> pd.DataFrame:
    path = REF2 / "Daraz FYP Dataset.csv"
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    out = pd.DataFrame(
        {
            TEXT_COLUMN: df["Review"],
            LABEL_COLUMN: df["Sentiment"].map(normalize_label),
            LANGUAGE_COLUMN: "unknown",
            "source": "daraz_fyp_local",
        }
    )
    return out.dropna(subset=[LABEL_COLUMN])


def _load_daraz_code_mixed() -> pd.DataFrame:
    path = REF2 / "daraz-code-mixed-product-reviews.csv"
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    out = pd.DataFrame(
        {
            TEXT_COLUMN: df["Reviews"],
            LABEL_COLUMN: df["Sentiments"].map(normalize_label),
            LANGUAGE_COLUMN: "code_mixed",
            "source": "daraz_code_mixed_local",
        }
    )
    return out.dropna(subset=[LABEL_COLUMN])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-references",
        action="store_true",
        help="Build binary CSV from local Daraz reference files (Neutral dropped).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROCESSED_DATA_DIR / "reviews_from_references.csv"),
    )
    args = parser.parse_args()
    if not args.from_references:
        raise SystemExit("Specify --from-references (see data/README.md).")

    if not REF2.exists():
        raise SystemExit(
            f"Reference folder missing: {REF2}. See references/README.md"
        )

    parts = [_load_daraz_fyp(), _load_daraz_code_mixed()]
    combined = pd.concat(parts, ignore_index=True)
    # Neutral already became None via normalize_label and was dropped
    validated = validate_reviews_dataframe(combined[[TEXT_COLUMN, LABEL_COLUMN, LANGUAGE_COLUMN]])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(validated)} binary-labelled rows to {out}")
    print(validated[LABEL_COLUMN].value_counts().to_dict())
    print(
        "NOTE: Do not publish this file unless dataset redistribution is permitted."
    )


if __name__ == "__main__":
    main()
