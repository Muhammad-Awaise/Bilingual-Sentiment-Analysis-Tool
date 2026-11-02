"""
Build data/processed/reviews_train_expanded.csv with language tags.

Recipe (documented, reproducible):

1. Always include the curated MIT dataset
   (data/raw/bilingual_product_reviews.csv) with author language tags.
2. Optionally add openly licensed / research CSVs from data/raw/external/
   (UCI Roman Urdu + Amazon/IMDB/Yelp, Kaggle Roman Urdu, TweetEval binary).
3. Always fold in a large project-authored Urdu-script corpus so Nastaliq
   reviews cannot collapse to the class prior at train time.
4. Optionally merge local Daraz reference exports
   (data/processed/reviews_from_references.csv) — NOT for public redistribution.

Language tags:
- Kept from source when present
- Otherwise restored via src.data_loader.infer_language

Examples:

    # Large open + Urdu-script set (curated + UCI/Kaggle/TweetEval if present)
    python scripts/build_expanded_dataset.py --open-licensed

    # Max workspace set (also merge local Daraz references)
    python scripts/build_expanded_dataset.py --open-licensed --include-local-references

Do not publish outputs that include --include-local-references unless licenses
are cleared.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_RAW_CSV,
    EXPANDED_TRAIN_CSV,
    EXTERNAL_DATA_DIR,
    LABEL_COLUMN,
    LANGUAGE_COLUMN,
    PROCESSED_DATA_DIR,
    TEXT_COLUMN,
)
from src.data_loader import (  # noqa: E402
    infer_language,
    normalize_label,
    validate_reviews_dataframe,
)
from src.urdu_augment import build_urdu_script_corpus  # noqa: E402


def _frame(text, label, language: str, source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            TEXT_COLUMN: text,
            LABEL_COLUMN: label,
            LANGUAGE_COLUMN: language,
            "source": source,
        }
    )


def load_curated() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_RAW_CSV, encoding="utf-8")
    out = _frame(
        df["text"],
        df["label"].map(normalize_label),
        df.get("language", "unknown"),
        "curated_mit",
    )
    return out.dropna(subset=[LABEL_COLUMN])


def load_uci_roman_urdu() -> pd.DataFrame | None:
    path = EXTERNAL_DATA_DIR / "uci_roman_urdu_binary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    text_col = "review" if "review" in df.columns else "text"
    label_col = "gold_label" if "gold_label" in df.columns else "label"
    out = _frame(
        df[text_col],
        df[label_col].map(normalize_label),
        "roman_urdu",
        "uci_roman_urdu_binary",
    )
    return out.dropna(subset=[LABEL_COLUMN])


def load_uci_english_reviews() -> pd.DataFrame | None:
    path = EXTERNAL_DATA_DIR / "uci_amazon_imdb_yelp_combined.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    text_col = "review" if "review" in df.columns else "text"
    label_col = "gold_label" if "gold_label" in df.columns else "label"
    out = _frame(
        df[text_col],
        df[label_col].map(normalize_label),
        "english",
        "uci_amazon_imdb_yelp",
    )
    return out.dropna(subset=[LABEL_COLUMN])


def load_kaggle_roman_urdu() -> pd.DataFrame | None:
    path = EXTERNAL_DATA_DIR / "kaggle_roman_urdu_binary_11000.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    text_col = "review" if "review" in df.columns else "text"
    label_col = "gold_label" if "gold_label" in df.columns else "label"
    out = _frame(
        df[text_col],
        df[label_col].map(normalize_label),
        "roman_urdu",
        "kaggle_roman_urdu_binary",
    )
    return out.dropna(subset=[LABEL_COLUMN])


def load_tweeteval_binary() -> pd.DataFrame | None:
    path = EXTERNAL_DATA_DIR / "tweeteval_sentiment_binary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    text_col = "review" if "review" in df.columns else "text"
    label_col = "gold_label" if "gold_label" in df.columns else "label"
    out = _frame(
        df[text_col],
        df[label_col].map(normalize_label),
        "english",
        "tweeteval_sentiment_binary",
    )
    return out.dropna(subset=[LABEL_COLUMN])


def load_urdu_script_corpus() -> pd.DataFrame:
    """Project-authored Nastaliq reviews — required so Urdu cannot collapse."""
    df = build_urdu_script_corpus()
    return _frame(
        df[TEXT_COLUMN],
        df[LABEL_COLUMN],
        "urdu",
        "synthetic_urdu_script",
    )


def load_local_references() -> pd.DataFrame | None:
    path = PROCESSED_DATA_DIR / "reviews_from_references.csv"
    if not path.exists():
        # Fall back to running prepare script output name variants
        alt = PROCESSED_DATA_DIR / "daraz_binary_for_testing.csv"
        if not alt.exists():
            return None
        df = pd.read_csv(alt, encoding="utf-8", on_bad_lines="skip")
        text_col = "review" if "review" in df.columns else "text"
        label_col = "gold_label" if "gold_label" in df.columns else "label"
        out = _frame(
            df[text_col],
            df[label_col].map(normalize_label),
            "unknown",
            "local_daraz_reference",
        )
        return out.dropna(subset=[LABEL_COLUMN])

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    lang = (
        df[LANGUAGE_COLUMN]
        if LANGUAGE_COLUMN in df.columns
        else pd.Series(["unknown"] * len(df))
    )
    out = _frame(
        df[TEXT_COLUMN],
        df[LABEL_COLUMN].map(normalize_label),
        lang,
        "local_daraz_reference",
    )
    return out.dropna(subset=[LABEL_COLUMN])


def _restore_languages(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working[LANGUAGE_COLUMN] = (
        working[LANGUAGE_COLUMN]
        .map(lambda v: "unknown" if pd.isna(v) else str(v).strip().lower())
    )
    blank = working[LANGUAGE_COLUMN].isin({"", "nan", "none", "unknown"})
    if blank.any():
        working.loc[blank, LANGUAGE_COLUMN] = working.loc[
            blank, TEXT_COLUMN
        ].map(infer_language)
    return working


def build_expanded(
    *,
    open_licensed: bool = True,
    include_local_references: bool = False,
    include_urdu_script: bool = True,
    out_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    parts: list[pd.DataFrame] = [load_curated()]
    sources_used = ["curated_mit"]

    if open_licensed:
        for loader, name in (
            (load_uci_roman_urdu, "uci_roman_urdu_binary"),
            (load_uci_english_reviews, "uci_amazon_imdb_yelp"),
            (load_kaggle_roman_urdu, "kaggle_roman_urdu_binary"),
            (load_tweeteval_binary, "tweeteval_sentiment_binary"),
        ):
            part = loader()
            if part is not None and not part.empty:
                parts.append(part)
                sources_used.append(name)

    if include_local_references:
        ref = load_local_references()
        if ref is not None:
            parts.append(ref)
            sources_used.append("local_daraz_reference")

    # Always bake Urdu-script rows into the on-disk expanded set so retrains
    # cannot silently drop Nastaliq coverage.
    if include_urdu_script:
        parts.append(load_urdu_script_corpus())
        sources_used.append("synthetic_urdu_script")

    combined = pd.concat(parts, ignore_index=True)
    combined = _restore_languages(combined)

    # validate_reviews_dataframe dedupes on text and keeps language
    validated = validate_reviews_dataframe(
        combined[[TEXT_COLUMN, LABEL_COLUMN, LANGUAGE_COLUMN]]
    )
    # Re-attach source via first occurrence text match
    source_map = (
        combined.drop_duplicates(subset=[TEXT_COLUMN], keep="first")
        .assign(_key=lambda d: d[TEXT_COLUMN].astype(str).str.strip().str.lower())
        .set_index("_key")["source"]
        .to_dict()
    )
    validated = validated.copy()
    validated["source"] = (
        validated[TEXT_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(source_map)
        .fillna("unknown")
    )

    out = Path(out_path) if out_path else EXPANDED_TRAIN_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    # Ship text/label/language for training; keep source in a sidecar manifest
    validated[[TEXT_COLUMN, LABEL_COLUMN, LANGUAGE_COLUMN]].to_csv(
        out, index=False, encoding="utf-8"
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(out),
        "n_rows": int(len(validated)),
        "class_counts": validated[LABEL_COLUMN].value_counts().to_dict(),
        "language_counts": validated[LANGUAGE_COLUMN].value_counts().to_dict(),
        "source_counts": validated["source"].value_counts().to_dict(),
        "sources_used": sources_used,
        "open_licensed": bool(open_licensed),
        "include_local_references": bool(include_local_references),
        "include_urdu_script": bool(include_urdu_script),
        "redistributable": bool(
            open_licensed
            and not include_local_references
            # synthetic Urdu is project-authored; still ok to redistribute
        ),
        "notes": (
            "Curated MIT rows and the synthetic Urdu-script corpus are always "
            "included when enabled. UCI / Kaggle / TweetEval files are for "
            "research use — verify terms before commercial redistribution. "
            "Local Daraz references must not be published without license clearance."
        ),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return validated, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reviews_train_expanded.csv with language tags"
    )
    parser.add_argument(
        "--no-open-licensed",
        action="store_true",
        help="Skip UCI open datasets (included by default when files exist)",
    )
    parser.add_argument(
        "--include-local-references",
        action="store_true",
        help="Merge local Daraz reference exports (not redistributable)",
    )
    parser.add_argument(
        "--no-urdu-script",
        action="store_true",
        help="Skip synthetic Urdu-script corpus (not recommended)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(EXPANDED_TRAIN_CSV),
    )
    args = parser.parse_args()

    open_licensed = not args.no_open_licensed
    _df, manifest = build_expanded(
        open_licensed=open_licensed,
        include_local_references=args.include_local_references,
        include_urdu_script=not args.no_urdu_script,
        out_path=Path(args.out),
    )
    print(json.dumps(manifest, indent=2))
    if not manifest["redistributable"]:
        print(
            "WARNING: Output includes local references — do not publish without "
            "license clearance."
        )


if __name__ == "__main__":
    main()
