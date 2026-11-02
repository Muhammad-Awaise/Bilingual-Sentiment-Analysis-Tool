"""
Build data/processed/reviews_train_expanded.csv with language + provenance tags.

Recipe (documented, reproducible):

1. Always include the curated MIT dataset
   (data/raw/bilingual_product_reviews.csv) with author language tags.
2. Optionally add openly licensed / research CSVs from data/raw/external/
   (UCI Roman Urdu + Amazon/IMDB/Yelp, Kaggle Roman Urdu, TweetEval binary).
3. Integrate *genuine* Neutral from UCI Roman Urdu 3-class (never invent Neutral
   from binary datasets).
4. Always fold in a large project-authored Urdu-script corpus so Nastaliq
   reviews cannot collapse to the class prior at train time.
5. Optionally merge local Daraz reference exports — NOT for public redistribution
   and NOT recommended for the primary training set when external Daraz eval
   is required (use --include-local-references only for in-domain experiments).

Language / provenance columns are retained on disk for source-aware evaluation.
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
from src.leakage import (  # noqa: E402
    ANNOTATION_STATUS_COLUMN,
    PROVENANCE_COLUMN,
    SOURCE_COLUMN,
    TEMPLATE_FAMILY_COLUMN,
)
from src.urdu_augment import build_urdu_script_corpus  # noqa: E402

META_COLS = [
    TEXT_COLUMN,
    LABEL_COLUMN,
    LANGUAGE_COLUMN,
    SOURCE_COLUMN,
    PROVENANCE_COLUMN,
    TEMPLATE_FAMILY_COLUMN,
    ANNOTATION_STATUS_COLUMN,
]


def _frame(
    text,
    label,
    language: str | pd.Series | list,
    source: str,
    *,
    provenance: str | pd.Series | list,
    annotation_status: str | pd.Series | list,
    template_family: str | pd.Series | list = "none",
) -> pd.DataFrame:
    n = len(pd.Series(text))
    if isinstance(language, str):
        language = [language] * n
    if isinstance(provenance, str):
        provenance = [provenance] * n
    if isinstance(annotation_status, str):
        annotation_status = [annotation_status] * n
    if isinstance(template_family, str):
        template_family = [template_family] * n
    return pd.DataFrame(
        {
            TEXT_COLUMN: list(text) if not isinstance(text, pd.Series) else text,
            LABEL_COLUMN: label,
            LANGUAGE_COLUMN: language,
            SOURCE_COLUMN: source,
            PROVENANCE_COLUMN: provenance,
            TEMPLATE_FAMILY_COLUMN: template_family,
            ANNOTATION_STATUS_COLUMN: annotation_status,
        }
    )


def load_curated() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_RAW_CSV, encoding="utf-8")
    # Real Nastaliq rows in curated set → provenance real_urdu; else curated
    langs = df.get("language", pd.Series(["unknown"] * len(df))).astype(str).str.lower()
    provenance = [
        "real_urdu" if lang == "urdu" else "curated" for lang in langs
    ]
    out = _frame(
        df["text"],
        df["label"].map(normalize_label),
        langs,
        "curated_mit",
        provenance=provenance,
        annotation_status="author_labelled",
    )
    return out.dropna(subset=[LABEL_COLUMN])


def load_uci_roman_urdu_binary() -> pd.DataFrame | None:
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
        provenance="external_real",
        annotation_status="dataset_labelled",
    )
    return out.dropna(subset=[LABEL_COLUMN])


def load_uci_roman_urdu_neutral() -> pd.DataFrame | None:
    """
    Genuine Neutral (+ full 3-class) from UCI Roman Urdu sentiment.

    This is NOT manufactured from a binary file — labels already include Neutral.
    """
    path = EXTERNAL_DATA_DIR / "uci_roman_urdu_sentiment.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    text_col = "review" if "review" in df.columns else "text"
    label_col = "gold_label" if "gold_label" in df.columns else "label"
    out = _frame(
        df[text_col],
        df[label_col].map(normalize_label),
        "roman_urdu",
        "uci_roman_urdu_sentiment",
        provenance="external_real",
        annotation_status="dataset_labelled",
    )
    out = out.dropna(subset=[LABEL_COLUMN])
    # Prefer keeping all three classes; caller may filter. Document Neutral gain.
    return out


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
        provenance="external_real",
        annotation_status="dataset_labelled",
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
        provenance="external_real",
        annotation_status="dataset_labelled",
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
        provenance="external_real",
        annotation_status="dataset_labelled",
    )
    return out.dropna(subset=[LABEL_COLUMN])


def load_urdu_script_corpus() -> pd.DataFrame:
    """Project-authored Nastaliq reviews — required so Urdu cannot collapse."""
    return build_urdu_script_corpus()


def load_local_references() -> pd.DataFrame | None:
    path = PROCESSED_DATA_DIR / "reviews_from_references.csv"
    if not path.exists():
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
            provenance="daraz_reference",
            annotation_status="reference_labelled",
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
        provenance="daraz_reference",
        annotation_status="reference_labelled",
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
    include_uci_neutral: bool = True,
    out_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    parts: list[pd.DataFrame] = [load_curated()]
    sources_used = ["curated_mit"]

    if open_licensed:
        for loader, name in (
            (load_uci_roman_urdu_binary, "uci_roman_urdu_binary"),
            (load_uci_english_reviews, "uci_amazon_imdb_yelp"),
            (load_kaggle_roman_urdu, "kaggle_roman_urdu_binary"),
            (load_tweeteval_binary, "tweeteval_sentiment_binary"),
        ):
            part = loader()
            if part is not None and not part.empty:
                parts.append(part)
                sources_used.append(name)

        if include_uci_neutral:
            neu = load_uci_roman_urdu_neutral()
            if neu is not None and not neu.empty:
                # Prefer Neutral rows to improve Neutral representation without
                # dumping the entire 19k 3-class set on top of the binary subset
                # (which would near-duplicate positives/negatives already loaded).
                neu_only = neu[neu[LABEL_COLUMN] == "neutral"]
                if not neu_only.empty:
                    parts.append(neu_only)
                    sources_used.append("uci_roman_urdu_sentiment_neutral")

    if include_local_references:
        ref = load_local_references()
        if ref is not None:
            parts.append(ref)
            sources_used.append("local_daraz_reference")

    if include_urdu_script:
        parts.append(load_urdu_script_corpus())
        sources_used.append("synthetic_urdu_script")

    combined = pd.concat(parts, ignore_index=True)
    combined = _restore_languages(combined)

    # Ensure required meta columns exist
    for col, default in (
        (SOURCE_COLUMN, "unknown"),
        (PROVENANCE_COLUMN, "unknown"),
        (TEMPLATE_FAMILY_COLUMN, "none"),
        (ANNOTATION_STATUS_COLUMN, "unknown"),
    ):
        if col not in combined.columns:
            combined[col] = default

    validated = validate_reviews_dataframe(combined)
    # validate may drop meta cols if not careful — re-merge keys
    for col in (
        SOURCE_COLUMN,
        PROVENANCE_COLUMN,
        TEMPLATE_FAMILY_COLUMN,
        ANNOTATION_STATUS_COLUMN,
    ):
        if col not in validated.columns and col in combined.columns:
            key_map = (
                combined.drop_duplicates(
                    subset=[TEXT_COLUMN], keep="first"
                )
                .assign(
                    _key=lambda d: d[TEXT_COLUMN].astype(str).str.strip().str.lower()
                )
                .set_index("_key")[col]
                .to_dict()
            )
            validated[col] = (
                validated[TEXT_COLUMN]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(key_map)
                .fillna("unknown" if col != TEMPLATE_FAMILY_COLUMN else "none")
            )

    out = Path(out_path) if out_path else EXPANDED_TRAIN_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [c for c in META_COLS if c in validated.columns]
    validated[keep_cols].to_csv(out, index=False, encoding="utf-8")

    # Provenance summary for Urdu
    urdu_mask = validated[LANGUAGE_COLUMN].astype(str).str.lower() == "urdu"
    urdu_prov = {}
    if PROVENANCE_COLUMN in validated.columns and urdu_mask.any():
        urdu_prov = (
            validated.loc[urdu_mask, PROVENANCE_COLUMN]
            .astype(str)
            .value_counts()
            .to_dict()
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(out),
        "n_rows": int(len(validated)),
        "class_counts": validated[LABEL_COLUMN].value_counts().to_dict(),
        "language_counts": validated[LANGUAGE_COLUMN].value_counts().to_dict(),
        "source_counts": validated[SOURCE_COLUMN].value_counts().to_dict()
        if SOURCE_COLUMN in validated.columns
        else {},
        "provenance_counts": validated[PROVENANCE_COLUMN].value_counts().to_dict()
        if PROVENANCE_COLUMN in validated.columns
        else {},
        "urdu_provenance_counts": urdu_prov,
        "annotation_status_counts": (
            validated[ANNOTATION_STATUS_COLUMN].value_counts().to_dict()
            if ANNOTATION_STATUS_COLUMN in validated.columns
            else {}
        ),
        "sources_used": sources_used,
        "open_licensed": bool(open_licensed),
        "include_local_references": bool(include_local_references),
        "include_urdu_script": bool(include_urdu_script),
        "include_uci_neutral": bool(include_uci_neutral),
        "columns": keep_cols,
        "redistributable": bool(open_licensed and not include_local_references),
        "notes": (
            "Curated MIT rows and the synthetic Urdu-script corpus are included "
            "when enabled. UCI Neutral rows are genuine dataset labels — not "
            "manufactured from binary data. Local Daraz references must not be "
            "published without license clearance and should be held out for "
            "external evaluation when measuring Daraz generalization."
        ),
        "warnings": [],
    }
    if include_local_references:
        manifest["warnings"].append(
            "Training set includes Daraz references — not a clean external Daraz eval."
        )
    if urdu_prov.get("synthetic", 0) > 10 * max(
        urdu_prov.get("real_urdu", 0) + urdu_prov.get("curated", 0), 1
    ):
        manifest["warnings"].append(
            "Synthetic Urdu dominates Urdu-tagged rows. Keep synthetic families "
            "out of evaluation partitions; report real vs synthetic Urdu separately."
        )

    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return validated, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reviews_train_expanded.csv with provenance tags"
    )
    parser.add_argument(
        "--no-open-licensed",
        action="store_true",
        help="Skip UCI open datasets (included by default when files exist)",
    )
    parser.add_argument(
        "--include-local-references",
        action="store_true",
        help="Merge local Daraz reference exports (not redistributable; breaks clean external Daraz eval)",
    )
    parser.add_argument(
        "--no-urdu-script",
        action="store_true",
        help="Skip synthetic Urdu-script corpus (not recommended)",
    )
    parser.add_argument(
        "--no-uci-neutral",
        action="store_true",
        help="Skip genuine Neutral rows from UCI Roman Urdu 3-class",
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
        include_uci_neutral=not args.no_uci_neutral,
        out_path=Path(args.out),
    )
    print(json.dumps(manifest, indent=2))
    if not manifest["redistributable"]:
        print(
            "WARNING: Output includes local references — do not publish without "
            "license clearance."
        )
    for w in manifest.get("warnings") or []:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
