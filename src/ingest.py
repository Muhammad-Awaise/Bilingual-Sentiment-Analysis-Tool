"""
Flexible org-data ingest for prediction and training.

Organizations do not need our curated schema. This module:
- Reads CSV / TSV / Excel / JSON / JSONL with encoding fallbacks
- Detects or accepts explicit column maps for text / label / language
- Normalizes labels (aliases + optional 1–5 star ratings)
- Emits a short validation report before prediction or training

Internal training schema remains: text, label[, language].
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    LABEL_COLUMN,
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_POSITIVE,
    LANGUAGE_COLUMN,
    TEXT_COLUMN,
)

# Column name hints (case-insensitive). Order = preference.
TEXT_COLUMN_CANDIDATES: tuple[str, ...] = (
    "text",
    "review",
    "reviews",
    "review_text",
    "review_body",
    "comment",
    "comments",
    "body",
    "content",
    "message",
    "feedback",
    "description",
    "tweet",
)
LABEL_COLUMN_CANDIDATES: tuple[str, ...] = (
    "label",
    "gold_label",
    "sentiment",
    "sentiments",
    "true_label",
    "actual",
    "actual_sentiment",
    "class",
    "rating",
    "star_rating",
    "stars",
    "score",
)
LANGUAGE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "language",
    "lang",
    "locale",
)

_LABEL_ALIASES = {
    "pos": LABEL_POSITIVE,
    "positive": LABEL_POSITIVE,
    "+": LABEL_POSITIVE,
    "plus": LABEL_POSITIVE,
    "good": LABEL_POSITIVE,
    "happy": LABEL_POSITIVE,
    "satisfied": LABEL_POSITIVE,
    "neg": LABEL_NEGATIVE,
    "negative": LABEL_NEGATIVE,
    "-": LABEL_NEGATIVE,
    "minus": LABEL_NEGATIVE,
    "bad": LABEL_NEGATIVE,
    "unhappy": LABEL_NEGATIVE,
    "dissatisfied": LABEL_NEGATIVE,
    "neu": LABEL_NEUTRAL,
    "neutral": LABEL_NEUTRAL,
    "ok": LABEL_NEUTRAL,
    "okay": LABEL_NEUTRAL,
    "average": LABEL_NEUTRAL,
    # "mixed" intentionally unmapped — ambiguous for training
    "true": LABEL_POSITIVE,
    "false": LABEL_NEGATIVE,
    "yes": LABEL_POSITIVE,
    "no": LABEL_NEGATIVE,
}

# Default product-review star → sentiment map (overridable).
DEFAULT_STAR_MAP: dict[int, str] = {
    1: LABEL_NEGATIVE,
    2: LABEL_NEGATIVE,
    3: LABEL_NEUTRAL,
    4: LABEL_POSITIVE,
    5: LABEL_POSITIVE,
}

_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_NON_TEXT_HINTS = frozenset(
    {
        "id",
        "review_id",
        "row_number",
        "index",
        "language",
        "lang",
        "locale",
        "category",
        "date",
        "time",
        "timestamp",
        "source",
        "store",
        "sku",
        "product_id",
    }
)


class IngestError(ValueError):
    """Raised when a file cannot be read or required columns are missing."""


@dataclass
class ColumnMap:
    text: str
    label: str | None = None
    language: str | None = None


@dataclass
class IngestReport:
    n_rows: int
    n_columns: int
    text_column: str
    label_column: str | None
    language_column: str | None
    encoding: str | None
    format: str
    empty_text_rows: int
    label_counts: dict[str, int] = field(default_factory=dict)
    detected_languages: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_sentiment_label(
    value: Any,
    *,
    star_map: dict[int, str] | None = None,
) -> str | None:
    """
    Map org label spellings / ratings to positive | negative | neutral.

    Returns None when the value cannot be interpreted (caller may drop the row).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    stars = star_map or DEFAULT_STAR_MAP

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value).is_integer():
            iv = int(value)
            if iv in stars:
                return stars[iv]
            if iv in (0, 1) and iv not in stars:
                # Binary 0/1 when not using a 5-star scheme exclusively
                return LABEL_POSITIVE if iv == 1 else LABEL_NEGATIVE

    key = str(value).strip().lower()
    if not key:
        return None

    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]

    # "5 stars", "star: 4", "rating 2/5"
    m = re.search(r"\b([1-5])\s*(?:star|stars|/5)?\b", key)
    if m:
        return stars.get(int(m.group(1)))

    # Pure digit string
    if re.fullmatch(r"[1-5]", key):
        return stars.get(int(key))
    if key in {"0", "1"}:
        return LABEL_POSITIVE if key == "1" else LABEL_NEGATIVE

    return None


def _lower_col_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def suggest_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the original column name for the first matching candidate."""
    lower_map = _lower_col_map(df)
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def detect_column_map(df: pd.DataFrame) -> ColumnMap:
    """Best-effort column detection for org tables."""
    text = suggest_column(df, TEXT_COLUMN_CANDIDATES)
    if text is None and len(df.columns) == 1:
        text = str(df.columns[0])
    if text is None:
        # Prefer first object/string-like column that is not an id/meta hint
        lower_map = _lower_col_map(df)
        for low, original in lower_map.items():
            if low in _NON_TEXT_HINTS:
                continue
            series = df[original]
            if pd.api.types.is_string_dtype(series) or series.dtype == object:
                text = original
                break
    if text is None and len(df.columns):
        text = str(df.columns[0])

    label = suggest_column(df, LABEL_COLUMN_CANDIDATES)
    if label is not None and text is not None and label == text:
        label = None
    language = suggest_column(df, LANGUAGE_COLUMN_CANDIDATES)
    return ColumnMap(text=text or "", label=label, language=language)


def _suffix(name: str | None) -> str:
    if not name:
        return ""
    return Path(str(name)).suffix.lower()


def _read_csv_bytes(data: bytes, *, sep: str = ",") -> tuple[pd.DataFrame, str]:
    last_err: Exception | None = None
    for enc in _ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep, on_bad_lines="skip")
            return df, enc
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise IngestError(f"Could not parse delimited text: {last_err}") from last_err


def _read_excel_bytes(data: bytes) -> tuple[pd.DataFrame, str]:
    try:
        df = pd.read_excel(io.BytesIO(data), engine="openpyxl")
    except ImportError as exc:
        raise IngestError(
            "Excel support requires openpyxl. Install with: pip install openpyxl"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"Could not parse Excel file: {exc}") from exc
    return df, "excel"


def _read_json_bytes(data: bytes) -> tuple[pd.DataFrame, str]:
    last_err: Exception | None = None
    for enc in _ENCODING_CANDIDATES:
        try:
            text = data.decode(enc)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            text = None
    else:
        raise IngestError(f"Could not decode JSON bytes: {last_err}") from last_err

    assert text is not None
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            df = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            # Single record or {"data": [...]}
            if any(isinstance(v, list) for v in payload.values()):
                for v in payload.values():
                    if isinstance(v, list):
                        df = pd.DataFrame(v)
                        break
                else:
                    df = pd.DataFrame([payload])
            else:
                df = pd.DataFrame([payload])
        else:
            raise IngestError("JSON root must be an object or array of objects.")
        return df, enc
    except IngestError:
        raise
    except Exception:
        # JSON Lines
        try:
            df = pd.read_json(io.StringIO(text), lines=True)
            return df, enc
        except Exception as exc:  # noqa: BLE001
            raise IngestError(f"Could not parse JSON/JSONL: {exc}") from exc


def read_tabular_bytes(
    data: bytes,
    *,
    filename: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a tabular file from bytes.

    Supported: .csv, .tsv, .txt (csv), .xlsx, .xls, .json, .jsonl
    Returns (dataframe, meta) where meta has format + encoding.
    """
    if not data:
        raise IngestError("Uploaded file is empty.")

    suffix = _suffix(filename)
    if suffix in {".xlsx", ".xls"}:
        df, enc = _read_excel_bytes(data)
        fmt = "excel"
    elif suffix in {".tsv"}:
        df, enc = _read_csv_bytes(data, sep="\t")
        fmt = "tsv"
    elif suffix in {".json", ".jsonl"}:
        df, enc = _read_json_bytes(data)
        fmt = "json"
    else:
        # Default / .csv / .txt / unknown → CSV with encoding fallback
        df, enc = _read_csv_bytes(data, sep=",")
        fmt = "csv"

    if df is None or df.empty:
        raise IngestError("The file has no data rows.")
    if len(df.columns) == 0:
        raise IngestError("The file has no columns.")

    return df, {"format": fmt, "encoding": enc, "filename": filename}


def read_tabular_path(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a tabular file from disk."""
    path = Path(path)
    if not path.exists():
        raise IngestError(f"File not found: {path}")
    data = path.read_bytes()
    return read_tabular_bytes(data, filename=path.name)


def build_ingest_report(
    df: pd.DataFrame,
    column_map: ColumnMap,
    *,
    encoding: str | None = None,
    fmt: str = "csv",
    language_series: pd.Series | None = None,
) -> IngestReport:
    warnings: list[str] = []
    text_col = column_map.text
    if not text_col or text_col not in df.columns:
        raise IngestError(
            f"Text column '{text_col}' not found. Available: {list(df.columns)}"
        )

    empty_text = int(
        df[text_col].map(
            lambda x: x is None
            or (isinstance(x, float) and pd.isna(x))
            or str(x).strip() == ""
        ).sum()
    )

    if str(text_col).strip().lower() in _NON_TEXT_HINTS:
        warnings.append(
            f"Column '{text_col}' looks like metadata (id/language/date), not review text."
        )

    label_counts: dict[str, int] = {}
    if column_map.label and column_map.label in df.columns:
        mapped = df[column_map.label].map(normalize_sentiment_label)
        label_counts = {
            str(k): int(v) for k, v in mapped.dropna().value_counts().to_dict().items()
        }
        raw_present = df[column_map.label].notna() & (
            df[column_map.label].astype(str).str.strip() != ""
        )
        failed = int((raw_present & mapped.isna()).sum())
        if failed:
            warnings.append(
                f"{failed} label value(s) could not be mapped to "
                "positive/negative/neutral (check stars or spellings)."
            )
        if not label_counts:
            warnings.append("Label column present but no usable sentiment labels found.")

    lang_counts: dict[str, int] = {}
    if language_series is not None and len(language_series):
        lang_counts = {
            str(k): int(v) for k, v in language_series.astype(str).value_counts().to_dict().items()
        }

    return IngestReport(
        n_rows=int(len(df)),
        n_columns=int(len(df.columns)),
        text_column=text_col,
        label_column=column_map.label,
        language_column=column_map.language,
        encoding=encoding,
        format=fmt,
        empty_text_rows=empty_text,
        label_counts=label_counts,
        detected_languages=lang_counts,
        warnings=warnings,
    )


def prepare_for_prediction(
    df: pd.DataFrame,
    *,
    text_column: str | None = None,
    label_column: str | None = None,
    language_column: str | None = None,
    encoding: str | None = None,
    fmt: str = "csv",
) -> tuple[pd.DataFrame, ColumnMap, IngestReport]:
    """
    Keep the original table; resolve which column holds review text.

    Does not require labels. Does not rename org columns.
    """
    if df is None or df.empty:
        raise IngestError("The table is empty.")

    detected = detect_column_map(df)
    cmap = ColumnMap(
        text=text_column or detected.text,
        label=label_column if label_column is not None else detected.label,
        language=language_column if language_column is not None else detected.language,
    )
    if not cmap.text:
        raise IngestError(
            "Could not detect a text column. Pass text_column explicitly. "
            f"Available: {list(df.columns)}"
        )
    if cmap.text not in df.columns:
        raise IngestError(
            f"Text column '{cmap.text}' not found. Available: {list(df.columns)}"
        )

    report = build_ingest_report(df, cmap, encoding=encoding, fmt=fmt)
    return df, cmap, report


def prepare_for_training(
    df: pd.DataFrame,
    *,
    text_column: str | None = None,
    label_column: str | None = None,
    language_column: str | None = None,
    star_map: dict[int, str] | None = None,
    infer_language_fn=None,
) -> pd.DataFrame:
    """
    Normalize any org table into internal columns: text, label[, language].

    Rows with empty text or unmapped labels are dropped by the caller via
    validate_reviews_dataframe — this function only reshapes and maps labels.
    """
    if df is None or df.empty:
        raise IngestError("The table is empty.")

    detected = detect_column_map(df)
    text_col = text_column or detected.text
    label_col = label_column or detected.label
    lang_col = language_column or detected.language

    if not text_col or text_col not in df.columns:
        raise IngestError(
            "Training requires a text column "
            f"(tried aliases {TEXT_COLUMN_CANDIDATES}). Found: {list(df.columns)}"
        )
    if not label_col or label_col not in df.columns:
        raise IngestError(
            "Training requires a label/sentiment column "
            f"(tried aliases {LABEL_COLUMN_CANDIDATES}). Found: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            TEXT_COLUMN: df[text_col],
            LABEL_COLUMN: df[label_col].map(
                lambda v: normalize_sentiment_label(v, star_map=star_map)
            ),
        }
    )

    if lang_col and lang_col in df.columns:
        out[LANGUAGE_COLUMN] = (
            df[lang_col]
            .map(lambda v: "unknown" if pd.isna(v) else str(v).strip().lower())
        )
        blank = out[LANGUAGE_COLUMN].isin({"", "nan", "none", "unknown"})
        if blank.any() and infer_language_fn is not None:
            out.loc[blank, LANGUAGE_COLUMN] = out.loc[blank, TEXT_COLUMN].map(
                infer_language_fn
            )
    elif infer_language_fn is not None:
        out[LANGUAGE_COLUMN] = out[TEXT_COLUMN].map(infer_language_fn)
    else:
        out[LANGUAGE_COLUMN] = "unknown"

    return out


def load_org_table(
    path: str | Path,
    *,
    text_column: str | None = None,
    label_column: str | None = None,
    language_column: str | None = None,
    for_training: bool = False,
    star_map: dict[int, str] | None = None,
    infer_language_fn=None,
) -> tuple[pd.DataFrame, IngestReport]:
    """
    One-shot load from path.

    for_training=True → internal schema text/label/language.
    for_training=False → original columns preserved + report.
    """
    df, meta = read_tabular_path(path)
    if for_training:
        normalized = prepare_for_training(
            df,
            text_column=text_column,
            label_column=label_column,
            language_column=language_column,
            star_map=star_map,
            infer_language_fn=infer_language_fn,
        )
        cmap = ColumnMap(
            text=TEXT_COLUMN,
            label=LABEL_COLUMN,
            language=LANGUAGE_COLUMN if LANGUAGE_COLUMN in normalized.columns else None,
        )
        report = build_ingest_report(
            normalized,
            cmap,
            encoding=meta.get("encoding"),
            fmt=meta.get("format", "csv"),
            language_series=normalized.get(LANGUAGE_COLUMN),
        )
        return normalized, report

    kept, cmap, report = prepare_for_prediction(
        df,
        text_column=text_column,
        label_column=label_column,
        language_column=language_column,
        encoding=meta.get("encoding"),
        fmt=meta.get("format", "csv"),
    )
    return kept, report
