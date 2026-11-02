"""
Train/test leakage detection beyond exact raw-text matches.

Checks:
- Exact strip+lower duplicates (existing behaviour)
- Normalized-text duplicates (clean_text without intensity/contrast markers)
- Near-duplicates via CPU-friendly token Jaccard (high threshold)
- Synthetic template-family overlap across partitions
- Source-aware partition validation

Near-duplicate detection is deliberately conservative so legitimate spelling
variants (achi/achhi, bohat/bht) are not treated as leakage by default.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

import pandas as pd

from src.config import RANDOM_SEED, TEXT_COLUMN
from src.preprocessing import clean_text

SOURCE_COLUMN = "source"
PROVENANCE_COLUMN = "provenance"
TEMPLATE_FAMILY_COLUMN = "template_family"
ANNOTATION_STATUS_COLUMN = "annotation_status"

# Marker tokens injected by clean_text — strip for normalized leakage keys
_MARKER_RE = re.compile(
    r"\b__(?:contrast|pos_intense|neg_intense|intensifier|neutral_cue|neg_scope)__\b"
)
_REPEAT_COLLAPSE_RE = re.compile(r"\b(\w+)(?:\s+\1){1,}\b")


def normalize_for_dedupe(text: Any) -> str:
    """
    Light normalized key for leakage / dedupe.

    Uses clean_text then strips engineered markers and collapsed repetitions so
    contrast/intensity emphasis does not create false uniqueness.
    Does NOT aggressively fold Roman Urdu spelling variants.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    cleaned = clean_text(str(text))
    if not cleaned:
        return ""
    cleaned = _MARKER_RE.sub(" ", cleaned)
    # Collapse consecutive repeated tokens from intensity emphasis
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _REPEAT_COLLAPSE_RE.sub(r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def exact_raw_key(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return str(text).strip().lower()


def _token_set(text: str) -> frozenset[str]:
    return frozenset(t for t in text.split() if t)


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def find_exact_normalized_duplicates(
    texts: list[str],
) -> dict[str, list[int]]:
    """Map normalized key → row indices for keys that appear more than once."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(texts):
        key = normalize_for_dedupe(t)
        if key:
            buckets[key].append(i)
    return {k: idxs for k, idxs in buckets.items() if len(idxs) > 1}


def find_near_duplicates(
    texts: list[str],
    *,
    threshold: float = 0.92,
    max_comparisons: int = 250_000,
) -> list[dict[str, Any]]:
    """
    Conservative near-duplicate pairs using token Jaccard on normalized text.

    Only compares candidates with similar length to keep CPU cost bounded.
    Spelling variants that share few exact tokens will NOT match.
    """
    norms = [normalize_for_dedupe(t) for t in texts]
    # Bucket by (length // 5) to limit pairwise work
    length_buckets: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(norms):
        if not n:
            continue
        length_buckets[len(n) // 5].append(i)

    pairs: list[dict[str, Any]] = []
    comparisons = 0
    for bucket_ids in length_buckets.values():
        # Also peek neighbouring length buckets
        ids = sorted(bucket_ids)
        for ai, i in enumerate(ids):
            for j in ids[ai + 1 :]:
                if comparisons >= max_comparisons:
                    return pairs
                comparisons += 1
                ni, nj = norms[i], norms[j]
                if ni == nj:
                    continue  # exact normalized handled elsewhere
                # Skip if token counts differ a lot (cheap filter)
                ti, tj = len(ni.split()), len(nj.split())
                if max(ti, tj) == 0:
                    continue
                if min(ti, tj) / max(ti, tj) < 0.75:
                    continue
                sim = jaccard_similarity(ni, nj)
                if sim >= threshold:
                    pairs.append(
                        {
                            "i": i,
                            "j": j,
                            "similarity": round(sim, 4),
                            "text_i": texts[i][:160],
                            "text_j": texts[j][:160],
                        }
                    )
    return pairs


def template_family_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    family_col: str = TEMPLATE_FAMILY_COLUMN,
) -> set[str]:
    """Return template_family ids present in both partitions."""
    if family_col not in train_df.columns or family_col not in test_df.columns:
        return set()
    train_f = set(
        train_df[family_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": None, "nan": None, "none": None})
        .dropna()
    )
    test_f = set(
        test_df[family_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": None, "nan": None, "none": None})
        .dropna()
    )
    # Ignore sentinel for non-templated rows
    train_f.discard("none")
    test_f.discard("none")
    train_f.discard("")
    test_f.discard("")
    return train_f & test_f


def partition_text_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    key_fn=exact_raw_key,
) -> set[str]:
    train_keys = {key_fn(t) for t in train_df[TEXT_COLUMN].tolist() if key_fn(t)}
    test_keys = {key_fn(t) for t in test_df[TEXT_COLUMN].tolist() if key_fn(t)}
    return train_keys & test_keys


def build_leakage_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    near_dup_threshold: float = 0.92,
    check_near_duplicates: bool = True,
) -> dict[str, Any]:
    """
    Produce a structured leakage report for train vs test (or any two partitions).

    Does not mutate data. Callers decide whether to raise.
    """
    raw_overlap = partition_text_overlap(train_df, test_df, key_fn=exact_raw_key)
    norm_overlap = partition_text_overlap(
        train_df, test_df, key_fn=normalize_for_dedupe
    )
    family_overlap = template_family_overlap(train_df, test_df)

    near_across: list[dict[str, Any]] = []
    if check_near_duplicates:
        # Cross-partition near-dups: compare test norms against a train index
        train_norms = {
            normalize_for_dedupe(t): i
            for i, t in enumerate(train_df[TEXT_COLUMN].tolist())
            if normalize_for_dedupe(t)
        }
        # Also index by frozenset of tokens for Jaccard candidates with same hash prefix
        train_by_prefix: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for key, idx in train_norms.items():
            toks = key.split()
            if not toks:
                continue
            prefix = toks[0][:4] if toks[0] else ""
            train_by_prefix[prefix].append((idx, key))

        for j, t in enumerate(test_df[TEXT_COLUMN].tolist()):
            nj = normalize_for_dedupe(t)
            if not nj or nj in train_norms:
                continue  # exact normalized already counted
            toks = nj.split()
            if not toks:
                continue
            prefix = toks[0][:4]
            for i, ni in train_by_prefix.get(prefix, []):
                ti, tj = len(ni.split()), len(toks)
                if max(ti, tj) == 0:
                    continue
                if min(ti, tj) / max(ti, tj) < 0.75:
                    continue
                sim = jaccard_similarity(ni, nj)
                if sim >= near_dup_threshold:
                    near_across.append(
                        {
                            "train_i": i,
                            "test_j": j,
                            "similarity": round(sim, 4),
                            "train_text": str(train_df.iloc[i][TEXT_COLUMN])[:160],
                            "test_text": str(t)[:160],
                        }
                    )
                    if len(near_across) >= 50:
                        break
            if len(near_across) >= 50:
                break

    source_train = (
        train_df[SOURCE_COLUMN].astype(str).value_counts().to_dict()
        if SOURCE_COLUMN in train_df.columns
        else {}
    )
    source_test = (
        test_df[SOURCE_COLUMN].astype(str).value_counts().to_dict()
        if SOURCE_COLUMN in test_df.columns
        else {}
    )

    report = {
        "random_seed_note": RANDOM_SEED,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "exact_raw_overlap_count": len(raw_overlap),
        "exact_raw_overlap_examples": list(raw_overlap)[:5],
        "normalized_overlap_count": len(norm_overlap),
        "normalized_overlap_examples": list(norm_overlap)[:5],
        "template_family_overlap_count": len(family_overlap),
        "template_family_overlap_examples": sorted(family_overlap)[:10],
        "near_duplicate_across_partitions": near_across[:25],
        "near_duplicate_across_count": len(near_across),
        "source_counts_train": {str(k): int(v) for k, v in source_train.items()},
        "source_counts_test": {str(k): int(v) for k, v in source_test.items()},
        "has_leakage": bool(
            raw_overlap or norm_overlap or family_overlap or near_across
        ),
        "severity_leakage": bool(raw_overlap or norm_overlap or family_overlap),
        "notes": (
            "Near-duplicates use high-threshold token Jaccard on lightly "
            "normalized text. Spelling variants that share few tokens are "
            "intentionally not treated as leakage."
        ),
    }
    return report


def assert_no_partition_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    allow_near_duplicates: bool = True,
) -> dict[str, Any]:
    """
    Raise ValueError on exact/normalized/template-family leakage.

    Near-duplicates are reported but do not raise by default (spelling variants).
    """
    from src.data_loader import DatasetError

    report = build_leakage_report(
        train_df,
        test_df,
        check_near_duplicates=True,
    )
    if report["exact_raw_overlap_count"]:
        raise DatasetError(
            f"Train/test leakage (exact text): {report['exact_raw_overlap_count']} "
            f"shared texts. Examples: {report['exact_raw_overlap_examples']}"
        )
    if report["normalized_overlap_count"]:
        raise DatasetError(
            f"Train/test leakage (normalized text): "
            f"{report['normalized_overlap_count']} shared keys. "
            f"Examples: {report['normalized_overlap_examples']}"
        )
    if report["template_family_overlap_count"]:
        raise DatasetError(
            f"Synthetic template-family leakage: "
            f"{report['template_family_overlap_count']} families in both "
            f"partitions. Examples: {report['template_family_overlap_examples']}"
        )
    if not allow_near_duplicates and report["near_duplicate_across_count"]:
        raise DatasetError(
            f"Near-duplicate leakage across partitions: "
            f"{report['near_duplicate_across_count']} pairs."
        )
    return report


def fingerprint_texts(texts: list[str]) -> str:
    """Stable short fingerprint of a text collection (order-insensitive)."""
    h = hashlib.sha256()
    for t in sorted(exact_raw_key(x) for x in texts):
        if t:
            h.update(t.encode("utf-8", errors="ignore"))
            h.update(b"\n")
    return h.hexdigest()[:16]


def dedupe_dataframe(
    df: pd.DataFrame,
    *,
    prefer_normalized: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Drop duplicate rows by raw and optionally normalized keys.

    Returns (deduped_df, stats).
    """
    working = df.copy()
    n_before = len(working)
    working["_raw_key"] = working[TEXT_COLUMN].map(exact_raw_key)
    working = working[working["_raw_key"] != ""]
    working = working.drop_duplicates(subset=["_raw_key"], keep="first")
    n_after_raw = len(working)
    if prefer_normalized:
        working["_norm_key"] = working[TEXT_COLUMN].map(normalize_for_dedupe)
        working = working[working["_norm_key"] != ""]
        working = working.drop_duplicates(subset=["_norm_key"], keep="first")
    n_after = len(working)
    working = working.drop(
        columns=[c for c in ("_raw_key", "_norm_key") if c in working.columns]
    ).reset_index(drop=True)
    stats = {
        "n_before": n_before,
        "dropped_exact_raw": n_before - n_after_raw,
        "dropped_normalized": n_after_raw - n_after,
        "n_after": n_after,
    }
    return working, stats
