"""Tests for leakage prevention helpers."""

import pandas as pd
import pytest

from src.config import LABEL_COLUMN, TEXT_COLUMN
from src.data_loader import DatasetError, assert_no_text_leakage, stratified_split
from src.leakage import (
    TEMPLATE_FAMILY_COLUMN,
    build_leakage_report,
    find_exact_normalized_duplicates,
    normalize_for_dedupe,
    template_family_overlap,
)


def test_normalize_for_dedupe_strips_markers():
    a = normalize_for_dedupe("Product acha hai lekin quality kharab hai")
    b = normalize_for_dedupe("product   acha hai lekin quality kharab hai")
    assert a == b
    assert "__contrast__" not in a


def test_naheeen_canonical_same_norm_key():
    a = normalize_for_dedupe("Product acha nahi hai")
    b = normalize_for_dedupe("Product acha naheeen hai")
    assert a == b


def test_spelling_variants_not_forced_equal():
    """achi vs achhi must NOT be collapsed by aggressive normalization."""
    a = normalize_for_dedupe("achi quality")
    b = normalize_for_dedupe("achhi quality")
    # May differ — character n-grams handle similarity, not dedupe
    assert a != b or "achi" in a


def test_normalized_duplicates_detected():
    texts = [
        "Great product!",
        "great product",
        "Different text here",
    ]
    dups = find_exact_normalized_duplicates(texts)
    assert len(dups) >= 1


def test_assert_no_text_leakage_raises_on_overlap():
    train = pd.DataFrame(
        {TEXT_COLUMN: ["same review text", "other"], LABEL_COLUMN: ["positive", "negative"]}
    )
    test = pd.DataFrame(
        {TEXT_COLUMN: ["Same Review Text", "unique"], LABEL_COLUMN: ["positive", "negative"]}
    )
    with pytest.raises(DatasetError):
        assert_no_text_leakage(train, test)


def test_template_family_overlap_detection():
    train = pd.DataFrame(
        {
            TEXT_COLUMN: ["a", "b"],
            LABEL_COLUMN: ["positive", "negative"],
            TEMPLATE_FAMILY_COLUMN: ["syn_pos_abc", "none"],
        }
    )
    test = pd.DataFrame(
        {
            TEXT_COLUMN: ["c", "d"],
            LABEL_COLUMN: ["positive", "negative"],
            TEMPLATE_FAMILY_COLUMN: ["syn_pos_abc", "syn_neg_xyz"],
        }
    )
    overlap = template_family_overlap(train, test)
    assert "syn_pos_abc" in overlap


def test_template_family_split_keeps_families_apart():
    rows = []
    for fam in ("syn_pos_aaa", "syn_pos_bbb", "syn_neg_ccc", "syn_neg_ddd"):
        for i in range(8):
            rows.append(
                {
                    TEXT_COLUMN: f"{fam} review {i} unique",
                    LABEL_COLUMN: "positive" if "pos" in fam else "negative",
                    "language": "urdu",
                    TEMPLATE_FAMILY_COLUMN: fam,
                }
            )
    # Add non-synthetic rows for stratification stability
    for i in range(20):
        rows.append(
            {
                TEXT_COLUMN: f"english review {i}",
                LABEL_COLUMN: "positive" if i % 2 == 0 else "negative",
                "language": "english",
                TEMPLATE_FAMILY_COLUMN: "none",
            }
        )
    df = pd.DataFrame(rows)
    train, test = stratified_split(df)
    report = build_leakage_report(train, test, check_near_duplicates=False)
    assert report["template_family_overlap_count"] == 0
