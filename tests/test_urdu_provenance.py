"""Synthetic Urdu corpus provenance / template family tags."""

from src.leakage import (
    PROVENANCE_COLUMN,
    SOURCE_COLUMN,
    TEMPLATE_FAMILY_COLUMN,
)
from src.urdu_augment import build_urdu_script_corpus, synthetic_template_summary


def test_synthetic_corpus_has_provenance_tags():
    df = build_urdu_script_corpus()
    assert PROVENANCE_COLUMN in df.columns
    assert SOURCE_COLUMN in df.columns
    assert TEMPLATE_FAMILY_COLUMN in df.columns
    assert (df[PROVENANCE_COLUMN] == "synthetic").all()
    assert (df[SOURCE_COLUMN] == "synthetic_urdu_script").all()
    assert df[TEMPLATE_FAMILY_COLUMN].nunique() >= 10


def test_template_summary():
    summary = synthetic_template_summary()
    assert summary["n_rows"] > 100
    assert summary["n_families"] >= 10
