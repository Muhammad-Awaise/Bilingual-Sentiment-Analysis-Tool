"""
Large, project-authored Urdu-script review corpus for training.

The expanded Latin-heavy CSVs contain little or no Nastaliq text. Without a
dedicated Urdu-script set, TF-IDF collapses every Urdu row to the class prior
(constant ~0.53 negative). This module builds thousands of unique templated
reviews so word/char features learn real Urdu polarity tokens.

Every synthetic row carries:
- provenance = synthetic
- source = synthetic_urdu_script
- template_family = stable id for template-family-aware splits
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd

from src.config import (
    LABEL_COLUMN,
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_POSITIVE,
    LANGUAGE_COLUMN,
    TEXT_COLUMN,
)
from src.leakage import (
    ANNOTATION_STATUS_COLUMN,
    PROVENANCE_COLUMN,
    SOURCE_COLUMN,
    TEMPLATE_FAMILY_COLUMN,
)

_ITEMS = (
    "پروڈکٹ",
    "مال",
    "آئٹم",
    "سامان",
    "آرڈر",
    "پیکج",
    "فون",
    "کپڑے",
    "جوتے",
    "ہیڈفون",
    "واچ",
    "بیگ",
)

_POS_ADJ = (
    "اچھا",
    "اچھی",
    "اچھے",
    "زبردست",
    "بہترین",
    "عمدہ",
    "شاندار",
    "مضبوط",
    "پیارا",
    "پیاری",
    "کمال",
    "صاف",
)

_NEG_ADJ = (
    "بیکار",
    "خراب",
    "برا",
    "بری",
    "برے",
    "جعلی",
    "کمزور",
    "سستا",
    "غلط",
    "فضول",
    "ناقص",
    "مایوس کن",
)

_NEU_ADJ = ("ٹھیک", "عام", "اوسط", "نارمل", "قابل قبول")

_INTENS = ("بہت", "بالکل", "انتہائی", "کافی")

_POS_TEMPLATES = (
    "یہ {item} {intens} {adj} ہے کوالٹی زبردست ہے",
    "{item} {adj} ہے میں خوش ہوں سفارش کرتا ہوں",
    "کوالٹی {adj} ہے پیکنگ بھی مضبوط تھی",
    "سیلر نے جلدی ڈیلیور کیا {item} بالکل درست آیا",
    "{item} کی کوالٹی {intens} {adj} ہے قیمت مناسب ہے",
    "میں اس {item} سے {intens} خوش ہوں",
    "آواز صاف ہے اور یہ {item} {adj} ہے",
    "مواد نرم ہے رنگ بھی {adj} ہے",
    "بیٹری کا وقت {adj} ہے میں سفارش کرتا ہوں",
    "جلدی ڈیلیور ہوا {item} {adj} ہے",
    "سائز بالکل درست آیا اور {item} {adj} ہے",
    "پیکنگ مضبوط تھی {item} {intens} {adj} نکلا",
)

_NEG_TEMPLATES = (
    "یہ {item} {intens} {adj} ہے ایک دن میں ٹوٹ گیا",
    "کوالٹی {adj} ہے میں خوش نہیں ہوں",
    "{item} {adj} ہے پیسے ضائع ہوئے مت خریدنا",
    "ڈیلیوری دیر سے ہوئی اور باکس {adj} تھا",
    "جعلی {item} ہے مت خریدنا میں خوش نہیں",
    "سائز غلط آیا اور یہ {item} {adj} ہے",
    "مواد سستا ہے سلائی {adj} ہے",
    "بیٹری {intens} جلدی ختم ہو جاتی ہے {item} {adj} ہے",
    "سیلر نے جواب نہیں دیا {item} بالکل {adj}",
    "میں اس خریداری سے بالکل خوش نہیں {item} {adj} ہے",
    "کوالٹی {adj} ہے اور پروڈکٹ ٹوٹ گیا",
    "واپس کرنا پڑا یہ {item} {intens} {adj} تھا",
)

_NEU_TEMPLATES = (
    "{item} {adj} ہے نہ بہت اچھا نہ برا",
    "عام سی چیز ہے کچھ خاص نہیں {item} {adj} ہے",
    "اوسط معیار ہے نہ تعریف نہ شکایت {item}",
    "نہ خوش نہ ناخوش بس چلے گا {item} {adj}",
    "{item} ٹھیک ہے عام سا تجربہ",
    "ڈیلیوری وقت پر ہوئی {item} {adj} سا ہے",
)

_EXTRAS: tuple[tuple[str, str], ...] = (
    (LABEL_POSITIVE, "میں اس خریداری سے بہت خوش ہوں"),
    (LABEL_POSITIVE, "آواز صاف ہے اور قیمت مناسب ہے"),
    (LABEL_POSITIVE, "مواد نرم ہے اور رنگ بھی پیارا ہے"),
    (LABEL_POSITIVE, "بیٹری کا وقت بہترین ہے میں سفارش کرتا ہوں"),
    (LABEL_POSITIVE, "سیلر بہت اچھا تھا جلدی ڈیلیور کیا"),
    (LABEL_POSITIVE, "صوتی معیار زبردست ہے قیمت مناسب"),
    (LABEL_POSITIVE, "پراڈکٹ اچھا ہے بیٹری بھی بہترین"),
    (LABEL_POSITIVE, "کوالٹی زبردست ہے بالکل درست آیا"),
    (LABEL_NEGATIVE, "جعلی مال ہے مت خریدنا میں خوش نہیں"),
    (LABEL_NEGATIVE, "بیٹری بہت جلدی ختم ہو جاتی ہے"),
    (LABEL_NEGATIVE, "سائز غلط آیا اور سیلر نے جواب نہیں دیا"),
    (LABEL_NEGATIVE, "مواد سستا ہے سلائی کمزور ہے"),
    (LABEL_NEGATIVE, "پیسے ضائع ہوئے کوالٹی بہت بری تھی"),
    (LABEL_NEGATIVE, "خراب کوالٹی جعلی مال مت خریدنا"),
    (LABEL_NEGATIVE, "ڈیلیوری بہت دیر سے ہوئی باکس بھی خراب"),
    (LABEL_NEGATIVE, "بہت برا تجربہ رہا بالکل بیکار"),
    (LABEL_NEUTRAL, "نہ خوش نہ ناخوش بس چلے گا"),
    (LABEL_NEUTRAL, "اوسط معیار ہے نہ تعریف نہ شکایت"),
    (LABEL_NEUTRAL, "پروڈکٹ ٹھیک ہے عام سا تجربہ"),
    (LABEL_NEUTRAL, "نہ اچھا نہ برا بس اوسط معیار"),
    (LABEL_NEUTRAL, "کچھ خاص نہیں عام چیز ہے"),
    (LABEL_NEUTRAL, "کام چل رہا ہے جذباتی رائے نہیں"),
)


def _family_id(kind: str, template: str) -> str:
    digest = hashlib.sha1(template.encode("utf-8")).hexdigest()[:10]
    return f"syn_{kind}_{digest}"


def _row(
    text: str,
    label: str,
    *,
    template_family: str,
) -> dict[str, str]:
    return {
        TEXT_COLUMN: text,
        LABEL_COLUMN: label,
        LANGUAGE_COLUMN: "urdu",
        SOURCE_COLUMN: "synthetic_urdu_script",
        PROVENANCE_COLUMN: "synthetic",
        TEMPLATE_FAMILY_COLUMN: template_family,
        ANNOTATION_STATUS_COLUMN: "synthetic_template",
    }


def _rows_from_templates(
    templates: Iterable[str],
    *,
    items: Iterable[str],
    adjs: Iterable[str],
    intensifiers: Iterable[str],
    label: str,
    kind: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tmpl in templates:
        family = _family_id(kind, tmpl)
        needs_intens = "{intens}" in tmpl
        for item in items:
            for adj in adjs:
                if needs_intens:
                    for intens in intensifiers:
                        rows.append(
                            _row(
                                tmpl.format(item=item, adj=adj, intens=intens),
                                label,
                                template_family=family,
                            )
                        )
                else:
                    rows.append(
                        _row(
                            tmpl.format(item=item, adj=adj, intens=""),
                            label,
                            template_family=family,
                        )
                    )
    return rows


def build_urdu_script_corpus() -> pd.DataFrame:
    """
    Return a large deduplicated Urdu-script sentiment DataFrame.

    Typical size is several thousand unique rows (pos/neg/neu).
    Synthetic rows are tagged for template-family-aware splits so the same
    template skeleton cannot dominate both train and evaluation.
    """
    rows: list[dict[str, str]] = []
    rows.extend(
        _rows_from_templates(
            _POS_TEMPLATES,
            items=_ITEMS,
            adjs=_POS_ADJ,
            intensifiers=_INTENS,
            label=LABEL_POSITIVE,
            kind="pos",
        )
    )
    rows.extend(
        _rows_from_templates(
            _NEG_TEMPLATES,
            items=_ITEMS,
            adjs=_NEG_ADJ,
            intensifiers=_INTENS,
            label=LABEL_NEGATIVE,
            kind="neg",
        )
    )
    rows.extend(
        _rows_from_templates(
            _NEU_TEMPLATES,
            items=_ITEMS,
            adjs=_NEU_ADJ,
            intensifiers=_INTENS,
            label=LABEL_NEUTRAL,
            kind="neu",
        )
    )
    for label, text in _EXTRAS:
        family = _family_id("extra", text)
        rows.append(_row(text, label, template_family=family))

    df = pd.DataFrame(rows)
    df["_dedupe"] = df[TEXT_COLUMN].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["_dedupe"], keep="first")
    return df.drop(columns=["_dedupe"]).reset_index(drop=True)


def synthetic_template_summary(df: pd.DataFrame | None = None) -> dict:
    """Counts by template family for research reporting."""
    if df is None:
        df = build_urdu_script_corpus()
    return {
        "n_rows": int(len(df)),
        "n_families": int(df[TEMPLATE_FAMILY_COLUMN].nunique())
        if TEMPLATE_FAMILY_COLUMN in df.columns
        else 0,
        "class_counts": df[LABEL_COLUMN].value_counts().to_dict(),
        "family_counts_head": (
            df[TEMPLATE_FAMILY_COLUMN].value_counts().head(15).to_dict()
            if TEMPLATE_FAMILY_COLUMN in df.columns
            else {}
        ),
    }
