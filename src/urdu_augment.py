"""
Large, project-authored Urdu-script review corpus for training.

The expanded Latin-heavy CSVs contain little or no Nastaliq text. Without a
dedicated Urdu-script set, TF-IDF collapses every Urdu row to the class prior
(constant ~0.53 negative). This module builds thousands of unique templated
reviews so word/char features learn real Urdu polarity tokens.
"""

from __future__ import annotations

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


def _rows_from_templates(
    templates: Iterable[str],
    *,
    items: Iterable[str],
    adjs: Iterable[str],
    intensifiers: Iterable[str],
    label: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tmpl in templates:
        needs_intens = "{intens}" in tmpl
        for item in items:
            for adj in adjs:
                if needs_intens:
                    for intens in intensifiers:
                        rows.append(
                            {
                                TEXT_COLUMN: tmpl.format(
                                    item=item, adj=adj, intens=intens
                                ),
                                LABEL_COLUMN: label,
                                LANGUAGE_COLUMN: "urdu",
                            }
                        )
                else:
                    rows.append(
                        {
                            TEXT_COLUMN: tmpl.format(
                                item=item, adj=adj, intens=""
                            ),
                            LABEL_COLUMN: label,
                            LANGUAGE_COLUMN: "urdu",
                        }
                    )
    return rows


def build_urdu_script_corpus() -> pd.DataFrame:
    """
    Return a large deduplicated Urdu-script sentiment DataFrame.

    Typical size is several thousand unique rows (pos/neg/neu).
    """
    rows: list[dict[str, str]] = []
    rows.extend(
        _rows_from_templates(
            _POS_TEMPLATES,
            items=_ITEMS,
            adjs=_POS_ADJ,
            intensifiers=_INTENS,
            label=LABEL_POSITIVE,
        )
    )
    rows.extend(
        _rows_from_templates(
            _NEG_TEMPLATES,
            items=_ITEMS,
            adjs=_NEG_ADJ,
            intensifiers=_INTENS,
            label=LABEL_NEGATIVE,
        )
    )
    rows.extend(
        _rows_from_templates(
            _NEU_TEMPLATES,
            items=_ITEMS,
            adjs=_NEU_ADJ,
            intensifiers=_INTENS,
            label=LABEL_NEUTRAL,
        )
    )
    for label, text in _EXTRAS:
        rows.append(
            {
                TEXT_COLUMN: text,
                LABEL_COLUMN: label,
                LANGUAGE_COLUMN: "urdu",
            }
        )

    df = pd.DataFrame(rows)
    df["_dedupe"] = df[TEXT_COLUMN].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["_dedupe"], keep="first")
    return df.drop(columns=["_dedupe"]).reset_index(drop=True)
