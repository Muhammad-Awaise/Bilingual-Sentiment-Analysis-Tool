"""
Language / script composition profiling for bilingual reviews.

Does not translate or rewrite text. Tags are multi-label composition, not a
single-language assumption from one cue word.
"""

from __future__ import annotations

import re
from typing import Any

from src.data_loader import _ENGLISH_FUNCTION, _ROMAN_URDU_CUES
from src.preprocessing import contains_urdu_script

_URDU_SCRIPT_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_script_composition(text: Any) -> dict[str, Any]:
    """
    Return composition flags and a coarse category.

    Categories:
      english_only | roman_urdu_only | urdu_script_only |
      english_roman_urdu | english_urdu_script | roman_urdu_urdu_script |
      three_way_mixed | unknown
    """
    if text is None or (isinstance(text, float) and str(text) == "nan"):
        return {
            "has_latin": False,
            "has_urdu_script": False,
            "has_english_cues": False,
            "has_roman_urdu_cues": False,
            "category": "unknown",
        }
    raw = str(text).strip()
    if not raw:
        return {
            "has_latin": False,
            "has_urdu_script": False,
            "has_english_cues": False,
            "has_roman_urdu_cues": False,
            "category": "unknown",
        }

    has_urdu = bool(_URDU_SCRIPT_RE.search(raw))
    has_latin = bool(_LATIN_RE.search(raw))
    lower = raw.lower()
    tokens = set(re.findall(r"[a-z']+", lower))
    ru_cues = _ROMAN_URDU_CUES - {"delivery"}
    ru_hits = len(tokens & ru_cues)
    en_hits = len(tokens & _ENGLISH_FUNCTION)
    has_en = en_hits >= 1
    has_ru = ru_hits >= 1

    # Composition category — do not collapse to one language from a single word
    if has_urdu and has_latin and (has_en or has_ru):
        if has_en and has_ru:
            category = "three_way_mixed"
        elif has_en:
            category = "english_urdu_script"
        elif has_ru:
            category = "roman_urdu_urdu_script"
        else:
            category = "english_urdu_script" if has_latin else "urdu_script_only"
    elif has_urdu and not has_latin:
        category = "urdu_script_only"
    elif has_urdu and has_latin and not has_en and not has_ru:
        # Latin present but no cue hits — still mixed script
        category = "english_urdu_script"
    elif has_en and has_ru:
        category = "english_roman_urdu"
    elif has_ru and not has_en:
        category = "roman_urdu_only"
    elif has_en and not has_ru:
        category = "english_only"
    elif has_latin:
        category = "english_only" if en_hits >= ru_hits else "roman_urdu_only"
    else:
        category = "unknown"

    return {
        "has_latin": has_latin,
        "has_urdu_script": has_urdu or contains_urdu_script(raw),
        "has_english_cues": has_en,
        "has_roman_urdu_cues": has_ru,
        "english_cue_hits": en_hits,
        "roman_urdu_cue_hits": ru_hits,
        "category": category,
    }


def composition_label(text: Any) -> str:
    return str(detect_script_composition(text)["category"])
