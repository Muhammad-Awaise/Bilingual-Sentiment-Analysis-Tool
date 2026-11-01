"""
Shared text preprocessing for training and inference.

Design goals (see docs/LINGUISTIC_DECISIONS.md):
- Handle null / empty / non-string input safely
- Normalize noise (URLs, mentions, excess punctuation) conservatively
- PRESERVE negation and contrast markers (not, no, nahi, nahin, نہیں, but, lekin, لیکن, ...)
- Apply a *closed-set* Roman Urdu negation spelling map (nahin → nahi, …)
- Keep Urdu (Arabic-script) letters so Nastaliq reviews are not wiped empty
- Lightly emphasize contrast verdict clauses for bag-of-ngrams models
- Do NOT strip a large stopword list that would erase polarity cues
"""

from __future__ import annotations

import re
from typing import Any

# Patterns compiled once
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
# Keep Latin letters/digits/apostrophe AND Arabic / Urdu *letters* + diacritics.
# Intentionally excludes Arabic punctuation (، ؛ ؟ ۔) so TF-IDF tokens stay clean.
_NON_TEXT_RE = re.compile(
    r"[^a-z0-9'\s"
    r"\u0621-\u063A\u0641-\u064A"  # Arabic letters
    r"\u064B-\u065F"  # Arabic diacritics / tatweel-adjacent marks
    r"\u066E-\u06D3"  # Extended Arabic / Urdu letters (incl. ے ہ ھ)
    r"\u06FA-\u06FF"  # Arabic more extended
    r"\u0750-\u077F"  # Arabic Supplement
    r"\u08A0-\u08FF"  # Arabic Extended-A
    r"]+",
    re.UNICODE,
)
# Extra pass for common Arabic/Urdu punctuation that may still appear
_ARABIC_PUNCT_RE = re.compile(r"[\u060C\u061B\u061F\u06D4\u066A\u066B\u066C\u066D]+")

# Closed-set informal Roman Urdu negation spellings → canonical "nahi"
# Intentionally narrow: does not invent broad phonetic normalization.
_NEGATION_SPELLING_MAP = {
    "nahin": "nahi",
    "naheen": "nahi",
    "nahe": "nahi",
    "nae": "nahi",
    "nai": "nahi",
    "naie": "nahi",
    "nhi": "nahi",
    "nhin": "nahi",
}

# Latin + Urdu-script contrast cues (word boundaries via Unicode-aware \b)
_CONTRAST_RE = re.compile(
    r"(?<!\w)(but|lekin|magar|however|though|لیکن|مگر|پر)(?!\w)",
    re.UNICODE,
)


def _canonicalize_negation_spellings(text: str) -> str:
    """Map known informal negation spellings to a single token."""
    tokens = text.split()
    return " ".join(_NEGATION_SPELLING_MAP.get(tok, tok) for tok in tokens)


def _emphasize_contrast_clause(text: str) -> str:
    """
    Re-weight contrastive reviews for bag-of-ngrams:

    Keep the pre-contrast clause once, mark the contrast cue, and repeat the
    post-contrast (verdict) clause so TF-IDF prefers the reviewer's final stance
    (e.g. *achi hai but delivery late* → late/delivery dominate).
    """
    matches = list(_CONTRAST_RE.finditer(text))
    if not matches:
        return text
    last = matches[-1]
    before = text[: last.start()].strip()
    after = text[last.end() :].strip()
    marker = last.group(1)
    if after:
        # Verdict clause repeated; setup clause kept once
        return f"{before} __contrast__ {marker} {after} {after} {after}".strip()
    if before:
        return f"{before} __contrast__ {marker}"
    return f"__contrast__ {marker}"


def clean_text(text: Any) -> str:
    """
    Clean a single review for vectorization.

    Returns an empty string for missing / non-string / whitespace-only input.
    Latin text is lowercased; Urdu-script letters are preserved.
    """
    if text is None:
        return ""
    if isinstance(text, float):  # pandas NaN
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    text = text.strip()
    if not text:
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    # Keep hashtag word content (often product tags) without the '#'
    text = _HASHTAG_RE.sub(r"\1", text)
    text = _ARABIC_PUNCT_RE.sub(" ", text)
    text = _NON_TEXT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    text = _canonicalize_negation_spellings(text)
    text = _emphasize_contrast_clause(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def preprocess_texts(texts) -> list[str]:
    """Apply clean_text to an iterable; returns a list of cleaned strings."""
    return [clean_text(t) for t in texts]


def is_effectively_empty(text: Any) -> bool:
    """True if cleaning would yield an empty string."""
    return clean_text(text) == ""


def contains_urdu_script(text: Any) -> bool:
    """True if the string contains Arabic/Urdu script letters."""
    if not isinstance(text, str) or not text:
        return False
    return bool(
        re.search(
            r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]",
            text,
        )
    )
