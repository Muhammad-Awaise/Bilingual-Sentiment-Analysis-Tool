"""Tests for src.preprocessing."""

from src.preprocessing import clean_text, is_effectively_empty, preprocess_texts


def test_empty_string():
    assert clean_text("") == ""


def test_whitespace_only():
    assert clean_text("   \t\n  ") == ""


def test_none():
    assert clean_text(None) == ""


def test_nan_float():
    assert clean_text(float("nan")) == ""


def test_english_text():
    out = clean_text("Great Product Quality!")
    assert "great" in out
    assert "product" in out
    assert "!" not in out


def test_roman_urdu_text():
    out = clean_text("Yeh Product Bohat Acha Hai")
    assert out == "yeh product bohat acha hai"


def test_code_mixed():
    out = clean_text("Quality achi hai but delivery late thi")
    assert "but" in out
    assert "achi" in out


def test_urls_and_mentions():
    out = clean_text("Buy now https://example.com contact @seller #deal")
    assert "http" not in out
    assert "@" not in out
    assert "deal" in out
    assert "buy" in out


def test_negation_preservation_english():
    out = clean_text("This is not good")
    assert "not" in out
    assert "good" in out


def test_negation_preservation_roman_urdu():
    out = clean_text("Product acha nahi hai mat")
    assert "nahi" in out
    # 'mat' is a negation cue; we intentionally keep it
    assert "mat" in out


def test_negation_spelling_canonicalization():
    """Closed-set map: nahin/nae/nhi → nahi (see LINGUISTIC_DECISIONS)."""
    out = clean_text("Product acha nahin hai")
    assert "nahi" in out
    assert "nahin" not in out
    assert "nahi" in clean_text("bohat acha nae hai")


def test_contrast_preservation():
    out = clean_text("Acha hai lekin quality kharab hai but delayed")
    assert "lekin" in out
    assert "but" in out
    assert "__contrast__" in out


def test_contrast_clause_emphasized():
    out = clean_text("Product acha hai lekin quality kharab hai")
    assert out.count("kharab") >= 3
    assert out.count("acha") == 1
    assert "__contrast__" in out


def test_unicode_input():
    out = clean_text("Café quality is good 🙂")
    assert "quality" in out
    assert "good" in out


def test_repeated_whitespace():
    assert clean_text("too    many     spaces") == "too many spaces"


def test_consistent_repeated_calls():
    text = "Bohat ACHA product!!! Visit www.test.com"
    assert clean_text(text) == clean_text(text)


def test_preprocess_texts_list():
    assert preprocess_texts([" Hi ", None, "OK"]) == ["hi", "", "ok"]


def test_is_effectively_empty():
    assert is_effectively_empty("   ")
    assert not is_effectively_empty("acha")


def test_non_string_number():
    assert clean_text(123) == "123"
