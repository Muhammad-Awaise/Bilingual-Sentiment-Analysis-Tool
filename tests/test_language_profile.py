"""Language / script composition tests."""

from src.language_profile import composition_label, detect_script_composition


def test_english_only():
    assert composition_label("This product is very good and useful") == "english_only"


def test_roman_urdu_only():
    assert composition_label("Yeh product bohat acha hai lekin mehnga hai") in {
        "roman_urdu_only",
        "english_roman_urdu",
    }


def test_urdu_script_only():
    assert composition_label("یہ پروڈکٹ بہت اچھی ہے") == "urdu_script_only"


def test_english_roman_mixed():
    out = detect_script_composition("Quality achi hai but delivery late thi")
    assert out["category"] in {"english_roman_urdu", "code_mixed"} or "roman" in out[
        "category"
    ]


def test_english_urdu_script():
    out = detect_script_composition("Quality اچھی hai but late")
    assert out["has_urdu_script"]
    assert out["has_latin"]
    assert out["category"] in {
        "english_urdu_script",
        "three_way_mixed",
        "roman_urdu_urdu_script",
    }
