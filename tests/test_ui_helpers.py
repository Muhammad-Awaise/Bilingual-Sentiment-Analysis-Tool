"""Smoke checks for UI helper module (avoid importing Streamlit in CI)."""

import ast
from pathlib import Path


def test_ui_module_defines_expected_helpers():
    path = Path(__file__).resolve().parents[1] / "src" / "ui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for required in (
        "inject_base_css",
        "page_header",
        "metric_row",
        "shimmer_metrics",
        "shimmer_block",
        "with_loading",
        "loading_placeholder",
    ):
        assert required in names


def test_ui_css_uses_bsa_prefix():
    text = (Path(__file__).resolve().parents[1] / "src" / "ui.py").read_text(
        encoding="utf-8"
    )
    assert "bsa-shimmer" in text
    assert "@keyframes bsa-shimmer" in text
    assert "#f5f7fb" in text
    # No blanket sidebar color / button overrides (sibling leaves widgets to theme)
    assert 'stSidebar"] *' not in text
    assert "baseButton-primary" not in text


def test_streamlit_theme_is_light():
    cfg = (
        Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    ).read_text(encoding="utf-8")
    assert 'base = "light"' in cfg
    assert 'backgroundColor = "#ffffff"' in cfg
    assert 'primaryColor = "#1e4fd6"' in cfg
    assert 'secondaryBackgroundColor = "#f5f7fb"' in cfg
    assert "[theme.sidebar]" in cfg
