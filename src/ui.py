"""
Shared Streamlit presentation helpers (UI polish + shimmer loading).

Matches sibling Urdu-English Code-Switching Analyzer light research UI.
Theme colors live in ``.streamlit/config.toml``; this module only adds layout
and shimmer — same approach as sibling ``streamlit_lib/ui.py`` (no button /
code / blanket color overrides that fight Streamlit's light theme).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

__all__ = [
    "inject_base_css",
    "page_header",
    "metric_row",
    "shimmer_metrics",
    "shimmer_block",
    "with_loading",
    "loading_placeholder",
]


def inject_base_css() -> None:
    """Inject sibling-style layout, shimmer, and toolbar chrome hiding.

    Same approach as sibling ``streamlit_lib/ui.py``: hide Share / star / edit /
    GitHub / ⋮ / Deploy via data-testid + attribute fallbacks, keep the sidebar
    collapse control usable, and soft-mask residual top-right chips.
    """
    st.markdown(
        """
        <style>
          /* Layout — same as sibling streamlit_lib/ui.py */
          .block-container { padding-top: 1.4rem; max-width: 1200px; }
          div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
          h1, h2, h3 { letter-spacing: -0.02em; }

          /* Sidebar surface = sibling secondaryBackgroundColor (#f5f7fb) */
          section[data-testid="stSidebar"],
          div[data-testid="stSidebar"] {
            background-color: #f5f7fb;
          }

          /* Soft metric cards on main canvas (not sidebar — too narrow) */
          div[data-testid="stAppViewContainer"]
            > .main div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.75rem 1rem;
          }

          /* Keep sibling-style page nav fully visible above model status */
          div[data-testid="stSidebarNav"] {
            min-height: auto;
          }

          hr { border-color: #e2e8f0; }

          /* --- Hide Cloud / app toolbar chrome (Share, star, edit, GitHub, ⋮, Deploy) --- */
          #MainMenu,
          footer,
          div[data-testid="stDecoration"],
          div[data-testid="stStatusWidget"],
          div[data-testid="stToolbar"],
          div[data-testid="stAppToolbar"],
          .stDeployButton,
          [data-testid="stAppDeployButton"],
          [data-testid="stToolbarActionButtonShare"],
          [data-testid="stToolbarActionButtonFavorite"],
          [data-testid="stToolbarActionButtonStar"],
          [data-testid="stToolbarActionButtonRename"],
          [data-testid="stToolbarActionButtonEdit"],
          [data-testid="stToolbarActionButtonGithub"],
          [data-testid="stToolbarActionButtonGitHub"],
          [data-testid^="stToolbarActionButton"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
            height: 0 !important;
            width: 0 !important;
            min-width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
          }

          /* Attribute / label fallbacks when testids differ by Streamlit version.
             Avoid blanketing all header buttons — sidebar collapse lives in header too. */
          header a[href*="github.com"],
          header a[href*="share.streamlit"],
          header [aria-label="Share"],
          header [aria-label="Share this app"],
          header [aria-label="Favorite"],
          header [aria-label="Star"],
          header [aria-label="Edit"],
          header [aria-label="Rename"],
          header [aria-label="GitHub"],
          header [aria-label="Source on GitHub"],
          header [title="Share"],
          header [title="GitHub"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
          }

          /* Soft mask over top-right if any residual Cloud chips remain inside the app */
          .bsa-toolbar-mask {
            position: fixed;
            top: 0;
            right: 0;
            width: min(320px, 42vw);
            height: 3.25rem;
            z-index: 1000000;
            background: #ffffff;
            pointer-events: auto;
          }

          /* Shimmer — sibling recipe (bsa- prefix) */
          @keyframes bsa-shimmer {
            0% { background-position: -400px 0; }
            100% { background-position: 400px 0; }
          }
          .bsa-shimmer {
            background: linear-gradient(90deg, #eef2f7 25%, #f8fafc 37%, #eef2f7 63%);
            background-size: 800px 100%;
            animation: bsa-shimmer 1.15s ease-in-out infinite;
            border-radius: 12px;
          }
          .bsa-shimmer-card {
            padding: 1rem;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            background: #fff;
            margin-bottom: 0.75rem;
          }
          .bsa-shimmer-line { height: 12px; margin-bottom: 10px; }
          .bsa-shimmer-line.lg { height: 28px; width: 55%; }
          .bsa-shimmer-line.sm { width: 35%; }
          .bsa-shimmer-line.md { width: 75%; }
        </style>
        <div class="bsa-toolbar-mask" aria-hidden="true"></div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def metric_row(items: list[tuple[str, Any]]) -> None:
    """Render a row of st.metric cards."""
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        display = "—" if value is None else value
        col.metric(label, display)


def shimmer_metrics(n: int = 4) -> None:
    """Skeleton placeholders for a metric row."""
    cols = st.columns(max(1, int(n)))
    for col in cols:
        col.markdown(
            """
            <div class="bsa-shimmer-card">
              <div class="bsa-shimmer bsa-shimmer-line sm"></div>
              <div class="bsa-shimmer bsa-shimmer-line lg"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def shimmer_block(rows: int = 3) -> None:
    """Skeleton placeholders for content panels / tables."""
    chunks: list[str] = []
    for _ in range(max(1, int(rows))):
        chunks.append(
            """
            <div class="bsa-shimmer-card">
              <div class="bsa-shimmer bsa-shimmer-line md"></div>
              <div class="bsa-shimmer bsa-shimmer-line sm"></div>
              <div class="bsa-shimmer bsa-shimmer-line" style="width:90%;"></div>
            </div>
            """
        )
    st.markdown("".join(chunks), unsafe_allow_html=True)


def with_loading(label: str = "Loading…"):
    """Context manager: show Streamlit spinner while work runs."""
    return st.spinner(label)


def loading_placeholder(
    *,
    metrics: int = 0,
    blocks: int = 2,
    spinner: str = "Loading…",
):
    """Show shimmer while work runs, then clear it."""

    class _LoadingCtx:
        def __init__(self) -> None:
            self._slot = st.empty()
            self._spinner_cm = None

        def __enter__(self):
            with self._slot.container():
                if metrics > 0:
                    shimmer_metrics(metrics)
                if blocks > 0:
                    shimmer_block(blocks)
            self._spinner_cm = st.spinner(spinner)
            self._spinner_cm.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            if self._spinner_cm is not None:
                self._spinner_cm.__exit__(exc_type, exc, tb)
            self._slot.empty()
            return False

    return _LoadingCtx()
