"""Keep a Streamlit Community Cloud app awake via a real browser session.

Plain HTTP (curl) only fetches the static HTML shell. Streamlit counts
activity after the page JS runs and the app WebSocket connects — so this
script uses Playwright/Chromium, clicks the wake button when needed, and
holds the page open long enough for the backend to register traffic.
"""

from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "APP_URL",
    "https://bilingual-sentiment-analysis-tool.streamlit.app/",
).rstrip("/") + "/"

# Community Cloud sleeps after ~12h without real browser traffic.
# Hold the tab open so JS / WebSocket have time to register activity.
HOLD_SECONDS = int(os.environ.get("KEEPALIVE_HOLD_SECONDS", "25"))
NAV_TIMEOUT_MS = int(os.environ.get("KEEPALIVE_NAV_TIMEOUT_MS", "60000"))
WAKE_TIMEOUT_MS = int(os.environ.get("KEEPALIVE_WAKE_TIMEOUT_MS", "180000"))


def _page_text(page) -> str:
    try:
        return page.inner_text("body") or ""
    except Exception:
        return ""


def _is_sleeping(page) -> bool:
    text = _page_text(page).lower()
    return "gone to sleep" in text or "get this app back up" in text


def _is_waking(page) -> bool:
    text = _page_text(page).lower()
    return "waking up" in text or "just a sec" in text


def _wake_if_needed(page) -> str:
    """Return status: already_awake | woken | still_sleeping."""
    page.wait_for_timeout(2500)

    if not _is_sleeping(page):
        return "already_awake"

    print("App is sleeping; clicking wake button…")
    button = page.get_by_role("button", name="Yes, get this app back up!")
    button.click(timeout=15_000)

    deadline = time.monotonic() + (WAKE_TIMEOUT_MS / 1000.0)
    while time.monotonic() < deadline:
        if _is_waking(page):
            print("Streamlit is booting the app…")
        elif not _is_sleeping(page):
            # Parent shell cleared sleep UI; app usually lives in an iframe.
            return "woken"
        page.wait_for_timeout(2000)

    return "still_sleeping"


def _wait_for_app_frame(page) -> bool:
    """Best-effort check that the Streamlit app iframe mounted."""
    try:
        page.wait_for_selector(
            'iframe[title="streamlitApp"], iframe[src*="/~/+/"]',
            timeout=60_000,
        )
        return True
    except PlaywrightTimeoutError:
        return False


def main() -> int:
    print(f"Keep-alive URL: {APP_URL}")
    print(f"Hold seconds: {HOLD_SECONDS}")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
        )
        page = context.new_page()

        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            print("ERROR: timed out loading the Streamlit URL", file=sys.stderr)
            browser.close()
            return 1

        status = _wake_if_needed(page)
        print(f"Wake status: {status}")
        if status == "still_sleeping":
            print(
                "ERROR: app still shows the sleep page after wake attempt",
                file=sys.stderr,
            )
            browser.close()
            return 1

        has_frame = _wait_for_app_frame(page)
        print(f"App iframe present: {has_frame}")

        # Keep the session open so JS and the app WebSocket count as traffic.
        print(f"Holding browser session for {HOLD_SECONDS}s…")
        page.wait_for_timeout(HOLD_SECONDS * 1000)

        if _is_sleeping(page):
            print("ERROR: app returned to the sleep page", file=sys.stderr)
            browser.close()
            return 1

        title = page.title()
        print(f"Page title: {title!r}")
        print("Keep-alive completed successfully.")
        browser.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
