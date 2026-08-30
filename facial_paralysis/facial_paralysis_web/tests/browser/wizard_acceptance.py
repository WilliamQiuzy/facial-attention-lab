"""Browser acceptance for the isolated five-stage A/B journey candidate."""

from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright


BASE_URL = os.environ.get("ACCEPTANCE_BASE_URL", "http://127.0.0.1:4174").rstrip("/")
SCREENSHOT_DIR = Path(os.environ.get("ACCEPTANCE_SCREENSHOT_DIR", "/tmp/faces-wizard-ab"))
READY_RESPONSE = {
    "status": "ready",
    "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
    "candidate_id": "BLV9-009",
    "ensemble_members": 3,
    "preprocessing": "faces-to-shared-v9/v1",
}


def _prepare(page: Page) -> None:
    page.route(
        "**/api/v1/facial-paralysis/ready",
        lambda route: route.fulfill(status=200, json=READY_RESPONSE),
    )
    page.add_init_script(
        """
        window.__journeyScrollCalls = []
        const original = Element.prototype.scrollIntoView
        Element.prototype.scrollIntoView = function(options) {
          window.__journeyScrollCalls.push({ options, className: this.className || '' })
          return original?.call(this, options)
        }
        class AcceptanceUtterance {
          constructor(text) { this.text = text; this.onend = null; this.onerror = null }
        }
        Object.defineProperty(window, 'SpeechSynthesisUtterance', {
          configurable: true, value: AcceptanceUtterance,
        })
        Object.defineProperty(window, 'speechSynthesis', {
          configurable: true,
          value: {
            speak(utterance) { window.__activeUtterance = utterance },
            cancel() { window.__activeUtterance = null },
          },
        })
        """
    )
    page.goto(BASE_URL, wait_until="networkidle")


def _assert_no_overflow(page: Page, label: str) -> None:
    metrics = page.evaluate(
        """() => ({
          viewport: window.innerWidth,
          document: document.documentElement.scrollWidth,
          body: document.body.scrollWidth,
        })"""
    )
    if max(metrics["document"], metrics["body"]) > metrics["viewport"] + 1:
        raise AssertionError(f"horizontal overflow at {label}: {metrics}")


def _assert_stage(page: Page, number: int, title: str) -> None:
    current = page.locator('.workflow-rail li[aria-current="step"]')
    expect(current.locator(".workflow-number")).to_have_text(str(number))
    expect(current).to_contain_text(title)
    expect(page.locator("[data-journey-heading]")).to_be_visible()
    if page.locator("[data-journey-heading]").evaluate("element => element !== document.activeElement"):
        raise AssertionError(f"stage {number} heading did not receive keyboard focus")
    visible_headings = page.locator(".journey-stage-heading:visible")
    expect(visible_headings).to_have_count(1)


def _assert_journey_controls_in_view(page: Page, label: str) -> None:
    box = page.get_by_role("navigation", name="Journey controls").bounding_box()
    viewport = page.viewport_size
    if box is None or viewport is None:
        raise AssertionError(f"journey controls are missing at {label}: {box}, {viewport}")
    if box["y"] < -1 or box["y"] + box["height"] > viewport["height"] + 1:
        raise AssertionError(f"journey controls are outside the viewport at {label}: {box}, {viewport}")


def _screenshot(page: Page, label: str) -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOT_DIR / f"{label}.png"), full_page=True)


def _run_viewport(page: Page, label: str, width: int, height: int) -> None:
    page.set_viewport_size({"width": width, "height": height})
    _prepare(page)

    journey = page.get_by_role("navigation", name="Assessment journey")
    expect(journey.locator("li")).to_have_count(5)
    expect(journey.get_by_role("button")).to_have_count(0)
    expect(page.get_by_role("heading", name="Prepare for a consistent capture")).to_be_visible()
    expect(page.get_by_role("tab", name="Use this device")).to_have_count(0)
    expect(page.locator(".analysis-section")).to_be_hidden()
    expect(page.get_by_role("button", name="Choose Step 8 above to continue")).to_be_disabled()
    page.get_by_role("radio", name="Step 8 not applicable", exact=False).check()
    expect(page.get_by_role("button", name="Continue to camera setup")).to_be_enabled()
    _assert_journey_controls_in_view(page, f"{label} prepare")
    _assert_no_overflow(page, f"{label} prepare")
    _screenshot(page, f"{label}-step-1-prepare")

    page.get_by_role("button", name="Continue to camera setup").click()
    _assert_stage(page, 2, "Set up")
    expect(page.get_by_role("tab", name="Use this device")).to_have_attribute("aria-selected", "true")
    expect(page.get_by_role("tab", name="Upload from LifeLink")).to_have_attribute("aria-selected", "false")
    expect(page.locator(".journey-preparation")).to_be_hidden()
    expect(page.locator(".analysis-section")).to_be_hidden()
    expect(page.get_by_role("button", name="Back to preparation")).to_be_visible()
    expect(page.get_by_role("button", name="Continue to recording")).to_be_disabled()
    _assert_journey_controls_in_view(page, f"{label} setup")
    _assert_no_overflow(page, f"{label} setup")
    _screenshot(page, f"{label}-step-2-setup")

    if label != "desktop":
        return

    page.get_by_role("button", name="Enable front camera").click()
    expect(page.get_by_role("button", name="Continue to recording")).to_be_enabled(timeout=10_000)
    page.get_by_role("button", name="Continue to recording").click()
    _assert_stage(page, 3, "Record")
    expect(page.get_by_role("heading", name="Ready for your guided recording")).to_be_visible()
    expect(page.get_by_role("button", name="Previous instruction")).to_have_count(0)
    expect(page.get_by_role("button", name="Next instruction")).to_have_count(0)

    page.get_by_role("button", name="Start guided recording").click()
    expect(page.get_by_role("button", name="Stop and discard guided recording")).to_be_visible()
    expect(page.get_by_role("navigation", name="Journey controls")).to_have_count(0)
    expect(page.get_by_role("region", name="Patient movement guidance")).to_be_visible()
    _assert_no_overflow(page, "desktop active recording")
    _screenshot(page, "desktop-step-3-recording")

    page.get_by_role("button", name="Stop and discard guided recording").click()
    expect(page.get_by_text("Guided recording stopped. The incomplete video was discarded.")).to_be_visible()
    expect(page.get_by_role("button", name="Back to camera setup")).to_be_enabled()

    calls = page.evaluate("window.__journeyScrollCalls")
    if len(calls) < 2:
        raise AssertionError(f"major-stage automatic positioning did not run: {calls}")

    for button in page.locator(".journey-actions .button:visible").all():
        box = button.bounding_box()
        if box is None or box["height"] < 48:
            raise AssertionError(f"journey touch target is too small: {box}")


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        )
        context = browser.new_context(permissions=["camera"])
        page = context.new_page()
        for label, width, height in (
            ("desktop", 1440, 1000),
            ("tablet", 900, 1100),
            ("mobile", 390, 844),
        ):
            _run_viewport(page, label, width, height)
        context.close()

        reduced = browser.new_context(reduced_motion="reduce", permissions=["camera"])
        reduced_page = reduced.new_page()
        _prepare(reduced_page)
        reduced_page.get_by_role("radio", name="Step 8 not applicable", exact=False).check()
        reduced_page.get_by_role("button", name="Continue to camera setup").click()
        calls = reduced_page.evaluate("window.__journeyScrollCalls")
        if not any(call.get("options", {}).get("behavior") == "auto" for call in calls):
            raise AssertionError(f"reduced-motion journey did not disable smooth positioning: {calls}")
        reduced.close()
        browser.close()

    print(f"PASS wizard A/B browser acceptance at {BASE_URL}")
    print(f"SCREENSHOTS {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
