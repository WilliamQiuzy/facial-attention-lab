"""Patient-journey edge acceptance for the five-stage capture experience."""

from __future__ import annotations

import argparse

from playwright.sync_api import Browser, Page, expect, sync_playwright


READY_RESPONSE = {
    "status": "ready",
    "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
    "candidate_id": "BLV9-009",
    "ensemble_members": 3,
    "preprocessing": "faces-to-shared-v9/v1",
}


def _route_ready(page: Page) -> None:
    page.route(
        "**/api/v1/facial-paralysis/ready",
        lambda route: route.fulfill(status=200, json=READY_RESPONSE),
    )


def _assert_clean_runtime(page: Page, console_errors: list[str], page_errors: list[str]) -> None:
    if console_errors or page_errors:
        raise AssertionError(
            f"browser runtime errors: console={console_errors}, page={page_errors}"
        )


def _new_page(context, *, init_script: str | None = None) -> tuple[Page, list[str], list[str]]:
    if init_script:
        context.add_init_script(init_script)
    page = context.new_page()
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    return page, console_errors, page_errors


def _choose_and_open_setup(page: Page, base_url: str, include_step_8: bool = False) -> None:
    _route_ready(page)
    page.goto(base_url, wait_until="networkidle")
    choice = "Yes — include reanimation smile" if include_step_8 else "No — standard assessment"
    page.get_by_role("radio", name=choice, exact=False).check()
    page.get_by_role("button", name="Continue to camera setup").click()
    expect(page.get_by_role("heading", name="Set up the camera")).to_be_visible()


def _case_initial_navigation(browser: Browser, base_url: str) -> None:
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    page, console_errors, page_errors = _new_page(context)
    _route_ready(page)
    page.goto(base_url, wait_until="networkidle")

    if page.evaluate("document.activeElement === document.body") is not True:
        raise AssertionError("initial render stole keyboard focus before user navigation")
    page.keyboard.press("Tab")
    skip = page.locator(".skip-link")
    expect(skip).to_be_focused()
    if skip.evaluate("element => getComputedStyle(element).clipPath") != "none":
        raise AssertionError("keyboard-focused skip link remained visually clipped")
    page.keyboard.press("Enter")
    expect(page.locator("#main-content")).to_be_focused()

    continue_button = page.get_by_role(
        "button", name="Choose the reanimation-smile option above to continue"
    )
    expect(continue_button).to_be_disabled()
    page.get_by_role("radio", name="No — standard assessment", exact=False).check()
    page.get_by_role("button", name="Continue to camera setup").click()
    expect(page.get_by_role("heading", name="Set up the camera")).to_be_focused()
    page.get_by_role("button", name="Back to preparation").click()
    expect(
        page.get_by_role("radio", name="No — standard assessment", exact=False)
    ).to_be_checked()

    page.reload(wait_until="networkidle")
    expect(page.get_by_role("button", name="Choose the reanimation-smile option above to continue")).to_be_disabled()
    expect(page.get_by_role("radio", name="No — standard assessment", exact=False)).not_to_be_checked()
    expect(page.get_by_role("radio", name="Yes — include reanimation smile", exact=False)).not_to_be_checked()
    _assert_clean_runtime(page, console_errors, page_errors)
    context.close()


def _case_endpoint_retry(browser: Browser, base_url: str) -> None:
    context = browser.new_context()
    page, console_errors, page_errors = _new_page(context)
    attempts = 0

    def readiness(route) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            route.fulfill(status=503, json={"detail": {"code": "model_not_ready"}})
        else:
            route.fulfill(status=200, json=READY_RESPONSE)

    page.route("**/api/v1/facial-paralysis/ready", readiness)
    page.goto(base_url, wait_until="networkidle")
    page.get_by_role("radio", name="No — standard assessment", exact=False).check()
    page.get_by_role("button", name="Continue to camera setup").click()
    expect(page.get_by_text("Research endpoint unavailable", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Continue to recording")).to_be_disabled()
    page.get_by_role("button", name="Retry endpoint check").click()
    expect(page.get_by_text("Analysis endpoint ready", exact=True)).to_be_visible()
    if attempts != 2:
        raise AssertionError(f"endpoint retry count drifted: {attempts}")
    expected_network_errors = [
        message for message in console_errors if "status of 503" in message
    ]
    if len(expected_network_errors) != 1:
        raise AssertionError(f"unexpected readiness network errors: {console_errors}")
    console_errors.clear()
    _assert_clean_runtime(page, console_errors, page_errors)
    context.close()


def _case_permission_denied(browser: Browser, base_url: str) -> None:
    context = browser.new_context()
    page, console_errors, page_errors = _new_page(
        context,
        init_script="""
          navigator.mediaDevices.getUserMedia = async () => {
            throw new DOMException('private device label', 'NotAllowedError')
          }
        """,
    )
    _choose_and_open_setup(page, base_url)
    page.get_by_role("button", name="Enable front camera").click()
    alert = page.get_by_role("alert")
    expect(alert).to_contain_text("Camera permission was denied")
    expect(alert).not_to_contain_text("private device label")
    expect(page.get_by_role("tab", name="Upload from LifeLink")).to_be_enabled()
    page.get_by_role("tab", name="Upload from LifeLink").click()
    expect(page.get_by_text("Browse video", exact=True)).to_be_visible()
    _assert_clean_runtime(page, console_errors, page_errors)
    context.close()


def _case_camera_missing(browser: Browser, base_url: str) -> None:
    context = browser.new_context()
    page, console_errors, page_errors = _new_page(
        context,
        init_script="""
          navigator.mediaDevices.getUserMedia = async () => {
            throw new DOMException('private device label', 'NotFoundError')
          }
        """,
    )
    _choose_and_open_setup(page, base_url)
    page.get_by_role("button", name="Enable front camera").click()
    alert = page.get_by_role("alert")
    expect(alert).to_contain_text("No front-facing camera was found")
    expect(alert).not_to_contain_text("private device label")
    expect(page.get_by_role("tab", name="Upload from LifeLink")).to_be_enabled()
    _assert_clean_runtime(page, console_errors, page_errors)
    context.close()


def _case_speech_unavailable(browser: Browser, base_url: str) -> None:
    context = browser.new_context(permissions=["camera"])
    page, console_errors, page_errors = _new_page(
        context,
        init_script="""
          Object.defineProperty(window, 'speechSynthesis', { configurable: true, value: undefined })
          Object.defineProperty(window, 'SpeechSynthesisUtterance', { configurable: true, value: undefined })
        """,
    )
    _choose_and_open_setup(page, base_url)
    page.get_by_role("button", name="Enable front camera").click()
    expect(page.get_by_text("This browser cannot play the guided voice sequence", exact=False)).to_be_visible()
    expect(page.get_by_role("button", name="Continue to recording")).to_be_disabled()
    expect(page.get_by_role("tab", name="Upload from LifeLink")).to_be_enabled()
    _assert_clean_runtime(page, console_errors, page_errors)
    context.close()


def _case_active_recording_lock(browser: Browser, base_url: str) -> None:
    context = browser.new_context(permissions=["camera"])
    page, console_errors, page_errors = _new_page(
        context,
        init_script="""
          class TestUtterance {
            constructor(text) { this.text = text; this.onstart = null; this.onend = null; this.onerror = null }
          }
          Object.defineProperty(window, 'SpeechSynthesisUtterance', { configurable: true, value: TestUtterance })
          Object.defineProperty(window, 'speechSynthesis', {
            configurable: true,
            value: {
              speak(utterance) { window.__heldUtterance = utterance; utterance.onstart?.() },
              cancel() { window.__heldUtterance = null },
            },
          })
        """,
    )
    _choose_and_open_setup(page, base_url, include_step_8=True)
    page.get_by_role("button", name="Enable front camera").click()
    next_button = page.get_by_role("button", name="Continue to recording")
    expect(next_button).to_be_enabled(timeout=10_000)
    next_button.click()
    start = page.get_by_role("button", name="Start guided recording")
    start.evaluate("button => { button.click(); button.click(); }")
    stop = page.get_by_role("button", name="Stop and discard guided recording")
    expect(stop).to_be_visible(timeout=10_000)
    source_tabs = page.locator('[role="tab"]')
    expect(source_tabs).to_have_count(2)
    if source_tabs.evaluate_all("tabs => tabs.some(tab => !tab.disabled)"):
        raise AssertionError("a hidden recording-source tab remained interactive")
    expect(page.get_by_role("navigation", name="Journey controls")).to_have_count(0)
    expect(page.get_by_role("region", name="Patient movement guidance")).to_be_visible()
    stop.click()
    expect(page.get_by_text("The incomplete video was discarded", exact=False)).to_be_visible()
    expect(page.get_by_role("button", name="Back to camera setup")).to_be_enabled()
    expect(page.get_by_label("Recorded camera preview")).to_have_count(0)
    _assert_clean_runtime(page, console_errors, page_errors)
    context.close()


def _case_direct_report_and_mobile(browser: Browser, base_url: str) -> None:
    context = browser.new_context(viewport={"width": 320, "height": 568})
    page, console_errors, page_errors = _new_page(context)
    _route_ready(page)
    page.goto(f"{base_url}/#research-report", wait_until="networkidle")
    expect(page.get_by_role("heading", name="Report not retained")).to_be_visible()
    expect(page.get_by_role("button", name="Run research analysis")).to_have_count(0)
    page.get_by_role("link", name="Return to research analysis").click()
    expect(page.get_by_role("heading", name="Before recording")).to_be_visible()
    overflow = page.evaluate(
        "Math.max(document.body.scrollWidth, document.documentElement.scrollWidth) - innerWidth"
    )
    if overflow > 1:
        raise AssertionError(f"small-phone journey has horizontal overflow: {overflow}")
    _assert_clean_runtime(page, console_errors, page_errors)
    context.close()


def _case_back_from_analysis(browser: Browser, base_url: str) -> None:
    context = browser.new_context()
    page, console_errors, page_errors = _new_page(context)
    _route_ready(page)
    page.goto(base_url, wait_until="networkidle")
    page.get_by_role("radio", name="No — standard assessment", exact=False).check()
    page.get_by_role("button", name="Continue to camera setup").click()
    page.get_by_role("tab", name="Upload from LifeLink").click()
    page.get_by_label("Choose LifeLink Face video").set_input_files(
        {
            "name": "faces-back-navigation.webm",
            "mimeType": "video/webm",
            "buffer": b"non-clinical-navigation-fixture",
        }
    )
    expect(page.get_by_role("heading", name="Review the recording and run analysis")).to_be_visible()
    page.get_by_role("button", name="Back to recording").click()
    expect(page.get_by_role("heading", name="Complete the automatic recording")).to_be_visible()
    expect(page.get_by_text("faces-back-navigation.webm", exact=True)).to_be_visible()
    page.get_by_role("button", name="Back to camera setup").click()
    expect(page.get_by_role("heading", name="Set up the camera")).to_be_visible()
    page.get_by_role("button", name="Back to preparation").click()
    expect(page.get_by_role("heading", name="Before recording")).to_be_visible()
    _assert_clean_runtime(page, console_errors, page_errors)
    context.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        )
        try:
            _case_initial_navigation(browser, base_url)
            _case_endpoint_retry(browser, base_url)
            _case_permission_denied(browser, base_url)
            _case_camera_missing(browser, base_url)
            _case_speech_unavailable(browser, base_url)
            _case_active_recording_lock(browser, base_url)
            _case_direct_report_and_mobile(browser, base_url)
            _case_back_from_analysis(browser, base_url)
        finally:
            browser.close()

    print("PASS journey edge acceptance: 8 cases")


if __name__ == "__main__":
    main()
