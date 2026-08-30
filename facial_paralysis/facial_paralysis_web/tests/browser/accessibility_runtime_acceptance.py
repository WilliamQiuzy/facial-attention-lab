"""Keyboard, assistive-status, touch-target, and contrast acceptance."""

from __future__ import annotations

import argparse

from playwright.sync_api import BrowserType, Page, expect, sync_playwright


READY_RESPONSE = {
    "status": "ready",
    "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
    "candidate_id": "BLV9-009",
    "ensemble_members": 3,
    "preprocessing": "faces-to-shared-v9/v1",
}


def _prepare(page: Page, base_url: str) -> tuple[list[str], list[str]]:
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.route(
        "**/api/v1/facial-paralysis/ready",
        lambda route: route.fulfill(status=200, json=READY_RESPONSE),
    )
    page.goto(base_url, wait_until="networkidle")
    return console_errors, page_errors


def _assert_runtime(console_errors: list[str], page_errors: list[str], label: str) -> None:
    if console_errors or page_errors:
        raise AssertionError(
            f"{label} runtime errors: console={console_errors}, page={page_errors}"
        )


def _keyboard_case(engine_name: str, engine: BrowserType, base_url: str) -> None:
    browser = engine.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()
    console_errors, page_errors = _prepare(page, base_url)

    if engine_name == "webkit":
        # Playwright WebKit follows the macOS system preference that may omit
        # links from Tab order; direct focus still verifies the Safari path.
        page.locator(".skip-link").focus()
    else:
        page.keyboard.press("Tab")
    expect(page.locator(".skip-link")).to_be_focused()
    page.keyboard.press("Enter")
    expect(page.get_by_role("main")).to_be_focused()

    choice = page.get_by_role("radio", name="Step 8 not applicable", exact=False)
    choice.focus()
    page.keyboard.press("Space")
    expect(choice).to_be_checked()
    continue_button = page.get_by_role("button", name="Continue to camera setup")
    expect(continue_button).to_be_enabled()
    continue_button.focus()
    page.keyboard.press("Enter")
    expect(page.get_by_role("heading", name="Set up the camera")).to_be_focused()

    camera_tab = page.get_by_role("tab", name="Use this device")
    camera_tab.focus()
    page.keyboard.press("ArrowRight")
    upload_tab = page.get_by_role("tab", name="Upload from LifeLink")
    expect(upload_tab).to_be_focused()
    expect(upload_tab).to_have_attribute("aria-selected", "true")
    page.keyboard.press("Home")
    expect(camera_tab).to_be_focused()
    expect(camera_tab).to_have_attribute("aria-selected", "true")

    status_nodes = page.locator('[role="status"]')
    if status_nodes.count() != 1:
        raise AssertionError(
            f"{engine_name}: setup exposes duplicate or missing live status: {status_nodes.count()}"
        )
    _assert_runtime(console_errors, page_errors, f"{engine_name} keyboard")
    context.close()
    browser.close()


def _mobile_touch_case(engine: BrowserType, base_url: str) -> None:
    browser = engine.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 320, "height": 568},
        is_mobile=True,
        has_touch=True,
    )
    page = context.new_page()
    console_errors, page_errors = _prepare(page, base_url)
    page.get_by_role("radio", name="Step 8 not applicable", exact=False).check()

    targets = page.locator(
        '.journey-actions button:visible, input[type="radio"] + span, .source-tab:visible'
    )
    boxes = targets.evaluate_all(
        "elements => elements.map(element => { const box = element.getBoundingClientRect(); return { width: box.width, height: box.height, text: element.textContent } })"
    )
    if not boxes or any(box["height"] < 48 or box["width"] < 48 for box in boxes):
        raise AssertionError(f"critical mobile touch target below 48px: {boxes}")
    overflow = page.evaluate(
        "Math.max(document.body.scrollWidth, document.documentElement.scrollWidth) - innerWidth"
    )
    if overflow > 1:
        raise AssertionError(f"mobile keyboard/touch page overflow: {overflow}")
    _assert_runtime(console_errors, page_errors, "chromium mobile touch")
    context.close()
    browser.close()


def _contrast_and_motion_case(engine: BrowserType, base_url: str) -> None:
    browser = engine.launch(headless=True)
    context = browser.new_context(
        forced_colors="active",
        reduced_motion="reduce",
        viewport={"width": 1000, "height": 800},
    )
    page = context.new_page()
    page.add_init_script(
        """
        window.__scrollCalls = []
        Element.prototype.scrollIntoView = function(options) { window.__scrollCalls.push(options) }
        """
    )
    console_errors, page_errors = _prepare(page, base_url)
    choice = page.get_by_role("radio", name="Step 8 not applicable", exact=False)
    choice.focus()
    page.keyboard.press("Space")
    next_button = page.get_by_role("button", name="Continue to camera setup")
    for _ in range(40):
        if next_button.evaluate("element => element === document.activeElement"):
            break
        page.keyboard.press("Tab")
    else:
        raise AssertionError("keyboard traversal did not reach the journey continue control")
    focus_style = next_button.evaluate(
        "element => ({ style: getComputedStyle(element).outlineStyle, width: parseFloat(getComputedStyle(element).outlineWidth) })"
    )
    if focus_style["style"] == "none" or focus_style["width"] < 2:
        raise AssertionError(f"forced-color keyboard focus is not visible: {focus_style}")
    next_button.click()
    calls = page.evaluate("window.__scrollCalls")
    if not calls or calls[-1].get("behavior") != "auto":
        raise AssertionError(f"reduced motion did not disable smooth positioning: {calls}")
    _assert_runtime(console_errors, page_errors, "forced color/reduced motion")
    context.close()
    browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    with sync_playwright() as playwright:
        _keyboard_case("chromium", playwright.chromium, base_url)
        _keyboard_case("webkit", playwright.webkit, base_url)
        _mobile_touch_case(playwright.chromium, base_url)
        _contrast_and_motion_case(playwright.chromium, base_url)

    print("PASS accessibility/runtime acceptance: 4 cases")


if __name__ == "__main__":
    main()
