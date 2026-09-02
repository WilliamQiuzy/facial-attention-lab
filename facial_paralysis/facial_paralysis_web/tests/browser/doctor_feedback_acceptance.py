"""Acceptance checks for the clinician-requested 8081 navigation and landing update."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Browser, expect, sync_playwright


VIEWPORTS = ((320, 800), (768, 900), (1440, 900), (1920, 1080))


def _run_viewport(
    browser: Browser,
    browser_name: str,
    base_url: str,
    width: int,
    height: int,
    screenshot_dir: Path | None,
) -> None:
    page = browser.new_page(viewport={"width": width, "height": height})
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(base_url, wait_until="networkidle")

    expect(page.get_by_role("heading", name="Capture the full facial movement story.")).to_have_count(0)
    expect(page.get_by_role("heading", name="Before recording")).to_be_visible()
    checklist = page.get_by_role("list", name="Before recording checklist")
    expect(checklist.get_by_role("listitem")).to_have_count(4)
    choice = page.get_by_role(
        "group",
        name="Should this assessment include a reanimation smile?",
    )
    expect(choice).to_be_visible()
    expect(choice).not_to_contain_text("Step 8")

    context = f"{browser_name} {width}x{height}"
    choice_box = choice.bounding_box()
    if choice_box is None or choice_box["y"] >= height - 96:
        raise AssertionError(
            f"{context} clinical choice begins below the useful first viewport: {choice_box}"
        )
    if width <= 375:
        first_option_box = choice.locator("label").first.bounding_box()
        if first_option_box is None or first_option_box["y"] >= height - 96:
            raise AssertionError(
                f"{context} first clinical option is below the useful first viewport: {first_option_box}"
            )

    rail_section = page.locator(".workflow-section")
    if rail_section.evaluate("element => getComputedStyle(element).position") != "sticky":
        raise AssertionError(f"{context} workflow rail is not sticky")
    page.evaluate(
        """() => {
          document.documentElement.style.scrollBehavior = 'auto';
          document.body.style.scrollBehavior = 'auto';
          const maximum = document.documentElement.scrollHeight - innerHeight - 1;
          window.scrollTo(0, Math.min(1000, maximum));
        }"""
    )
    page.wait_for_timeout(50)
    rail_box = rail_section.bounding_box()
    if rail_box is None or abs(rail_box["y"]) > 1:
        raise AssertionError(f"{context} workflow rail did not remain at the top: {rail_box}")

    overflow = page.evaluate(
        "Math.max(document.body.scrollWidth, document.documentElement.scrollWidth) - innerWidth"
    )
    if overflow > 1:
        raise AssertionError(f"{context} page has horizontal overflow: {overflow}")
    if console_errors or page_errors:
        raise AssertionError(
            f"{context} runtime errors: console={console_errors}, page={page_errors}"
        )

    if screenshot_dir and browser_name == "chromium":
        screenshot_page = browser.new_page(viewport={"width": width, "height": height})
        screenshot_page.goto(base_url, wait_until="networkidle")
        screenshot_page.screenshot(
            path=screenshot_dir / f"doctor-feedback-{width}x{height}.png",
            full_page=False,
        )
        screenshot_page.close()
    page.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--screenshot-dir", type=Path)
    args = parser.parse_args()
    if args.screenshot_dir:
        args.screenshot_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser_types = (
            ("chromium", playwright.chromium),
            ("firefox", playwright.firefox),
            ("webkit", playwright.webkit),
        )
        for browser_name, browser_type in browser_types:
            browser = browser_type.launch(headless=True)
            try:
                for width, height in VIEWPORTS:
                    _run_viewport(
                        browser,
                        browser_name,
                        args.base_url,
                        width,
                        height,
                        args.screenshot_dir,
                    )
            finally:
                browser.close()

    combinations = len(VIEWPORTS) * len(browser_types)
    print(f"PASS doctor feedback acceptance: {combinations} browser/viewport combinations")


if __name__ == "__main__":
    main()
