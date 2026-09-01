"""Elderly-friendly readability acceptance for the five-stage workflow rail."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


VIEWPORTS = (320, 375, 620, 640, 768, 900, 1024, 1280, 1440, 1920)


def _channel(value: int) -> float:
    normalized = value / 255
    return normalized / 12.92 if normalized <= 0.04045 else ((normalized + 0.055) / 1.055) ** 2.4


def _luminance(rgb: list[int]) -> float:
    red, green, blue = (_channel(value) for value in rgb[:3])
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(foreground: list[int], background: list[int]) -> float:
    lighter = max(_luminance(foreground), _luminance(background))
    darker = min(_luminance(foreground), _luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


def _rgb(locator, property_name: str) -> list[int]:
    return locator.evaluate(
        r"""(element, propertyName) =>
          getComputedStyle(element)[propertyName].match(/\d+/g).slice(0, 3).map(Number)""",
        property_name,
    )


def _font_metrics(locator) -> tuple[float, float]:
    metrics = locator.evaluate(
        """element => ({
          fontSize: parseFloat(getComputedStyle(element).fontSize),
          lineHeight: parseFloat(getComputedStyle(element).lineHeight),
        })"""
    )
    return metrics["fontSize"], metrics["lineHeight"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--screenshot-dir", type=Path)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    if args.screenshot_dir:
        args.screenshot_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for width in VIEWPORTS:
                page = browser.new_page(viewport={"width": width, "height": 900})
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

                rail = page.get_by_role("navigation", name="Assessment journey")
                items = rail.get_by_role("listitem")
                expect(items).to_have_count(5)
                expect(items.nth(0)).to_have_attribute("aria-current", "step")
                for index in range(5):
                    accessible_name = items.nth(index).get_attribute("aria-label") or ""
                    if f"Step {index + 1} of 5" not in accessible_name:
                        raise AssertionError(
                            f"{width}px step {index + 1} lacks total/status semantics: {accessible_name!r}"
                        )

                overflow = page.evaluate(
                    "Math.max(document.body.scrollWidth, document.documentElement.scrollWidth) - innerWidth"
                )
                if overflow > 1:
                    raise AssertionError(f"{width}px page has horizontal overflow: {overflow}")

                circles = rail.locator(".workflow-number")
                for index in range(circles.count()):
                    circle = circles.nth(index)
                    box = circle.bounding_box()
                    if box is None or box["width"] < 40 or box["height"] < 40:
                        raise AssertionError(f"{width}px step {index + 1} marker is too small: {box}")
                    border = _rgb(circle, "borderTopColor")
                    background = _rgb(circle, "backgroundColor") if index == 0 else [255, 255, 255]
                    if index != 0 and _contrast(border, background) < 3:
                        raise AssertionError(
                            f"{width}px step {index + 1} marker contrast is below 3:1"
                        )

                summary = rail.locator(".workflow-current-summary")
                if width <= 959:
                    expect(summary).to_be_visible()
                    expect(summary).to_contain_text("Step 1 of 5")
                    expect(summary).to_contain_text("Prepare")
                    expect(summary).to_contain_text("Review movements")
                    summary_title = summary.locator("strong")
                    summary_detail = summary.locator("small")
                    title_size, title_line = _font_metrics(summary_title)
                    detail_size, detail_line = _font_metrics(summary_detail)
                    if title_size < 18 or title_line / title_size < 1.35:
                        raise AssertionError(f"{width}px current-step title is too small/tight")
                    if detail_size < 16 or detail_line / detail_size < 1.45:
                        raise AssertionError(f"{width}px current-step detail is too small/tight")
                else:
                    expect(summary).to_be_hidden()

                labels = rail.locator(".workflow-rail strong")
                details = rail.locator(".workflow-rail small")
                if width >= 960:
                    for index in range(5):
                        expect(labels.nth(index)).to_be_visible()
                        expect(details.nth(index)).to_be_visible()
                        label_size, label_line = _font_metrics(labels.nth(index))
                        detail_size, detail_line = _font_metrics(details.nth(index))
                        if label_size < 17 or label_line / label_size < 1.35:
                            raise AssertionError(f"{width}px step {index + 1} title is too small/tight")
                        if detail_size < 16 or detail_line / detail_size < 1.45:
                            raise AssertionError(f"{width}px step {index + 1} detail is too small/tight")
                        foreground = _rgb(details.nth(index), "color")
                        background = [243, 248, 253] if index == 0 else [255, 255, 255]
                        if _contrast(foreground, background) < 7:
                            raise AssertionError(f"{width}px step {index + 1} text contrast is below 7:1")
                elif width >= 621:
                    for index in range(5):
                        expect(labels.nth(index)).to_be_visible()
                        label_size, _ = _font_metrics(labels.nth(index))
                        if label_size < 16:
                            raise AssertionError(f"{width}px step {index + 1} title is below 16px")

                if console_errors or page_errors:
                    raise AssertionError(
                        f"{width}px runtime errors: console={console_errors}, page={page_errors}"
                    )
                if args.screenshot_dir and width in (320, 640, 900, 1280, 1920):
                    rail.screenshot(path=args.screenshot_dir / f"workflow-readability-{width}.png")
                page.close()
        finally:
            browser.close()

    print(f"PASS workflow readability acceptance: {len(VIEWPORTS)} viewports")


if __name__ == "__main__":
    main()
