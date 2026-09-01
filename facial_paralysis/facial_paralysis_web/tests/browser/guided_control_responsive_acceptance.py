"""Responsive acceptance checks for the guided recording control."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


VIEWPORTS = (320, 375, 768, 820, 1024, 1280, 1350, 1440, 1920)
EXPECTED_INSTRUCTION_COLOR = "rgb(44, 111, 163)"
EXPECTED_HEADING_COLOR = "rgb(18, 79, 127)"


def _assert_inside(parent: dict[str, float], child: dict[str, float], label: str) -> None:
    tolerance = 1.0
    if child["x"] < parent["x"] - tolerance:
        raise AssertionError(f"{label} extends past the left edge")
    if child["x"] + child["width"] > parent["x"] + parent["width"] + tolerance:
        raise AssertionError(f"{label} extends past the right edge")
    if child["y"] < parent["y"] - tolerance:
        raise AssertionError(f"{label} extends past the top edge")
    if child["y"] + child["height"] > parent["y"] + parent["height"] + tolerance:
        raise AssertionError(f"{label} extends past the bottom edge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
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

                control = page.locator(".guided-session-control")
                flow = page.locator(".guided-flow")
                copy = page.locator(".guided-control-copy")
                action = page.locator(".guided-control-action")
                items = page.locator(".guided-flow li")
                expect(control).to_be_visible()
                expect(items).to_have_count(4)

                overflow = page.evaluate(
                    "Math.max(document.body.scrollWidth, document.documentElement.scrollWidth) - innerWidth"
                )
                if overflow > 1:
                    raise AssertionError(f"{width}px viewport has horizontal overflow: {overflow}")

                control_box = control.bounding_box()
                flow_box = flow.bounding_box()
                copy_box = copy.bounding_box()
                action_box = action.bounding_box()
                if not all((control_box, flow_box, copy_box, action_box)):
                    raise AssertionError(f"{width}px viewport is missing a control region")

                if width >= 1024:
                    if flow_box["width"] < control_box["width"] - 80:
                        raise AssertionError(
                            f"{width}px flow is still squeezed beside copy/action: "
                            f"flow={flow_box['width']}, control={control_box['width']}"
                        )
                    if flow_box["y"] < max(
                        copy_box["y"] + copy_box["height"],
                        action_box["y"] + action_box["height"],
                    ) + 8:
                        raise AssertionError(f"{width}px flow did not receive its own lower row")

                for index in range(items.count()):
                    item = items.nth(index)
                    item_box = item.bounding_box()
                    icon_box = item.locator("svg").bounding_box()
                    text_box = item.locator("span").bounding_box()
                    heading_box = item.locator("strong").bounding_box()
                    if not all((item_box, icon_box, text_box, heading_box)):
                        raise AssertionError(f"{width}px item {index} is missing visible content")
                    _assert_inside(item_box, icon_box, f"{width}px item {index} icon")
                    _assert_inside(item_box, text_box, f"{width}px item {index} text")
                    _assert_inside(item_box, heading_box, f"{width}px item {index} heading")

                    instruction_color = item.evaluate("element => getComputedStyle(element).color")
                    heading_color = item.locator("strong").evaluate(
                        "element => getComputedStyle(element).color"
                    )
                    if instruction_color != EXPECTED_INSTRUCTION_COLOR:
                        raise AssertionError(
                            f"{width}px item {index} uses {instruction_color}, "
                            f"expected stable blue {EXPECTED_INSTRUCTION_COLOR}"
                        )
                    if heading_color != EXPECTED_HEADING_COLOR:
                        raise AssertionError(
                            f"{width}px item {index} heading uses {heading_color}, "
                            f"expected stable blue {EXPECTED_HEADING_COLOR}"
                        )

                if console_errors or page_errors:
                    raise AssertionError(
                        f"{width}px runtime errors: console={console_errors}, page={page_errors}"
                    )
                if args.screenshot_dir and width in (320, 820, 1350, 1920):
                    control.screenshot(path=args.screenshot_dir / f"guided-control-{width}.png")
                page.close()
        finally:
            browser.close()

    print(f"PASS guided control responsive acceptance: {len(VIEWPORTS)} viewports")


if __name__ == "__main__":
    main()
