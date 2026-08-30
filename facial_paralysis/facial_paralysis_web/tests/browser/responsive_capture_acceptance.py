"""Cross-browser geometry checks for camera framing and the four setup cues."""

from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import BrowserType, Page, expect, sync_playwright


BASE_URL = os.environ.get("ACCEPTANCE_BASE_URL", "http://127.0.0.1:4174").rstrip("/")
SCREENSHOT_DIR = Path(os.environ.get("ACCEPTANCE_SCREENSHOT_DIR", "/tmp/faces-responsive-capture"))
VIEWPORTS = (
    ("desktop", 1440, 1000),
    ("narrow-desktop", 1000, 800),
    ("tablet-portrait", 768, 1024),
    ("phone-portrait", 390, 844),
    ("phone-landscape", 844, 390),
    ("small-phone", 320, 568),
)


def _open_setup(page: Page) -> None:
    page.goto(BASE_URL, wait_until="networkidle")
    expect(page.get_by_role("button", name="Choose Step 8 above to continue")).to_be_disabled()
    page.get_by_role("radio", name="Step 8 not applicable", exact=False).check()
    expect(page.get_by_role("button", name="Continue to camera setup")).to_be_enabled()
    page.get_by_role("button", name="Continue to camera setup").click()
    page.get_by_role("heading", name="Set up the camera").wait_for(state="visible")


def _intersects(first: dict[str, float], second: dict[str, float]) -> bool:
    return (
        min(first["right"], second["right"]) - max(first["left"], second["left"]) > 0.5
        and min(first["bottom"], second["bottom"]) - max(first["top"], second["top"]) > 0.5
    )


def _assert_geometry(page: Page, label: str, width: int, height: int) -> None:
    metrics = page.locator(".camera-stage").evaluate(
        """element => {
          const stage = element.getBoundingClientRect()
          const guide = getComputedStyle(element, '::after')
          const video = getComputedStyle(element.querySelector('video'))
          const cueItems = [...document.querySelectorAll('.guided-flow li')].map(item => {
            const rect = item.getBoundingClientRect()
            return {
              left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom,
              scrollWidth: item.scrollWidth, clientWidth: item.clientWidth,
              scrollHeight: item.scrollHeight, clientHeight: item.clientHeight,
            }
          })
          const navigation = document.querySelector('.journey-actions')?.getBoundingClientRect()
          const navigationButtons = [...document.querySelectorAll('.journey-actions .button')].map(item => {
            const rect = item.getBoundingClientRect()
            return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom }
          })
          return {
            stage: { width: stage.width, height: stage.height },
            guide: { width: parseFloat(guide.width), height: parseFloat(guide.height) },
            objectFit: video.objectFit,
            cueItems,
            navigation: navigation ? { height: navigation.height } : null,
            navigationButtons,
            documentWidth: document.documentElement.scrollWidth,
            viewportWidth: window.innerWidth,
          }
        }"""
    )

    stage_ratio = metrics["stage"]["width"] / metrics["stage"]["height"]
    portrait_phone = width <= 560 and height > width
    expected_ratio = 3 / 4 if portrait_phone else 4 / 3
    if abs(stage_ratio - expected_ratio) > 0.04:
        raise AssertionError(
            f"{label}: camera frame ratio {stage_ratio:.3f} is not stable at {expected_ratio:.3f}"
        )
    if metrics["stage"]["height"] > height * 0.76 + 1:
        raise AssertionError(f"{label}: camera frame is taller than the usable viewport")

    guide_ratio = metrics["guide"]["width"] / metrics["guide"]["height"]
    if not 0.68 <= guide_ratio <= 0.76:
        raise AssertionError(f"{label}: face guide is stretched to ratio {guide_ratio:.3f}")
    if metrics["guide"]["width"] > metrics["stage"]["width"] - 24:
        raise AssertionError(f"{label}: face guide is wider than the safe camera area")
    if metrics["guide"]["height"] > metrics["stage"]["height"] - 24:
        raise AssertionError(f"{label}: face guide is taller than the safe camera area")
    if metrics["objectFit"] != "contain":
        raise AssertionError(f"{label}: camera preview may crop the face ({metrics['objectFit']})")

    cue_items = metrics["cueItems"]
    if len(cue_items) != 4:
        raise AssertionError(f"{label}: expected four setup cues, found {len(cue_items)}")
    for index, item in enumerate(cue_items):
        if item["scrollWidth"] > item["clientWidth"] + 1:
            raise AssertionError(f"{label}: cue {index + 1} overflows horizontally: {item}")
        if item["scrollHeight"] > item["clientHeight"] + 1:
            raise AssertionError(f"{label}: cue {index + 1} overflows vertically: {item}")
    for index, first in enumerate(cue_items):
        for second in cue_items[index + 1 :]:
            if _intersects(first, second):
                raise AssertionError(f"{label}: setup cue boxes overlap")
    if metrics["documentWidth"] > metrics["viewportWidth"] + 1:
        raise AssertionError(f"{label}: page has horizontal overflow")
    if width <= 560:
        navigation = metrics["navigation"]
        if navigation is None or navigation["height"] > 96:
            raise AssertionError(f"{label}: mobile journey controls obscure too much content: {navigation}")
        for index, first in enumerate(metrics["navigationButtons"]):
            for second in metrics["navigationButtons"][index + 1 :]:
                if _intersects(first, second):
                    raise AssertionError(f"{label}: mobile journey buttons overlap")


def _run_engine(engine_name: str, engine: BrowserType) -> None:
    browser = engine.launch(headless=True)
    try:
        for label, width, height in VIEWPORTS:
            page = browser.new_page(viewport={"width": width, "height": height})
            _open_setup(page)
            _assert_geometry(page, f"{engine_name}/{label}", width, height)
            if engine_name == "chromium":
                SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
                page.locator(".capture-card").screenshot(
                    path=str(SCREENSHOT_DIR / f"{label}-camera.png"),
                    animations="disabled",
                )
                page.locator(".guided-session-control").screenshot(
                    path=str(SCREENSHOT_DIR / f"{label}-setup-cues.png"),
                    animations="disabled",
                )
            page.close()
    finally:
        browser.close()


def main() -> None:
    with sync_playwright() as playwright:
        for engine_name in ("chromium", "firefox", "webkit"):
            _run_engine(engine_name, getattr(playwright, engine_name))
    print(f"PASS responsive capture acceptance at {BASE_URL}")
    print(f"SCREENSHOTS {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
