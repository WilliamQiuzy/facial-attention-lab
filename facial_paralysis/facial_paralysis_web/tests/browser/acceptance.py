"""Browser acceptance for the FACES research capture web application.

The web server is intentionally managed outside this script. Build with
``VITE_ENABLE_DEMONSTRATION=true`` so the explicitly labelled demonstration
path is available to acceptance testing.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import Browser, BrowserContext, Page, Route, expect, sync_playwright


BASE_URL = os.environ.get("ACCEPTANCE_BASE_URL", "http://127.0.0.1:4173").rstrip("/")
BASE_PARTS = urlsplit(BASE_URL)
if BASE_PARTS.scheme not in {"http", "https"} or not BASE_PARTS.netloc:
    raise ValueError(f"ACCEPTANCE_BASE_URL must be an HTTP(S) origin, received {BASE_URL!r}")
BASE_ORIGIN = f"{BASE_PARTS.scheme}://{BASE_PARTS.netloc}"
SCREENSHOT_DIR = Path(os.environ.get("ACCEPTANCE_SCREENSHOT_DIR", "/tmp/faces-browser-acceptance"))

VIEWPORTS = (
    ("desktop", 1440, 1000),
    ("tablet", 900, 1100),
    ("mobile", 390, 844),
)

# 16 x 16 blue, 200 ms VP8/WebM generated from a solid-color source. It contains
# no face, person, voice, patient identifier, or other PHI. Keeping the valid
# media bytes in the Playwright upload payload avoids a filesystem dependency.
SYNTHETIC_WEBM = base64.b64decode(
    "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAAK2"
    "EU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZUrmtTrIHWTbuMU6uEElTDZ1OsggEj"
    "TbuMU6uEHFO7a1OsggKg7AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAVSalmsCrXsYMPQkBNgIxMYXZmNjIuMy4xMDBXQYxMYXZmNjIuMy4xMDBE"
    "iYhAaQAAAAAAABZUrmvIrgEAAAAAAAA/14EBc8WIlkdYgkmqpP2cgQAitZyDdW5k"
    "iIEAhoVWX1ZQOIOBASPjg4QCYloA4JCwgRC6gRCagQJVsIRVuYEBElTDZ/tzc59j"
    "wIBnyJlFo4dFTkNPREVSRIeMTGF2ZjYyLjMuMTAwc3PWY8CLY8WIlkdYgkmqpP1n"
    "yKFFo4dFTkNPREVSRIeUTGF2YzYyLjExLjEwMCBsaWJ2cHhnyKFFo4hEVVJBVElP"
    "TkSHkzAwOjAwOjAwLjIwMDAwMDAwMAAfQ7Z1QPfngQCjvIEAAIDwAgCdASoQABAA"
    "AEcIhYWImYSIAgICdaoD+AP6Ag1NtgD++F9//5Z+wd7d/y0L/81zuQiN/5oMAKOs"
    "gQAoADECAAEQKAAYBz//8BBqf+WagbgA/vD6K/9lXwmR/7Vf/2tAcGq/0bijrIEA"
    "UADxAQABEBQAGAc4CDU/8s1A3AD+9r4H/wpWSgHv/Cmv/itYdZxf+JxAo6uBAHgA"
    "sQEAARAQABgAMD/0DADCAP74X3//ln7B3t3/LQv/zXO5CI3/mgwAo6uBAKAAsQEA"
    "ARAQABgAMD/0DADCAP74X3//ln7B3t3/LQv/zXO5CI3/mgwAHFO7a5G7j7OBALeK"
    "94EB8YIBo/CBAw=="
)
SYNTHETIC_UPLOAD = {
    "name": "synthetic-no-phi-faces-session.webm",
    "mimeType": "video/webm",
    "buffer": SYNTHETIC_WEBM,
}


def _allowed_url(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme in {"about", "blob", "data"}:
        return True
    if parts.scheme in {"http", "https"}:
        return parts.scheme == BASE_PARTS.scheme and parts.netloc == BASE_PARTS.netloc
    expected_websocket_scheme = "wss" if BASE_PARTS.scheme == "https" else "ws"
    return parts.scheme == expected_websocket_scheme and parts.netloc == BASE_PARTS.netloc


def _focus_style(page: Page, selector: str) -> dict[str, object]:
    return page.locator(selector).evaluate(
        """element => {
          const style = getComputedStyle(element)
          return {
            outlineStyle: style.outlineStyle,
            outlineWidth: style.outlineWidth,
            outlineColor: style.outlineColor,
            boxShadow: style.boxShadow,
            focusVisible: element.matches(':focus-visible'),
          }
        }"""
    )


def _has_visible_focus_indicator(style: dict[str, object]) -> bool:
    outline_width = float(str(style["outlineWidth"]).removesuffix("px") or 0)
    has_outline = style["outlineStyle"] != "none" and outline_width >= 2
    has_shadow = style["boxShadow"] != "none"
    return bool(style["focusVisible"] and (has_outline or has_shadow))


def _capture_screenshot(page: Page, label: str) -> str:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    path = SCREENSHOT_DIR / f"{safe_label}.png"
    scroll_position = page.evaluate("() => ({ x: window.scrollX, y: window.scrollY })")
    page.evaluate("document.activeElement?.blur()")
    page.locator(".skip-link").evaluate("element => { element.style.visibility = 'hidden' }")
    page.screenshot(path=str(path), full_page=True)
    page.evaluate("position => window.scrollTo(position.x, position.y)", scroll_position)
    print(f"SCREENSHOT {path}")
    return str(path)


def _assert_no_page_overflow(page: Page, checkpoint: str) -> None:
    metrics = page.evaluate(
        """() => {
          const viewportWidth = window.innerWidth
          const documentWidth = document.documentElement.scrollWidth
          const bodyWidth = document.body.scrollWidth
          const offenders = [...document.querySelectorAll('body *')]
            .map(element => {
              const rect = element.getBoundingClientRect()
              return {
                tag: element.tagName.toLowerCase(),
                className: typeof element.className === 'string' ? element.className : '',
                left: Math.round(rect.left * 10) / 10,
                right: Math.round(rect.right * 10) / 10,
              }
            })
            .filter(item => item.left < -1 || item.right > viewportWidth + 1)
            .slice(0, 8)
          return { viewportWidth, documentWidth, bodyWidth, offenders }
        }"""
    )
    widest = max(metrics["documentWidth"], metrics["bodyWidth"])
    if widest > metrics["viewportWidth"] + 1:
        raise AssertionError(
            f"horizontal page overflow at {checkpoint}: {metrics}; "
            "an intentionally scrollable inner workflow rail is allowed, page-level overflow is not"
        )


def _assert_hero_stat_clear(page: Page) -> None:
    collisions = page.locator(".hero-visual").evaluate(
        """hero => {
          const stat = hero.querySelector('.hero-stat')
          const value = stat?.querySelector('strong')
          const label = stat?.querySelector('span')
          const markers = [...hero.querySelectorAll('.face-orbit > b')]
          if (!stat || !value || !label || markers.length !== 8) {
            return [{ reason: 'missing hero statistic or protocol marker' }]
          }
          const intersects = (a, b) => (
            Math.min(a.right, b.right) > Math.max(a.left, b.left) &&
            Math.min(a.bottom, b.bottom) > Math.max(a.top, b.top)
          )
          const statBox = stat.getBoundingClientRect()
          const collisions = markers
            .map((marker, index) => ({ marker: index + 1, box: marker.getBoundingClientRect() }))
            .filter(item => intersects(statBox, item.box))
            .map(item => ({ reason: 'stat-marker overlap', marker: item.marker }))
          if (intersects(value.getBoundingClientRect(), label.getBoundingClientRect())) {
            collisions.push({ reason: 'stat value-label overlap' })
          }
          return collisions
        }"""
    )
    if collisions:
        raise AssertionError(f"hero statistic overlaps protocol content: {collisions}")


def _assert_active_coach_in_view(page: Page) -> None:
    selectors = {
        "camera": ".workspace.is-guided-active .camera-stage",
        "coach": ".patient-guidance",
        "stop": '[aria-label="Stop and discard guided recording"]',
    }
    metrics = page.evaluate(
        """selectors => {
          const viewport = { width: window.innerWidth, height: window.innerHeight }
          const boxes = Object.fromEntries(Object.entries(selectors).map(([name, selector]) => {
            const element = document.querySelector(selector)
            if (!element) return [name, null]
            const rect = element.getBoundingClientRect()
            return [name, {
              left: rect.left,
              right: rect.right,
              top: rect.top,
              bottom: rect.bottom,
              width: rect.width,
              height: rect.height,
            }]
          }))
          return { viewport, boxes }
        }""",
        selectors,
    )
    for name, box in metrics["boxes"].items():
        if box is None:
            raise AssertionError(f"active guided recording is missing its {name} region: {metrics}")
        if (
            box["left"] < -1
            or box["right"] > metrics["viewport"]["width"] + 1
            or box["top"] < -1
            or box["bottom"] > metrics["viewport"]["height"] + 1
        ):
            raise AssertionError(f"active {name} region is not fully visible in the viewport: {metrics}")

    def intersects(first: dict[str, float], second: dict[str, float]) -> bool:
        return (
            min(first["right"], second["right"]) > max(first["left"], second["left"])
            and min(first["bottom"], second["bottom"]) > max(first["top"], second["top"])
        )

    for first_name, second_name in (("camera", "coach"), ("camera", "stop"), ("coach", "stop")):
        if intersects(metrics["boxes"][first_name], metrics["boxes"][second_name]):
            raise AssertionError(
                f"active {first_name} and {second_name} regions overlap: {metrics}"
            )


def _assert_core_page(page: Page) -> None:
    expect(page).to_have_title("FACES Research Capture")
    expect(page.get_by_text("Research use only", exact=True)).to_be_visible()
    expect(page.get_by_text("FACES protocol · Source script v0.01", exact=True)).to_be_attached()
    expect(
        page.get_by_role("heading", name="Capture the full facial movement story.")
    ).to_be_visible()
    expect(page.get_by_role("heading", name="Bring in one complete session")).to_be_visible()
    expect(
        page.get_by_role("heading", name="Record and coach in one continuous flow.")
    ).to_be_visible()
    expect(
        page.get_by_role("heading", name="Validate the path before any result appears.")
    ).to_be_visible()
    expect(page.get_by_text("Interpretation boundary", exact=True)).to_have_count(0)
    expect(
        page.get_by_role("heading", name="Designed to support clinician review.")
    ).to_be_visible()
    expect(
        page.get_by_text("No patient data is persisted by this browser prototype.", exact=True)
    ).to_be_visible()
    expect(page.get_by_role("tab", name="Upload from LifeLink")).to_be_visible()
    expect(page.get_by_role("tab", name="Use this device")).to_be_visible()
    tabs = page.get_by_role("tab")
    expect(tabs.nth(0)).to_have_text("Use this device")
    expect(tabs.nth(0)).to_have_attribute("aria-selected", "true")
    expect(tabs.nth(1)).to_have_text("Upload from LifeLink")

    capture_layout = page.evaluate(
        """() => {
          const box = selector => {
            const rect = document.querySelector(selector).getBoundingClientRect()
            return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom }
          }
          return {
            viewportWidth: window.innerWidth,
            guidance: box('#protocol'),
            capture: box('.capture-card'),
            controls: box('.guided-session-control'),
          }
        }"""
    )
    guidance = capture_layout["guidance"]
    capture = capture_layout["capture"]
    controls = capture_layout["controls"]
    if capture_layout["viewportWidth"] > 1080:
        if guidance["left"] >= capture["left"] or abs(guidance["top"] - capture["top"]) > 2:
            raise AssertionError(f"guidance is not left of capture: {capture_layout}")
    elif guidance["bottom"] > capture["top"] + 2:
        raise AssertionError(f"guidance is not above capture in the responsive layout: {capture_layout}")
    if controls["top"] < max(guidance["bottom"], capture["bottom"]) - 2:
        raise AssertionError(f"recording controls are not below guidance and capture: {capture_layout}")

    instruction_palette = page.locator(".guided-flow li").first.evaluate(
        """element => {
          const rgb = getComputedStyle(element).color.match(/\\d+(?:\\.\\d+)?/g).map(Number)
          const label = element.querySelector('strong')
          const labelRgb = getComputedStyle(label).color.match(/\\d+(?:\\.\\d+)?/g).map(Number)
          return { rgb, labelRgb }
        }"""
    )
    if instruction_palette["rgb"][2] - instruction_palette["rgb"][0] < 40:
        raise AssertionError(f"instruction details lost their muted blue palette: {instruction_palette}")
    if instruction_palette["labelRgb"][2] - instruction_palette["labelRgb"][0] < 40:
        raise AssertionError(f"instruction labels lost their blue palette: {instruction_palette}")

    body_text = page.locator("body").inner_text().lower()
    if "house-brackmann" in body_text or "heatmap" in body_text:
        raise AssertionError("unsupported clinical grade or spatial heatmap language is visible")
    if "shared v9" in body_text or "blv9-009" in body_text or "target release" in body_text:
        raise AssertionError("the internal model release is visible in the product interface")

    hero_type = page.locator("#hero-title").evaluate(
        "element => ({ fontSize: parseFloat(getComputedStyle(element).fontSize), lineHeight: parseFloat(getComputedStyle(element).lineHeight) })"
    )
    if hero_type["lineHeight"] < hero_type["fontSize"] * 1.05:
        raise AssertionError(f"hero title lines are too tight: {hero_type}")
    guided_type = page.locator(".guided-flow li").first.evaluate(
        "element => { const label = element.querySelector('strong'); const detail = element.querySelector('span'); const icon = element.querySelector('svg').getBoundingClientRect(); return { label: parseFloat(getComputedStyle(label).fontSize), detail: parseFloat(getComputedStyle(detail).fontSize), iconWidth: icon.width, iconHeight: icon.height }; }"
    )
    if guided_type["label"] < 14 or guided_type["detail"] < 13 or min(guided_type["iconWidth"], guided_type["iconHeight"]) < 21:
        raise AssertionError(f"guided-flow labels or icons are too small: {guided_type}")

    page.evaluate("document.activeElement?.blur()")
    page.keyboard.press("Tab")
    skip_link = page.get_by_role("link", name="Skip to main content")
    expect(skip_link).to_be_focused()
    expect(skip_link).to_be_visible()
    page.wait_for_timeout(100)
    focus_style = _focus_style(page, ".skip-link")
    if not _has_visible_focus_indicator(focus_style):
        raise AssertionError(
            f"the keyboard-focused skip link has no visible focus indicator: {focus_style}"
        )
    page.evaluate("document.activeElement?.blur()")

    _assert_no_page_overflow(page, "initial render")
    _assert_hero_stat_clear(page)


def _upload_synthetic_video(page: Page) -> None:
    page.get_by_role("tab", name="Upload from LifeLink").click()
    file_input = page.get_by_label("Choose LifeLink Face video")
    file_input.set_input_files(SYNTHETIC_UPLOAD)
    expect(page.get_by_text(SYNTHETIC_UPLOAD["name"], exact=True)).to_be_visible()
    expect(page.get_by_text("Ready for protocol review", exact=True)).to_be_visible()


def _show_demonstration_result(page: Page) -> None:
    expect(page.get_by_text("Interface demonstration", exact=True)).to_be_visible()
    expect(
        page.locator(".demo-action")
    ).to_contain_text("Generated locally from file metadata; never model output")
    demo_button = page.get_by_role("button", name="Preview demonstration results")
    expect(demo_button).to_be_enabled()
    demo_button.click()

    expect(page.get_by_text("DEMONSTRATION - NOT MODEL OUTPUT", exact=True)).to_be_visible()
    expect(page.get_by_role("heading", name="Movement summary")).to_be_visible()
    expect(page.get_by_role("heading", name="Demonstration probability layout")).to_be_visible()
    expect(page.get_by_role("heading", name="Eye region")).to_be_visible()
    expect(page.get_by_role("heading", name="Mouth region")).to_be_visible()
    expect(page.get_by_text("not model output", exact=True)).to_be_visible()
    body_text = page.locator(".results-section").inner_text()
    if "Binary model output" in body_text or "Ordinal region output" in body_text:
        raise AssertionError("demonstration cards must not describe metadata values as model output")
    expect(page.locator(".workflow-rail .is-active strong")).to_have_text("Review")
    _assert_no_page_overflow(page, "demonstration result")
    viewport = page.viewport_size
    if viewport is not None:
        _capture_screenshot(
            page,
            f"demonstration-result-{viewport['width']}x{viewport['height']}",
        )


def _assert_demonstration_and_reset(page: Page) -> None:
    _upload_synthetic_video(page)
    _show_demonstration_result(page)

    page.get_by_role("button", name="Start a new session").click()
    expect(page.get_by_text("DEMONSTRATION - NOT MODEL OUTPUT", exact=True)).to_have_count(0)
    expect(page.get_by_role("heading", name="Movement summary")).to_have_count(0)
    expect(page.get_by_text(SYNTHETIC_UPLOAD["name"], exact=True)).to_have_count(0)
    expect(page.get_by_label("Choose LifeLink Face video")).to_have_count(0)
    expect(page.get_by_role("tab", name="Use this device")).to_have_attribute("aria-selected", "true")
    expect(page.get_by_role("button", name="Enable front camera")).to_be_visible()
    reset_color = page.locator(".guided-flow li").first.evaluate("element => getComputedStyle(element).color")
    if reset_color != "rgb(44, 111, 163)":
        raise AssertionError(f"new-session instructions changed away from accessible blue: {reset_color}")
    expect(page.get_by_role("button", name="Preview demonstration results")).to_be_disabled()
    expect(page.locator(".workflow-rail .is-active strong")).to_have_text("Prepare")
    _assert_no_page_overflow(page, "reset state")


def _assert_reload_drops_session_state(page: Page) -> None:
    _upload_synthetic_video(page)
    _show_demonstration_result(page)

    page.reload(wait_until="domcontentloaded")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_label("Choose LifeLink Face video")).to_have_count(0)
    expect(page.get_by_text(SYNTHETIC_UPLOAD["name"], exact=True)).to_have_count(0)
    expect(page.get_by_text("DEMONSTRATION - NOT MODEL OUTPUT", exact=True)).to_have_count(0)
    expect(page.get_by_role("heading", name="Movement summary")).to_have_count(0)
    expect(page.get_by_role("tab", name="Use this device")).to_have_attribute("aria-selected", "true")
    expect(page.get_by_role("button", name="Enable front camera")).to_be_visible()
    expect(page.get_by_role("button", name="Preview demonstration results")).to_be_disabled()
    expect(page.locator(".workflow-rail .is-active strong")).to_have_text("Prepare")
    _assert_no_page_overflow(page, "reload without session persistence")


def _assert_manual_voice_preview_cancel(page: Page) -> None:
    page.evaluate("() => { window.__acceptancePauseSpeech = true }")
    preview_button = page.get_by_role("button", name="Preview voice instruction")
    preview_button.click()
    stop_button = page.get_by_role("button", name="Stop voice preview")
    expect(stop_button).to_be_visible()
    stop_button.click()

    page.wait_for_timeout(150)
    expect(page.get_by_role("button", name="Preview voice instruction")).to_be_visible()
    expect(page.get_by_role("alert")).to_have_count(0)


def _prepare_guided_camera(page: Page, include_step_8: bool = False) -> None:
    upload_tab = page.get_by_role("tab", name="Upload from LifeLink")
    camera_tab = page.get_by_role("tab", name="Use this device")

    camera_tab.focus()
    page.keyboard.press("ArrowRight")
    expect(upload_tab).to_be_focused()
    expect(upload_tab).to_have_attribute("aria-selected", "true")
    page.keyboard.press("ArrowLeft")
    expect(camera_tab).to_be_focused()
    page.wait_for_timeout(100)
    focus_style = _focus_style(page, '[role="tab"][aria-selected="true"]')
    if not _has_visible_focus_indicator(focus_style):
        raise AssertionError(
            f"the keyboard-focused camera tab has no visible focus indicator: {focus_style}"
        )
    expect(camera_tab).to_have_attribute("aria-selected", "true")

    page.get_by_role("button", name="Enable front camera").click()
    start_button = page.get_by_role("button", name="Start guided recording")
    expect(start_button).to_be_visible(timeout=10_000)
    expect(
        page.get_by_text(re.compile(r"^Camera ready\. Resolve Step 8"))
    ).to_be_visible(timeout=10_000)
    expect(start_button).to_be_disabled()
    has_live_track = page.get_by_label("Live front camera preview").evaluate(
        """video => Boolean(
          video.srcObject &&
          video.srcObject.getVideoTracks().some(track => track.readyState === 'live')
        )"""
    )
    if not has_live_track:
        raise AssertionError("fake camera permission succeeded but no live video track was attached")
    page.get_by_label("Live front camera preview").evaluate(
        """video => {
          window.__acceptanceCameraTracks = [...video.srcObject.getVideoTracks()]
        }"""
    )

    choice_name = re.compile("include step 8", re.I) if include_step_8 else re.compile(
        "step 8 not applicable", re.I
    )
    page.get_by_role("radio", name=choice_name).check()
    expect(start_button).to_be_enabled()


def _assert_guided_camera_path(page: Page, include_step_8: bool) -> None:
    _prepare_guided_camera(page, include_step_8=include_step_8)
    start_button = page.get_by_role("button", name="Start guided recording")

    page.evaluate("() => { window.__acceptancePauseSpeech = true }")
    start_button.click()
    stop_button = page.get_by_role("button", name="Stop and discard guided recording")
    expect(stop_button).to_be_visible()
    source_tabs = page.locator(".source-tab")
    expect(source_tabs).to_have_count(2)
    for index in range(2):
        expect(source_tabs.nth(index)).to_be_disabled()
        expect(source_tabs.nth(index)).to_be_hidden()
    expect(page.get_by_role("button", name=re.compile("voice instruction", re.I))).to_have_count(0)

    patient_guide = page.get_by_role("region", name="Patient movement guidance")
    expect(patient_guide).to_be_visible()
    expect(patient_guide.locator(".patient-sequence-label strong")).to_have_text(
        "Step 1 of 8" if include_step_8 else "Step 1 of 7"
    )
    expect(patient_guide.get_by_role("heading", name="Neutral Expression (Repose)")).to_be_visible()
    expect(
        patient_guide.locator(".patient-guidance-copy > p")
    ).to_be_visible()
    expect(patient_guide.locator(".patient-guidance-copy > p")).to_contain_text(
        re.compile("Keep your face relaxed", re.I)
    )
    repose_avatar = patient_guide.get_by_role(
        "img", name="Neutral Expression (Repose) movement demonstration"
    )
    expect(repose_avatar).to_be_visible()
    expect(repose_avatar).to_have_class(re.compile(r"\bis-active\b"))
    expect(patient_guide.get_by_text("Voice prompt playing", exact=True)).to_be_visible()
    expect(patient_guide.locator('[aria-current="step"]')).to_have_attribute(
        "aria-label", "Repose"
    )
    page.wait_for_timeout(250)
    guided_scroll_position = page.evaluate("() => ({ x: window.scrollX, y: window.scrollY })")
    stop_button.focus()
    page.keyboard.press("Tab")
    page.keyboard.press("Shift+Tab")
    expect(stop_button).to_be_focused()
    page.wait_for_timeout(100)
    stop_focus = _focus_style(page, '[aria-label="Stop and discard guided recording"]')
    if not _has_visible_focus_indicator(stop_focus):
        raise AssertionError(
            f"the active Stop and discard control has no visible focus indicator: {stop_focus}"
        )
    page.evaluate("position => window.scrollTo(position.x, position.y)", guided_scroll_position)
    _assert_active_coach_in_view(page)
    _assert_no_page_overflow(page, "active guided voice prompt")

    viewport = page.viewport_size or {"width": 0, "height": 0}
    path_label = "eight-step" if include_step_8 else "seven-step"
    _capture_screenshot(
        page,
        f"guided-camera-active-speaking-{path_label}-{viewport['width']}x{viewport['height']}",
    )

    page.evaluate("() => window.__acceptanceReleaseSpeech?.()")
    expect(patient_guide.locator(".patient-timer")).to_have_attribute(
        "aria-label", "3 seconds remaining", timeout=2_000
    )
    expect(patient_guide.get_by_text("Hold the pose", exact=True)).to_be_visible()
    expect(repose_avatar).not_to_have_class(re.compile(r"\bis-active\b"))
    page.evaluate("() => { window.__acceptancePauseSpeech = true }")
    stable_snapshot = patient_guide.evaluate(
        """guide => {
          const title = guide.querySelector('h2')
          const avatar = guide.querySelector('.movement-avatar')
          const titleBox = title.getBoundingClientRect()
          const avatarBox = avatar.getBoundingClientRect()
          return {
            title: title.textContent,
            action: avatar.dataset.action,
            titleBox: { x: titleBox.x, y: titleBox.y, width: titleBox.width, height: titleBox.height },
            avatarBox: { x: avatarBox.x, y: avatarBox.y, width: avatarBox.width, height: avatarBox.height },
          }
        }"""
    )
    for remaining in (2, 1):
        page.wait_for_function(
            """remaining => {
              const guide = document.querySelector('[aria-label="Patient movement guidance"]')
              const timer = guide?.querySelector('.patient-timer')
              if (timer?.getAttribute('aria-label') !== `${remaining} seconds remaining`) return false
              const title = guide.querySelector('h2')
              const avatar = guide.querySelector('.movement-avatar')
              const titleBox = title.getBoundingClientRect()
              const avatarBox = avatar.getBoundingClientRect()
              window.__acceptanceCountdownSnapshot = {
                title: title.textContent,
                action: avatar.dataset.action,
                titleBox: { x: titleBox.x, y: titleBox.y, width: titleBox.width, height: titleBox.height },
                avatarBox: { x: avatarBox.x, y: avatarBox.y, width: avatarBox.width, height: avatarBox.height },
              }
              return true
            }""",
            arg=remaining,
            timeout=2_000,
        )
        current_snapshot = page.evaluate("() => window.__acceptanceCountdownSnapshot")
        if current_snapshot["title"] != stable_snapshot["title"] or current_snapshot["action"] != stable_snapshot["action"]:
            raise AssertionError(
                f"movement cue changed during the 3-second hold: {stable_snapshot} -> {current_snapshot}"
            )
        for box_name in ("titleBox", "avatarBox"):
            drift = max(
                abs(current_snapshot[box_name][field] - stable_snapshot[box_name][field])
                for field in ("x", "y", "width", "height")
            )
            if drift > 2:
                raise AssertionError(
                    f"{box_name} shifted {drift:.2f}px during the countdown: "
                    f"{stable_snapshot} -> {current_snapshot}"
                )
    expect(patient_guide.get_by_role("heading", name="Eyebrow Raise")).to_be_visible(timeout=2_000)
    expect(patient_guide.locator('[aria-current="step"]')).to_have_attribute("aria-label", "Brows")
    expect(
        patient_guide.get_by_role("img", name="Eyebrow Raise movement demonstration")
    ).to_be_visible()
    page.evaluate("() => window.__acceptanceReleaseSpeech?.()")
    _assert_active_coach_in_view(page)

    expect(page.get_by_role("status")).to_contain_text("Guided recording complete", timeout=20_000)
    expect(page.locator(".camera-ready")).to_contain_text("faces-capture-", timeout=10_000)
    expect(page.locator(".camera-ready")).to_contain_text("is ready.")
    expect(page.get_by_label("Recorded camera preview")).to_be_visible()
    expect(page.get_by_label("Live front camera preview")).to_have_count(0)
    tracks_released = page.evaluate(
        """() => window.__acceptanceCameraTracks.every(track => track.readyState === 'ended')"""
    )
    if not tracks_released:
        raise AssertionError("camera tracks remained live after guided recording completed")

    spoken = page.evaluate("() => window.__acceptanceSpokenInstructions")
    expected_prompts = 8 if include_step_8 else 7
    if len(spoken) != expected_prompts:
        raise AssertionError(
            f"expected {expected_prompts} automatic voice prompts, received {len(spoken)}: {spoken}"
        )
    spoken_text = " ".join(spoken).lower()
    if include_step_8 and "reanimation surgery" not in spoken[-1].lower():
        raise AssertionError(f"Step 8 was not spoken in the eight-step path: {spoken}")
    if not include_step_8 and "reanimation surgery" in spoken_text:
        raise AssertionError("Step 8 was spoken after the clinician marked it not applicable")
    expected_last_phrase = "reanimation surgery" if include_step_8 else "bottom teeth"
    if "Keep your face relaxed" not in spoken[0] or expected_last_phrase not in spoken[-1]:
        raise AssertionError(f"guided prompts were not spoken in protocol order: {spoken}")

    _assert_no_page_overflow(page, "guided camera recording")
    _capture_screenshot(
        page,
        f"guided-camera-{path_label}-{viewport['width']}x{viewport['height']}",
    )

    page.get_by_role("button", name="Record again").click()
    expect(page.get_by_label("Recorded camera preview")).to_have_count(0)
    expect(page.locator(".camera-ready")).to_have_count(0)
    expect(page.get_by_role("button", name="Enable camera")).to_be_visible()
    expect(page.get_by_role("button", name="Preview demonstration results")).to_be_disabled()
    expect(page.locator(".workflow-rail .is-active strong")).to_have_text("Prepare")
    expect(page.get_by_role("radio", name=re.compile("step 8 not applicable", re.I))).not_to_be_checked()
    expect(page.get_by_role("radio", name=re.compile("include step 8", re.I))).not_to_be_checked()


def _assert_guided_camera_cancel(page: Page) -> None:
    _prepare_guided_camera(page, include_step_8=True)
    page.get_by_role("button", name="Start guided recording").click()
    stop_button = page.get_by_role("button", name="Stop and discard guided recording")
    expect(stop_button).to_be_visible()
    stop_button.click()

    expect(page.get_by_text("Guided recording stopped. The incomplete video was discarded.")).to_be_visible()
    expect(page.get_by_label("Recorded camera preview")).to_have_count(0)
    expect(page.locator(".camera-ready")).to_have_count(0)
    expect(page.get_by_role("button", name="Preview demonstration results")).to_be_disabled()
    expect(page.get_by_role("tab", name="Upload from LifeLink")).to_be_enabled()
    expect(page.get_by_role("tab", name="Use this device")).to_be_enabled()
    expect(page.get_by_role("radio", name=re.compile("step 8 not applicable", re.I))).to_be_enabled()
    tracks_released = page.evaluate(
        """() => window.__acceptanceCameraTracks.every(track => track.readyState === 'ended')"""
    )
    if not tracks_released:
        raise AssertionError("camera tracks remained live after the guided recording was discarded")

    page.evaluate("() => window.__acceptanceLastUtterance?.onend?.()")
    page.wait_for_timeout(500)
    expect(page.get_by_label("Recorded camera preview")).to_have_count(0)
    expect(page.locator(".camera-ready")).to_have_count(0)
    expect(page.get_by_role("button", name="Preview demonstration results")).to_be_disabled()
    _assert_no_page_overflow(page, "cancelled guided camera recording")


def _new_instrumented_page(
    context: BrowserContext,
    case_name: str,
    runtime_errors: list[str],
    external_urls: list[str],
) -> Page:
    page = context.new_page()

    def record_console_error(message: object) -> None:
        if getattr(message, "type", "") == "error":
            runtime_errors.append(f"{case_name} console.error: {getattr(message, 'text', message)}")

    page.on("console", record_console_error)
    page.on("pageerror", lambda error: runtime_errors.append(f"{case_name} pageerror: {error}"))
    page.on(
        "websocket",
        lambda websocket: external_urls.append(f"WEBSOCKET {websocket.url}")
        if not _allowed_url(websocket.url)
        else None,
    )
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.wait_for_load_state("networkidle")
    return page


def _run_case(
    context: BrowserContext,
    case_name: str,
    check: Callable[[Page], None],
    failures: list[str],
    runtime_errors: list[str],
    external_urls: list[str],
) -> None:
    page: Page | None = None
    try:
        page = _new_instrumented_page(context, case_name, runtime_errors, external_urls)
        check(page)
        print(f"PASS {case_name}")
    except Exception as error:  # Keep running independent viewports and scenarios.
        failures.append(f"{case_name}: {error}")
        print(f"FAIL {case_name}: {error}")
        if page is not None:
            try:
                _capture_screenshot(page, f"failure-{case_name}")
            except Exception as screenshot_error:
                failures.append(f"{case_name} screenshot capture: {screenshot_error}")
    finally:
        if page is not None:
            page.close()


def _run_viewport(browser: Browser, name: str, width: int, height: int, failures: list[str]) -> None:
    runtime_errors: list[str] = []
    external_urls: list[str] = []
    context = browser.new_context(
        viewport={"width": width, "height": height},
        reduced_motion="reduce",
        locale="en-US",
    )
    context.add_init_script(
        """
        (() => {
          const nativeSetTimeout = window.setTimeout.bind(window)
          let guidedClock = 0
          Object.defineProperty(performance, 'now', {
            configurable: true,
            value: () => guidedClock,
          })
          window.setTimeout = (callback, delay = 0, ...args) => {
            if (delay === 1000) {
              return nativeSetTimeout(() => {
                guidedClock += 1000
                callback(...args)
              }, 400)
            }
            return nativeSetTimeout(callback, delay, ...args)
          }

          class AcceptanceUtterance {
            constructor(text) {
              this.text = text
              this.rate = 1
              this.pitch = 1
              this.volume = 1
              this.onstart = null
              this.onend = null
              this.onerror = null
            }
          }

          let speechGeneration = 0
          window.__acceptanceSpokenInstructions = []
          window.__acceptancePauseSpeech = false
          window.__acceptancePendingSpeechEnd = null
          window.__acceptanceReleaseSpeech = () => {
            window.__acceptancePauseSpeech = false
            const finish = window.__acceptancePendingSpeechEnd
            window.__acceptancePendingSpeechEnd = null
            finish?.()
          }
          Object.defineProperty(window, 'SpeechSynthesisUtterance', {
            configurable: true,
            value: AcceptanceUtterance,
          })
          Object.defineProperty(window, 'speechSynthesis', {
            configurable: true,
            value: {
              speak(utterance) {
                const generation = ++speechGeneration
                window.__acceptanceLastUtterance = utterance
                window.__acceptanceSpokenInstructions.push(utterance.text)
                nativeSetTimeout(() => {
                  if (generation !== speechGeneration) return
                  utterance.onstart?.()
                  const finishSpeech = () => {
                    if (generation !== speechGeneration) return
                    utterance.onend?.()
                  }
                  if (window.__acceptancePauseSpeech) {
                    window.__acceptancePendingSpeechEnd = finishSpeech
                  } else {
                    nativeSetTimeout(finishSpeech, 60)
                  }
                }, 0)
              },
              cancel() {
                const interruptedUtterance = window.__acceptanceLastUtterance
                window.__acceptanceLastUtterance = null
                speechGeneration += 1
                window.__acceptancePendingSpeechEnd = null
                nativeSetTimeout(() => {
                  interruptedUtterance?.onerror?.({ error: 'interrupted' })
                }, 0)
              },
            },
          })
        })()
        """
    )
    context.grant_permissions(["camera"], origin=BASE_ORIGIN)

    def guard_request(route: Route) -> None:
        request = route.request
        if _allowed_url(request.url):
            route.continue_()
            return
        external_urls.append(f"{request.method} {request.resource_type} {request.url}")
        route.abort("blockedbyclient")

    context.route("**/*", guard_request)
    try:
        _run_case(
            context,
            f"{name} {width}x{height} core-responsive-keyboard",
            _assert_core_page,
            failures,
            runtime_errors,
            external_urls,
        )
        _run_case(
            context,
            f"{name} {width}x{height} upload-demonstration-results-reset",
            _assert_demonstration_and_reset,
            failures,
            runtime_errors,
            external_urls,
        )
        _run_case(
            context,
            f"{name} {width}x{height} reload-clears-session",
            _assert_reload_drops_session_state,
            failures,
            runtime_errors,
            external_urls,
        )
        if name == "desktop":
            _run_case(
                context,
                f"{name} {width}x{height} manual-voice-preview-cancel",
                _assert_manual_voice_preview_cancel,
                failures,
                runtime_errors,
                external_urls,
            )
            _run_case(
                context,
                f"{name} {width}x{height} guided-camera-seven-step-recording",
                lambda page: _assert_guided_camera_path(page, include_step_8=False),
                failures,
                runtime_errors,
                external_urls,
            )
            _run_case(
                context,
                f"{name} {width}x{height} guided-camera-eight-step-recording",
                lambda page: _assert_guided_camera_path(page, include_step_8=True),
                failures,
                runtime_errors,
                external_urls,
            )
            _run_case(
                context,
                f"{name} {width}x{height} guided-camera-cancel",
                _assert_guided_camera_cancel,
                failures,
                runtime_errors,
                external_urls,
            )
        else:
            _run_case(
                context,
                f"{name} {width}x{height} guided-camera-eight-step-recording",
                lambda page: _assert_guided_camera_path(page, include_step_8=True),
                failures,
                runtime_errors,
                external_urls,
            )
    finally:
        context.close()

    if runtime_errors:
        failures.append(f"{name} runtime errors:\n  " + "\n  ".join(sorted(set(runtime_errors))))
    if external_urls:
        failures.append(
            f"{name} unexpected external requests (blocked when routable):\n  "
            + "\n  ".join(sorted(set(external_urls)))
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local browser acceptance for FACES Research Capture.")
    parser.parse_args()
    failures: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                "--mute-audio",
            ],
        )
        try:
            for name, width, height in VIEWPORTS:
                _run_viewport(browser, name, width, height, failures)
        finally:
            browser.close()

    if failures:
        raise AssertionError("\n\n".join(failures))
    print(f"PASS all browser acceptance checks against {BASE_URL}")


if __name__ == "__main__":
    main()
