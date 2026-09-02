"""State-transition acceptance for camera/upload recovery in the setup journey."""

from __future__ import annotations

import os

from playwright.sync_api import Browser, Page, expect, sync_playwright


BASE_URL = os.environ.get("ACCEPTANCE_BASE_URL", "http://127.0.0.1:4174").rstrip("/")
READY_RESPONSE = {
    "status": "ready",
    "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
    "candidate_id": "BLV9-009",
    "ensemble_members": 3,
    "preprocessing": "faces-to-shared-v9/v1",
}


def _prepare(page: Page, include_step_8: bool) -> None:
    page.route(
        "**/api/v1/facial-paralysis/ready",
        lambda route: route.fulfill(status=200, json=READY_RESPONSE),
    )
    page.add_init_script(
        """
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
        const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
        navigator.mediaDevices.getUserMedia = async (...args) => {
          const stream = await originalGetUserMedia(...args)
          window.__captureSourceRecoveryStream = stream
          return stream
        }
        """
    )
    page.goto(BASE_URL, wait_until="networkidle")
    choice = "Yes — include reanimation smile" if include_step_8 else "No — standard assessment"
    page.get_by_role("radio", name=choice, exact=False).check()
    page.get_by_role("button", name="Continue to camera setup").click()
    expect(page.get_by_role("heading", name="Set up the camera")).to_be_visible()


def _enable_camera(page: Page) -> None:
    page.get_by_role("button", name="Enable front camera").click()
    expect(page.get_by_role("button", name="Continue to recording")).to_be_enabled(timeout=10_000)


def _assert_camera_closed(page: Page) -> None:
    track_states = page.evaluate(
        "window.__captureSourceRecoveryStream?.getTracks().map(track => track.readyState) ?? []"
    )
    if not track_states or any(state != "ended" for state in track_states):
        raise AssertionError(f"camera tracks remained active after switching to upload: {track_states}")


def _run_case(browser: Browser, width: int, height: int, include_step_8: bool) -> None:
    context = browser.new_context(permissions=["camera"], viewport={"width": width, "height": height})
    page = context.new_page()
    _prepare(page, include_step_8)
    _enable_camera(page)

    for _ in range(5):
        page.get_by_role("tab", name="Upload from LifeLink").click()
        expect(page.get_by_role("button", name="Continue to recording")).to_have_count(0)
        expect(page.get_by_role("button", name="Return to live camera")).to_be_enabled()
        _assert_camera_closed(page)

        page.get_by_role("button", name="Return to live camera").click()
        expect(page.get_by_role("tab", name="Use this device")).to_have_attribute("aria-selected", "true")
        expect(page.get_by_role("button", name="Enable front camera")).to_be_enabled()
        _enable_camera(page)

    page.get_by_role("tab", name="Upload from LifeLink").click()
    with page.expect_file_chooser() as chooser_info:
        page.get_by_text("Browse video", exact=True).click()
    chooser_info.value.set_files([])
    expect(page.get_by_role("alert")).to_have_count(0)
    expect(page.get_by_role("button", name="Return to live camera")).to_be_enabled()

    page.get_by_label("Choose LifeLink Face video").set_input_files({
        "name": "not-a-video.png",
        "mimeType": "image/png",
        "buffer": b"not-video",
    })
    expect(page.get_by_role("alert")).to_contain_text("supported video")
    expect(page.get_by_role("button", name="Return to live camera")).to_be_enabled()
    page.get_by_role("button", name="Return to live camera").click()

    page.get_by_role("button", name="Back to preparation").click()
    choice = "Yes — include reanimation smile" if include_step_8 else "No — standard assessment"
    expect(page.get_by_role("radio", name=choice, exact=False)).to_be_checked()
    context.close()


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        )
        for width, height in ((1440, 1000), (390, 844)):
            for include_step_8 in (False, True):
                _run_case(browser, width, height, include_step_8)
        browser.close()

    print(f"PASS camera/upload recovery acceptance at {BASE_URL}: 4 cases, 20 recovery loops")


if __name__ == "__main__":
    main()
