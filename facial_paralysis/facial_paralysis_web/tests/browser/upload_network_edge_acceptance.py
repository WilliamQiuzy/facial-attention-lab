"""Upload, sidecar, and inference-recovery edge acceptance in a real browser."""

from __future__ import annotations

import argparse
import hashlib
import json

from playwright.sync_api import Browser, Page, expect, sync_playwright

from live_full_loop import _stub_response


VIDEO_BYTES = b"non-clinical synthetic browser fixture"
VIDEO_SHA256 = hashlib.sha256(VIDEO_BYTES).hexdigest()
READY_RESPONSE = {
    "status": "ready",
    "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
    "candidate_id": "BLV9-009",
    "ensemble_members": 3,
    "preprocessing": "faces-to-shared-v9/v1",
}
ACTION_IDS = (
    "neutral_repose",
    "eyebrow_raise",
    "gentle_eye_closure",
    "tight_eye_squeeze",
    "relaxed_smile",
    "lip_pucker",
    "lower_teeth_show",
    "reanimated_smile",
)


def _sidecar(*, steps: int, digest: str = VIDEO_SHA256) -> bytes:
    return json.dumps(
        {
            "schema_version": "faces-action-timeline/v1",
            "script_version": "faces-script/24-004956-v1",
            "recording_sha256": digest,
            "timing_source": "capture_event_log",
            "recording_duration_ms": steps * 4_000,
            "actions": [
                {
                    "action": action,
                    "status": "completed",
                    "prompt_start_ms": index * 4_000,
                    "hold_start_ms": index * 4_000 + 500,
                    "hold_end_ms": index * 4_000 + 3_500,
                    "completion_ms": index * 4_000 + 3_750,
                }
                for index, action in enumerate(ACTION_IDS[:steps])
            ],
        },
        separators=(",", ":"),
    ).encode()


def _new_page(context) -> tuple[Page, list[str], list[str]]:
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


def _open_upload(page: Page, base_url: str) -> None:
    page.route(
        "**/api/v1/facial-paralysis/ready",
        lambda route: route.fulfill(status=200, json=READY_RESPONSE),
    )
    page.goto(base_url, wait_until="networkidle")
    page.get_by_role("radio", name="Step 8 not applicable", exact=False).check()
    page.get_by_role("button", name="Continue to camera setup").click()
    expect(page.get_by_text("Analysis endpoint ready", exact=True)).to_be_visible()
    page.get_by_role("tab", name="Upload from LifeLink").click()


def _upload_video(page: Page) -> None:
    page.get_by_label("Choose LifeLink Face video").set_input_files(
        {"name": "faces-session.webm", "mimeType": "video/webm", "buffer": VIDEO_BYTES}
    )
    expect(page.get_by_role("heading", name="Review the recording and run analysis")).to_be_visible()


def _upload_timeline(page: Page, source: bytes, name: str = "faces-session.timeline.json") -> None:
    page.get_by_label("Choose FACES action timeline").set_input_files(
        {"name": name, "mimeType": "application/json", "buffer": source}
    )


def _authorize(page: Page) -> None:
    checkbox = page.get_by_role(
        "checkbox", name="I confirm this is an authorized research endpoint."
    )
    if not checkbox.is_checked():
        checkbox.check()


def _assert_runtime(
    console_errors: list[str], page_errors: list[str], *, expected_statuses: tuple[int, ...] = ()
) -> None:
    remaining = list(console_errors)
    for status in expected_statuses:
        matches = [message for message in remaining if f"status of {status}" in message]
        if len(matches) != 1:
            raise AssertionError(
                f"expected one browser network message for HTTP {status}: {console_errors}"
            )
        remaining.remove(matches[0])
    if remaining or page_errors:
        raise AssertionError(f"browser runtime errors: console={remaining}, page={page_errors}")


def _case_file_and_timeline_validation(browser: Browser, base_url: str) -> None:
    context = browser.new_context()
    page, console_errors, page_errors = _new_page(context)
    _open_upload(page, base_url)

    with page.expect_file_chooser() as chooser_info:
        page.get_by_text("Browse video", exact=True).click()
    chooser_info.value.set_files([])
    expect(page.get_by_role("alert")).to_have_count(0)

    page.get_by_label("Choose LifeLink Face video").set_input_files(
        {"name": "patient.png", "mimeType": "image/png", "buffer": b"not-video"}
    )
    expect(page.get_by_role("alert")).to_contain_text("supported video")
    page.get_by_label("Choose LifeLink Face video").set_input_files(
        {"name": "empty.webm", "mimeType": "video/webm", "buffer": b""}
    )
    expect(page.get_by_role("alert")).to_contain_text("selected video is empty")

    _upload_video(page)
    expect(page.get_by_role("button", name="Run research analysis")).to_be_disabled()
    _upload_timeline(page, b"{bad-json")
    expect(page.get_by_role("alert")).to_contain_text("invalid JSON")
    _upload_timeline(page, b"x" * (256 * 1024 + 1))
    expect(page.get_by_role("alert")).to_contain_text("bounded JSON")

    _upload_timeline(page, _sidecar(steps=7, digest="0" * 64))
    _authorize(page)
    inference_requests: list[str] = []
    page.route(
        "**/api/v1/facial-paralysis/infer",
        lambda route, request: (
            inference_requests.append(request.url),
            route.fulfill(status=500, json={"detail": {"private_path": "/patient/name"}}),
        ),
    )
    page.get_by_role("button", name="Run research analysis").click()
    expect(page.get_by_role("alert")).to_contain_text("recording SHA-256 differs")
    if inference_requests:
        raise AssertionError("digest-mismatched upload reached the inference endpoint")

    _upload_timeline(page, _sidecar(steps=7))
    _authorize(page)
    expect(page.get_by_role("button", name="Run research analysis")).to_be_enabled()
    _assert_runtime(console_errors, page_errors)
    context.close()


def _case_retryable_server_failures(browser: Browser, base_url: str) -> None:
    context = browser.new_context()
    page, console_errors, page_errors = _new_page(context)
    attempts = 0

    def infer(route) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            route.fulfill(
                status=500,
                json={"detail": {"private_path": "/patient/name/video.mov"}},
            )
        elif attempts == 2:
            route.fulfill(
                status=200,
                json={"private_path": "/patient/name/video.mov"},
            )
        else:
            route.fulfill(status=200, json=_stub_response(7))

    page.route("**/api/v1/facial-paralysis/infer", infer)
    _open_upload(page, base_url)
    _upload_video(page)
    _upload_timeline(page, _sidecar(steps=7))
    _authorize(page)
    run = page.get_by_role("button", name="Run research analysis")

    run.click()
    expect(page.get_by_role("alert")).to_have_text(
        "Research endpoint returned HTTP 500. No result was accepted."
    )
    expect(page.locator("body")).not_to_contain_text("/patient/name")
    expect(run).to_be_enabled()
    expect(page.get_by_label("Selected recording preview")).to_be_visible()

    run.click()
    expect(page.get_by_role("alert")).to_have_text(
        "The analysis response did not pass validation. The same recording is still available; wait briefly and retry."
    )
    expect(page.locator("body")).not_to_contain_text("/patient/name")
    expect(run).to_be_enabled()

    run.evaluate("button => { button.click(); button.click(); }")
    expect(page.get_by_text("Analysis report ready", exact=True)).to_be_visible()
    if attempts != 3:
        raise AssertionError(f"duplicate or missing inference requests: {attempts}")
    _assert_runtime(console_errors, page_errors, expected_statuses=(500,))
    context.close()


def _case_nonretryable_capture_rejection(browser: Browser, base_url: str) -> None:
    context = browser.new_context()
    page, console_errors, page_errors = _new_page(context)
    attempts = 0

    def reject(route) -> None:
        nonlocal attempts
        attempts += 1
        route.fulfill(status=422, json={"detail": {"code": "video_timing_mismatch"}})

    page.route("**/api/v1/facial-paralysis/infer", reject)
    _open_upload(page, base_url)
    _upload_video(page)
    _upload_timeline(page, _sidecar(steps=7))
    _authorize(page)
    page.get_by_role("button", name="Run research analysis").click()

    expect(page.get_by_role("alert")).to_contain_text(
        "recorded video timing did not match the guided action timeline"
    )
    expect(page.get_by_role("button", name="New recording required")).to_be_disabled()
    expect(page.get_by_role("button", name="Download recorded video")).to_be_enabled()
    expect(page.get_by_role("button", name="Clear recording and start over")).to_be_enabled()
    page.get_by_role("button", name="New recording required").evaluate(
        "button => { button.click(); button.click(); }"
    )
    if attempts != 1:
        raise AssertionError(f"permanent rejection was resubmitted: {attempts}")
    _assert_runtime(console_errors, page_errors, expected_statuses=(422,))
    context.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    args = parser.parse_args()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            _case_file_and_timeline_validation(browser, args.base_url.rstrip("/"))
            _case_retryable_server_failures(browser, args.base_url.rstrip("/"))
            _case_nonretryable_capture_rejection(browser, args.base_url.rstrip("/"))
        finally:
            browser.close()

    print("PASS upload/network edge acceptance: 3 cases")


if __name__ == "__main__":
    main()
