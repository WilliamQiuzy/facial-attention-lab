"""Live Docker/browser acceptance for one guided Shared V9 recording.

The caller supplies a non-clinical Y4M camera fixture. The test records through
the browser's real MediaRecorder, posts through Nginx and the gateway, and
asserts either an accepted pinned result or a recoverable tracking rejection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


def _speech_mock() -> str:
    return """
    (() => {
      class TestUtterance {
        constructor(text) {
          this.text = text;
          this.rate = 1;
          this.pitch = 1;
          this.volume = 1;
          this.onstart = null;
          this.onend = null;
          this.onerror = null;
        }
      }
      let generation = 0;
      Object.defineProperty(window, 'SpeechSynthesisUtterance', {
        configurable: true,
        value: TestUtterance,
      });
      Object.defineProperty(window, 'speechSynthesis', {
        configurable: true,
        value: {
          speak(utterance) {
            const current = ++generation;
            window.setTimeout(() => {
              if (current !== generation) return;
              utterance.onstart?.();
              window.setTimeout(() => {
                if (current === generation) utterance.onend?.();
              }, 60);
            }, 0);
          },
          cancel() { generation += 1; },
        },
      });
    })();
    """


def _object_url_audit() -> str:
    return """
    (() => {
      const create = URL.createObjectURL.bind(URL);
      const revoke = URL.revokeObjectURL.bind(URL);
      const live = new Set();
      window.__objectUrlAudit = { created: 0, revoked: 0, live };
      URL.createObjectURL = (value) => {
        const url = create(value);
        window.__objectUrlAudit.created += 1;
        live.add(url);
        return url;
      };
      URL.revokeObjectURL = (url) => {
        window.__objectUrlAudit.revoked += 1;
        live.delete(url);
        revoke(url);
      };
    })();
    """


def _stub_response(steps: int) -> dict[str, object]:
    count = 7 if steps == 8 else 6
    ids = (
        "eyebrow_raise", "gentle_eye_closure", "tight_eye_squeeze",
        "relaxed_smile", "lip_pucker", "lower_teeth_show", "reanimated_smile",
    )[:count]
    v9_actions = (
        "BROW_RAISE", "EYE_GENTLE", "EYE_FORCEFUL", "SMILE_GENTLE",
        "LIP_PUCKER", "SHOW_BOTTOM_TEETH", "SMILE_FULL",
    )[:count]
    metrics = (
        ("brow_height_asymmetry_iod", "brow_height_change_from_rest_iod"),
        ("eye_aperture_asymmetry_iod", "residual_eye_aperture_iod", "eye_closure_change_from_rest_iod"),
        ("eye_aperture_asymmetry_iod", "residual_eye_aperture_iod", "eye_closure_change_from_rest_iod"),
        ("mouth_corner_vertical_asymmetry_iod", "mouth_corner_vertical_change_from_rest_iod"),
        ("mouth_corner_horizontal_asymmetry_iod", "mouth_width_change_from_rest_iod"),
        ("mouth_corner_vertical_asymmetry_iod", "lower_lip_change_from_rest_iod", "mouth_open_change_from_rest_iod"),
        ("mouth_corner_vertical_asymmetry_iod", "mouth_corner_vertical_change_from_rest_iod"),
    )[:count]
    return {
        "schema_version": "facial-paralysis-shared-v9-inference/v2",
        "model": {
            "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
            "candidate_id": "BLV9-009",
            "release_manifest_sha256": "81e396954090a0da6b99519909c1af15b6df5d1585ba27a642539352fe0a0c64",
            "ensemble_members": 3,
        },
        "preprocessing": {
            "version": "faces-to-shared-v9/v1",
            "face_landmarker_sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
            "mirror_method": "horizontal_flip_and_redetect",
            "protocol": "cue_aligned_action",
            "timing_source": "capture_event_log",
        },
        "quality": {
            "eligible": True,
            "actions_used": count,
            "optional_actions_unavailable": [] if count == 7 else ["reanimated_smile"],
            "actions": [
                {
                    "id": action_id,
                    "v9_action": v9_action,
                    "hold_start_ms": index * 4_000 + 4_500,
                    "hold_end_ms": index * 4_000 + 7_500,
                    "valid_samples": 26 + index,
                }
                for index, (action_id, v9_action) in enumerate(zip(ids, v9_actions))
            ],
        },
        "prediction": {
            "probability": 0.48,
            "member_probabilities": [0.45, 0.48, 0.51],
            "predicted_class": 0,
            "threshold": 0.5,
            "interpretation": "class_1_research_score_only",
            "endpoint_semantics": "meei_facial_palsy_vs_healthy_control_development_head",
            "class_0_label": "meei_healthy_control",
            "class_1_label": "meei_facial_palsy",
        },
        "report_evidence": {
            "normalization": "original_view_centered_eye_axis_aligned_interocular_scaled",
            "interpretation": "measured_movement_observation_not_causal_or_severity",
            "context_frame_method": "registered_hold_midpoint_not_model_selected",
            "actions": [
                {
                    "id": action_id,
                    "context_frame_ms": index * 4_000 + 6_000,
                    "observations": [
                        {
                            "metric": metric,
                            "value": round(0.01 * (metric_index + 1), 3),
                            "unit": "interocular_distance",
                        }
                        for metric_index, metric in enumerate(action_metrics)
                    ],
                }
                for index, (action_id, action_metrics) in enumerate(zip(ids, metrics))
            ],
        },
        "clinical_use_eligible": False,
    }


def run(
    base_url: str,
    camera_file: Path,
    steps: int,
    expected: str,
    screenshot: Path,
    *,
    pdf: Path | None,
    stub_success: bool,
    viewport_width: int,
    viewport_height: int,
) -> None:
    if not camera_file.is_file() or camera_file.suffix.casefold() != ".y4m":
        raise ValueError("camera fixture must be one existing Y4M file")
    if steps not in {7, 8} or expected not in {"accepted", "tracking-rejected"}:
        raise ValueError("steps/expected differ from the closed test matrix")
    if stub_success and expected != "accepted":
        raise ValueError("stub success is valid only for the accepted report path")

    console_errors: list[str] = []
    page_errors: list[str] = []
    inference_requests: list[str] = []
    route_failures: list[str] = []
    requested_urls: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-video-capture={camera_file}",
                "--mute-audio",
            ],
        )
        if not 320 <= viewport_width <= 2_560 or not 568 <= viewport_height <= 1_600:
            raise ValueError("viewport differs from the closed browser test range")
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height}
        )
        context.grant_permissions(["camera"], origin=base_url)
        context.add_init_script(_speech_mock())
        context.add_init_script(_object_url_audit())
        page = context.new_page()
        page.on("request", lambda request: requested_urls.append(request.url))
        if stub_success:
            def fulfill_inference(route, request) -> None:
                if request.method != "POST":
                    route.continue_()
                    return
                key = request.headers.get("idempotency-key", "")
                if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
                    route_failures.append("missing or malformed Idempotency-Key")
                inference_requests.append(key)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(_stub_response(steps), separators=(",", ":")),
                )
            page.route("**/api/v1/facial-paralysis/infer", fulfill_inference)
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        try:
            page.goto(base_url, wait_until="networkidle")
            expect(page.get_by_text("Analysis endpoint ready", exact=True)).to_be_visible(
                timeout=15_000
            )
            page.keyboard.press("Tab")
            skip_focus = page.evaluate(
                "({ activeClass: document.activeElement?.className, width: getComputedStyle(document.activeElement).width, clipPath: getComputedStyle(document.activeElement).clipPath })"
            )
            if skip_focus["activeClass"] != "skip-link" or skip_focus["width"] == "1px" or skip_focus["clipPath"] != "none":
                raise AssertionError(f"keyboard-focused skip link was not revealed: {skip_focus}")

            page.get_by_role("tab", name="Use this device").click()
            page.get_by_role("button", name="Enable front camera").click()
            start = page.get_by_role("button", name="Start guided recording")
            expect(start).to_be_visible(timeout=10_000)
            choice = "Include Step 8" if steps == 8 else "Step 8 not applicable"
            page.get_by_role("radio", name=choice).check()
            expect(start).to_be_enabled()
            start.click()

            expect(
                page.get_by_text(
                    "Guided recording complete. Review the video below before analysis.",
                    exact=True,
                )
            ).to_be_visible(timeout=50_000)
            expect(page.get_by_label("Recorded camera preview")).to_be_visible()
            page.get_by_role("checkbox", name="I confirm this is an authorized research endpoint.").check()
            run_button = page.get_by_role("button", name="Run research analysis")
            expect(run_button).to_be_enabled()
            if stub_success:
                run_button.evaluate("button => { button.click(); button.click(); }")
            else:
                run_button.click()

            if expected == "accepted":
                expect(page.get_by_text("Research report ready", exact=True)).to_be_visible(
                    timeout=120_000
                )
                expect(page.get_by_role("button", name="Run research analysis")).to_have_count(0)
                expect(page.get_by_role("link", name="View full research report")).to_be_visible()
                baseline_url_audit = page.evaluate("({ created: window.__objectUrlAudit.created, revoked: window.__objectUrlAudit.revoked, live: window.__objectUrlAudit.live.size })")
                page.get_by_role("link", name="View full research report").click()
                expect(page.get_by_role("heading", name="Research Movement Report")).to_be_visible()
                if page.evaluate("document.activeElement?.id") != "research-report-title":
                    raise AssertionError("opening the report did not move focus to its title")
                skip_style = page.locator(".skip-link").evaluate(
                    "element => ({ width: getComputedStyle(element).width, height: getComputedStyle(element).height, clipPath: getComputedStyle(element).clipPath })"
                )
                if skip_style != {"width": "1px", "height": "1px", "clipPath": "inset(50%)"}:
                    raise AssertionError(f"inactive skip link is not visually hidden: {skip_style}")
                expect(page.get_by_text("48 / 100").first).to_be_visible()
                expect(page.get_by_text("MEEI facial-movement classification score", exact=True)).to_be_visible()
                expect(page.get_by_text("Below MEEI research cutpoint", exact=True)).to_be_visible()
                expect(page.get_by_role("heading", name="How the model formed the score")).to_have_count(0)
                expect(page.get_by_role("heading", name="Recorded action evidence")).to_be_visible()
                expect(page.get_by_role("heading", name="Recording coverage")).to_be_visible()
                expect(page.get_by_text(f"Neutral baseline + all {7 if steps == 8 else 6} active movements", exact=True)).to_be_visible()
                expect(page.get_by_text(f"All {steps} recorded steps in this session were used", exact=False)).to_be_visible()
                expect(page.get_by_text("Measurements are scaled to the same eye-to-eye reference width", exact=False)).to_be_visible()
                expect(page.get_by_text("Side-to-side difference", exact=True).first).to_be_visible()
                expect(page.get_by_text("Change from neutral", exact=True).first).to_be_visible()
                expect(page.get_by_text("Action tracking", exact=True).first).to_be_visible()
                evidence_type = page.locator(".evidence-copy dl > div").first.evaluate(
                    "element => ({ label: parseFloat(getComputedStyle(element.querySelector('dt')).fontSize), kind: parseFloat(getComputedStyle(element.querySelector('dt > span')).fontSize), normalized: parseFloat(getComputedStyle(element.querySelector('dd > span')).fontSize), explanation: parseFloat(getComputedStyle(element.querySelector('dd > small')).fontSize) })"
                )
                if evidence_type["label"] < 14 or evidence_type["kind"] < 12 or evidence_type["normalized"] < 13 or evidence_type["explanation"] < 13:
                    raise AssertionError(f"evidence supporting text is too small: {evidence_type}")
                expect(page.get_by_role("heading", name="Clinical scale status")).to_have_count(0)
                expect(page.get_by_text("Validated response provenance", exact=True)).to_have_count(0)
                expect(page.get_by_role("heading", name="Interpretation limits")).to_have_count(0)
                expect(page.get_by_text("not calibrated on FACES recordings", exact=False)).to_have_count(0)
                expect(page.get_by_role("heading", name="Clinical review note")).to_be_visible()
                expect(page.get_by_text("MediaPipe 478-point facial landmarks", exact=False)).to_be_visible()
                if page.locator(".report-clinical-note svg").get_attribute("width") != "28":
                    raise AssertionError("clinical-review icon is not the approved 28px size")
                report_text = page.locator("body").inner_text().lower()
                if "shared v9" in report_text or "blv9-009" in report_text or "target release" in report_text:
                    raise AssertionError("the internal model release is visible in the report")
                expect(page.get_by_role("button", name="Run research analysis")).to_have_count(0)
                expect(page.get_by_role("button", name="Save PDF")).to_be_visible()
                expect(page.get_by_text("PDF includes the recorded evidence images", exact=False)).to_be_visible()
                action_boxes = page.locator(".report-action-control .button").evaluate_all(
                    "buttons => buttons.map(button => { const box = button.getBoundingClientRect(); return { top: box.top, left: box.left, width: box.width, height: box.height } })"
                )
                desktop_misaligned = (
                    viewport_width > 620
                    and max(box["top"] for box in action_boxes) - min(box["top"] for box in action_boxes) > 1
                )
                mobile_misaligned = (
                    viewport_width <= 620
                    and (
                        max(box["left"] for box in action_boxes) - min(box["left"] for box in action_boxes) > 1
                        or max(box["width"] for box in action_boxes) - min(box["width"] for box in action_boxes) > 1
                    )
                )
                if len(action_boxes) != 3 or desktop_misaligned or mobile_misaligned or max(box["height"] for box in action_boxes) - min(box["height"] for box in action_boxes) > 1:
                    raise AssertionError(f"report actions are not aligned: {action_boxes}")
                with page.expect_download() as pdf_download_info:
                    page.get_by_role("button", name="Save PDF").click()
                report_pdf_download = pdf_download_info.value
                if report_pdf_download.suggested_filename != "faces-research-movement-report.pdf":
                    raise AssertionError("Save PDF did not directly download the fixed report filename")
                if pdf is not None:
                    pdf.parent.mkdir(parents=True, exist_ok=True)
                    report_pdf_download.save_as(str(pdf))
                expect(page.locator(".evidence-frame img")).to_have_count(
                    7 if steps == 8 else 6,
                    timeout=20_000,
                )
                report_url_audit = page.evaluate("({ created: window.__objectUrlAudit.created, revoked: window.__objectUrlAudit.revoked, live: window.__objectUrlAudit.live.size })")
                if report_url_audit != {
                    "created": baseline_url_audit["created"] + 2,
                    "revoked": baseline_url_audit["revoked"] + 1,
                    "live": baseline_url_audit["live"] + 1,
                }:
                    raise AssertionError(
                        f"the report Blob URL lifecycle drifted: baseline={baseline_url_audit} audit={report_url_audit}"
                    )
                with page.expect_download() as download_info:
                    page.get_by_role("button", name="Download recorded video").first.click()
                if download_info.value.suggested_filename != "faces-research-recording.webm":
                    raise AssertionError("recording download did not use the de-identified fixed filename")
                page.wait_for_function(
                    "minimum => window.__objectUrlAudit.revoked >= minimum",
                    arg=report_url_audit["revoked"] + 1,
                )
                post_download_url_audit = page.evaluate("({ created: window.__objectUrlAudit.created, revoked: window.__objectUrlAudit.revoked, live: window.__objectUrlAudit.live.size })")
                if post_download_url_audit != {
                    "created": report_url_audit["created"] + 1,
                    "revoked": report_url_audit["revoked"] + 1,
                    "live": report_url_audit["live"],
                }:
                    raise AssertionError(f"recording download URL was not released: before={report_url_audit} after={post_download_url_audit}")
                report_url_audit = post_download_url_audit
                page.emulate_media(media="print")
                expect(page.locator(".evidence-frame").first).to_be_visible()
                page.emulate_media(media="screen")
                if stub_success:
                    if len(inference_requests) != 1:
                        raise AssertionError("rapid duplicate clicks submitted more than one request")
                    if route_failures:
                        raise AssertionError(f"stub route contract failures: {route_failures}")
            else:
                alert = page.get_by_role("alert")
                expect(alert).to_be_visible(timeout=120_000)
                expect(alert).to_contain_text("Face tracking was insufficient")
                expect(alert).to_contain_text("of 26 required samples")
                expect(page.get_by_label("Recorded camera preview")).to_be_visible()
                expect(
                    page.get_by_role("button", name="Clear recording and start over")
                ).to_be_enabled()

            screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot), full_page=True)
            if pdf is not None and expected != "accepted":
                raise AssertionError("PDF acceptance is valid only for an accepted report")
            if expected == "accepted":
                page.get_by_role("link", name="Back to session summary").click()
                expect(page.get_by_text("Research report ready", exact=True)).to_be_visible()
                summary_url_audit = page.evaluate("({ created: window.__objectUrlAudit.created, revoked: window.__objectUrlAudit.revoked, live: window.__objectUrlAudit.live.size })")
                if summary_url_audit != {
                    "created": report_url_audit["created"],
                    "revoked": report_url_audit["revoked"] + 1,
                    "live": report_url_audit["live"] - 1,
                }:
                    raise AssertionError(
                        f"closing the report did not revoke its Blob URL: report={report_url_audit} summary={summary_url_audit}"
                    )
                page.go_back()
                expect(page.get_by_role("heading", name="Research Movement Report")).to_be_visible()
                page.reload(wait_until="networkidle")
                expect(page.get_by_role("heading", name="Report not retained")).to_be_visible()
                expect(page.get_by_role("button", name="Run research analysis")).to_have_count(0)
                if stub_success and len(inference_requests) != 1:
                    raise AssertionError("report navigation or reload resubmitted inference")
            base = urlsplit(base_url)
            unexpected_origins = sorted({
                f"{parts.scheme}://{parts.netloc}"
                for url in requested_urls
                if (parts := urlsplit(url)).scheme in {"http", "https"}
                and (parts.scheme, parts.netloc) != (base.scheme, base.netloc)
            })
            if unexpected_origins:
                raise AssertionError(f"unexpected external browser requests: {unexpected_origins}")
            unexpected_console = [
                message
                for message in console_errors
                if not (
                    expected == "tracking-rejected"
                    and "server responded with a status of 422" in message
                )
            ]
            if unexpected_console or page_errors:
                raise AssertionError(
                    f"browser runtime errors: console={unexpected_console}, page={page_errors}"
                )
        finally:
            context.close()
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--camera-file", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int, choices=(7, 8))
    parser.add_argument(
        "--expected", required=True, choices=("accepted", "tracking-rejected")
    )
    parser.add_argument("--screenshot", required=True, type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument(
        "--stub-success",
        action="store_true",
        help="exercise the browser report using a strict synthetic v2 response",
    )
    parser.add_argument("--viewport-width", type=int, default=1_440)
    parser.add_argument("--viewport-height", type=int, default=1_000)
    args = parser.parse_args()
    run(
        args.base_url.rstrip("/"),
        args.camera_file,
        args.steps,
        args.expected,
        args.screenshot,
        pdf=args.pdf,
        stub_success=args.stub_success,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
    )
    print(f"PASS live full loop: steps={args.steps} expected={args.expected}")


if __name__ == "__main__":
    main()
