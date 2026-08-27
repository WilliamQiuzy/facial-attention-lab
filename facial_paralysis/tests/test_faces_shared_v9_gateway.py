"""HTTP boundary tests for the Facial Process Web Shared V9 gateway."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from test_faces_shared_v9_pipeline import (  # noqa: E402
    _explained_prediction,
    _mesh_stream,
    _payloads,
)
from src.deployment.faces_shared_v9_gateway import (  # noqa: E402
    FACE_LANDMARKER_SHA256,
    GatewayPreparedCapture,
    MediaPipeCaptureProcessor,
    SharedV9HttpClient,
    create_app,
)
from src.deployment import faces_shared_v9_gateway as gateway_module  # noqa: E402
from src.deployment.shared_v9_attribution import (  # noqa: E402
    load_attribution_request_npz,
)
from src.models.dense_clinical_shared_encoder_v1 import ACTION_VOCAB  # noqa: E402
from src.preprocessing.faces_shared_v9_pipeline import (  # noqa: E402
    build_v9_action_arrays,
    parse_capture_evidence,
)
from src.preprocessing import faces_shared_v9_pipeline as pipeline_module  # noqa: E402


def _processor(video, filename, manifest, timeline):
    if filename != "capture.mp4":
        raise ValueError("unexpected fixture filename")
    evidence = parse_capture_evidence(video, manifest, timeline)
    stream = _mesh_stream(evidence)
    prepared = build_v9_action_arrays(
        evidence,
        frame_timestamps_ms=stream[0],
        source_frame_indices=stream[1],
        original_meshes=stream[2],
        mirrored_meshes=stream[3],
        pair_valid_mask=stream[4],
        source_fps=stream[5],
    )
    return GatewayPreparedCapture(
        evidence=evidence,
        prepared=prepared,
        face_landmarker_sha256="6" * 64,
    )


def _inference(payload: bytes):
    arrays, neutral_original, neutral_mirrored = load_attribution_request_npz(
        payload, protocol="cue_aligned_action"
    )
    c_actions = tuple(ACTION_VOCAB[int(code)] for code in arrays["action_codes"])
    assert neutral_original.shape == (len(c_actions), 110)
    assert neutral_mirrored.shape == (len(c_actions), 110)
    return _explained_prediction(c_actions)


def _client(inference=_inference):
    client = TestClient(create_app(
        processor=_processor,
        inference_client=inference,
        readiness_client=lambda: {
            "status": "ready",
            "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
            "candidate_id": "BLV9-009",
            "ensemble_members": 3,
        },
    ))
    video, manifest, timeline = _payloads()
    client.headers["Idempotency-Key"] = gateway_module._idempotency_key(
        video, manifest, timeline
    )
    return client


def _idempotency_headers(video: bytes, manifest: bytes, timeline: bytes):
    return {
        "Idempotency-Key": gateway_module._idempotency_key(
            video, manifest, timeline
        )
    }


def test_gateway_replays_one_single_flight_result_and_rejects_key_drift(c: Check):
    processor_calls = 0
    inference_calls = 0
    inference_entered = threading.Event()
    release_inference = threading.Event()

    def counted_processor(*args):
        nonlocal processor_calls
        processor_calls += 1
        return _processor(*args)

    def counted_inference(payload):
        nonlocal inference_calls
        inference_calls += 1
        inference_entered.set()
        release_inference.wait(timeout=5)
        return _inference(payload)

    client = TestClient(create_app(
        processor=counted_processor,
        inference_client=counted_inference,
        readiness_client=lambda: {
            "status": "ready",
            "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
            "candidate_id": "BLV9-009",
            "ensemble_members": 3,
        },
    ))
    video, manifest, timeline = _payloads()
    headers = _idempotency_headers(video, manifest, timeline)

    def submit():
        return client.post(
            "/api/v1/facial-paralysis/infer",
            files={"video": ("capture.mp4", video, "video/mp4")},
            data={
                "manifest": manifest.decode("utf-8"),
                "timeline": timeline.decode("utf-8"),
            },
            headers=headers,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(submit)
        c.true(inference_entered.wait(timeout=5))
        second = executor.submit(submit)
        release_inference.set()
        first_response = first.result(timeout=10)
        second_response = second.result(timeout=10)
    c.eq(first_response.status_code, 200)
    c.eq(second_response.status_code, 200)
    c.eq(first_response.json(), second_response.json())
    c.eq(processor_calls, 1)
    c.eq(inference_calls, 1)

    replay = submit()
    c.eq(replay.json(), first_response.json())
    c.eq(processor_calls, 1)
    c.eq(inference_calls, 1)

    missing = client.post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", video, "video/mp4")},
        data={
            "manifest": manifest.decode("utf-8"),
            "timeline": timeline.decode("utf-8"),
        },
    )
    c.eq(missing.status_code, 428)
    c.eq(missing.json(), {"detail": {"code": "idempotency_key_required"}})

    drift = client.post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", video + b"changed", "video/mp4")},
        data={
            "manifest": manifest.decode("utf-8"),
            "timeline": timeline.decode("utf-8"),
        },
        headers=headers,
    )
    c.eq(drift.status_code, 409)
    c.eq(drift.json(), {"detail": {"code": "idempotency_key_conflict"}})


def test_gateway_health_ready_and_exact_multipart_inference(c: Check):
    client = _client()
    c.eq(client.get("/healthz").json(), {"status": "ok"})
    ready = client.get("/readyz")
    c.eq(ready.status_code, 200)
    c.eq(ready.json(), {
        "status": "ready",
        "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
        "candidate_id": "BLV9-009",
        "ensemble_members": 3,
        "preprocessing": "faces-to-shared-v9/v1",
    })
    video, manifest, timeline = _payloads()
    response = client.post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", video, "video/mp4")},
        data={
            "manifest": manifest.decode("utf-8"),
            "timeline": timeline.decode("utf-8"),
        },
    )
    c.eq(response.status_code, 200)
    c.eq(response.json()["model"]["candidate_id"], "BLV9-009")
    c.eq(response.json()["quality"]["actions_used"], 7)
    c.eq(response.json()["prediction"]["probability"], 0.73)
    c.eq(
        response.json()["report_evidence"]["actions"][0]["model_influence"]["status"],
        "stable",
    )
    c.true("capture.mp4" not in response.text)


def test_gateway_exposes_a_closed_error_for_every_http_failure_class(c: Check):
    client = _client()
    video, manifest, timeline = _payloads()
    wrong_content = client.post(
        "/api/v1/facial-paralysis/infer",
        content=b"not multipart",
        headers={"content-type": "application/octet-stream"},
    )
    c.eq(wrong_content.status_code, 415)
    c.eq(wrong_content.json(), {"detail": {"code": "multipart_required"}})

    empty_video = client.post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", b"", "video/mp4")},
        data={
            "manifest": manifest.decode("utf-8"),
            "timeline": timeline.decode("utf-8"),
        },
    )
    c.eq(empty_video.status_code, 400)
    c.eq(empty_video.json(), {"detail": {"code": "video_required"}})

    missing_field = client.post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", video, "video/mp4")},
        data={"manifest": manifest.decode("utf-8")},
    )
    c.eq(missing_field.status_code, 400)
    c.eq(missing_field.json(), {"detail": {"code": "invalid_capture_request"}})

    def unavailable(_payload: bytes):
        raise TimeoutError("/private/patient-name must not escape")

    downstream = _client(unavailable).post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", video, "video/mp4")},
        data={
            "manifest": manifest.decode("utf-8"),
            "timeline": timeline.decode("utf-8"),
        },
    )
    c.eq(downstream.status_code, 502)
    c.eq(downstream.json(), {"detail": {"code": "inference_unavailable"}})
    c.true("patient-name" not in downstream.text)

    not_ready = TestClient(create_app(
        processor=_processor,
        inference_client=_inference,
        readiness_client=lambda: (_ for _ in ()).throw(RuntimeError("private")),
    )).get("/readyz")
    c.eq(not_ready.status_code, 503)
    c.eq(not_ready.json(), {"detail": {"code": "model_not_ready"}})

    invalid_processor = TestClient(create_app(
        processor=lambda *_args: object(),
        inference_client=_inference,
        readiness_client=lambda: {
            "status": "ready",
            "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
            "candidate_id": "BLV9-009",
            "ensemble_members": 3,
        },
    )).post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", video, "video/mp4")},
        data={
            "manifest": manifest.decode("utf-8"),
            "timeline": timeline.decode("utf-8"),
        },
        headers=_idempotency_headers(video, manifest, timeline),
    )
    c.eq(invalid_processor.status_code, 500)
    c.eq(invalid_processor.json(), {"detail": {"code": "gateway_unavailable"}})


def test_gateway_rejects_open_forms_unusable_script_and_bad_downstream(c: Check):
    client = _client()
    video, manifest, timeline = _payloads()
    missing = client.post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", video, "video/mp4")},
        data={"manifest": manifest.decode("utf-8")},
    )
    c.eq(missing.status_code, 400)
    extra = client.post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", video, "video/mp4")},
        data={
            "manifest": manifest.decode("utf-8"),
            "timeline": timeline.decode("utf-8"),
            "patient_id": "must-not-be-accepted",
        },
    )
    c.eq(extra.status_code, 400)
    video, manifest, timeline = _payloads(include_optional=False)
    seven_step = client.post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", video, "video/mp4")},
        data={
            "manifest": manifest.decode("utf-8"),
            "timeline": timeline.decode("utf-8"),
        },
        headers=_idempotency_headers(video, manifest, timeline),
    )
    c.eq(seven_step.status_code, 200)
    c.eq(seven_step.json()["quality"]["actions_used"], 6)

    def wrong_model(_payload: bytes):
        result = _inference(_payload)
        result["model_id"] = "wrong"
        return result

    bad_downstream = _client(wrong_model).post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", _payloads()[0], "video/mp4")},
        data={
            "manifest": _payloads()[1].decode("utf-8"),
            "timeline": _payloads()[2].decode("utf-8"),
        },
    )
    c.eq(bad_downstream.status_code, 502)


def test_gateway_closes_multipart_uploads_on_success_and_failure(c: Check):
    original_close = gateway_module.UploadFile.close
    closed: list[str] = []

    async def tracked_close(upload):
        closed.append(str(upload.filename))
        await original_close(upload)

    gateway_module.UploadFile.close = tracked_close
    try:
        video, manifest, timeline = _payloads()
        success = _client().post(
            "/api/v1/facial-paralysis/infer",
            files={"video": ("capture.mp4", video, "video/mp4")},
            data={
                "manifest": manifest.decode("utf-8"),
                "timeline": timeline.decode("utf-8"),
            },
        )
        c.eq(success.status_code, 200)

        changed_manifest = json.loads(manifest)
        changed_manifest["video_sha256"] = "0" * 64
        invalid = _client().post(
            "/api/v1/facial-paralysis/infer",
            files={"video": ("capture.mp4", video, "video/mp4")},
            data={
                "manifest": json.dumps(changed_manifest),
                "timeline": timeline.decode("utf-8"),
            },
            headers=_idempotency_headers(
                video,
                json.dumps(changed_manifest).encode("utf-8"),
                timeline,
            ),
        )
        c.eq(invalid.status_code, 422)
    finally:
        gateway_module.UploadFile.close = original_close
    c.eq(closed, ["capture.mp4", "capture.mp4"])


def test_gateway_closes_oversized_upload_without_calling_processor(c: Check):
    original_limit = gateway_module.MAX_VIDEO_BYTES
    original_close = gateway_module.UploadFile.close
    closed: list[str] = []
    processor_calls: list[bool] = []

    async def tracked_close(upload):
        closed.append(str(upload.filename))
        await original_close(upload)

    def forbidden_processor(*_args):
        processor_calls.append(True)
        raise AssertionError("oversized bytes reached the processor")

    gateway_module.MAX_VIDEO_BYTES = 8
    gateway_module.UploadFile.close = tracked_close
    try:
        exact_video, exact_manifest, exact_timeline = _payloads(video=b"12345678")
        exact = _client().post(
            "/api/v1/facial-paralysis/infer",
            files={"video": ("capture.mp4", exact_video, "video/mp4")},
            data={
                "manifest": exact_manifest.decode("utf-8"),
                "timeline": exact_timeline.decode("utf-8"),
            },
            headers=_idempotency_headers(
                exact_video, exact_manifest, exact_timeline
            ),
        )
        c.eq(exact.status_code, 200)

        video, manifest, timeline = _payloads(video=b"123456789")
        client = TestClient(create_app(
            processor=forbidden_processor,
            inference_client=_inference,
            readiness_client=lambda: {
                "status": "ready",
                "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
                "candidate_id": "BLV9-009",
                "ensemble_members": 3,
            },
        ))
        response = client.post(
            "/api/v1/facial-paralysis/infer",
            files={"video": ("capture.mp4", video, "video/mp4")},
            data={
                "manifest": manifest.decode("utf-8"),
                "timeline": timeline.decode("utf-8"),
            },
        )
        c.eq(response.status_code, 413)
    finally:
        gateway_module.UploadFile.close = original_close
        gateway_module.MAX_VIDEO_BYTES = original_limit
    c.eq(closed, ["capture.mp4", "capture.mp4"])
    c.eq(processor_calls, [])


def test_gateway_returns_only_safe_preprocessing_failure_codes(c: Check):
    video, manifest, timeline = _payloads(include_optional=False)

    def post_with(error: ValueError):
        def rejected_processor(*_args):
            raise error

        client = TestClient(create_app(
            processor=rejected_processor,
            inference_client=_inference,
            readiness_client=lambda: {
                "status": "ready",
                "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
                "candidate_id": "BLV9-009",
                "ensemble_members": 3,
            },
        ))
        return client.post(
            "/api/v1/facial-paralysis/infer",
            files={"video": ("capture.mp4", video, "video/mp4")},
            data={
                "manifest": manifest.decode("utf-8"),
                "timeline": timeline.decode("utf-8"),
            },
            headers=_idempotency_headers(video, manifest, timeline),
        )

    tracking_error_type = getattr(pipeline_module, "CaptureTrackingError")
    tracking = post_with(
        tracking_error_type("lower_teeth_show", valid_samples=25)
    )
    c.eq(tracking.status_code, 422)
    c.eq(tracking.json(), {
        "detail": {
            "code": "face_tracking_insufficient",
            "action": "lower_teeth_show",
            "valid_samples": 25,
            "required_samples": 26,
        },
    })
    timing = post_with(ValueError("decoded duration contradicts the FACES timeline"))
    c.eq(timing.json(), {"detail": {"code": "video_timing_mismatch"}})
    categories = {
        "capture manifest fields differ from the closed schema": "capture_evidence_invalid",
        "video container extension is unsupported": "video_format_unsupported",
        "video frame rate is below the gateway minimum": "video_frame_rate_too_low",
        "video dimensions are outside the gateway bounds": "video_dimensions_unsupported",
        "interocular distance is degenerate": "face_geometry_invalid",
    }
    for message, code in categories.items():
        response = post_with(ValueError(message))
        c.eq(response.status_code, 422)
        c.eq(response.json(), {"detail": {"code": code}})
    unknown = post_with(ValueError("/private/patient-name.mov internal secret"))
    c.eq(unknown.json(), {"detail": {"code": "preprocessing_failed"}})
    c.true("patient-name" not in unknown.text)


class _Response:
    def __init__(self, payload: dict[str, object]):
        self._payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def read(self, size: int = -1):
        if size < 0:
            return self._payload
        result, self._payload = self._payload[:size], self._payload[size:]
        return result

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_shared_v9_http_client_uses_closed_internal_routes(c: Check):
    requests = []

    def opener(request, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/readyz"):
            return _Response({
                "status": "ready",
                "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
                "candidate_id": "BLV9-009",
                "ensemble_members": 3,
                "device": "cpu",
                "weight_sha256": ["1" * 64, "2" * 64, "3" * 64],
            })
        c.eq(request.get_header("Content-type"), "application/octet-stream")
        c.eq(request.data, b"npz")
        return _Response(_explained_prediction((
            "BROW_RAISE", "EYE_GENTLE", "EYE_FORCEFUL",
            "SMILE_GENTLE", "LIP_PUCKER", "SHOW_BOTTOM_TEETH",
            "SMILE_FULL",
        )))

    client = SharedV9HttpClient("http://shared-v9:8080", opener=opener)
    c.eq(client.ready()["candidate_id"], "BLV9-009")
    c.eq(client.infer(b"npz")["probability"], 0.73)
    c.eq([item[0].full_url for item in requests], [
        "http://shared-v9:8080/readyz",
        "http://shared-v9:8080/v1/explain/cue_aligned_action",
    ])
    c.true(all(item[1] == 10.0 for item in requests))


def test_runtime_processor_requires_the_frozen_face_landmarker(c: Check):
    c.eq(
        FACE_LANDMARKER_SHA256,
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    )
    with tempfile.TemporaryDirectory() as temporary:
        model = Path(temporary) / "face_landmarker.task"
        model.write_bytes(b"not the frozen model")
        c.raises(
            lambda: MediaPipeCaptureProcessor(model),
            ValueError,
            "runtime cannot start with a different MediaPipe model",
        )


if __name__ == "__main__":
    run_all("test_faces_shared_v9_gateway", dict(globals()))
