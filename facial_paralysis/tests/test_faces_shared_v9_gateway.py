"""HTTP boundary tests for the Facial Process Web Shared V9 gateway."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile

from fastapi.testclient import TestClient
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from test_faces_shared_v9_pipeline import _mesh_stream, _payloads  # noqa: E402
from src.deployment.faces_shared_v9_gateway import (  # noqa: E402
    FACE_LANDMARKER_SHA256,
    GatewayPreparedCapture,
    MediaPipeCaptureProcessor,
    SharedV9HttpClient,
    create_app,
)
from src.deployment.shared_v8_release import validate_request_arrays  # noqa: E402
from src.preprocessing.faces_shared_v9_pipeline import (  # noqa: E402
    build_v9_action_arrays,
    parse_capture_evidence,
)


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
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        arrays = {name: np.array(saved[name], copy=True) for name in saved.files}
    validate_request_arrays("cue_aligned_action", arrays)
    return {
        "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
        "protocol": "cue_aligned_action",
        "probability": 0.73,
        "member_probabilities": [0.71, 0.74, 0.74],
        "predicted_class": 1,
        "threshold": 0.5,
    }


def _client(inference=_inference):
    return TestClient(create_app(
        processor=_processor,
        inference_client=inference,
        readiness_client=lambda: {
            "status": "ready",
            "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
            "candidate_id": "BLV9-009",
            "ensemble_members": 3,
        },
    ))


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
    c.true("capture.mp4" not in response.text)


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
    unavailable = client.post(
        "/api/v1/facial-paralysis/infer",
        files={"video": ("capture.mp4", video, "video/mp4")},
        data={
            "manifest": manifest.decode("utf-8"),
            "timeline": timeline.decode("utf-8"),
        },
    )
    c.eq(unavailable.status_code, 422)

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
        return _Response({
            "model_id": "broad_literature_shared_v9_blv9_009_ensemble",
            "protocol": "cue_aligned_action",
            "probability": 0.73,
            "member_probabilities": [0.71, 0.74, 0.74],
            "predicted_class": 1,
            "threshold": 0.5,
        })

    client = SharedV9HttpClient("http://shared-v9:8080", opener=opener)
    c.eq(client.ready()["candidate_id"], "BLV9-009")
    c.eq(client.infer(b"npz")["probability"], 0.73)
    c.eq([item[0].full_url for item in requests], [
        "http://shared-v9:8080/readyz",
        "http://shared-v9:8080/v1/predict/cue_aligned_action",
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
