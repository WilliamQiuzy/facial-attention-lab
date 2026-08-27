from __future__ import annotations

import io
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient
import numpy as np

from _testlib import run_all
from test_shared_v9_research_release import _models, _provenance, _request

from src.deployment.shared_v9_research_release import write_release
from src.deployment.shared_v9_attribution import encode_attribution_request_npz
from src.deployment.shared_v9_service import create_app, encode_request_npz


def _client():
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name) / "release"
    write_release(
        root,
        models=_models(),
        scaler_mean=np.zeros(110, dtype=np.float64),
        scaler_scale=np.ones(110, dtype=np.float64),
        provenance=_provenance(),
    )
    return temporary, TestClient(create_app(root, device="cpu"))


def test_v9_health_readiness_and_three_member_prediction(c):
    temporary, client = _client()
    try:
        c.eq(client.get("/healthz").json(), {"status": "ok"})
        ready = client.get("/readyz")
        c.eq(ready.status_code, 200)
        c.eq(set(ready.json()), {
            "status", "model_id", "candidate_id", "ensemble_members",
            "weight_sha256", "device",
        })
        c.eq(ready.json()["status"], "ready")
        c.eq(ready.json()["candidate_id"], "BLV9-009")
        c.eq(ready.json()["ensemble_members"], 3)
        c.eq(len(ready.json()["weight_sha256"]), 3)

        payload = encode_request_npz(
            "scripted_three_action", _request("scripted_three_action")
        )
        first = client.post(
            "/v1/predict/scripted_three_action",
            content=payload,
            headers={"content-type": "application/octet-stream"},
        )
        second = client.post(
            "/v1/predict/scripted_three_action",
            content=payload,
            headers={"content-type": "application/octet-stream"},
        )
        c.eq(first.status_code, 200)
        c.eq(first.json(), second.json())
        c.eq(set(first.json()), {
            "model_id", "protocol", "probability", "member_probabilities",
            "predicted_class", "threshold",
        })
        c.eq(len(first.json()["member_probabilities"]), 3)
        c.true(abs(
            first.json()["probability"]
            - float(np.mean(first.json()["member_probabilities"]))
        ) < 1e-8)
    finally:
        temporary.cleanup()


def test_v9_service_rejects_transport_and_archive_drift(c):
    temporary, client = _client()
    try:
        request = _request("free_motion_four_window")
        payload = encode_request_npz("free_motion_four_window", request)
        wrong_type = client.post(
            "/v1/predict/free_motion_four_window",
            content=payload,
            headers={"content-type": "application/json"},
        )
        c.eq(wrong_type.status_code, 415)
        malformed = client.post(
            "/v1/predict/free_motion_four_window",
            content=b"PK\x03\x04",
            headers={"content-type": "application/octet-stream"},
        )
        c.eq(malformed.status_code, 400)
        extra_buffer = io.BytesIO()
        np.savez_compressed(extra_buffer, **request, patient_id=np.asarray([7]))
        extra = client.post(
            "/v1/predict/free_motion_four_window",
            content=extra_buffer.getvalue(),
            headers={"content-type": "application/octet-stream"},
        )
        c.eq(extra.status_code, 400)
        unknown = client.post(
            "/v1/predict/unknown",
            content=payload,
            headers={"content-type": "application/octet-stream"},
        )
        c.eq(unknown.status_code, 404)
    finally:
        temporary.cleanup()


def test_v9_service_explains_cue_aligned_actions_without_changing_prediction(c):
    temporary, client = _client()
    try:
        request = _request("cue_aligned_action")
        actions = request["clinical_original"].shape[0]
        neutral_original = np.zeros((actions, 110), dtype=np.float32)
        neutral_mirrored = np.zeros((actions, 110), dtype=np.float32)
        explain_payload = encode_attribution_request_npz(
            "cue_aligned_action",
            request,
            neutral_original,
            neutral_mirrored,
        )
        explained = client.post(
            "/v1/explain/cue_aligned_action",
            content=explain_payload,
            headers={"content-type": "application/octet-stream"},
        )
        predicted = client.post(
            "/v1/predict/cue_aligned_action",
            content=encode_request_npz("cue_aligned_action", request),
            headers={"content-type": "application/octet-stream"},
        )
        c.eq(explained.status_code, 200)
        c.eq(predicted.status_code, 200)
        body = explained.json()
        c.eq(set(body), {
            "model_id", "protocol", "probability", "member_probabilities",
            "predicted_class", "threshold", "attribution",
        })
        c.eq({key: body[key] for key in predicted.json()}, predicted.json())
        c.eq(body["attribution"]["schema_version"], "shared_v9_action_token_attribution/v1")
        c.eq(body["attribution"]["method"], "integrated_gradients_shared_action_tokens")
        c.eq(len(body["attribution"]["actions"]), actions)

        malformed = client.post(
            "/v1/explain/cue_aligned_action",
            content=b"PK\x03\x04",
            headers={"content-type": "application/octet-stream"},
        )
        c.eq(malformed.status_code, 400)
        wrong_protocol = client.post(
            "/v1/explain/scripted_three_action",
            content=explain_payload,
            headers={"content-type": "application/octet-stream"},
        )
        c.eq(wrong_protocol.status_code, 404)
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    run_all("test_shared_v9_service", dict(globals()))
