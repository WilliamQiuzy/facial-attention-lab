from __future__ import annotations

import io
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient
import numpy as np

from _testlib import run_all
from test_shared_v9_research_release import _models, _provenance, _request

from src.deployment.shared_v9_research_release import write_release
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


if __name__ == "__main__":
    run_all("test_shared_v9_service", dict(globals()))
