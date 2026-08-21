from __future__ import annotations

import io
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient
import numpy as np
import torch

from _testlib import run_all
from test_shared_v8_release import _candidate, _provenance, _request

from src.deployment.shared_v8_release import write_release
from src.deployment.shared_v8_service import create_app, encode_request_npz


def _client():
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name) / "release"
    torch.manual_seed(91)
    write_release(
        root,
        model=__import__(
            "src.models.residual_shared_router_v8", fromlist=["ResidualSharedRouterV8"]
        ).ResidualSharedRouterV8(_candidate()).eval(),
        scaler_mean=np.zeros(110, dtype=np.float64),
        scaler_scale=np.ones(110, dtype=np.float64),
        provenance=_provenance(),
    )
    return temporary, TestClient(create_app(root, device="cpu"))


def test_health_readiness_and_binary_prediction_contract(c):
    temporary, client = _client()
    try:
        c.eq(client.get("/healthz").json(), {"status": "ok"})
        ready = client.get("/readyz")
        c.eq(ready.status_code, 200)
        c.eq(ready.json()["status"], "ready")
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
            "model_id", "protocol", "probability", "predicted_class", "threshold"
        })
        c.eq(first.json()["protocol"], "scripted_three_action")
    finally:
        temporary.cleanup()


def test_service_fails_closed_on_transport_and_archive_drift(c):
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
        np.savez_compressed(extra_buffer, **request, participant_id=np.asarray([7]))
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
    run_all("test_shared_v8_service", dict(globals()))
