"""Small fail-closed HTTP surface for the public Shared V9 ensemble."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from src.deployment.shared_v8_release import PROTOCOL_TASK_CODES
from src.deployment.shared_v8_service import (
    MAX_REQUEST_BYTES,
    _bounded_body,
    _load_request_npz,
    encode_request_npz,
)
from src.deployment.shared_v9_research_release import (
    RESEARCH_CANDIDATE_ID,
    RESEARCH_MODEL_ID,
    load_release,
)
from src.deployment.shared_v9_attribution import (
    explain_prediction,
    load_attribution_request_npz,
)


def create_app(release_root: Path, *, device: str) -> FastAPI:
    predictor = load_release(release_root, device=device)
    app = FastAPI(
        title="Shared V9 Research Service",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        weights = predictor.manifest["weights"]
        return {
            "status": "ready",
            "model_id": RESEARCH_MODEL_ID,
            "candidate_id": RESEARCH_CANDIDATE_ID,
            "ensemble_members": len(predictor.models),
            "weight_sha256": [row["sha256"] for row in weights],
            "device": predictor.device.type,
        }

    @app.post("/v1/predict/{protocol}")
    async def predict(protocol: str, request: Request):
        if protocol not in PROTOCOL_TASK_CODES:
            raise HTTPException(status_code=404, detail="unknown protocol")
        if request.headers.get("content-type") != "application/octet-stream":
            raise HTTPException(status_code=415, detail="binary NPZ is required")
        payload = await _bounded_body(request)
        try:
            arrays = _load_request_npz(payload)
            prediction = predictor.predict(protocol, arrays)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid model input") from exc
        return asdict(prediction)

    @app.post("/v1/explain/{protocol}")
    async def explain(protocol: str, request: Request):
        if protocol != "cue_aligned_action":
            raise HTTPException(status_code=404, detail="unknown attribution protocol")
        if request.headers.get("content-type") != "application/octet-stream":
            raise HTTPException(status_code=415, detail="binary NPZ is required")
        payload = await _bounded_body(request)
        try:
            arrays, neutral_original, neutral_mirrored = (
                load_attribution_request_npz(payload, protocol=protocol)
            )
            explained = explain_prediction(
                predictor,
                protocol,
                arrays,
                neutral_original,
                neutral_mirrored,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid attribution input") from exc
        result = asdict(explained)
        return {**result["prediction"], "attribution": result["attribution"]}

    return app


__all__ = [
    "MAX_REQUEST_BYTES",
    "create_app",
    "encode_request_npz",
]
