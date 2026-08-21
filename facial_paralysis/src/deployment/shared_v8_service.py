"""Small fail-closed HTTP surface for the locked shared V8 release."""
from __future__ import annotations

from dataclasses import asdict
import io
from pathlib import Path, PurePosixPath
import zipfile

from fastapi import FastAPI, HTTPException, Request
import numpy as np

from src.deployment.shared_v8_release import (
    DEPLOYMENT_MODEL_ID,
    PROTOCOL_TASK_CODES,
    load_release,
    validate_request_arrays,
)


MAX_REQUEST_BYTES = 16 * 1024 * 1024
_REQUEST_FIELDS = frozenset({
    "clinical_original",
    "clinical_mirrored",
    "dense_original",
    "dense_mirrored",
    "dense_valid_mask",
    "dense_available",
    "dense_timestamps",
    "action_mask",
    "action_codes",
})


def _load_request_npz(payload: bytes) -> dict[str, np.ndarray]:
    if type(payload) is not bytes or not payload or len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("request size is outside the deployment bound")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            canonical = {name + ".npy" for name in _REQUEST_FIELDS}
            if (
                len(names) != len(set(names))
                or set(names) != canonical
                or any(
                    PurePosixPath(name).name != name
                    or name.startswith(".")
                    or member.file_size <= 0
                    or member.file_size > 16 * 1024 * 1024
                    for name, member in zip(names, members)
                )
                or sum(member.file_size for member in members) > 32 * 1024 * 1024
            ):
                raise ValueError("request archive differs from the closed schema")
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            if (
                len(saved.files) != len(set(saved.files))
                or set(saved.files) != _REQUEST_FIELDS
            ):
                raise ValueError("request fields differ from the closed schema")
            return {name: np.array(saved[name], copy=True) for name in saved.files}
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError) as exc:
        raise ValueError("request archive is invalid") from exc


def encode_request_npz(protocol: str, arrays: dict[str, np.ndarray]) -> bytes:
    """Reference client encoder; validates before serialization."""
    validate_request_arrays(protocol, arrays)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("encoded request exceeds the transport bound")
    return payload


async def _bounded_body(request: Request) -> bytes:
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="request is too large")
    return bytes(payload)


def create_app(release_root: Path, *, device: str) -> FastAPI:
    predictor = load_release(release_root, device=device)
    app = FastAPI(
        title="Shared V8 Research Deployment",
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
        return {
            "status": "ready",
            "model_id": DEPLOYMENT_MODEL_ID,
            "weights_sha256": predictor.manifest["weights_sha256"],
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

    return app


__all__ = ["MAX_REQUEST_BYTES", "create_app", "encode_request_npz"]
