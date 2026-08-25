"""Browser-facing, fail-closed gateway for FACES video and Shared V9."""
from __future__ import annotations

from dataclasses import dataclass
import hmac
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import threading
import time
from typing import Callable, Mapping
import urllib.parse
import urllib.request

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from src.preprocessing.faces_shared_v9_pipeline import (
    CaptureEvidence,
    CaptureTimingError,
    CaptureTrackingError,
    MAX_VIDEO_BYTES,
    PREPROCESSING_VERSION,
    PreparedV9Request,
    SHARED_V9_CANDIDATE_ID,
    SHARED_V9_MODEL_ID,
    build_gateway_response,
    build_v9_action_arrays,
    decode_capture_samples,
    encode_v9_request_npz,
    extract_paired_meshes,
    parse_capture_evidence,
)


FACE_LANDMARKER_SHA256 = (
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
)

_LOGGER = logging.getLogger(__name__)


def _safe_preprocessing_failure_detail(error: ValueError) -> dict[str, object]:
    """Reduce internal preprocessing failures to non-identifying public evidence."""
    if isinstance(error, CaptureTrackingError):
        return {
            "code": "face_tracking_insufficient",
            "action": error.action,
            "valid_samples": error.valid_samples,
            "required_samples": error.required_samples,
        }
    message = str(error)
    if (
        "capture manifest" in message
        or "action timeline" in message
        or "video digest differs from capture evidence" in message
        or "timeline action count contradicts" in message
        or "timeline identity differs" in message
        or "validated capture evidence is required" in message
    ):
        return {"code": "capture_evidence_invalid"}
    if "video container extension is unsupported" in message:
        return {"code": "video_format_unsupported"}
    if (
        "video frame rate is below the gateway minimum" in message
        or "video frame rate cannot provide unique hold samples" in message
    ):
        return {"code": "video_frame_rate_too_low"}
    if (
        "video dimensions are outside the gateway bounds" in message
        or "decoded frame dimensions changed" in message
    ):
        return {"code": "video_dimensions_unsupported"}
    if (
        "decoded duration contradicts the FACES timeline" in message
        or "does not cover every FACES hold" in message
        or "sampling exceeds half a source frame" in message
    ):
        return {"code": "video_timing_mismatch"}
    if (
        "video cannot be decoded" in message
    ):
        return {"code": "video_decode_failed"}
    if (
        "interocular distance is degenerate" in message
        or "normalized landmark geometry" in message
        or "degenerate landmark geometry" in message
    ):
        return {"code": "face_geometry_invalid"}
    return {"code": "preprocessing_failed"}


@dataclass(frozen=True)
class GatewayPreparedCapture:
    """One authenticated capture after video decoding and landmark extraction."""

    evidence: CaptureEvidence
    prepared: PreparedV9Request
    face_landmarker_sha256: str


_IDEMPOTENCY_DOMAIN = b"facial-process-shared-v9-idempotency/v1\n"


def _idempotency_key(
    video_payload: bytes,
    manifest_payload: bytes,
    timeline_payload: bytes,
) -> str:
    """Bind one browser retry key to exact evidence and the frozen release."""
    if any(type(value) is not bytes or not value for value in (
        video_payload, manifest_payload, timeline_payload
    )):
        raise ValueError("idempotency evidence must be nonempty exact bytes")
    components = (
        hashlib.sha256(video_payload).hexdigest(),
        hashlib.sha256(manifest_payload).hexdigest(),
        hashlib.sha256(timeline_payload).hexdigest(),
        SHARED_V9_MODEL_ID,
        SHARED_V9_CANDIDATE_ID,
        PREPROCESSING_VERSION,
    )
    return hashlib.sha256(
        _IDEMPOTENCY_DOMAIN + "\n".join(components).encode("ascii")
    ).hexdigest()


@dataclass
class _IdempotencyEntry:
    commitment: str
    created_at: float
    event: threading.Event
    response: dict[str, object] | None = None


class _IdempotencyRegistry:
    """Short-lived single-flight response replay without retaining video bytes."""

    def __init__(self, *, ttl_seconds: float = 900.0, max_entries: int = 128):
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[str, _IdempotencyEntry] = {}
        self._lock = threading.Lock()

    def claim(self, key: str, commitment: str):
        now = time.monotonic()
        with self._lock:
            expired = [
                stored_key for stored_key, item in self._entries.items()
                if item.response is not None
                and now - item.created_at >= self._ttl_seconds
            ]
            for stored_key in expired:
                del self._entries[stored_key]
            existing = self._entries.get(key)
            if existing is not None:
                if not hmac.compare_digest(existing.commitment, commitment):
                    raise ValueError("idempotency key is bound to different evidence")
                return ("replay" if existing.response is not None else "wait", existing)
            if len(self._entries) >= self._max_entries:
                raise RuntimeError("idempotency registry is at capacity")
            entry = _IdempotencyEntry(
                commitment=commitment,
                created_at=now,
                event=threading.Event(),
            )
            self._entries[key] = entry
            return "owner", entry

    def complete(
        self,
        key: str,
        entry: _IdempotencyEntry,
        response: dict[str, object],
    ) -> None:
        with self._lock:
            if self._entries.get(key) is not entry:
                raise RuntimeError("idempotency owner changed before completion")
            entry.response = response
            entry.event.set()

    def abort(self, key: str, entry: _IdempotencyEntry) -> None:
        with self._lock:
            if self._entries.get(key) is entry:
                del self._entries[key]
            entry.event.set()

    @staticmethod
    def wait(entry: _IdempotencyEntry) -> dict[str, object]:
        if not entry.event.wait(timeout=330.0) or entry.response is None:
            raise RuntimeError("idempotent inference did not complete")
        return entry.response


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("downstream JSON contains a duplicate key")
        result[key] = value
    return result


class SharedV9HttpClient:
    """Bounded client for the one internal Shared V9 service surface."""

    def __init__(self, base_url: str, *, opener=urllib.request.urlopen):
        if type(base_url) is not str or not callable(opener):
            raise ValueError("Shared V9 URL and opener are required")
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Shared V9 URL must be one internal HTTP origin")
        self._base_url = base_url.rstrip("/")
        self._opener = opener

    def _request(self, request: urllib.request.Request) -> dict[str, object]:
        with self._opener(request, timeout=10.0) as response:
            payload = response.read(64 * 1024 + 1)
            if len(payload) > 64 * 1024 or response.read(1):
                raise ValueError("Shared V9 response exceeded the gateway bound")
        try:
            value = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Shared V9 response is not strict UTF-8 JSON") from exc
        if type(value) is not dict:
            raise ValueError("Shared V9 response is not a JSON object")
        return value

    def ready(self) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self._base_url}/readyz",
            method="GET",
            headers={"Accept": "application/json"},
        )
        return self._request(request)

    def infer(self, payload: bytes) -> dict[str, object]:
        if type(payload) is not bytes or not payload:
            raise ValueError("Shared V9 request must be nonempty exact bytes")
        request = urllib.request.Request(
            f"{self._base_url}/v1/predict/cue_aligned_action",
            data=payload,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/octet-stream",
            },
        )
        return self._request(request)


def _sha256_regular_file(path: Path) -> str:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("Face Landmarker path must be absolute")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16 * 1024 * 1024:
            raise ValueError("Face Landmarker asset is not a bounded regular file")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


class MediaPipeCaptureProcessor:
    """Decode one FACES recording and produce the frozen Shared V9 tensors."""

    def __init__(self, face_landmarker_path: Path):
        path = Path(face_landmarker_path)
        if _sha256_regular_file(path) != FACE_LANDMARKER_SHA256:
            raise ValueError("Face Landmarker asset differs from the frozen release")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision as mpv
        except ImportError as exc:
            raise RuntimeError("MediaPipe runtime is unavailable") from exc
        options = mpv.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(path)),
            running_mode=mpv.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = mpv.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    def _detect_mesh(self, rgb):
        import numpy as np

        result = self._landmarker.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        )
        if len(result.face_landmarks) != 1 or len(result.face_landmarks[0]) != 478:
            return None
        return np.asarray(
            [(point.x, point.y, point.z) for point in result.face_landmarks[0]],
            dtype=np.float64,
        )

    def __call__(
        self,
        video_payload: bytes,
        filename: str,
        manifest_payload: bytes,
        timeline_payload: bytes,
    ) -> GatewayPreparedCapture:
        evidence = parse_capture_evidence(
            video_payload,
            manifest_payload,
            timeline_payload,
        )
        decoded = decode_capture_samples(
            video_payload,
            evidence.timeline,
            filename=filename,
        )
        with self._lock:
            stream = extract_paired_meshes(decoded, self._detect_mesh)
        prepared = build_v9_action_arrays(
            evidence,
            frame_timestamps_ms=stream.frame_timestamps_ms,
            source_frame_indices=stream.source_frame_indices,
            original_meshes=stream.original_meshes,
            mirrored_meshes=stream.mirrored_meshes,
            pair_valid_mask=stream.pair_valid_mask,
            source_fps=stream.source_fps,
        )
        return GatewayPreparedCapture(
            evidence=evidence,
            prepared=prepared,
            face_landmarker_sha256=FACE_LANDMARKER_SHA256,
        )

    def close(self) -> None:
        with self._lock:
            self._landmarker.close()


def _validate_ready(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("Shared V9 readiness is not an object")
    required = {
        "status": "ready",
        "model_id": SHARED_V9_MODEL_ID,
        "candidate_id": SHARED_V9_CANDIDATE_ID,
        "ensemble_members": 3,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("Shared V9 readiness identity drifted")
    return {
        **required,
        "preprocessing": PREPROCESSING_VERSION,
    }


async def _bounded_video(upload: UploadFile) -> bytes:
    payload = await upload.read(MAX_VIDEO_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail={"code": "video_required"})
    if len(payload) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail={"code": "video_too_large"})
    if await upload.read(1):
        raise HTTPException(status_code=413, detail={"code": "video_too_large"})
    return bytes(payload)


def create_app(
    *,
    processor: Callable[[bytes, str, bytes, bytes], GatewayPreparedCapture],
    inference_client: Callable[[bytes], Mapping[str, object]],
    readiness_client: Callable[[], Mapping[str, object]],
) -> FastAPI:
    """Create the exact multipart HTTP boundary used by Facial Process Web."""
    if not all(callable(item) for item in (processor, inference_client, readiness_client)):
        raise ValueError("gateway dependencies must be callable")
    app = FastAPI(
        title="Facial Process Web Shared V9 Gateway",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    idempotency_registry = _IdempotencyRegistry()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        try:
            return _validate_ready(readiness_client())
        except Exception as exc:  # downstream is outside this trust boundary
            raise HTTPException(
                status_code=503,
                detail={"code": "model_not_ready"},
            ) from exc

    @app.post("/api/v1/facial-paralysis/infer")
    async def infer(request: Request):
        content_type = request.headers.get("content-type", "")
        if not content_type.casefold().startswith("multipart/form-data;"):
            raise HTTPException(
                status_code=415,
                detail={"code": "multipart_required"},
            )
        form = None
        try:
            try:
                form = await request.form(
                    max_files=1,
                    max_fields=2,
                    max_part_size=256 * 1024,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "invalid_capture_request"},
                ) from exc
            items = list(form.multi_items())
            names = [name for name, _value in items]
            if sorted(names) != ["manifest", "timeline", "video"]:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "invalid_capture_request"},
                )
            values = {name: value for name, value in items}
            upload = values["video"]
            manifest = values["manifest"]
            timeline = values["timeline"]
            if (
                not isinstance(upload, UploadFile)
                or type(manifest) is not str
                or type(timeline) is not str
                or not upload.filename
            ):
                raise HTTPException(
                    status_code=400,
                    detail={"code": "invalid_capture_request"},
                )
            video = await _bounded_video(upload)
            filename = str(upload.filename)
            manifest_payload = manifest.encode("utf-8")
            timeline_payload = timeline.encode("utf-8")
        finally:
            if form is not None:
                await form.close()

        supplied_key = request.headers.get("idempotency-key", "")
        if (
            len(supplied_key) != 64
            or any(character not in "0123456789abcdef" for character in supplied_key)
        ):
            raise HTTPException(
                status_code=428,
                detail={"code": "idempotency_key_required"},
            )
        expected_key = _idempotency_key(video, manifest_payload, timeline_payload)
        if not hmac.compare_digest(supplied_key, expected_key):
            raise HTTPException(
                status_code=409,
                detail={"code": "idempotency_key_conflict"},
            )
        try:
            ownership, idempotency_entry = idempotency_registry.claim(
                supplied_key, expected_key
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "idempotency_key_conflict"},
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "gateway_unavailable"},
            ) from exc
        if ownership == "replay":
            if idempotency_entry.response is None:
                raise HTTPException(
                    status_code=503,
                    detail={"code": "gateway_unavailable"},
                )
            return idempotency_entry.response
        if ownership == "wait":
            try:
                return await run_in_threadpool(
                    idempotency_registry.wait, idempotency_entry
                )
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=502,
                    detail={"code": "inference_unavailable"},
                ) from exc
        try:
            prepared = await run_in_threadpool(
                processor,
                video,
                filename,
                manifest_payload,
                timeline_payload,
            )
        except ValueError as exc:
            detail = _safe_preprocessing_failure_detail(exc)
            if isinstance(exc, CaptureTimingError):
                _LOGGER.warning(
                    "capture_timing_rejected reason=%s decoded_duration_ms=%d "
                    "timeline_duration_ms=%d last_hold_ms=%d source_fps=%.6f "
                    "decoded_frame_count=%d tolerance_ms=%d",
                    exc.reason,
                    exc.decoded_duration_ms,
                    exc.timeline_duration_ms,
                    exc.last_hold_ms,
                    exc.source_fps,
                    exc.decoded_frame_count,
                    exc.tolerance_ms,
                )
            else:
                _LOGGER.warning("capture_gate_rejected code=%s", detail["code"])
            idempotency_registry.abort(supplied_key, idempotency_entry)
            raise HTTPException(status_code=422, detail=detail) from exc
        if not isinstance(prepared, GatewayPreparedCapture):
            idempotency_registry.abort(supplied_key, idempotency_entry)
            raise HTTPException(
                status_code=500,
                detail={"code": "gateway_unavailable"},
            )
        try:
            request_payload = encode_v9_request_npz(prepared.prepared.arrays)
            prediction = await run_in_threadpool(inference_client, request_payload)
            response = build_gateway_response(
                prediction,
                evidence=prepared.evidence,
                valid_samples_per_action=prepared.prepared.valid_samples_per_action,
                descriptive_evidence_per_action=(
                    prepared.prepared.descriptive_evidence_per_action
                ),
                preprocessing_version=PREPROCESSING_VERSION,
                face_landmarker_sha256=prepared.face_landmarker_sha256,
            )
        except Exception as exc:
            idempotency_registry.abort(supplied_key, idempotency_entry)
            raise HTTPException(
                status_code=502,
                detail={"code": "inference_unavailable"},
            ) from exc
        idempotency_registry.complete(supplied_key, idempotency_entry, response)
        return response

    return app


def create_runtime_app(
    *,
    shared_v9_url: str,
    face_landmarker_path: Path,
) -> FastAPI:
    """Build the production dependency graph from two explicit runtime pins."""
    processor = MediaPipeCaptureProcessor(face_landmarker_path)
    client = SharedV9HttpClient(shared_v9_url)
    return create_app(
        processor=processor,
        inference_client=client.infer,
        readiness_client=client.ready,
    )


__all__ = [
    "FACE_LANDMARKER_SHA256",
    "GatewayPreparedCapture",
    "MediaPipeCaptureProcessor",
    "SharedV9HttpClient",
    "create_app",
    "create_runtime_app",
]
