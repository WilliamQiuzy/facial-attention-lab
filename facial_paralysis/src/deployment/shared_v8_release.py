"""Checksum-bound release and inference contract for shared V8 / RSR8-001."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Mapping
import zipfile

import numpy as np
import torch

from src.models.dense_clinical_shared_encoder_v1 import ACTION_VOCAB
from src.models.residual_shared_router_v8 import (
    ResidualSharedRouterV8,
    candidate_registry_v8,
)


DEPLOYMENT_SCHEMA = "shared_v8_deployment_v1"
DEPLOYMENT_MODEL_ID = "residual_shared_router_v8_rsr8_001"
DEPLOYMENT_CANDIDATE_ID = "RSR8-001"
PROTOCOL_TASK_CODES = {
    "free_motion_four_window": 0,
    "scripted_three_action": 1,
    "cue_aligned_action": 2,
}
_PROTOCOL_ACTION_COUNTS = {
    "free_motion_four_window": (4,),
    "scripted_three_action": (3,),
    "cue_aligned_action": (7, 8),
}
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
_STATE_PREFIX = "state__"
_SCALER_MEAN = "scaler__mean"
_SCALER_SCALE = "scaler__scale"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IMPLEMENTATION_FILES = (
    "src/deployment/shared_v8_release.py",
    "src/models/residual_shared_router_v8.py",
    "src/models/script_aware_shared_router_v6.py",
    "src/models/medically_gated_shared_encoder_v2.py",
    "src/models/medical_shared_candidate_registry_v2.py",
    "src/models/dense_clinical_shared_encoder_v1.py",
)


def _candidate():
    rows = tuple(
        row for row in candidate_registry_v8()
        if row.candidate_id == DEPLOYMENT_CANDIDATE_ID
    )
    if len(rows) != 1:
        raise RuntimeError("the frozen deployment candidate is unavailable")
    return rows[0]


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_FILES:
        path = _PROJECT_ROOT / relative
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains a duplicate key")
        result[key] = value
    return result


def _load_json(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("release manifest is not canonical UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError("release manifest must be an object")
    return value


def _validate_provenance(provenance: Mapping[str, object]) -> dict[str, object]:
    if type(provenance) is not dict or set(provenance) != {
        "git_commit", "training_seed", "training_epochs", "training_device",
        "source_counts", "source_commitments",
    }:
        raise ValueError("release provenance differs from the closed schema")
    counts = provenance["source_counts"]
    commitments = provenance["source_commitments"]
    if (
        type(provenance["git_commit"]) is not str
        or _COMMIT.fullmatch(provenance["git_commit"]) is None
        or provenance["training_seed"] != 0
        or provenance["training_epochs"] != 20
        or provenance["training_device"] != "NVIDIA H200"
        or type(counts) is not dict
        or counts != {"palsynet": 38, "neuroface": 36, "meei": 56}
        or type(commitments) is not dict
        or set(commitments) != {"palsynet", "neuroface", "meei"}
        or any(
            type(value) is not str or _SHA256.fullmatch(value) is None
            for value in commitments.values()
        )
    ):
        raise ValueError("release provenance values are not frozen")
    return {
        "git_commit": provenance["git_commit"],
        "training_seed": 0,
        "training_epochs": 20,
        "training_device": "NVIDIA H200",
        "source_counts": dict(counts),
        "source_commitments": dict(commitments),
    }


def _validate_scaler(mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (
        type(mean) is not np.ndarray
        or mean.shape != (110,)
        or mean.dtype != np.dtype(np.float64)
        or type(scale) is not np.ndarray
        or scale.shape != (110,)
        or scale.dtype != np.dtype(np.float64)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("deployment scaler differs from the frozen 110D contract")
    return np.array(mean, copy=True), np.array(scale, copy=True)


def _state_arrays(model: ResidualSharedRouterV8) -> dict[str, np.ndarray]:
    if (
        type(model) is not ResidualSharedRouterV8
        or model.candidate.candidate_id != DEPLOYMENT_CANDIDATE_ID
    ):
        raise ValueError("only RSR8-001 may be published as deployment v1")
    return {
        _STATE_PREFIX + name: tensor.detach().cpu().contiguous().numpy()
        for name, tensor in model.state_dict().items()
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def write_release(
    output: Path,
    *,
    model: ResidualSharedRouterV8,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Publish one immutable directory without serializing Python objects."""
    if not isinstance(output, Path) or output.exists() or output.is_symlink():
        raise FileExistsError("deployment release already exists")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("deployment release parent must be an existing real directory")
    mean, scale = _validate_scaler(scaler_mean, scaler_scale)
    metadata = _validate_provenance(provenance)
    arrays = _state_arrays(model)
    arrays[_SCALER_MEAN] = mean
    arrays[_SCALER_SCALE] = scale
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    weights_payload = buffer.getvalue()
    manifest = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "model_id": DEPLOYMENT_MODEL_ID,
        "candidate_id": DEPLOYMENT_CANDIDATE_ID,
        "status": "locked_research_deployment_not_clinically_validated",
        "input_schema": "shared_clinical_action_bag_npz_v1",
        "protocol_task_codes": dict(PROTOCOL_TASK_CODES),
        "threshold": 0.5,
        "weights_file": "weights.npz",
        "weights_sha256": _sha256(weights_payload),
        "implementation_sha256": implementation_sha256(),
        "provenance": metadata,
        "claims": {
            "clinical_validation": False,
            "hb_grade": False,
            "mayo_accuracy": False,
        },
    }
    manifest_payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        os.chmod(stage, 0o700)
        _write_bytes(stage / "weights.npz", weights_payload)
        _write_bytes(stage / "manifest.json", manifest_payload)
        os.rename(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("release member must be a regular file")
    stat_result = path.stat()
    if stat_result.st_size <= 0 or stat_result.st_size > maximum_bytes:
        raise ValueError("release member size is outside the frozen bound")
    payload = path.read_bytes()
    if len(payload) != stat_result.st_size:
        raise ValueError("release member changed while reading")
    return payload


def _load_npz(payload: bytes, expected: set[str]) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = [member.filename for member in archive.infolist()]
            canonical = {name + ".npy" for name in expected}
            if (
                len(names) != len(set(names))
                or set(names) != canonical
                or any(
                    PurePosixPath(name).name != name
                    or name.startswith(".")
                    or member.file_size > 16 * 1024 * 1024
                    for name, member in zip(names, archive.infolist())
                )
                or sum(member.file_size for member in archive.infolist()) > 64 * 1024 * 1024
            ):
                raise ValueError("weights archive members differ from the closed schema")
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            if len(saved.files) != len(set(saved.files)) or set(saved.files) != expected:
                raise ValueError("weights archive fields differ from the closed schema")
            return {name: np.array(saved[name], copy=True) for name in saved.files}
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError) as exc:
        raise ValueError("weights archive is invalid") from exc


def _validate_manifest(manifest: dict[str, object]) -> None:
    if set(manifest) != {
        "schema_version", "model_id", "candidate_id", "status", "input_schema",
        "protocol_task_codes", "threshold", "weights_file", "weights_sha256",
        "implementation_sha256", "provenance", "claims",
    }:
        raise ValueError("release manifest fields differ from the closed schema")
    if (
        manifest["schema_version"] != DEPLOYMENT_SCHEMA
        or manifest["model_id"] != DEPLOYMENT_MODEL_ID
        or manifest["candidate_id"] != DEPLOYMENT_CANDIDATE_ID
        or manifest["status"] != "locked_research_deployment_not_clinically_validated"
        or manifest["input_schema"] != "shared_clinical_action_bag_npz_v1"
        or manifest["protocol_task_codes"] != PROTOCOL_TASK_CODES
        or manifest["threshold"] != 0.5
        or manifest["weights_file"] != "weights.npz"
        or type(manifest["weights_sha256"]) is not str
        or _SHA256.fullmatch(manifest["weights_sha256"]) is None
        or manifest["implementation_sha256"] != implementation_sha256()
        or manifest["claims"] != {
            "clinical_validation": False, "hb_grade": False, "mayo_accuracy": False,
        }
    ):
        raise ValueError("release manifest values are not frozen")
    _validate_provenance(manifest["provenance"])


@dataclass(frozen=True)
class Prediction:
    model_id: str
    protocol: str
    probability: float
    predicted_class: int
    threshold: float


class SharedV8Predictor:
    def __init__(self, model, mean, scale, manifest, device):
        self.model = model
        self.mean = mean
        self.scale = scale
        self.manifest = manifest
        self.device = device

    def predict(self, protocol: str, arrays: Mapping[str, np.ndarray]) -> Prediction:
        normalized = validate_request_arrays(protocol, arrays)
        original = (
            normalized["clinical_original"].astype(np.float64) - self.mean[None, None, :]
        ) / self.scale[None, None, :]
        mirrored = (
            normalized["clinical_mirrored"].astype(np.float64) - self.mean[None, None, :]
        ) / self.scale[None, None, :]
        inputs = (
            original.astype(np.float32), mirrored.astype(np.float32),
            normalized["dense_original"], normalized["dense_mirrored"],
            normalized["dense_valid_mask"], normalized["dense_available"],
            normalized["dense_timestamps"], normalized["action_mask"],
            normalized["action_codes"],
        )
        tensors = tuple(torch.from_numpy(np.array(value, copy=True)).to(self.device) for value in inputs)
        task_codes = torch.tensor(
            [PROTOCOL_TASK_CODES[protocol]], dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            tokens = self.model.shared_action_tokens(*tensors)
            logit = self.model.routed_logits(tokens, tensors[-2], task_codes)
            probability = float(torch.sigmoid(logit)[0].cpu())
        if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise RuntimeError("model produced an invalid probability")
        return Prediction(
            model_id=DEPLOYMENT_MODEL_ID,
            protocol=protocol,
            probability=probability,
            predicted_class=int(probability >= 0.5),
            threshold=0.5,
        )


def load_release(root: Path, *, device: str) -> SharedV8Predictor:
    if not isinstance(root, Path) or root.is_symlink() or not root.is_dir():
        raise ValueError("release root must be a real directory")
    manifest_payload = _read_regular_file(root / "manifest.json", maximum_bytes=64 * 1024)
    manifest = _load_json(manifest_payload)
    _validate_manifest(manifest)
    weights_payload = _read_regular_file(root / "weights.npz", maximum_bytes=64 * 1024 * 1024)
    if _sha256(weights_payload) != manifest["weights_sha256"]:
        raise ValueError("weights checksum differs from the release manifest")
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested deployment device is unavailable")
    model = ResidualSharedRouterV8(_candidate())
    state = model.state_dict()
    expected = {_STATE_PREFIX + name for name in state} | {_SCALER_MEAN, _SCALER_SCALE}
    arrays = _load_npz(weights_payload, expected)
    mean, scale = _validate_scaler(arrays.pop(_SCALER_MEAN), arrays.pop(_SCALER_SCALE))
    loaded = {}
    for name, tensor in state.items():
        array = arrays[_STATE_PREFIX + name]
        expected_array = tensor.detach().cpu().numpy()
        if array.shape != expected_array.shape or array.dtype != expected_array.dtype:
            raise ValueError("a model tensor differs from the frozen state schema")
        loaded[name] = torch.from_numpy(np.array(array, copy=True))
    model.load_state_dict(loaded, strict=True)
    model.to(runtime).eval()
    return SharedV8Predictor(model, mean, scale, manifest, runtime)


def validate_request_arrays(
    protocol: str,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if protocol not in PROTOCOL_TASK_CODES or type(arrays) is not dict or set(arrays) != _REQUEST_FIELDS:
        raise ValueError("request protocol or fields differ from the deployment contract")
    actions = arrays["clinical_original"].shape[0] if isinstance(arrays["clinical_original"], np.ndarray) and arrays["clinical_original"].ndim else -1
    if actions not in _PROTOCOL_ACTION_COUNTS[protocol]:
        raise ValueError("request action count differs from the protocol contract")
    expected = {
        "clinical_original": ((actions, 110), np.dtype(np.float32)),
        "clinical_mirrored": ((actions, 110), np.dtype(np.float32)),
        "dense_original": ((actions, 32, 478, 3), np.dtype(np.float32)),
        "dense_mirrored": ((actions, 32, 478, 3), np.dtype(np.float32)),
        "dense_valid_mask": ((actions, 32), np.dtype(bool)),
        "dense_available": ((actions,), np.dtype(bool)),
        "dense_timestamps": ((actions, 32), np.dtype(np.float32)),
        "action_mask": ((actions,), np.dtype(bool)),
        "action_codes": ((actions,), np.dtype(np.int64)),
    }
    result = {}
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if type(value) is not np.ndarray or value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{name} differs from the deployment tensor contract")
        result[name] = np.array(value, copy=True)[None, ...]
    if (
        not np.isfinite(arrays["clinical_original"]).all()
        or not np.isfinite(arrays["clinical_mirrored"]).all()
        or not np.isfinite(arrays["dense_original"]).all()
        or not np.isfinite(arrays["dense_mirrored"]).all()
        or not np.isfinite(arrays["dense_timestamps"]).all()
        or not arrays["action_mask"].all()
        or np.any(arrays["action_codes"] < 0)
        or np.any(arrays["action_codes"] >= len(ACTION_VOCAB))
        or len(np.unique(arrays["action_codes"])) != actions
        or np.any(arrays["dense_available"] & ~arrays["action_mask"])
        or np.any(arrays["dense_valid_mask"] != arrays["dense_available"][:, None])
    ):
        raise ValueError("request failed finite, mask, or action-code validation")
    available = arrays["dense_available"]
    if protocol == "free_motion_four_window":
        if available.any():
            raise ValueError("free-motion protocol cannot carry dense trajectories")
    elif not available.all():
        raise ValueError("scripted deployment protocols require all dense actions")
    unavailable = ~available
    if (
        np.any(arrays["dense_original"][unavailable] != 0.0)
        or np.any(arrays["dense_mirrored"][unavailable] != 0.0)
        or np.any(arrays["dense_timestamps"][unavailable] != 0.0)
    ):
        raise ValueError("unavailable dense actions cannot hide signal")
    if available.any() and np.any(np.diff(arrays["dense_timestamps"][available], axis=1) <= 0.0):
        raise ValueError("available dense actions require increasing seconds")
    return result


__all__ = [
    "DEPLOYMENT_MODEL_ID",
    "DEPLOYMENT_SCHEMA",
    "PROTOCOL_TASK_CODES",
    "Prediction",
    "SharedV8Predictor",
    "implementation_sha256",
    "load_release",
    "validate_request_arrays",
    "write_release",
]
