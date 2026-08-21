"""Public, checksum-bound three-seed release contract for Shared V9 / BLV9-009."""
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
from typing import Mapping, Sequence
import zipfile

import numpy as np
import torch

from src.deployment.shared_v8_release import (
    PROTOCOL_TASK_CODES,
    validate_request_arrays,
)
from src.models.broad_literature_candidate_registry_v9 import candidate_registry_v9
from src.models.broad_literature_shared_router_v9 import BroadLiteratureSharedRouterV9


RESEARCH_SCHEMA = "shared_v9_research_release_v1"
RESEARCH_MODEL_ID = "broad_literature_shared_v9_blv9_009_ensemble"
RESEARCH_CANDIDATE_ID = "BLV9-009"
RESEARCH_SEEDS = (0, 1, 2)
_STATE_PREFIX = "state__"
_SCALER_MEAN = "scaler__mean"
_SCALER_SCALE = "scaler__scale"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IMPLEMENTATION_FILES = (
    "src/deployment/shared_v9_research_release.py",
    "src/deployment/shared_v8_release.py",
    "src/models/broad_literature_candidate_registry_v9.py",
    "src/models/broad_literature_shared_router_v9.py",
    "src/models/residual_shared_router_v8.py",
    "src/models/script_aware_shared_router_v6.py",
    "src/models/medically_gated_shared_encoder_v2.py",
    "src/models/medical_shared_candidate_registry_v2.py",
    "src/models/dense_clinical_shared_encoder_v1.py",
)


def _candidate():
    rows = tuple(
        row for row in candidate_registry_v9()
        if row.candidate_id == RESEARCH_CANDIDATE_ID
    )
    if len(rows) != 1 or rows[0].mechanism != "masked_clinical_reconstruction":
        raise RuntimeError("the exact frozen Shared V9 candidate is unavailable")
    return rows[0]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_FILES:
        payload = (_PROJECT_ROOT / relative).read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


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
        raise ValueError("V9 scaler differs from the frozen 110D contract")
    return np.array(mean, copy=True), np.array(scale, copy=True)


def _validate_provenance(provenance: Mapping[str, object]) -> dict[str, object]:
    if type(provenance) is not dict or set(provenance) != {
        "git_commit", "training_seeds", "training_epochs", "training_device",
        "source_counts", "source_commitments",
    }:
        raise ValueError("V9 release provenance differs from the closed schema")
    counts = provenance["source_counts"]
    commitments = provenance["source_commitments"]
    if (
        type(provenance["git_commit"]) is not str
        or _COMMIT.fullmatch(provenance["git_commit"]) is None
        or provenance["training_seeds"] != list(RESEARCH_SEEDS)
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
        raise ValueError("V9 release provenance values are not frozen")
    return {
        "git_commit": provenance["git_commit"],
        "training_seeds": list(RESEARCH_SEEDS),
        "training_epochs": 20,
        "training_device": "NVIDIA H200",
        "source_counts": dict(counts),
        "source_commitments": dict(commitments),
    }


def _state_arrays(model: BroadLiteratureSharedRouterV9) -> dict[str, np.ndarray]:
    if (
        type(model) is not BroadLiteratureSharedRouterV9
        or model.candidate.candidate_id != RESEARCH_CANDIDATE_ID
    ):
        raise ValueError("only BLV9-009 may be published as Shared V9")
    return {
        _STATE_PREFIX + name: tensor.detach().cpu().contiguous().numpy()
        for name, tensor in model.state_dict().items()
    }


def _npz_payload(
    model: BroadLiteratureSharedRouterV9,
    mean: np.ndarray,
    scale: np.ndarray,
) -> bytes:
    arrays = _state_arrays(model)
    arrays[_SCALER_MEAN] = mean
    arrays[_SCALER_SCALE] = scale
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


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
    models: Sequence[BroadLiteratureSharedRouterV9],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Publish one immutable public V9 ensemble directory."""
    if not isinstance(output, Path) or output.exists() or output.is_symlink():
        raise FileExistsError("V9 research release already exists")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("V9 release parent must be an existing real directory")
    if type(models) not in {tuple, list} or len(models) != len(RESEARCH_SEEDS):
        raise ValueError("V9 release requires the exact three-seed ensemble")
    mean, scale = _validate_scaler(scaler_mean, scaler_scale)
    metadata = _validate_provenance(provenance)
    payloads = tuple(_npz_payload(model, mean, scale) for model in models)
    weights = [
        {
            "seed": seed,
            "file": f"weights-seed{seed}.npz",
            "sha256": _sha256(payload),
            "bytes": len(payload),
        }
        for seed, payload in zip(RESEARCH_SEEDS, payloads)
    ]
    manifest = {
        "schema_version": RESEARCH_SCHEMA,
        "model_id": RESEARCH_MODEL_ID,
        "candidate_id": RESEARCH_CANDIDATE_ID,
        "mechanism": "masked_clinical_reconstruction",
        "status": "locked_research_model_not_clinically_validated",
        "input_schema": "shared_clinical_action_bag_npz_v1",
        "protocol_task_codes": dict(PROTOCOL_TASK_CODES),
        "ensemble_seeds": list(RESEARCH_SEEDS),
        "aggregation": "arithmetic_mean_probability",
        "threshold": 0.5,
        "weights": weights,
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
        for row, payload in zip(weights, payloads):
            _write_bytes(stage / row["file"], payload)
        _write_bytes(stage / "manifest.json", manifest_payload)
        os.rename(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("release member must be a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError("release member size is outside the frozen bound")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError("release member exceeded the frozen bound")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev, after.st_ino, after.st_size
        ) or total != before.st_size:
            raise ValueError("release member changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_npz(payload: bytes, expected: set[str]) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            canonical = {name + ".npy" for name in expected}
            if (
                len(names) != len(set(names))
                or set(names) != canonical
                or any(
                    PurePosixPath(name).name != name
                    or name.startswith(".")
                    or member.file_size > 32 * 1024 * 1024
                    for name, member in zip(names, members)
                )
                or sum(member.file_size for member in members) > 128 * 1024 * 1024
            ):
                raise ValueError("V9 weights members differ from the closed schema")
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            if len(saved.files) != len(set(saved.files)) or set(saved.files) != expected:
                raise ValueError("V9 weights fields differ from the closed schema")
            return {name: np.array(saved[name], copy=True) for name in saved.files}
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError) as exc:
        raise ValueError("V9 weights archive is invalid") from exc


def _validate_manifest(manifest: dict[str, object]) -> None:
    if set(manifest) != {
        "schema_version", "model_id", "candidate_id", "mechanism", "status",
        "input_schema", "protocol_task_codes", "ensemble_seeds", "aggregation",
        "threshold", "weights", "implementation_sha256", "provenance", "claims",
    }:
        raise ValueError("V9 release manifest fields differ from the closed schema")
    weights = manifest["weights"]
    if (
        manifest["schema_version"] != RESEARCH_SCHEMA
        or manifest["model_id"] != RESEARCH_MODEL_ID
        or manifest["candidate_id"] != RESEARCH_CANDIDATE_ID
        or manifest["mechanism"] != "masked_clinical_reconstruction"
        or manifest["status"] != "locked_research_model_not_clinically_validated"
        or manifest["input_schema"] != "shared_clinical_action_bag_npz_v1"
        or manifest["protocol_task_codes"] != PROTOCOL_TASK_CODES
        or manifest["ensemble_seeds"] != list(RESEARCH_SEEDS)
        or manifest["aggregation"] != "arithmetic_mean_probability"
        or manifest["threshold"] != 0.5
        or type(weights) is not list
        or len(weights) != len(RESEARCH_SEEDS)
        or manifest["implementation_sha256"] != implementation_sha256()
        or manifest["claims"] != {
            "clinical_validation": False, "hb_grade": False, "mayo_accuracy": False,
        }
    ):
        raise ValueError("V9 release manifest values are not frozen")
    for seed, row in zip(RESEARCH_SEEDS, weights):
        if (
            type(row) is not dict
            or set(row) != {"seed", "file", "sha256", "bytes"}
            or row["seed"] != seed
            or row["file"] != f"weights-seed{seed}.npz"
            or type(row["sha256"]) is not str
            or _SHA256.fullmatch(row["sha256"]) is None
            or isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or not 0 < row["bytes"] <= 64 * 1024 * 1024
        ):
            raise ValueError("V9 weight manifest row is invalid")
    _validate_provenance(manifest["provenance"])


@dataclass(frozen=True)
class V9Prediction:
    model_id: str
    protocol: str
    probability: float
    member_probabilities: tuple[float, float, float]
    predicted_class: int
    threshold: float


class SharedV9Predictor:
    def __init__(self, models, mean, scale, manifest, device):
        self.models = models
        self.mean = mean
        self.scale = scale
        self.manifest = manifest
        self.device = device

    def predict(self, protocol: str, arrays: Mapping[str, np.ndarray]) -> V9Prediction:
        normalized = validate_request_arrays(protocol, arrays)
        original = (
            normalized["clinical_original"].astype(np.float64)
            - self.mean[None, None, :]
        ) / self.scale[None, None, :]
        mirrored = (
            normalized["clinical_mirrored"].astype(np.float64)
            - self.mean[None, None, :]
        ) / self.scale[None, None, :]
        inputs = (
            original.astype(np.float32), mirrored.astype(np.float32),
            normalized["dense_original"], normalized["dense_mirrored"],
            normalized["dense_valid_mask"], normalized["dense_available"],
            normalized["dense_timestamps"], normalized["action_mask"],
            normalized["action_codes"],
        )
        tensors = tuple(
            torch.from_numpy(np.array(value, copy=True)).to(self.device)
            for value in inputs
        )
        task_codes = torch.tensor(
            [PROTOCOL_TASK_CODES[protocol]], dtype=torch.long, device=self.device
        )
        probabilities = []
        with torch.inference_mode():
            for model in self.models:
                tokens = model.shared_action_tokens(*tensors)
                logit = model.routed_logits(tokens, tensors[-2], task_codes)
                probabilities.append(float(torch.sigmoid(logit)[0].cpu()))
        members = tuple(probabilities)
        probability = float(np.mean(members))
        if len(members) != 3 or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("V9 ensemble produced an invalid probability")
        return V9Prediction(
            model_id=RESEARCH_MODEL_ID,
            protocol=protocol,
            probability=probability,
            member_probabilities=members,
            predicted_class=int(probability >= 0.5),
            threshold=0.5,
        )


def load_release(root: Path, *, device: str) -> SharedV9Predictor:
    if not isinstance(root, Path) or root.is_symlink() or not root.is_dir():
        raise ValueError("V9 release root must be a real directory")
    manifest_payload = _read_regular_file(root / "manifest.json", maximum_bytes=64 * 1024)
    manifest = _load_json(manifest_payload)
    _validate_manifest(manifest)
    runtime = torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (
        runtime.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested V9 device is unavailable")
    loaded_models = []
    common_mean = None
    common_scale = None
    for row in manifest["weights"]:
        payload = _read_regular_file(root / row["file"], maximum_bytes=64 * 1024 * 1024)
        if len(payload) != row["bytes"] or _sha256(payload) != row["sha256"]:
            raise ValueError("V9 weights differ from the release manifest")
        model = BroadLiteratureSharedRouterV9(_candidate())
        state = model.state_dict()
        expected = {_STATE_PREFIX + name for name in state} | {
            _SCALER_MEAN, _SCALER_SCALE,
        }
        arrays = _load_npz(payload, expected)
        mean, scale = _validate_scaler(
            arrays.pop(_SCALER_MEAN), arrays.pop(_SCALER_SCALE)
        )
        if common_mean is None:
            common_mean, common_scale = mean, scale
        elif not np.array_equal(mean, common_mean) or not np.array_equal(scale, common_scale):
            raise ValueError("V9 ensemble members do not share one frozen scaler")
        loaded = {}
        for name, tensor in state.items():
            array = arrays[_STATE_PREFIX + name]
            expected_array = tensor.detach().cpu().numpy()
            if array.shape != expected_array.shape or array.dtype != expected_array.dtype:
                raise ValueError("a V9 tensor differs from the frozen state schema")
            loaded[name] = torch.from_numpy(np.array(array, copy=True))
        model.load_state_dict(loaded, strict=True)
        model.to(runtime).eval()
        loaded_models.append(model)
    return SharedV9Predictor(
        tuple(loaded_models), common_mean, common_scale, manifest, runtime
    )


__all__ = [
    "RESEARCH_CANDIDATE_ID",
    "RESEARCH_MODEL_ID",
    "RESEARCH_SCHEMA",
    "RESEARCH_SEEDS",
    "SharedV9Predictor",
    "V9Prediction",
    "implementation_sha256",
    "load_release",
    "write_release",
]
