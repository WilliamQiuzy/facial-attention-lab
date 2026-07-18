"""Synthetic-safe masked-span pretraining core for dynamic facial geometry.

The module defines source boundaries, split/scaling contracts, checkpoint
validation, and the reconstruction model.  It intentionally has no real-data
training loop and no outer-test path.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import stat
import sys
import weakref
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from ..models.dynamic_landmark import DynamicLandmarkModel


CHECKPOINT_RAVDESS_ONLY = "ravdess_only"
CHECKPOINT_RAVDESS_MAYO = "ravdess_then_mayo"
CHECKPOINT_TYPES = (CHECKPOINT_RAVDESS_ONLY, CHECKPOINT_RAVDESS_MAYO)
SSL_CHECKPOINT_SCHEMA = "dynamic_landmark_ssl_v1"
SSL_CHECKPOINT_RECEIPT_SCHEMA = "dynamic_landmark_ssl_checkpoint_receipt_v1"
SSL_CHECKPOINT_RECEIPT_V2_SCHEMA = "dynamic_landmark_ssl_checkpoint_receipt_v2"
SSL_STAGE_EVIDENCE_V1_SCHEMA = "dynamic_landmark_ssl_stage_evidence_v1"
SSL_STAGE_EVIDENCE_SCHEMA = "dynamic_landmark_ssl_stage_evidence_v2"
SSL_TRAINING_RECEIPT_SCHEMA = "dynamic_landmark_ssl_training_receipt_v1"
SSL_SEEDS = (0, 1, 2)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RAVDESS_STAGE = "ravdess"
MAYO_DEVELOPMENT_STAGE = "mayo"
RAVDESS_SOURCE = "ravdess_openface_semantic23"
MAYO_SOURCE = "mayo_mediapipe_clinical23_development_only"
SSL_MANIFEST_SCHEMA = "dynamic_landmark_ssl_manifest_v1"
SSL_CONFIG_SCHEMA = "dynamic_landmark_ssl_config_v1"
SSL_SPLIT_SCHEMA = "dynamic_landmark_ssl_split_v1"
SSL_SCALER_SCHEMA = "dynamic_landmark_ssl_scaler_v1"
SSL_MANIFEST_V2_SCHEMA = "dynamic_landmark_ssl_manifest_v2"
SSL_CONFIG_V2_SCHEMA = "dynamic_landmark_ssl_config_v2"
SSL_SPLIT_V2_SCHEMA = "dynamic_landmark_ssl_split_v2"
SSL_SCALER_V2_SCHEMA = "dynamic_landmark_ssl_scaler_v2"
BRIDGE_RECEIPT_SCHEMA = "dynamic_landmark_bridge_receipt_v1"
_MAX_FROZEN_JSON_BYTES = 64 * 1024 * 1024
_MAX_FROZEN_BUNDLE_BYTES = 100 * 1024 * 1024
_AUTHORIZATION_MARKER = object()


class PretrainingLockedError(RuntimeError):
    """Real pretraining inputs are not yet frozen and authorized."""


@dataclass(frozen=True)
class SSLGroupSplit:
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    unit: str
    claim_unit: str
    patient_held_out: bool


@dataclass(frozen=True)
class ResampledTrajectory:
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    source_frame_indices: np.ndarray
    source_step: int
    sample_rate_hz: float = 30.0


@dataclass(frozen=True)
class SourceScaler:
    source: str
    mean: torch.Tensor
    scale: torch.Tensor
    fit_indices: tuple[int, ...]

    def transform(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        source: str,
    ) -> torch.Tensor:
        if source != self.source:
            raise ValueError("source scaler cannot be applied across source boundaries")
        if features.ndim < 3 or features.shape[-1] != self.mean.numel():
            raise ValueError("source scaler feature shape is incompatible")
        if valid_mask.shape != features.shape[:-1] or valid_mask.dtype != torch.bool:
            raise ValueError("source scaler mask must be boolean with feature leading shape")
        if features.device != valid_mask.device:
            raise ValueError("source scaler inputs must share one device")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("source scaler input must contain finite floating values")
        if (
            self.mean.ndim != 1
            or self.scale.shape != self.mean.shape
            or not self.mean.is_floating_point()
            or not self.scale.is_floating_point()
            or not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.scale).all()
            or bool((self.scale <= 0).any())
        ):
            raise ValueError("source scaler state must be finite with positive scale")
        mean = self.mean.to(device=features.device, dtype=features.dtype)
        scale = self.scale.to(device=features.device, dtype=features.dtype)
        transformed = (features - mean) / scale
        return torch.where(
            valid_mask.unsqueeze(-1), transformed, torch.zeros_like(transformed)
        )


@dataclass(frozen=True)
class SSLStageEvidence:
    schema_version: str
    stage: str
    source: str
    manifest_sha256: str
    cache_commitment_sha256: str
    cache_count: int
    config_sha256: str
    split_unit: str
    claim_unit: str
    patient_held_out: bool
    train_indices_sha256: str
    heldout_indices_sha256: str
    group_ids_sha256: str
    scaler_sha256: str
    split_artifact_sha256: str
    scaler_artifact_sha256: str
    train_count: int
    heldout_count: int
    development_only: bool
    prior_checkpoint_sha256: str | None
    evidence_sha256: str
    mode: str | None = None
    bridge_receipt_sha256: str | None = None
    receipt_hmac: str | None = None
    receipt_file_identity_sha256: str | None = None
    canonical_key_identity_sha256: str | None = None
    sample_ids_sha256: str | None = None
    source_unit_ids_sha256: str | None = None
    cache_integrity_ids_sha256: str | None = None
    original_mapping_sha256: str | None = None
    bundle_sha256: str | None = None
    bundle_size_bytes: int | None = None
    bundle_file_count: int | None = None
    sample_count: int | None = None
    source_unit_count: int | None = None
    unique_group_count: int | None = None
    upstream_cache_count: int | None = None
    exclusion_count: int | None = None
    feature_names_sha256: str | None = None
    adapter_sha256: str | None = None
    temporal_policy_sha256: str | None = None
    bridge_generation_sha256: str | None = None
    upstream_manifest_commitments_sha256: str | None = None
    upstream_generation_closure_hmac: str | None = None
    source_schema: str | None = None
    _runtime_authorization: object | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }


@dataclass(frozen=True)
class _SSLStageAuthorization:
    marker: object
    evidence_sha256: str
    manifest_path: Path
    config_path: Path
    split_artifact_path: Path
    scaler_artifact_path: Path
    split: SSLGroupSplit
    scaler: SourceScaler
    group_ids: tuple[str, ...]
    training_config: Mapping[str, object]
    cache_artifacts: tuple[_SSLCacheArtifactAuthorization, ...]
    cache_commitment_sha256: str
    cache_count: int
    prior_ravdess_checkpoint: Mapping[str, object] | None
    prior_ravdess_evidence: SSLStageEvidence | None
    frozen_stage: _FrozenSSLStageAuthorization | None = None


@dataclass(frozen=True)
class _SSLCacheArtifactAuthorization:
    path: Path
    sha256: str
    identity: _RegularFileIdentity

    @property
    def inode(self) -> int:
        return self.identity.inode


@dataclass(frozen=True)
class _RegularFileIdentity:
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _FrozenStageSnapshot:
    stage: str
    mode: str
    files: Mapping[str, _SSLCacheArtifactAuthorization]
    receipt: Mapping[str, object]
    manifest: Mapping[str, object]
    config: Mapping[str, object]
    split: Mapping[str, object]
    scaler: Mapping[str, object]


@dataclass(frozen=True)
class _FrozenSSLStageAuthorization:
    marker: object
    stage: str
    mode: str
    inputs_root: Path
    bridge_root: Path
    ravdess_authorizer: Callable[[], object]
    mayo_authorizer: Callable[[], object]
    producer_sha256: str
    snapshot: _FrozenStageSnapshot


@dataclass(frozen=True)
class SSLTrainingReceipt:
    schema_version: str
    stage: str
    source: str
    seed: int
    stage_evidence_sha256: str
    cache_binding_sha256: str
    cache_count: int
    manifest_sha256: str
    config_sha256: str
    split_artifact_sha256: str
    scaler_artifact_sha256: str
    train_indices_sha256: str
    heldout_indices_sha256: str
    group_ids_sha256: str
    scaler_sha256: str
    prior_checkpoint_sha256: str | None
    pre_state_sha256: str
    post_state_sha256: str
    baseline_state_sha256: str
    fresh_untrained_state_sha256: str
    train_mask_schedule_sha256: str
    heldout_mask_schedule_sha256: str
    train_trace_sha256: str
    heldout_report_sha256: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_policy: str
    span_length: int
    spans_per_window: int
    device: str
    optimizer_steps: int
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class SSLTrainingResult:
    model: DynamicLandmarkSSLModel
    stage_evidence: SSLStageEvidence
    training_receipt: SSLTrainingReceipt
    heldout_report: Mapping[str, object]
    _runtime_authorization: object | None = field(
        default=None, init=False, repr=False, compare=False,
    )


@dataclass(frozen=True)
class _SSLTrainingResultAuthorization:
    marker: object
    result_reference: weakref.ReferenceType[SSLTrainingResult]
    model_reference: weakref.ReferenceType[DynamicLandmarkSSLModel]
    receipt_reference: weakref.ReferenceType[SSLTrainingReceipt]
    heldout_report: Mapping[str, object]
    stage_evidence_sha256: str
    cache_binding_sha256: str
    post_state_sha256: str
    heldout_report_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class _MayoModelAuthorization:
    marker: object
    model_reference: weakref.ReferenceType[DynamicLandmarkSSLModel]
    prior_checkpoint_sha256: str


class SSLCheckpointPayload(dict):
    """Checkpoint mapping carrying non-serialized runtime authorization."""

    def __init__(self, value: Mapping[str, object]):
        super().__init__(value)
        self._runtime_authorization: _SSLCheckpointPayloadAuthorization | None = None


@dataclass(frozen=True)
class _SSLCheckpointPayloadAuthorization:
    marker: object
    payload_reference: weakref.ReferenceType[SSLCheckpointPayload]
    checkpoint_fingerprint: str
    stage_evidence_sha256: str
    checkpoint_path: Path | None = None
    receipt_reference: weakref.ReferenceType[SSLCheckpointReceipt] | None = None


@dataclass(frozen=True)
class SSLCheckpointReceipt:
    schema_version: str
    checkpoint_name: str
    checkpoint_type: str
    checkpoint_fingerprint: str
    checkpoint_file_sha256: str
    checkpoint_file_identity_sha256: str
    receipt_file_identity_sha256: str
    stage_evidence_sha256: str
    receipt_sha256: str
    stage_authority_hmac: str | None = None
    _runtime_authorization: object | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }


@dataclass(frozen=True)
class _SSLCheckpointReceiptAuthorization:
    marker: object
    receipt_reference: weakref.ReferenceType[SSLCheckpointReceipt]
    checkpoint_path: Path
    receipt_path: Path
    receipt_file_sha256: str
    checkpoint_identity: _RegularFileIdentity
    receipt_identity: _RegularFileIdentity
    stage_evidence_sha256: str


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def make_contiguous_span_mask(
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
    source_frame_indices: torch.Tensor,
    *,
    expected_source_step: int,
    span_length: int,
    spans_per_window: int,
    seed: int,
) -> torch.Tensor:
    """Choose spans contiguous in validity, time, and the source-frame time base."""
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.ndim != 3:
        raise ValueError("valid_mask must have shape (batch, windows, frames)")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must have bool dtype")
    if not isinstance(timestamps, torch.Tensor) or not isinstance(
        source_frame_indices, torch.Tensor
    ):
        raise ValueError("span provenance must use torch tensors")
    if timestamps.shape != valid_mask.shape or source_frame_indices.shape != valid_mask.shape:
        raise ValueError("span timestamps and source indices must match valid_mask")
    if timestamps.device != valid_mask.device or source_frame_indices.device != valid_mask.device:
        raise ValueError("span mask and temporal provenance must share one device")
    if not timestamps.is_floating_point() or not torch.isfinite(timestamps).all():
        raise ValueError("span timestamps must be finite floating values")
    if source_frame_indices.dtype != torch.int64:
        raise ValueError("span source indices must have int64 dtype")
    if bool((source_frame_indices < 0).any()):
        raise ValueError("span source indices must be nonnegative")
    if bool((timestamps[..., 1:] <= timestamps[..., :-1]).any()):
        raise ValueError("span timestamps must increase strictly within each window")
    if bool((source_frame_indices[..., 1:] <= source_frame_indices[..., :-1]).any()):
        raise ValueError("span source indices must increase strictly within each window")
    expected_source_step = _positive_integer(
        expected_source_step, "expected_source_step"
    )
    span_length = _positive_integer(span_length, "span_length")
    spans_per_window = _positive_integer(spans_per_window, "spans_per_window")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if bool((~valid_mask.reshape(valid_mask.shape[0], -1).any(dim=1)).any()):
        raise ValueError("every recording requires at least one valid frame")

    valid_cpu = valid_mask.detach().to("cpu").numpy()
    timestamps_cpu = timestamps.detach().to("cpu").numpy()
    indices_cpu = source_frame_indices.detach().to("cpu").numpy()
    selected = np.zeros_like(valid_cpu, dtype=bool)
    rng = np.random.default_rng(int(seed))
    for batch_index in range(valid_cpu.shape[0]):
        for window_index in range(valid_cpu.shape[1]):
            row = valid_cpu[batch_index, window_index]
            valid_count = int(row.sum())
            candidates = [
                start
                for start in range(0, row.shape[0] - span_length + 1)
                if (
                    row[start:start + span_length].all()
                    and (
                        span_length == 1
                        or (
                            np.all(
                                np.diff(indices_cpu[
                                    batch_index, window_index,
                                    start:start + span_length,
                                ]) == expected_source_step
                            )
                            and np.allclose(
                                np.diff(timestamps_cpu[
                                    batch_index, window_index,
                                    start:start + span_length,
                                ]),
                                1.0 / 30.0,
                                rtol=1e-4,
                                atol=1e-5,
                            )
                        )
                    )
                )
            ]
            if not candidates or span_length >= valid_count:
                continue
            order = rng.permutation(candidates).tolist()

            def find_feasible(
                position: int,
                starts: tuple[int, ...],
                occupied: np.ndarray,
            ) -> np.ndarray | None:
                if len(starts) == spans_per_window:
                    if int((occupied & row).sum()) < valid_count:
                        return occupied
                    return None
                remaining_needed = spans_per_window - len(starts)
                if len(order) - position < remaining_needed:
                    return None
                for candidate_index in range(position, len(order)):
                    start = order[candidate_index]
                    candidate = np.zeros_like(row, dtype=bool)
                    candidate[start:start + span_length] = True
                    if bool((occupied & candidate).any()):
                        continue
                    result = find_feasible(
                        candidate_index + 1,
                        starts + (start,),
                        occupied | candidate,
                    )
                    if result is not None:
                        return result
                return None

            solution = find_feasible(0, (), np.zeros_like(row, dtype=bool))
            if solution is None:
                raise ValueError("cannot place the requested number of nonoverlapping spans")
            selected[batch_index, window_index] = solution
    if any(not selected[index].any() for index in range(selected.shape[0])):
        raise ValueError("every recording must contain at least one eligible masked span")
    return torch.as_tensor(selected, dtype=torch.bool, device=valid_mask.device)


def ssl_gap_safe_per_second_differences(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    timestamps: torch.Tensor,
    source_frame_indices: torch.Tensor,
    *,
    expected_source_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate a 30-Hz SSL view without discarding its source time base."""
    if not all(isinstance(item, torch.Tensor) for item in (
        features, valid_mask, timestamps, source_frame_indices,
    )):
        raise ValueError("SSL temporal inputs must be torch tensors")
    if features.ndim < 3:
        raise ValueError("SSL features require batch, time, and feature dimensions")
    expected_shape = features.shape[:-1]
    if (
        valid_mask.shape != expected_shape
        or timestamps.shape != expected_shape
        or source_frame_indices.shape != expected_shape
    ):
        raise ValueError("SSL temporal provenance must match feature leading dimensions")
    if len({
        features.device, valid_mask.device, timestamps.device,
        source_frame_indices.device,
    }) != 1:
        raise ValueError("SSL temporal inputs must share one device")
    if valid_mask.dtype != torch.bool:
        raise ValueError("SSL temporal valid_mask must have bool dtype")
    if not features.is_floating_point() or not timestamps.is_floating_point():
        raise ValueError("SSL features and timestamps must use floating dtype")
    if source_frame_indices.dtype != torch.int64:
        raise ValueError("SSL source frame indices must have int64 dtype")
    if not torch.isfinite(features).all() or not torch.isfinite(timestamps).all():
        raise ValueError("SSL temporal inputs must be finite")
    if bool((source_frame_indices < 0).any()):
        raise ValueError("SSL source frame indices must be nonnegative")
    if bool((source_frame_indices[..., 1:] <= source_frame_indices[..., :-1]).any()):
        raise ValueError("SSL source frame indices must increase strictly")
    if bool((timestamps[..., 1:] <= timestamps[..., :-1]).any()):
        raise ValueError("SSL timestamps must increase strictly")
    expected_source_step = _positive_integer(
        expected_source_step, "expected_source_step"
    )

    differences = torch.zeros_like(features)
    difference_mask = torch.zeros_like(valid_mask)
    if features.shape[-2] < 2:
        return differences, difference_mask
    elapsed = timestamps[..., 1:] - timestamps[..., :-1]
    index_step = (
        source_frame_indices[..., 1:] - source_frame_indices[..., :-1]
    )
    time_contiguous = torch.isclose(
        elapsed,
        torch.full_like(elapsed, 1.0 / 30.0),
        rtol=1e-4,
        atol=1e-5,
    )
    endpoint_valid = valid_mask[..., :-1] & valid_mask[..., 1:]
    valid_difference = (
        (index_step == expected_source_step) & time_contiguous & endpoint_valid
    )
    safe_elapsed = torch.where(valid_difference, elapsed, torch.ones_like(elapsed))
    raw = (
        features[..., 1:, :] - features[..., :-1, :]
    ) / safe_elapsed.unsqueeze(-1)
    raw = torch.where(valid_difference.unsqueeze(-1), raw, torch.zeros_like(raw))
    if not torch.isfinite(raw).all():
        raise ValueError("valid SSL per-second differences must be finite")
    differences[..., 1:, :] = raw
    difference_mask[..., 1:] = valid_difference
    return differences, difference_mask


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reconstruction_mask: torch.Tensor,
) -> torch.Tensor:
    """SmoothL1 over masked valid positions only; unmasked values are ignored."""
    if not all(isinstance(item, torch.Tensor) for item in (
        prediction, target, reconstruction_mask
    )):
        raise ValueError("reconstruction loss inputs must be tensors")
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same feature shape")
    if reconstruction_mask.shape != target.shape[:-1] or reconstruction_mask.dtype != torch.bool:
        raise ValueError("reconstruction mask must match target leading dimensions")
    if len({prediction.device, target.device, reconstruction_mask.device}) != 1:
        raise ValueError("reconstruction loss inputs must share one device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("prediction and target must use floating dtype")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("reconstruction values must be finite")
    if not bool(reconstruction_mask.any()):
        raise ValueError("reconstruction mask must select at least one position")
    return functional.smooth_l1_loss(
        prediction[reconstruction_mask], target[reconstruction_mask], reduction="mean"
    )


def deterministic_group_split(
    group_ids: Sequence[str],
    *,
    heldout_fraction: float,
    seed: int,
    unit: str,
) -> SSLGroupSplit:
    """Deterministically hold out complete actors or complete recordings."""
    groups = tuple(group_ids)
    if unit not in {"actor", "recording"}:
        raise ValueError("split unit must be actor or recording")
    if len(groups) < 2 or any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("group_ids must contain at least two nonempty strings")
    if not math.isfinite(heldout_fraction) or not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must lie strictly within (0, 1)")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("at least two independent groups are required")
    ordered = sorted(
        unique,
        key=lambda group: hashlib.sha256(
            f"dynamic-landmark-ssl:{unit}:{int(seed)}:{group}".encode("utf-8")
        ).digest(),
    )
    heldout_count = max(1, min(len(unique) - 1, int(math.floor(
        len(unique) * heldout_fraction + 0.5
    ))))
    heldout_groups = set(ordered[:heldout_count])
    heldout = np.asarray(
        [index for index, group in enumerate(groups) if group in heldout_groups],
        dtype=np.int64,
    )
    train = np.asarray(
        [index for index, group in enumerate(groups) if group not in heldout_groups],
        dtype=np.int64,
    )
    claim = (
        "actor_held_out" if unit == "actor"
        else "recording_held_out_not_patient_held_out"
    )
    return SSLGroupSplit(
        train_indices=train,
        heldout_indices=heldout,
        unit=unit,
        claim_unit=claim,
        patient_held_out=False,
    )


def resample_trajectory_30hz(
    features: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    source_frame_indices: np.ndarray,
) -> ResampledTrajectory:
    """Select exact source-frame positions on a 30-Hz grid without filling gaps."""
    values = np.asarray(features)
    mask = np.asarray(valid_mask)
    times = np.asarray(timestamps)
    source_indices = np.asarray(source_frame_indices)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("features must have shape (time, feature) with at least two rows")
    if mask.shape != (values.shape[0],) or mask.dtype != np.bool_:
        raise ValueError("valid_mask must be a boolean time vector")
    if times.shape != (values.shape[0],) or times.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("timestamps must be a numeric time vector")
    if source_indices.shape != (values.shape[0],) or source_indices.dtype.kind not in {
        "i", "u",
    }:
        raise ValueError("source_frame_indices must be an integer time vector")
    if values.dtype.kind not in {"f", "i", "u"} or not np.isfinite(values).all():
        raise ValueError("trajectory features must be finite numeric values")
    if not np.isfinite(times).all() or not np.all(times[1:] > times[:-1]):
        raise ValueError("source timestamps must be finite and strictly increasing")
    source_indices = source_indices.astype(np.int64, copy=False)
    if np.any(source_indices < 0) or not np.all(
        source_indices[1:] > source_indices[:-1]
    ):
        raise ValueError("source frame indices must be nonnegative and strictly increasing")
    if not mask.any():
        raise ValueError("resampling rejects an all-masked trajectory")

    rate = 30.0
    source_index_span = float(source_indices[-1] - source_indices[0])
    source_time_span = float(times[-1] - times[0])
    source_rate_hz = source_index_span / source_time_span
    if not math.isfinite(source_rate_hz) or source_rate_hz <= 0:
        raise ValueError("cannot infer a positive source-frame time base")
    source_step = int(round(source_rate_hz / rate))
    if source_step < 1 or not math.isclose(
        source_rate_hz / rate, source_step, rel_tol=0.0, abs_tol=0.02
    ):
        raise ValueError(
            "no-interpolation 30-Hz resampling requires an integer source-rate multiple"
        )
    measured_intercept = float(np.median(
        source_indices.astype(np.float64) - source_rate_hz * times.astype(np.float64)
    ))
    reconstructed_times = (
        source_indices.astype(np.float64) - measured_intercept
    ) / source_rate_hz
    source_time_tolerance = max(1e-6, 0.2 / source_rate_hz)
    if np.max(np.abs(reconstructed_times - times)) > source_time_tolerance:
        raise ValueError("source timestamps and frame indices do not share a stable time base")

    nominal_source_rate_hz = rate * source_step
    phase = int(round(float(np.median(
        source_indices.astype(np.float64)
        - nominal_source_rate_hz * times.astype(np.float64)
    ))))
    first_tick = int(math.ceil(
        (int(source_indices[0]) - phase) / source_step
    ))
    last_tick = int(math.floor(
        (int(source_indices[-1]) - phase) / source_step
    ))
    if last_tick < first_tick:
        raise ValueError("trajectory does not span one 30-Hz sample")
    ticks = np.arange(first_tick, last_tick + 1, dtype=np.int64)
    grid = ticks.astype(np.float64) / rate
    output = np.zeros((grid.size, values.shape[1]), dtype=np.float32)
    output_mask = np.zeros(grid.size, dtype=bool)
    expected_source_indices = phase + ticks * source_step
    canonical_tolerance = max(1e-6, 0.49 / nominal_source_rate_hz)
    canonical_source_times = (
        source_indices.astype(np.float64) - phase
    ) / nominal_source_rate_hz
    if np.max(np.abs(canonical_source_times - times)) > canonical_tolerance:
        raise ValueError("source timestamps cannot be mapped safely to canonical 30 Hz")
    source_row_by_index = {
        int(source_index): row_index
        for row_index, source_index in enumerate(source_indices.tolist())
    }
    for output_index, (timestamp, expected_source_index) in enumerate(zip(
        grid, expected_source_indices.tolist()
    )):
        source_row = source_row_by_index.get(int(expected_source_index))
        if source_row is None:
            continue
        if (
            abs(float(times[source_row]) - float(timestamp)) <= canonical_tolerance
            and mask[source_row]
        ):
            output[output_index] = values[source_row].astype(np.float32, copy=False)
            output_mask[output_index] = True
    if not output_mask.any():
        raise ValueError("30-Hz view contains no observed frames")
    return ResampledTrajectory(
        features=output,
        valid_mask=output_mask,
        timestamps=grid,
        source_frame_indices=expected_source_indices,
        source_step=source_step,
    )


def _index_array(values: Sequence[int] | np.ndarray, n_samples: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = np.asarray([int(value) for value in array.tolist()], dtype=np.int64)
    if result.size == 0 or np.unique(result).size != result.size:
        raise ValueError(f"{name} must be nonempty and unique")
    if np.any(result < 0) or np.any(result >= n_samples):
        raise ValueError(f"{name} is outside the dataset")
    return result


def fit_source_scaler(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    source: str,
    fit_indices: Sequence[int] | np.ndarray,
    heldout_indices: Sequence[int] | np.ndarray,
) -> SourceScaler:
    """Fit one detector/source scaler using its explicit training partition only."""
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a nonempty identifier")
    if features.ndim < 3 or valid_mask.shape != features.shape[:-1]:
        raise ValueError("source scaler feature/mask shapes are incompatible")
    if valid_mask.dtype != torch.bool or features.device != valid_mask.device:
        raise ValueError("source scaler requires same-device boolean mask")
    fit = _index_array(fit_indices, features.shape[0], "fit_indices")
    heldout = _index_array(heldout_indices, features.shape[0], "heldout_indices")
    if set(fit.tolist()).intersection(heldout.tolist()):
        raise ValueError("heldout samples cannot enter source scaler state")
    if set(fit.tolist()).union(heldout.tolist()) != set(range(features.shape[0])):
        raise ValueError("source scaler train/heldout partition must cover every row")
    device_indices = torch.as_tensor(fit, dtype=torch.int64, device=features.device)
    train_features = features.index_select(0, device_indices)
    train_mask = valid_mask.index_select(0, device_indices)
    if not train_features.is_floating_point() or not torch.isfinite(train_features).all():
        raise ValueError("source scaler training rows must be finite floating values")
    rows = train_features[train_mask]
    if rows.shape[0] < 2:
        raise ValueError("source scaler requires at least two valid training frames")
    mean = rows.mean(dim=0)
    scale = rows.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return SourceScaler(
        source=source,
        mean=mean.detach().cpu(),
        scale=scale.detach().cpu(),
        fit_indices=tuple(int(index) for index in fit.tolist()),
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_file_snapshot(
    path: str | Path,
    name: str,
) -> tuple[Path, bytes, str]:
    resolved, payload, digest, _ = _regular_file_snapshot_with_identity(path, name)
    return resolved, payload, digest


def _identity_from_stat(status: os.stat_result) -> _RegularFileIdentity:
    return _RegularFileIdentity(
        device=int(status.st_dev),
        inode=int(status.st_ino),
        mode=int(stat.S_IMODE(status.st_mode)),
        uid=int(status.st_uid),
        gid=int(status.st_gid),
        links=int(status.st_nlink),
        size=int(status.st_size),
        mtime_ns=int(status.st_mtime_ns),
        ctime_ns=int(status.st_ctime_ns),
    )


def _regular_file_snapshot_with_identity(
    path: str | Path,
    name: str,
    *,
    max_bytes: int | None = None,
) -> tuple[Path, bytes, str, _RegularFileIdentity]:
    source = Path(path)
    try:
        before = source.lstat()
    except OSError as exc:
        raise ValueError(f"{name} must be a readable regular file") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} must be a nonempty regular file")
    if (
        max_bytes is not None
        and (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
            or before.st_size > max_bytes
        )
    ):
        raise ValueError(f"{name} exceeds its byte bound")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"{name} cannot be opened without following a symlink") from exc
    try:
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or _identity_from_stat(before) != _identity_from_stat(opened_before)
        ):
            raise ValueError(f"{name} changed while it was being authorized")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
        identity = _identity_from_stat(opened_before)
        if identity != _identity_from_stat(opened_after):
            raise ValueError(f"{name} changed while it was being read")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    try:
        after = source.lstat()
    except OSError as exc:
        raise ValueError(f"{name} disappeared while it was being authorized") from exc
    if (
        _identity_from_stat(after) != identity
        or len(payload) != identity.size
    ):
        raise ValueError(f"{name} changed while it was being authorized")
    if not payload:
        raise ValueError(f"{name} must be a nonempty regular file")
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} cannot be resolved") from exc
    return resolved, payload, hashlib.sha256(payload).hexdigest(), identity


def _private_regular_file_snapshot(
    path: str | Path,
    name: str,
) -> tuple[Path, bytes, str, _RegularFileIdentity]:
    resolved, payload, digest, identity = _regular_file_snapshot_with_identity(
        path, name,
    )
    if (
        identity.uid != os.geteuid()
        or identity.mode != 0o600
        or identity.links != 1
    ):
        raise ValueError(f"{name} must be an owner-only single-link file")
    return resolved, payload, digest, identity


def _authorize_cache_artifact(path: str | Path) -> _SSLCacheArtifactAuthorization:
    resolved, _, digest, identity = _regular_file_snapshot_with_identity(
        path, "SSL cache artifact"
    )
    return _SSLCacheArtifactAuthorization(
        path=resolved,
        sha256=digest,
        identity=identity,
    )


def _reopen_authorized_cache_artifacts(
    artifacts: Sequence[_SSLCacheArtifactAuthorization],
) -> tuple[bytes, ...]:
    if not artifacts:
        raise ValueError("authorized stage requires at least one exact cache artifact")
    payloads: list[bytes] = []
    for artifact in artifacts:
        if not isinstance(artifact, _SSLCacheArtifactAuthorization):
            raise ValueError("SSL cache authorization has the wrong type")
        resolved, payload, digest, identity = _regular_file_snapshot_with_identity(
            artifact.path, "SSL cache artifact"
        )
        identity = (
            resolved,
            digest,
            identity,
        )
        expected = (
            artifact.path,
            artifact.sha256,
            artifact.identity,
        )
        if identity != expected:
            raise ValueError("SSL cache artifact changed after stage authorization")
        payloads.append(payload)
    return tuple(payloads)


def _cache_commitment_sha256(
    group_ids: Sequence[str],
    payloads: Sequence[bytes],
) -> str:
    """Commit to keyed row groups and ordered cache bytes as one aggregate."""
    digest = hashlib.sha256()
    digest.update(b"dynamic-landmark-ssl-cache-commitment-v1\x00")
    encoded_groups = json.dumps(
        list(group_ids), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    digest.update(len(encoded_groups).to_bytes(8, "big"))
    digest.update(encoded_groups)
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _parse_training_cache_payloads(
    payloads: Sequence[bytes],
    *,
    stage: str,
    group_ids: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse the exact bytes read from authorized cache descriptors."""
    if stage not in {RAVDESS_STAGE, MAYO_DEVELOPMENT_STAGE}:
        raise ValueError("SSL cache stage is unsupported")
    if not payloads:
        raise ValueError("authorized stage requires at least one cache payload")
    expected_keys = {
        "features", "valid_mask", "timestamps", "source_frame_indices",
        "group_ids",
    }
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in expected_keys}
    for payload in payloads:
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                if set(archive.files) != expected_keys:
                    raise ValueError("SSL cache NPZ schema is not exact")
                for name in expected_keys:
                    arrays[name].append(np.asarray(archive[name]).copy())
        except (OSError, ValueError, TypeError) as exc:
            if (
                isinstance(exc, ValueError)
                and str(exc) == "SSL cache NPZ schema is not exact"
            ):
                raise
            raise ValueError(
                "SSL cache artifact is not a safe exact NPZ bundle"
            ) from exc
    try:
        features = np.concatenate(arrays["features"], axis=0)
        valid_mask = np.concatenate(arrays["valid_mask"], axis=0)
        timestamps = np.concatenate(arrays["timestamps"], axis=0)
        source_indices = np.concatenate(arrays["source_frame_indices"], axis=0)
        observed_group_ids = np.concatenate(arrays["group_ids"], axis=0)
    except ValueError as exc:
        raise ValueError("SSL cache shards have incompatible array shapes") from exc

    width = 23 if stage == RAVDESS_STAGE else 95
    sample_count = len(group_ids)
    expected_leading = (sample_count, 4, 32)
    if features.shape != expected_leading + (width,) or features.dtype != np.float32:
        raise ValueError("SSL cache features violate the frozen shape/dtype contract")
    if valid_mask.shape != expected_leading or valid_mask.dtype != np.bool_:
        raise ValueError("SSL cache valid mask violates the frozen shape/dtype contract")
    if timestamps.shape != expected_leading or timestamps.dtype != np.float32:
        raise ValueError("SSL cache timestamps violate the frozen shape/dtype contract")
    if source_indices.shape != expected_leading or source_indices.dtype != np.int64:
        raise ValueError("SSL cache source indices violate the frozen shape/dtype contract")
    if (
        observed_group_ids.shape != (sample_count,)
        or observed_group_ids.dtype.kind != "U"
        or tuple(str(item) for item in observed_group_ids.tolist()) != group_ids
    ):
        raise ValueError("SSL cache row groups do not match the authorized stage order")
    if not np.isfinite(features).all() or not np.isfinite(timestamps).all():
        raise ValueError("SSL cache values must be finite")
    if not valid_mask.reshape(sample_count, -1).any(axis=1).all():
        raise ValueError("every SSL cache recording requires observed frames")
    expected_timestamps = np.arange(32, dtype=np.float32) / np.float32(30.0)
    if not np.array_equal(
        timestamps,
        np.broadcast_to(expected_timestamps, expected_leading),
    ):
        raise ValueError("SSL cache timestamps must be the exact local 30-Hz axis")
    expected_indices = np.arange(32, dtype=np.int64)
    if not np.array_equal(
        source_indices,
        np.broadcast_to(expected_indices, expected_leading),
    ):
        raise ValueError("SSL cache source indices must be the exact local canonical axis")
    if np.any(features[~valid_mask] != np.float32(0.0)):
        raise ValueError("SSL cache invalid feature rows must be canonical zero")
    return (
        torch.from_numpy(features),
        torch.from_numpy(valid_mask),
        torch.from_numpy(timestamps),
        torch.from_numpy(source_indices),
    )


def _exact_json_value(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _exact_json_value(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _exact_json_value(left, right)
            for left, right in zip(observed, expected)
        )
    if isinstance(expected, float):
        return math.isfinite(observed) and observed == expected
    return observed == expected


def _read_json_artifact(
    path: str | Path,
    name: str,
) -> tuple[Path, dict[str, object], str]:
    resolved, payload, digest = _regular_file_snapshot(path, name)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be one valid UTF-8 JSON object") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} must be one exact JSON object")
    return resolved, value, digest


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("frozen SSL metadata is not canonical JSON") from exc


def _strict_json_mapping(payload: bytes, name: str) -> dict[str, object]:
    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                raise ValueError(f"{name} contains a duplicate or non-string key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=exact_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be one exact ASCII JSON object") from exc
    if type(value) is not dict or _canonical_json_bytes(value) != payload:
        raise ValueError(f"{name} is not exact canonical JSON")
    return value


def _private_file_snapshot(
    path: Path,
    name: str,
    *,
    max_bytes: int,
) -> tuple[_SSLCacheArtifactAuthorization, bytes]:
    resolved, payload, digest, identity = _regular_file_snapshot_with_identity(
        path, name, max_bytes=max_bytes,
    )
    if (
        identity.mode != 0o600
        or identity.uid != os.geteuid()
        or identity.links != 1
    ):
        raise ValueError(f"{name} must be singly-linked owner-only mode 0600")
    return (
        _SSLCacheArtifactAuthorization(
            path=resolved,
            sha256=digest,
            identity=identity,
        ),
        payload,
    )


def _file_identity_sha256(identity: _RegularFileIdentity) -> str:
    return hashlib.sha256(
        b"dynamic-landmark-ssl-file-identity-v1\0"
        + _canonical_json_bytes({
            "device": identity.device,
            "inode": identity.inode,
            "mode": identity.mode,
            "uid": identity.uid,
            "gid": identity.gid,
            "links": identity.links,
            "size": identity.size,
            "mtime_ns": identity.mtime_ns,
            "ctime_ns": identity.ctime_ns,
        })
    ).hexdigest()


def _stable_storage_identity_sha256(identity: _RegularFileIdentity) -> str:
    return hashlib.sha256(
        b"dynamic-landmark-ssl-stable-storage-identity-v1\0"
        + _canonical_json_bytes({
            "device": identity.device,
            "inode": identity.inode,
            "mode": identity.mode,
            "uid": identity.uid,
            "gid": identity.gid,
            "links": identity.links,
        })
    ).hexdigest()


_BRIDGE_RECEIPT_FIELDS = {
    "schema", "stage", "mode", "producer_sha256", "source_schema",
    "upstream_manifest_commitments", "upstream_generation_closure_hmac",
    "sample_ids", "source_unit_ids", "group_ids", "cache_integrity_ids",
    "window_starts", "original_mapping_sha256", "feature_names_sha256",
    "adapter_sha256", "bundle_file_count", "sample_count",
    "source_unit_count", "unique_group_count", "upstream_cache_count",
    "packet_policy", "overlap_pair_count", "covered_canonical_position_count",
    "exclusion_count", "bundle_sha256", "bundle_size_bytes",
    "bridge_stage_closure_hmac", "bridge_generation_sha256",
    "artifact_core_sha256", "canonical_key_identity_sha256",
    "original_canonical_frame_indices", "original_source_frame_indices",
    "original_timestamps", "receipt_hmac",
}
_V2_MANIFEST_FIELDS = {
    "schema_version", "stage", "mode", "source", "source_schema",
    "sample_ids", "source_unit_ids", "group_ids", "sample_count",
    "source_unit_count", "unique_group_count", "upstream_cache_count",
    "exclusion_count",
    "bundle_file_count", "bundle_sha256", "bundle_size_bytes",
    "feature_names_sha256", "adapter_sha256", "temporal_policy_sha256",
    "bridge_generation_sha256", "upstream_manifest_commitments",
    "upstream_generation_closure_hmac", "bridge_receipt_sha256",
    "receipt_hmac",
}
_V2_CONFIG_FIELDS = {
    "schema_version", "stage", "mode", "source", "objective",
    "sample_rate_hz", "seeds", "development_only", "optimizer",
    "learning_rate", "weight_decay", "epochs", "batch_policy",
    "span_length", "spans_per_window", "device",
    "bridge_receipt_sha256", "receipt_hmac",
}
_V2_SPLIT_FIELDS = {
    "schema_version", "stage", "mode", "source", "split_seed",
    "heldout_fraction", "unit", "claim_unit", "train_group_ids",
    "heldout_group_ids", "train_indices", "heldout_indices",
    "bridge_receipt_sha256", "receipt_hmac",
}
_V2_SCALER_FIELDS = {
    "schema_version", "stage", "mode", "source", "fit_indices",
    "fit_source_unit_ids", "unique_frame_key", "fit_unique_frame_count",
    "mean", "scale", "bridge_receipt_sha256", "receipt_hmac",
}


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256 digest")
    return value


def _exact_string_list(value: object, name: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or not value
        or any(type(item) is not str or not item for item in value)
    ):
        raise ValueError(f"{name} must be a nonempty ordered string list")
    return tuple(value)


def _validate_v2_training_config(
    value: Mapping[str, object],
    *,
    stage: str,
    mode: str,
    source: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _V2_CONFIG_FIELDS:
        raise ValueError("receipt-bound training config schema is not exact")
    expected = {
        "schema_version": SSL_CONFIG_V2_SCHEMA,
        "stage": stage,
        "mode": mode,
        "source": source,
        "objective": "masked_span_smooth_l1_only",
        "sample_rate_hz": 30.0,
        "seeds": [0] if mode == "smoke" else [0, 1, 2],
        "development_only": stage == "mayo",
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 1 if mode == "smoke" else 30,
        "batch_policy": "full_train_partition",
        "span_length": 4,
        "spans_per_window": 2,
        "device": "cpu",
    }
    if any(not _exact_json_value(value[name], expected_item)
           for name, expected_item in expected.items()):
        raise ValueError("receipt-bound training config is not mode-exact")
    return dict(value)


def _fit_receipt_bound_source_scaler(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    source: str,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    source_unit_ids: tuple[str, ...],
    original_canonical_frame_indices: object,
) -> tuple[SourceScaler, int, tuple[str, ...]]:
    fit = _index_array(train_indices, features.shape[0], "fit_indices")
    heldout = _index_array(
        heldout_indices, features.shape[0], "heldout_indices"
    )
    if set(fit.tolist()).intersection(heldout.tolist()):
        raise ValueError("heldout samples cannot enter receipt-bound scaler state")
    if set(fit.tolist()).union(heldout.tolist()) != set(range(features.shape[0])):
        raise ValueError("receipt-bound scaler partition must cover every packet")
    if len(source_unit_ids) != features.shape[0]:
        raise ValueError("receipt source-unit order does not align with bundle packets")
    canonical = np.asarray(original_canonical_frame_indices)
    if canonical.dtype.kind not in {"i", "u"} or canonical.shape != valid_mask.shape:
        raise ValueError("receipt original canonical mapping is noncanonical")
    values = features.detach().cpu().numpy()
    masks = valid_mask.detach().cpu().numpy()
    observations: list[np.ndarray] = []
    seen: dict[tuple[str, int], tuple[bool, np.ndarray]] = {}
    fit_sources: list[str] = []
    fit_source_set: set[str] = set()
    for sample_index in fit.tolist():
        source_unit = source_unit_ids[sample_index]
        if source_unit not in fit_source_set:
            fit_source_set.add(source_unit)
            fit_sources.append(source_unit)
        for window_index in range(4):
            for frame_index in range(32):
                key = (
                    source_unit,
                    int(canonical[sample_index, window_index, frame_index]),
                )
                present = bool(masks[sample_index, window_index, frame_index])
                row = values[sample_index, window_index, frame_index]
                prior = seen.get(key)
                if prior is not None:
                    if prior[0] is not present or not np.array_equal(prior[1], row):
                        raise ValueError(
                            "repeated canonical frame has conflicting bundle values"
                        )
                    continue
                seen[key] = (present, row.copy())
                if present:
                    observations.append(row.astype(np.float64, copy=True))
    if not observations:
        raise ValueError("receipt-bound train scaler has no unique valid frames")
    stacked = np.stack(observations, axis=0)
    mean = stacked.mean(axis=0, dtype=np.float64)
    scale = stacked.std(axis=0, dtype=np.float64)
    scale[scale < np.finfo(np.float32).eps] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("receipt-bound train scaler is nonfinite")
    return (
        SourceScaler(
            source=source,
            mean=torch.from_numpy(mean.copy()),
            scale=torch.from_numpy(scale.copy()),
            fit_indices=tuple(int(index) for index in fit.tolist()),
        ),
        len(observations),
        tuple(fit_sources),
    )


def _snapshot_frozen_ssl_stage(
    *,
    stage: str,
    mode: str,
    inputs_root: Path,
    bridge_root: Path,
    live_authorization: object,
) -> _FrozenStageSnapshot:
    if stage not in {"ravdess", "mayo"} or mode not in {"smoke", "formal"}:
        raise ValueError("frozen SSL stage or mode is unsupported")
    source = (
        "ravdess_openface_semantic23"
        if stage == "ravdess"
        else "mayo_mediapipe_clinical23_development_only"
    )
    paths = {
        "receipt": inputs_root / "receipts" / f"{stage}.json",
        "manifest": inputs_root / "artifacts" / stage / "manifest.json",
        "config": inputs_root / "artifacts" / stage / "config.json",
        "split": inputs_root / "artifacts" / stage / "split.json",
        "scaler": inputs_root / "artifacts" / stage / "scaler.json",
        "bundle": bridge_root / "bundles" / f"{stage}_bundle.npz",
    }
    files: dict[str, _SSLCacheArtifactAuthorization] = {}
    payloads: dict[str, bytes] = {}
    for name, path in paths.items():
        file_authorization, payload = _private_file_snapshot(
            path,
            f"frozen {stage} {name}",
            max_bytes=(
                _MAX_FROZEN_BUNDLE_BYTES
                if name == "bundle"
                else _MAX_FROZEN_JSON_BYTES
            ),
        )
        files[name] = file_authorization
        payloads[name] = payload

    receipt = _strict_json_mapping(payloads["receipt"], f"{stage} bridge receipt")
    if set(receipt) != _BRIDGE_RECEIPT_FIELDS:
        raise ValueError(f"{stage} bridge receipt schema is not exact")
    if (
        receipt.get("schema") != BRIDGE_RECEIPT_SCHEMA
        or receipt.get("stage") != stage
        or receipt.get("mode") != mode
    ):
        raise ValueError(f"{stage} bridge receipt contradicts its stage or mode")
    private_key = getattr(live_authorization, "private_key", None)
    key_identity = getattr(
        live_authorization, "key_file_identity_sha256", None
    )
    if type(private_key) is not bytes or len(private_key) != 32:
        raise ValueError(f"{stage} live authorization lacks its canonical key")
    key_identity = _require_sha256(
        key_identity, f"{stage} canonical key identity"
    )
    if receipt.get("canonical_key_identity_sha256") != key_identity:
        raise ValueError(f"{stage} receipt names a different canonical key")
    unsigned_receipt = dict(receipt)
    observed_hmac = unsigned_receipt.pop("receipt_hmac", None)
    expected_hmac = hmac.new(
        private_key,
        b"dynamic-landmark-bridge-receipt-v1\0"
        + _canonical_json_bytes(unsigned_receipt),
        hashlib.sha256,
    ).hexdigest()
    if type(observed_hmac) is not str or not hmac.compare_digest(
        observed_hmac, expected_hmac
    ):
        raise ValueError(f"{stage} bridge receipt HMAC is invalid")
    receipt_sha256 = hashlib.sha256(payloads["receipt"]).hexdigest()

    artifacts: dict[str, dict[str, object]] = {}
    field_sets = {
        "manifest": _V2_MANIFEST_FIELDS,
        "config": _V2_CONFIG_FIELDS,
        "split": _V2_SPLIT_FIELDS,
        "scaler": _V2_SCALER_FIELDS,
    }
    schemas = {
        "manifest": SSL_MANIFEST_V2_SCHEMA,
        "config": SSL_CONFIG_V2_SCHEMA,
        "split": SSL_SPLIT_V2_SCHEMA,
        "scaler": SSL_SCALER_V2_SCHEMA,
    }
    core_digests = receipt.get("artifact_core_sha256")
    if type(core_digests) is not dict or set(core_digests) != set(field_sets):
        raise ValueError(f"{stage} receipt artifact digest set is not exact")
    for name, expected_fields in field_sets.items():
        value = _strict_json_mapping(
            payloads[name], f"{stage} {name} artifact"
        )
        if set(value) != expected_fields:
            raise ValueError(f"{stage} {name} artifact schema is not exact")
        if (
            value.get("schema_version") != schemas[name]
            or value.get("stage") != stage
            or value.get("mode") != mode
            or value.get("source") != source
            or value.get("bridge_receipt_sha256") != receipt_sha256
            or value.get("receipt_hmac") != observed_hmac
        ):
            raise ValueError(f"{stage} {name} artifact cross-link is invalid")
        core = dict(value)
        core.pop("bridge_receipt_sha256")
        core.pop("receipt_hmac")
        expected_core_sha256 = _require_sha256(
            core_digests.get(name), f"{stage} {name} artifact core"
        )
        if hashlib.sha256(_canonical_json_bytes(core)).hexdigest() != expected_core_sha256:
            raise ValueError(f"{stage} {name} artifact core digest is invalid")
        artifacts[name] = value

    manifest = artifacts["manifest"]
    config = artifacts["config"]
    split_value = artifacts["split"]
    scaler_value = artifacts["scaler"]
    _validate_v2_training_config(
        config, stage=stage, mode=mode, source=source,
    )
    sample_ids = _exact_string_list(receipt.get("sample_ids"), "sample IDs")
    source_unit_ids = _exact_string_list(
        receipt.get("source_unit_ids"), "source-unit IDs"
    )
    group_ids = _exact_string_list(receipt.get("group_ids"), "group IDs")
    cache_integrity_ids = _exact_string_list(
        receipt.get("cache_integrity_ids"), "cache-integrity IDs"
    )
    sample_count = _positive_integer(receipt.get("sample_count"), "sample count")
    source_unit_count = _positive_integer(
        receipt.get("source_unit_count"), "source-unit count"
    )
    unique_group_count = _positive_integer(
        receipt.get("unique_group_count"), "unique-group count"
    )
    upstream_cache_count = _positive_integer(
        receipt.get("upstream_cache_count"), "upstream-cache count"
    )
    exclusion_count = receipt.get("exclusion_count")
    expected_exclusion_count = 0 if stage == "ravdess" else 2
    if (
        len(sample_ids) != sample_count
        or len(set(sample_ids)) != sample_count
        or len(source_unit_ids) != sample_count
        or len(group_ids) != sample_count
        or len(cache_integrity_ids) != sample_count
        or len(set(source_unit_ids)) != source_unit_count
        or len(set(group_ids)) != unique_group_count
        or len(set(cache_integrity_ids)) != upstream_cache_count
        or receipt.get("bundle_file_count") != 1
        or isinstance(exclusion_count, (bool, np.bool_))
        or not isinstance(exclusion_count, (int, np.integer))
        or int(exclusion_count) != expected_exclusion_count
    ):
        raise ValueError(f"{stage} receipt aggregate counts are inconsistent")
    common_claims = {
        "source_schema", "sample_ids", "source_unit_ids", "group_ids",
        "sample_count", "source_unit_count", "unique_group_count",
        "upstream_cache_count", "exclusion_count", "bundle_file_count", "bundle_sha256",
        "bundle_size_bytes", "feature_names_sha256", "adapter_sha256",
        "bridge_generation_sha256", "upstream_manifest_commitments",
        "upstream_generation_closure_hmac",
    }
    if any(
        not _exact_json_value(manifest.get(name), receipt.get(name))
        for name in common_claims
    ):
        raise ValueError(f"{stage} manifest contradicts its bridge receipt")
    expected_temporal = hashlib.sha256(
        _canonical_json_bytes(receipt.get("packet_policy"))  # type: ignore[arg-type]
    ).hexdigest() if type(receipt.get("packet_policy")) is dict else None
    if manifest.get("temporal_policy_sha256") != expected_temporal:
        raise ValueError(f"{stage} temporal policy digest is invalid")
    bundle_sha256 = _require_sha256(
        receipt.get("bundle_sha256"), f"{stage} bundle"
    )
    if (
        files["bundle"].sha256 != bundle_sha256
        or files["bundle"].identity.size != receipt.get("bundle_size_bytes")
    ):
        raise ValueError(f"{stage} bundle bytes contradict the receipt")
    parse_stage = RAVDESS_STAGE if stage == "ravdess" else MAYO_DEVELOPMENT_STAGE
    _parse_training_cache_payloads(
        (payloads["bundle"],), stage=parse_stage, group_ids=group_ids,
    )
    return _FrozenStageSnapshot(
        stage=stage,
        mode=mode,
        files=files,
        receipt=receipt,
        manifest=manifest,
        config=config,
        split=split_value,
        scaler=scaler_value,
    )


def _capture_frozen_ssl_stage(
    *,
    stage: str,
    mode: str,
    inputs_root: Path,
    bridge_root: Path,
    ravdess_authorizer: Callable[[], object],
    mayo_authorizer: Callable[[], object],
    producer_sha256: str,
) -> _FrozenStageSnapshot:
    from .dynamic_landmark_ssl_bridge import verify_frozen_bridge_stage

    captured: dict[str, list[object]] = {"ravdess": [], "mayo": []}
    snapshots: list[_FrozenStageSnapshot] = []

    def capture_ravdess() -> object:
        value = ravdess_authorizer()
        captured["ravdess"].append(value)
        return value

    def capture_mayo() -> object:
        value = mayo_authorizer()
        captured["mayo"].append(value)
        return value

    def finalize_locked() -> None:
        if not captured[stage]:
            raise ValueError(f"{stage} live authorization was not captured")
        snapshots.append(_snapshot_frozen_ssl_stage(
            stage=stage,
            mode=mode,
            inputs_root=inputs_root,
            bridge_root=bridge_root,
            live_authorization=captured[stage][-1],
        ))

    verify_frozen_bridge_stage(
        inputs_root,
        bridge_root,
        mode=mode,
        ravdess_authorizer=capture_ravdess,
        mayo_authorizer=capture_mayo,
        producer_sha256=producer_sha256,
        finalize_locked=finalize_locked,
    )
    if len(snapshots) != 1:
        raise RuntimeError("frozen SSL verifier did not finalize exactly once")
    return snapshots[0]


def _snapshot_semantic_facts(snapshot: _FrozenStageSnapshot) -> dict[str, object]:
    return {
        "stage": snapshot.stage,
        "mode": snapshot.mode,
        "files": {
            name: {
                "path": str(item.path),
                "sha256": item.sha256,
                "identity": item.identity,
            }
            for name, item in snapshot.files.items()
        },
        "receipt": dict(snapshot.receipt),
        "manifest": dict(snapshot.manifest),
        "config": dict(snapshot.config),
        "split": dict(snapshot.split),
        "scaler": dict(snapshot.scaler),
    }


def _validate_split_partition(
    split: SSLGroupSplit,
    group_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not isinstance(split, SSLGroupSplit):
        raise ValueError("split must use SSLGroupSplit")
    groups = tuple(group_ids)
    if len(groups) < 2 or any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("group_ids must contain at least two nonempty strings")
    train = _index_array(split.train_indices, len(groups), "split.train_indices")
    heldout = _index_array(
        split.heldout_indices, len(groups), "split.heldout_indices"
    )
    if set(train.tolist()).intersection(heldout.tolist()):
        raise ValueError("split train and heldout indices must be disjoint")
    if set(train.tolist()).union(heldout.tolist()) != set(range(len(groups))):
        raise ValueError("split train and heldout indices must cover every sample")
    expected_claim = (
        "actor_held_out"
        if split.unit == "actor"
        else "recording_held_out_not_patient_held_out"
        if split.unit == "recording"
        else None
    )
    if (
        expected_claim is None
        or split.claim_unit != expected_claim
        or split.patient_held_out is not False
    ):
        raise ValueError("split claim metadata is inconsistent or overclaims patients")
    train_groups = {groups[index] for index in train.tolist()}
    heldout_groups = {groups[index] for index in heldout.tolist()}
    if train_groups.intersection(heldout_groups):
        raise ValueError("one split group cannot cross train and heldout partitions")
    return train, heldout, groups


def _validate_scaler_artifact(
    scaler: SourceScaler,
    *,
    source: str,
    train_indices: np.ndarray,
    feature_width: int | None = None,
) -> None:
    if not isinstance(scaler, SourceScaler) or scaler.source != source:
        raise ValueError("baseline/scaler source does not match the evaluation source")
    if scaler.fit_indices != tuple(int(index) for index in train_indices.tolist()):
        raise ValueError("baseline/scaler was not fitted on the exact training partition")
    if (
        scaler.mean.ndim != 1
        or scaler.scale.shape != scaler.mean.shape
        or (feature_width is not None and scaler.mean.numel() != feature_width)
        or not scaler.mean.is_floating_point()
        or not scaler.scale.is_floating_point()
        or not torch.isfinite(scaler.mean).all()
        or not torch.isfinite(scaler.scale).all()
        or bool((scaler.scale <= 0).any())
    ):
        raise ValueError("baseline/scaler state is malformed or nonfinite")


def _tensor_fingerprint_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to("cpu").contiguous()
    header = json.dumps({
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    raw = value.view(torch.uint8).numpy().tobytes()
    return len(header).to_bytes(8, "big") + header + raw


def _scaler_sha256(scaler: SourceScaler) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_sha256({
        "source": scaler.source,
        "fit_indices": list(scaler.fit_indices),
    }).encode("ascii"))
    digest.update(_tensor_fingerprint_bytes(scaler.mean))
    digest.update(_tensor_fingerprint_bytes(scaler.scale))
    return digest.hexdigest()


def _ssl_model_state_schema() -> dict[str, tuple[torch.Size, torch.dtype]]:
    """Return the frozen architecture schema without advancing caller RNG."""
    with torch.random.fork_rng(devices=[]):
        state = DynamicLandmarkSSLModel().state_dict()
    return {
        name: (value.shape, value.dtype)
        for name, value in state.items()
    }


def _model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    expected = _ssl_model_state_schema()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("SSL model state must be exact and complete")
    digest = hashlib.sha256()
    digest.update(b"dynamic-landmark-ssl-model-state-v1\x00")
    for name in sorted(expected):
        value = state[name]
        expected_shape, expected_dtype = expected[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected_shape
            or value.dtype != expected_dtype
            or (value.is_floating_point() and not torch.isfinite(value).all())
        ):
            raise ValueError(f"SSL model state tensor {name!r} is malformed")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_fingerprint_bytes(value))
    return digest.hexdigest()


def _clone_model_state(model: DynamicLandmarkSSLModel) -> dict[str, torch.Tensor]:
    if not isinstance(model, DynamicLandmarkSSLModel):
        raise ValueError("training model must use the frozen SSL architecture")
    return {
        name: value.detach().to("cpu").clone()
        for name, value in model.state_dict().items()
    }


def _expected_stage_artifacts(
    *,
    stage: str,
    source: str,
    development_only: bool,
    split: SSLGroupSplit,
    scaler: SourceScaler,
    group_ids: tuple[str, ...],
    cache_commitment_sha256: str,
    cache_count: int,
    training_config: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    return {
        "manifest": {
            "schema_version": SSL_MANIFEST_SCHEMA,
            "stage": stage,
            "source": source,
            "sample_count": len(group_ids),
            "group_ids": list(group_ids),
            "cache_commitment_sha256": cache_commitment_sha256,
            "cache_count": cache_count,
        },
        "config": dict(training_config),
        "split": {
            "schema_version": SSL_SPLIT_SCHEMA,
            "stage": stage,
            "source": source,
            "unit": split.unit,
            "claim_unit": split.claim_unit,
            "patient_held_out": split.patient_held_out,
            "train_indices": split.train_indices.tolist(),
            "heldout_indices": split.heldout_indices.tolist(),
        },
        "scaler": {
            "schema_version": SSL_SCALER_SCHEMA,
            "stage": stage,
            "source": source,
            "fit_indices": list(scaler.fit_indices),
            "mean": scaler.mean.tolist(),
            "scale": scaler.scale.tolist(),
        },
    }


def _validate_training_config(
    value: object,
    *,
    stage: str,
    source: str,
    development_only: bool,
) -> dict[str, object]:
    if (
        type(value) is dict
        and value.get("schema_version") == SSL_CONFIG_V2_SCHEMA
    ):
        mode = value.get("mode")
        if type(mode) is not str or mode not in {"smoke", "formal"}:
            raise ValueError("receipt-bound training config mode is invalid")
        validated = _validate_v2_training_config(
            value, stage=stage, mode=mode, source=source,
        )
        if validated["development_only"] is not development_only:
            raise ValueError("receipt-bound config development claim is invalid")
        return validated
    expected_fields = {
        "schema_version", "stage", "source", "objective", "sample_rate_hz",
        "seeds", "development_only", "optimizer", "learning_rate",
        "weight_decay", "epochs", "batch_policy", "span_length",
        "spans_per_window", "device",
    }
    if type(value) is not dict or set(value) != expected_fields:
        raise ValueError("training config schema is not exact")
    fixed = {
        "schema_version": SSL_CONFIG_SCHEMA,
        "stage": stage,
        "source": source,
        "objective": "masked_span_smooth_l1_only",
        "sample_rate_hz": 30.0,
        "seeds": list(SSL_SEEDS),
        "development_only": development_only,
        "optimizer": "adamw",
        "batch_policy": "full_train_partition",
        "device": "cpu",
    }
    if any(not _exact_json_value(value[name], expected) for name, expected in fixed.items()):
        raise ValueError("training config contradicts the stage or fixed algorithm")
    learning_rate = value["learning_rate"]
    weight_decay = value["weight_decay"]
    if (
        type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or not 0.0 < learning_rate <= 0.1
        or type(weight_decay) is not float
        or not math.isfinite(weight_decay)
        or not 0.0 <= weight_decay <= 1.0
    ):
        raise ValueError("training config optimizer values are outside safe bounds")
    for name, upper in (
        ("epochs", 1000), ("span_length", 31), ("spans_per_window", 8),
    ):
        item = value[name]
        if (
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            or not 1 <= int(item) <= upper
        ):
            raise ValueError(f"training config {name} is outside safe bounds")
    if int(value["span_length"]) * int(value["spans_per_window"]) >= 32:
        raise ValueError("training config masks cannot consume an entire window")
    return dict(value)


def _validate_stage_artifact_files(
    *,
    stage: str,
    source: str,
    development_only: bool,
    manifest_path: str | Path,
    config_path: str | Path,
    split_artifact_path: str | Path,
    scaler_artifact_path: str | Path,
    split: SSLGroupSplit,
    scaler: SourceScaler,
    group_ids: tuple[str, ...],
    cache_commitment_sha256: str,
    cache_count: int,
    training_config: Mapping[str, object],
) -> dict[str, tuple[Path, str]]:
    expected = _expected_stage_artifacts(
        stage=stage,
        source=source,
        development_only=development_only,
        split=split,
        scaler=scaler,
        group_ids=group_ids,
        cache_commitment_sha256=cache_commitment_sha256,
        cache_count=cache_count,
        training_config=training_config,
    )
    paths = {
        "manifest": manifest_path,
        "config": config_path,
        "split": split_artifact_path,
        "scaler": scaler_artifact_path,
    }
    validated: dict[str, tuple[Path, str]] = {}
    for name, path in paths.items():
        resolved, observed, digest = _read_json_artifact(
            path, f"{name} artifact"
        )
        if not _exact_json_value(observed, expected[name]):
            raise ValueError(
                f"{name} artifact does not match the canonical {stage} schema and claims"
            )
        validated[name] = (resolved, digest)
    return validated


def _stage_evidence_without_digest(evidence: SSLStageEvidence) -> dict[str, object]:
    value = evidence.to_dict()
    value.pop("evidence_sha256")
    if evidence.mode is None:
        value = {
            name: item for name, item in value.items()
            if name in {
                "schema_version", "stage", "source", "manifest_sha256",
                "cache_commitment_sha256", "cache_count", "config_sha256",
                "split_unit", "claim_unit", "patient_held_out",
                "train_indices_sha256", "heldout_indices_sha256",
                "group_ids_sha256", "scaler_sha256",
                "split_artifact_sha256", "scaler_artifact_sha256",
                "train_count", "heldout_count", "development_only",
                "prior_checkpoint_sha256",
            }
        }
    return value


def _validate_stage_evidence(evidence: SSLStageEvidence) -> None:
    if not isinstance(evidence, SSLStageEvidence):
        raise ValueError("checkpoint stage evidence has the wrong type")
    expected_schema = (
        SSL_STAGE_EVIDENCE_V1_SCHEMA
        if evidence.mode is None
        else SSL_STAGE_EVIDENCE_SCHEMA
    )
    if evidence.schema_version != expected_schema:
        raise ValueError("checkpoint stage evidence schema is unsupported")
    for name in (
        "manifest_sha256", "cache_commitment_sha256", "config_sha256",
        "train_indices_sha256",
        "heldout_indices_sha256", "group_ids_sha256", "scaler_sha256",
        "split_artifact_sha256", "scaler_artifact_sha256",
        "evidence_sha256",
    ):
        value = getattr(evidence, name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"checkpoint stage evidence {name} is malformed")
    if (
        isinstance(evidence.cache_count, (bool, np.bool_))
        or not isinstance(evidence.cache_count, (int, np.integer))
        or int(evidence.cache_count) < 1
        or isinstance(evidence.train_count, (bool, np.bool_))
        or isinstance(evidence.heldout_count, (bool, np.bool_))
        or not isinstance(evidence.train_count, (int, np.integer))
        or not isinstance(evidence.heldout_count, (int, np.integer))
        or int(evidence.train_count) < 1
        or int(evidence.heldout_count) < 1
    ):
        raise ValueError("checkpoint stage evidence split counts are invalid")
    if evidence.patient_held_out is not False:
        raise ValueError("unlabeled SSL evidence cannot claim patient-held-out validation")
    if evidence.mode is not None:
        if evidence.mode not in {"smoke", "formal"}:
            raise ValueError("receipt-bound evidence mode is unsupported")
        for name in (
            "bridge_receipt_sha256", "receipt_hmac",
            "receipt_file_identity_sha256", "canonical_key_identity_sha256",
            "sample_ids_sha256", "source_unit_ids_sha256",
            "cache_integrity_ids_sha256", "original_mapping_sha256",
            "bundle_sha256", "feature_names_sha256", "adapter_sha256",
            "temporal_policy_sha256", "bridge_generation_sha256",
            "upstream_manifest_commitments_sha256",
            "upstream_generation_closure_hmac",
        ):
            _require_sha256(getattr(evidence, name), f"stage evidence {name}")
        for name in (
            "bundle_size_bytes", "bundle_file_count", "sample_count",
            "source_unit_count", "unique_group_count", "upstream_cache_count",
        ):
            _positive_integer(getattr(evidence, name), f"stage evidence {name}")
        expected_exclusion_count = 0 if evidence.stage == "ravdess" else 2
        if (
            isinstance(evidence.exclusion_count, (bool, np.bool_))
            or not isinstance(evidence.exclusion_count, (int, np.integer))
            or int(evidence.exclusion_count) != expected_exclusion_count
        ):
            raise ValueError(
                "receipt-bound evidence exclusion count contradicts its stage"
            )
        if evidence.bundle_file_count != 1 or evidence.cache_count != 1:
            raise ValueError("receipt-bound evidence requires one exact bundle")
        expected_source = (
            "ravdess_openface_semantic23"
            if evidence.stage == "ravdess"
            else "mayo_mediapipe_clinical23_development_only"
            if evidence.stage == "mayo"
            else None
        )
        expected_unit = "actor" if evidence.stage == "ravdess" else "recording"
        expected_claim = (
            "actor_held_out"
            if evidence.stage == "ravdess"
            else "recording_held_out_not_patient_held_out"
        )
        if (
            expected_source is None
            or evidence.source != expected_source
            or evidence.split_unit != expected_unit
            or evidence.claim_unit != expected_claim
            or evidence.development_only is not (evidence.stage == "mayo")
            or type(evidence.source_schema) is not str
            or not evidence.source_schema
        ):
            raise ValueError("receipt-bound stage evidence contradicts its stage")
        if evidence.stage == "ravdess":
            if evidence.prior_checkpoint_sha256 is not None:
                raise ValueError("RAVDESS receipt-bound evidence cannot name prior state")
        else:
            _require_sha256(
                evidence.prior_checkpoint_sha256,
                "Mayo prior checkpoint fingerprint",
            )
        if evidence.evidence_sha256 != _canonical_sha256(
            _stage_evidence_without_digest(evidence)
        ):
            raise ValueError("receipt-bound stage evidence digest is invalid")
        return
    if evidence.stage == RAVDESS_STAGE:
        expected = (
            RAVDESS_SOURCE, "actor", "actor_held_out", False,
        )
        if evidence.prior_checkpoint_sha256 is not None:
            raise ValueError("RAVDESS stage evidence cannot name a prior checkpoint")
    elif evidence.stage == MAYO_DEVELOPMENT_STAGE:
        expected = (
            MAYO_SOURCE, "recording",
            "recording_held_out_not_patient_held_out", True,
        )
        if (
            not isinstance(evidence.prior_checkpoint_sha256, str)
            or _SHA256.fullmatch(evidence.prior_checkpoint_sha256) is None
        ):
            raise ValueError("Mayo stage evidence requires a prior RAVDESS fingerprint")
    else:
        raise ValueError("checkpoint stage evidence uses an unknown stage")
    observed = (
        evidence.source, evidence.split_unit, evidence.claim_unit,
        evidence.development_only,
    )
    if observed != expected:
        raise ValueError("checkpoint stage evidence contradicts its stage")
    if evidence.evidence_sha256 != _canonical_sha256(
        _stage_evidence_without_digest(evidence)
    ):
        raise ValueError("checkpoint stage evidence digest does not match its contents")


def _stage_evidence_from_mapping(value: object) -> SSLStageEvidence:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint stage evidence must be a mapping")
    fields = {
        name for name in SSLStageEvidence.__dataclass_fields__
        if not name.startswith("_")
    }
    if set(value) != fields:
        raise ValueError("checkpoint stage evidence schema is not exact")
    try:
        evidence = SSLStageEvidence(**dict(value))
    except TypeError as exc:
        raise ValueError("checkpoint stage evidence cannot be constructed") from exc
    _validate_stage_evidence(evidence)
    return evidence


def _require_receipt_bound_stage_authorization(
    evidence: SSLStageEvidence,
    authorization: _SSLStageAuthorization,
) -> None:
    frozen = authorization.frozen_stage
    if (
        not isinstance(frozen, _FrozenSSLStageAuthorization)
        or frozen.marker is not _AUTHORIZATION_MARKER
        or frozen.stage != evidence.stage
        or frozen.mode != evidence.mode
    ):
        raise ValueError("receipt-bound evidence lacks its frozen-input authority")
    current = _capture_frozen_ssl_stage(
        stage=frozen.stage,
        mode=frozen.mode,
        inputs_root=frozen.inputs_root,
        bridge_root=frozen.bridge_root,
        ravdess_authorizer=frozen.ravdess_authorizer,
        mayo_authorizer=frozen.mayo_authorizer,
        producer_sha256=frozen.producer_sha256,
    )
    if _snapshot_semantic_facts(current) != _snapshot_semantic_facts(
        frozen.snapshot
    ):
        raise ValueError("frozen receipt-bound inputs changed after authorization")
    train, heldout, groups = _validate_split_partition(
        authorization.split, authorization.group_ids
    )
    bundle_payload = _reopen_authorized_cache_artifacts(
        authorization.cache_artifacts
    )
    if len(bundle_payload) != 1:
        raise ValueError("receipt-bound evidence requires one exact bundle file")
    cache_commitment = _cache_commitment_sha256(groups, bundle_payload)
    if (
        cache_commitment != evidence.cache_commitment_sha256
        or cache_commitment != authorization.cache_commitment_sha256
        or authorization.cache_count != 1
    ):
        raise ValueError("receipt-bound bundle commitment changed")
    parse_stage = (
        RAVDESS_STAGE if evidence.stage == "ravdess" else MAYO_DEVELOPMENT_STAGE
    )
    features, valid_mask, _, _ = _parse_training_cache_payloads(
        bundle_payload, stage=parse_stage, group_ids=groups,
    )
    source_units = tuple(
        str(value) for value in current.receipt["source_unit_ids"]
    )
    recomputed, unique_count, fit_sources = _fit_receipt_bound_source_scaler(
        features,
        valid_mask,
        source=evidence.source,
        train_indices=train,
        heldout_indices=heldout,
        source_unit_ids=source_units,
        original_canonical_frame_indices=current.receipt[
            "original_canonical_frame_indices"
        ],
    )
    scaler_value = current.scaler
    if (
        _scaler_sha256(recomputed) != evidence.scaler_sha256
        or _scaler_sha256(recomputed) != _scaler_sha256(authorization.scaler)
        or scaler_value.get("fit_unique_frame_count") != unique_count
        or scaler_value.get("fit_source_unit_ids") != list(fit_sources)
    ):
        raise ValueError("receipt-bound scaler changed after authorization")
    prior_payload = authorization.prior_ravdess_checkpoint
    prior_evidence = authorization.prior_ravdess_evidence
    if evidence.stage == "ravdess":
        if prior_payload is not None or prior_evidence is not None:
            raise ValueError("RAVDESS receipt-bound authority carries prior state")
        return
    if prior_payload is None or prior_evidence is None:
        raise ValueError("Mayo receipt-bound authority lacks prior RAVDESS state")
    _require_exact_ravdess_only_prior(
        prior_payload,
        prior_evidence,
        expected_mode=evidence.mode,
        require_persisted=True,
    )
    if ssl_checkpoint_fingerprint(prior_payload) != evidence.prior_checkpoint_sha256:
        raise ValueError("Mayo prior checkpoint changed after authorization")


def _require_authorized_stage_evidence(evidence: SSLStageEvidence) -> None:
    """Revalidate private artifact provenance immediately before a gated action."""
    _validate_stage_evidence(evidence)
    authorization = evidence._runtime_authorization
    if (
        not isinstance(authorization, _SSLStageAuthorization)
        or authorization.marker is not _AUTHORIZATION_MARKER
        or authorization.evidence_sha256 != evidence.evidence_sha256
    ):
        raise ValueError(
            "stage evidence lacks live authorization from actual artifact files"
        )
    if evidence.mode is not None:
        _require_receipt_bound_stage_authorization(evidence, authorization)
        return
    train, heldout, groups = _validate_split_partition(
        authorization.split, authorization.group_ids
    )
    if evidence.stage == RAVDESS_STAGE:
        source = RAVDESS_SOURCE
        development_only = False
        expected_width = 23
        if authorization.split.unit != "actor":
            raise ValueError("authorized RAVDESS evidence requires an actor split")
    else:
        source = MAYO_SOURCE
        development_only = True
        expected_width = 95
        if authorization.split.unit != "recording":
            raise ValueError("authorized Mayo evidence requires a recording split")
    _validate_scaler_artifact(
        authorization.scaler,
        source=source,
        train_indices=train,
        feature_width=expected_width,
    )
    _validate_training_config(
        authorization.training_config,
        stage=evidence.stage,
        source=source,
        development_only=development_only,
    )
    artifacts = _validate_stage_artifact_files(
        stage=evidence.stage,
        source=source,
        development_only=development_only,
        manifest_path=authorization.manifest_path,
        config_path=authorization.config_path,
        split_artifact_path=authorization.split_artifact_path,
        scaler_artifact_path=authorization.scaler_artifact_path,
        split=authorization.split,
        scaler=authorization.scaler,
        group_ids=groups,
        cache_commitment_sha256=evidence.cache_commitment_sha256,
        cache_count=evidence.cache_count,
        training_config=authorization.training_config,
    )
    observed_hashes = {
        "manifest_sha256": artifacts["manifest"][1],
        "config_sha256": artifacts["config"][1],
        "split_artifact_sha256": artifacts["split"][1],
        "scaler_artifact_sha256": artifacts["scaler"][1],
        "train_indices_sha256": _canonical_sha256(train.tolist()),
        "heldout_indices_sha256": _canonical_sha256(heldout.tolist()),
        "group_ids_sha256": _canonical_sha256(list(groups)),
        "scaler_sha256": _scaler_sha256(authorization.scaler),
    }
    if any(
        getattr(evidence, name) != value
        for name, value in observed_hashes.items()
    ):
        raise ValueError("authorized stage artifacts changed after evidence creation")
    cache_payloads = _reopen_authorized_cache_artifacts(
        authorization.cache_artifacts
    )
    cache_commitment = _cache_commitment_sha256(groups, cache_payloads)
    if (
        evidence.train_count != int(train.size)
        or evidence.heldout_count != int(heldout.size)
        or evidence.source != source
        or evidence.development_only is not development_only
        or evidence.cache_commitment_sha256 != cache_commitment
        or evidence.cache_count != len(cache_payloads)
        or authorization.cache_commitment_sha256 != cache_commitment
        or authorization.cache_count != len(cache_payloads)
    ):
        raise ValueError("authorized stage claims changed after evidence creation")

    prior_payload = authorization.prior_ravdess_checkpoint
    prior_evidence = authorization.prior_ravdess_evidence
    if evidence.stage == RAVDESS_STAGE:
        if prior_payload is not None or prior_evidence is not None:
            raise ValueError("RAVDESS authorization cannot carry prior-stage state")
        return
    if prior_payload is None or prior_evidence is None:
        raise ValueError("Mayo authorization requires the actual prior RAVDESS stage")
    _require_authorized_checkpoint_payload(prior_payload, prior_evidence)
    if ssl_checkpoint_fingerprint(prior_payload) != evidence.prior_checkpoint_sha256:
        raise ValueError("the prior RAVDESS checkpoint changed after Mayo authorization")


def _require_public_receipt_bound_stage(evidence: SSLStageEvidence) -> None:
    _require_authorized_stage_evidence(evidence)
    authorization = evidence._runtime_authorization
    if (
        evidence.mode not in {"smoke", "formal"}
        or not isinstance(authorization, _SSLStageAuthorization)
        or not isinstance(
            authorization.frozen_stage, _FrozenSSLStageAuthorization,
        )
    ):
        raise PretrainingLockedError(
            "production SSL actions require receipt-bound frozen v2 authority"
        )


def _build_synthetic_ssl_stage_evidence_v1(
    *,
    stage: str,
    manifest_path: str | Path,
    config_path: str | Path,
    split_artifact_path: str | Path,
    scaler_artifact_path: str | Path,
    cache_paths: Sequence[str | Path],
    split: SSLGroupSplit,
    scaler: SourceScaler,
    group_ids: Sequence[str],
    prior_ravdess_checkpoint: Mapping[str, object] | None = None,
    prior_ravdess_evidence: SSLStageEvidence | None = None,
) -> SSLStageEvidence:
    """Test-only compatibility constructor for the retired v1 evidence schema."""
    if isinstance(cache_paths, (str, bytes, Path)):
        raise ValueError("cache_paths must be an ordered sequence of regular files")
    cache_artifacts = tuple(
        _authorize_cache_artifact(path) for path in tuple(cache_paths)
    )
    if not cache_artifacts:
        raise ValueError("stage evidence requires at least one exact cache artifact")
    if len({artifact.path for artifact in cache_artifacts}) != len(cache_artifacts):
        raise ValueError("stage evidence cache paths must be unique and ordered")
    train, heldout, groups = _validate_split_partition(split, group_ids)
    if stage == RAVDESS_STAGE:
        source = RAVDESS_SOURCE
        development_only = False
        if (
            split.unit != "actor"
            or prior_ravdess_checkpoint is not None
            or prior_ravdess_evidence is not None
        ):
            raise ValueError("RAVDESS evidence requires actor split and no prior checkpoint")
        prior_sha256 = None
    elif stage == MAYO_DEVELOPMENT_STAGE:
        source = MAYO_SOURCE
        development_only = True
        if (
            split.unit != "recording"
            or prior_ravdess_checkpoint is None
            or prior_ravdess_evidence is None
        ):
            raise ValueError(
                "Mayo development evidence requires recording split and authorized prior RAVDESS checkpoint"
            )
        _require_authorized_checkpoint_payload(
            prior_ravdess_checkpoint,
            prior_ravdess_evidence,
        )
        prior_sha256 = ssl_checkpoint_fingerprint(prior_ravdess_checkpoint)
    else:
        raise ValueError("stage must be ravdess or mayo")
    _, config_value, _ = _read_json_artifact(config_path, "config artifact")
    training_config = _validate_training_config(
        config_value,
        stage=stage,
        source=source,
        development_only=development_only,
    )
    cache_payloads = _reopen_authorized_cache_artifacts(cache_artifacts)
    cache_commitment = _cache_commitment_sha256(groups, cache_payloads)
    cache_count = len(cache_payloads)
    cache_features, cache_valid_mask, _, _ = _parse_training_cache_payloads(
        cache_payloads, stage=stage, group_ids=groups,
    )
    recomputed_scaler = fit_source_scaler(
        cache_features,
        cache_valid_mask,
        source=source,
        fit_indices=train,
        heldout_indices=heldout,
    )
    _validate_scaler_artifact(
        scaler, source=source, train_indices=train, feature_width=(23 if stage == RAVDESS_STAGE else 95)
    )
    if _scaler_sha256(scaler) != _scaler_sha256(recomputed_scaler):
        raise ValueError(
            "scaler must exactly match repository recomputation from authorized train rows"
        )
    artifacts = _validate_stage_artifact_files(
        stage=stage,
        source=source,
        development_only=development_only,
        manifest_path=manifest_path,
        config_path=config_path,
        split_artifact_path=split_artifact_path,
        scaler_artifact_path=scaler_artifact_path,
        split=split,
        scaler=recomputed_scaler,
        group_ids=groups,
        cache_commitment_sha256=cache_commitment,
        cache_count=cache_count,
        training_config=training_config,
    )
    values: dict[str, object] = {
        "schema_version": SSL_STAGE_EVIDENCE_V1_SCHEMA,
        "stage": stage,
        "source": source,
        "manifest_sha256": artifacts["manifest"][1],
        "cache_commitment_sha256": cache_commitment,
        "cache_count": cache_count,
        "config_sha256": artifacts["config"][1],
        "split_unit": split.unit,
        "claim_unit": split.claim_unit,
        "patient_held_out": split.patient_held_out,
        "train_indices_sha256": _canonical_sha256(train.tolist()),
        "heldout_indices_sha256": _canonical_sha256(heldout.tolist()),
        "group_ids_sha256": _canonical_sha256(list(groups)),
        "scaler_sha256": _scaler_sha256(recomputed_scaler),
        "split_artifact_sha256": artifacts["split"][1],
        "scaler_artifact_sha256": artifacts["scaler"][1],
        "train_count": int(train.size),
        "heldout_count": int(heldout.size),
        "development_only": development_only,
        "prior_checkpoint_sha256": prior_sha256,
    }
    values["evidence_sha256"] = _canonical_sha256(values)
    evidence = SSLStageEvidence(**values)
    _validate_stage_evidence(evidence)
    authorization = _SSLStageAuthorization(
        marker=_AUTHORIZATION_MARKER,
        evidence_sha256=evidence.evidence_sha256,
        manifest_path=artifacts["manifest"][0],
        config_path=artifacts["config"][0],
        split_artifact_path=artifacts["split"][0],
        scaler_artifact_path=artifacts["scaler"][0],
        split=split,
        scaler=recomputed_scaler,
        group_ids=groups,
        training_config=training_config,
        cache_artifacts=cache_artifacts,
        cache_commitment_sha256=cache_commitment,
        cache_count=cache_count,
        prior_ravdess_checkpoint=prior_ravdess_checkpoint,
        prior_ravdess_evidence=prior_ravdess_evidence,
    )
    object.__setattr__(evidence, "_runtime_authorization", authorization)
    _require_authorized_stage_evidence(evidence)
    return evidence


def build_ssl_stage_evidence(*_args: object, **_kwargs: object) -> SSLStageEvidence:
    """Reject the retired public v1 evidence-minting path."""
    raise PretrainingLockedError(
        "public v1 SSL evidence is retired; authorize a frozen receipt-bound "
        "Task 2 stage instead"
    )


def authorize_frozen_ssl_stage(
    *,
    stage: str,
    mode: str,
    inputs_root: str | Path,
    bridge_root: str | Path,
    ravdess_authorizer: Callable[[], object],
    mayo_authorizer: Callable[[], object],
    producer_sha256: str,
    prior_ravdess_checkpoint: Mapping[str, object] | None = None,
    prior_ravdess_evidence: SSLStageEvidence | None = None,
) -> SSLStageEvidence:
    """Authorize one Task 2 frozen stage as receipt-bound v2 evidence."""
    if stage not in {"ravdess", "mayo"} or mode not in {"smoke", "formal"}:
        raise ValueError("frozen SSL stage and mode must be exact")
    if not callable(ravdess_authorizer) or not callable(mayo_authorizer):
        raise ValueError("frozen SSL live authorizers must be callable")
    producer_sha256 = _require_sha256(producer_sha256, "bridge producer")
    inputs = Path(inputs_root).resolve(strict=True)
    bridge = Path(bridge_root).resolve(strict=True)
    snapshot = _capture_frozen_ssl_stage(
        stage=stage,
        mode=mode,
        inputs_root=inputs,
        bridge_root=bridge,
        ravdess_authorizer=ravdess_authorizer,
        mayo_authorizer=mayo_authorizer,
        producer_sha256=producer_sha256,
    )
    receipt = snapshot.receipt
    manifest = snapshot.manifest
    split_value = snapshot.split
    scaler_value = snapshot.scaler
    source = str(manifest["source"])
    group_ids = tuple(str(value) for value in manifest["group_ids"])
    source_unit_ids = tuple(str(value) for value in manifest["source_unit_ids"])
    sample_ids = tuple(str(value) for value in manifest["sample_ids"])
    train = _index_array(
        split_value.get("train_indices"), len(group_ids), "train indices"
    )
    heldout = _index_array(
        split_value.get("heldout_indices"), len(group_ids), "heldout indices"
    )
    unit = "actor" if stage == "ravdess" else "recording"
    claim_unit = (
        "actor_held_out"
        if stage == "ravdess"
        else "recording_held_out_not_patient_held_out"
    )
    if (
        split_value.get("schema_version") != SSL_SPLIT_V2_SCHEMA
        or split_value.get("stage") != stage
        or split_value.get("mode") != mode
        or split_value.get("source") != source
        or split_value.get("split_seed") != 0
        or not _exact_json_value(split_value.get("heldout_fraction"), 0.20)
        or split_value.get("unit") != unit
        or split_value.get("claim_unit") != claim_unit
    ):
        raise ValueError("receipt-bound split is not exact")
    split = SSLGroupSplit(
        train_indices=train,
        heldout_indices=heldout,
        unit=unit,
        claim_unit=claim_unit,
        patient_held_out=False,
    )
    train, heldout, group_ids = _validate_split_partition(split, group_ids)

    def ordered_unique(indices: np.ndarray) -> list[str]:
        return list(dict.fromkeys(group_ids[int(index)] for index in indices))

    if (
        split_value.get("train_group_ids") != ordered_unique(train)
        or split_value.get("heldout_group_ids") != ordered_unique(heldout)
    ):
        raise ValueError("receipt-bound split group order is inconsistent")
    bundle_payload = _reopen_authorized_cache_artifacts(
        (snapshot.files["bundle"],)
    )[0]
    parse_stage = RAVDESS_STAGE if stage == "ravdess" else MAYO_DEVELOPMENT_STAGE
    features, valid_mask, _, _ = _parse_training_cache_payloads(
        (bundle_payload,), stage=parse_stage, group_ids=group_ids,
    )
    scaler, unique_frame_count, fit_source_units = (
        _fit_receipt_bound_source_scaler(
            features,
            valid_mask,
            source=source,
            train_indices=train,
            heldout_indices=heldout,
            source_unit_ids=source_unit_ids,
            original_canonical_frame_indices=receipt.get(
                "original_canonical_frame_indices"
            ),
        )
    )
    if (
        scaler_value.get("schema_version") != SSL_SCALER_V2_SCHEMA
        or scaler_value.get("stage") != stage
        or scaler_value.get("mode") != mode
        or scaler_value.get("source") != source
        or scaler_value.get("fit_indices") != train.tolist()
        or scaler_value.get("fit_source_unit_ids") != list(fit_source_units)
        or scaler_value.get("unique_frame_key")
        != "source_unit_id_plus_original_canonical_30hz_index"
        or scaler_value.get("fit_unique_frame_count") != unique_frame_count
        or not _exact_json_value(scaler_value.get("mean"), scaler.mean.tolist())
        or not _exact_json_value(scaler_value.get("scale"), scaler.scale.tolist())
    ):
        raise ValueError(
            "receipt-bound scaler does not match unique train-frame recomputation"
        )
    cache_commitment = _cache_commitment_sha256(group_ids, (bundle_payload,))
    if stage == "ravdess":
        if prior_ravdess_checkpoint is not None or prior_ravdess_evidence is not None:
            raise ValueError("RAVDESS receipt-bound evidence cannot carry prior state")
        prior_sha256 = None
    else:
        if prior_ravdess_checkpoint is None or prior_ravdess_evidence is None:
            raise ValueError("Mayo receipt-bound evidence requires prior RAVDESS state")
        _require_exact_ravdess_only_prior(
            prior_ravdess_checkpoint,
            prior_ravdess_evidence,
            expected_mode=mode,
            require_persisted=True,
        )
        prior_sha256 = ssl_checkpoint_fingerprint(prior_ravdess_checkpoint)
    files = snapshot.files
    values: dict[str, object] = {
        "schema_version": SSL_STAGE_EVIDENCE_SCHEMA,
        "stage": stage,
        "source": source,
        "manifest_sha256": files["manifest"].sha256,
        "cache_commitment_sha256": cache_commitment,
        "cache_count": 1,
        "config_sha256": files["config"].sha256,
        "split_unit": split.unit,
        "claim_unit": split.claim_unit,
        "patient_held_out": False,
        "train_indices_sha256": _canonical_sha256(train.tolist()),
        "heldout_indices_sha256": _canonical_sha256(heldout.tolist()),
        "group_ids_sha256": _canonical_sha256(list(group_ids)),
        "scaler_sha256": _scaler_sha256(scaler),
        "split_artifact_sha256": files["split"].sha256,
        "scaler_artifact_sha256": files["scaler"].sha256,
        "train_count": int(train.size),
        "heldout_count": int(heldout.size),
        "development_only": stage == "mayo",
        "prior_checkpoint_sha256": prior_sha256,
        "mode": mode,
        "bridge_receipt_sha256": files["receipt"].sha256,
        "receipt_hmac": receipt["receipt_hmac"],
        "receipt_file_identity_sha256": _file_identity_sha256(
            files["receipt"].identity
        ),
        "canonical_key_identity_sha256": receipt[
            "canonical_key_identity_sha256"
        ],
        "sample_ids_sha256": _canonical_sha256(list(sample_ids)),
        "source_unit_ids_sha256": _canonical_sha256(list(source_unit_ids)),
        "cache_integrity_ids_sha256": _canonical_sha256(
            receipt["cache_integrity_ids"]
        ),
        "original_mapping_sha256": receipt["original_mapping_sha256"],
        "bundle_sha256": receipt["bundle_sha256"],
        "bundle_size_bytes": receipt["bundle_size_bytes"],
        "bundle_file_count": receipt["bundle_file_count"],
        "sample_count": receipt["sample_count"],
        "source_unit_count": receipt["source_unit_count"],
        "unique_group_count": receipt["unique_group_count"],
        "upstream_cache_count": receipt["upstream_cache_count"],
        "exclusion_count": receipt["exclusion_count"],
        "feature_names_sha256": receipt["feature_names_sha256"],
        "adapter_sha256": receipt["adapter_sha256"],
        "temporal_policy_sha256": manifest["temporal_policy_sha256"],
        "bridge_generation_sha256": receipt["bridge_generation_sha256"],
        "upstream_manifest_commitments_sha256": _canonical_sha256(
            receipt["upstream_manifest_commitments"]
        ),
        "upstream_generation_closure_hmac": receipt[
            "upstream_generation_closure_hmac"
        ],
        "source_schema": receipt["source_schema"],
    }
    values["evidence_sha256"] = _canonical_sha256(values)
    evidence = SSLStageEvidence(**values)
    frozen_stage = _FrozenSSLStageAuthorization(
        marker=_AUTHORIZATION_MARKER,
        stage=stage,
        mode=mode,
        inputs_root=inputs,
        bridge_root=bridge,
        ravdess_authorizer=ravdess_authorizer,
        mayo_authorizer=mayo_authorizer,
        producer_sha256=producer_sha256,
        snapshot=snapshot,
    )
    authorization = _SSLStageAuthorization(
        marker=_AUTHORIZATION_MARKER,
        evidence_sha256=evidence.evidence_sha256,
        manifest_path=files["manifest"].path,
        config_path=files["config"].path,
        split_artifact_path=files["split"].path,
        scaler_artifact_path=files["scaler"].path,
        split=split,
        scaler=scaler,
        group_ids=group_ids,
        training_config=dict(snapshot.config),
        cache_artifacts=(files["bundle"],),
        cache_commitment_sha256=cache_commitment,
        cache_count=1,
        prior_ravdess_checkpoint=prior_ravdess_checkpoint,
        prior_ravdess_evidence=prior_ravdess_evidence,
        frozen_stage=frozen_stage,
    )
    object.__setattr__(evidence, "_runtime_authorization", authorization)
    _require_authorized_stage_evidence(evidence)
    return evidence


class DynamicLandmarkSSLModel(nn.Module):
    """Masked-frame reconstruction model with Task4-compatible shared modules."""

    def __init__(self):
        super().__init__()
        self._mayo_lineage_authorization: _MayoModelAuthorization | None = None
        self.ravdess_proj_x = nn.Linear(23, 32, bias=False)
        self.ravdess_proj_dx = nn.Linear(23, 32, bias=False)
        self.proj_bs_x = nn.Linear(72, 32, bias=False)
        self.proj_bs_dx = nn.Linear(72, 32, bias=False)
        self.proj_lm_x = nn.Linear(23, 32, bias=False)
        self.proj_lm_dx = nn.Linear(23, 32, bias=False)
        self.temporal = nn.GRU(
            input_size=64, hidden_size=32, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.attention_score = nn.Linear(64, 1)
        self.pool_projection = nn.Linear(128, 32)
        self.ravdess_decoder = nn.Linear(96, 23)
        self.mayo_decoder = nn.Linear(96, 95)

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Fail closed: an arbitrary state replacement invalidates stage lineage."""
        self._mayo_lineage_authorization = None
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _source_width(source: str) -> int:
        if source == "ravdess":
            return 23
        if source == "mayo":
            return 95
        raise ValueError("SSL source must be ravdess or mayo")

    def build_gru_input(
        self,
        features: torch.Tensor,
        observed_mask: torch.Tensor,
        timestamps: torch.Tensor,
        source_frame_indices: torch.Tensor,
        *,
        source: str,
    ) -> torch.Tensor:
        width = self._source_width(source)
        if features.ndim != 4 or features.shape[1:3] != (4, 32) or features.shape[-1] != width:
            raise ValueError(f"{source} SSL features require shape (batch, 4, 32, {width})")
        if len({features.device, observed_mask.device, timestamps.device,
                source_frame_indices.device}) != 1:
            raise ValueError("SSL features, mask, timestamps, and indices must share one device")
        if observed_mask.shape != features.shape[:-1] or observed_mask.dtype != torch.bool:
            raise ValueError("observed mask must be boolean with feature leading shape")
        if bool((~observed_mask.reshape(features.shape[0], -1).any(dim=1)).any()):
            raise ValueError("each SSL recording requires observed context")
        x = features * observed_mask.unsqueeze(-1)
        dx, _ = ssl_gap_safe_per_second_differences(
            x,
            observed_mask,
            timestamps,
            source_frame_indices,
            expected_source_step=1,
        )
        if source == "ravdess":
            landmark = self.ravdess_proj_x(x) + self.ravdess_proj_dx(dx)
            base = torch.zeros_like(landmark)
        else:
            bs, lm = x[..., :72], x[..., 72:]
            dbs, dlm = dx[..., :72], dx[..., 72:]
            base = self.proj_bs_x(bs) + self.proj_bs_dx(dbs)
            landmark = self.proj_lm_x(lm) + self.proj_lm_dx(dlm)
        result = torch.cat((base, landmark), dim=-1)
        if result.shape[-1] != 64 or not torch.isfinite(result).all():
            raise RuntimeError("SSL adapter violated the full 64-d GRU contract")
        return result

    def _pooled_context(
        self,
        encoded: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        present = observed_mask.any(dim=1)
        masked = encoded.masked_fill(
            ~observed_mask.unsqueeze(-1), torch.finfo(encoded.dtype).min
        )
        max_pool = masked.max(dim=1).values
        max_pool = torch.where(present.unsqueeze(-1), max_pool, torch.zeros_like(max_pool))
        scores = self.attention_score(encoded).squeeze(-1).masked_fill(
            ~observed_mask, -1e9
        )
        weights = torch.softmax(scores, dim=1) * observed_mask.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        attention = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        attention = torch.where(present.unsqueeze(-1), attention, torch.zeros_like(attention))
        return torch.tanh(self.pool_projection(torch.cat((max_pool, attention), dim=-1)))

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        timestamps: torch.Tensor,
        source_frame_indices: torch.Tensor,
        *,
        reconstruction_mask: torch.Tensor,
        source: str,
    ) -> torch.Tensor:
        if valid_mask.shape != features.shape[:-1] or valid_mask.dtype != torch.bool:
            raise ValueError("valid mask must be boolean with feature leading shape")
        if reconstruction_mask.shape != valid_mask.shape or reconstruction_mask.dtype != torch.bool:
            raise ValueError("reconstruction mask must match valid mask")
        if len({features.device, valid_mask.device, reconstruction_mask.device,
                timestamps.device, source_frame_indices.device}) != 1:
            raise ValueError("all SSL model inputs must share one device")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("SSL feature input must be finite floating values")
        if bool((reconstruction_mask & ~valid_mask).any()) or not bool(reconstruction_mask.any()):
            raise ValueError("reconstruction mask must select valid target frames only")
        if bool((~valid_mask.reshape(features.shape[0], -1).any(dim=1)).any()):
            raise ValueError("all-masked SSL recordings are rejected")
        observed = valid_mask & ~reconstruction_mask
        gru_input = self.build_gru_input(
            features, observed, timestamps, source_frame_indices, source=source
        )
        batch, windows, frames, _ = gru_input.shape
        encoded, _ = self.temporal(gru_input.reshape(batch * windows, frames, 64))
        flat_observed = observed.reshape(batch * windows, frames)
        pooled = self._pooled_context(encoded, flat_observed)
        decoder_input = torch.cat(
            (encoded, pooled.unsqueeze(1).expand(-1, frames, -1)), dim=-1
        )
        decoder = self.ravdess_decoder if source == "ravdess" else self.mayo_decoder
        reconstruction = decoder(decoder_input).reshape(batch, windows, frames, -1)
        if reconstruction.shape != features.shape or not torch.isfinite(reconstruction).all():
            raise RuntimeError("SSL decoder produced an invalid reconstruction")
        return reconstruction


def _initialize_mayo_ssl_model_impl(
    prior_checkpoint: Mapping[str, object],
    *,
    prior_stage_evidence: SSLStageEvidence,
) -> DynamicLandmarkSSLModel:
    """Load and mark the exact authorized RAVDESS state used to start Mayo SSL."""
    _require_exact_ravdess_only_prior(
        prior_checkpoint,
        prior_stage_evidence,
        expected_mode=prior_stage_evidence.mode,
        require_persisted=prior_stage_evidence.mode is not None,
    )
    fingerprint = ssl_checkpoint_fingerprint(prior_checkpoint)
    with torch.random.fork_rng(devices=[]):
        model = DynamicLandmarkSSLModel()
    model.load_state_dict(prior_checkpoint["model_state"], strict=True)
    model._mayo_lineage_authorization = _MayoModelAuthorization(
        marker=_AUTHORIZATION_MARKER,
        model_reference=weakref.ref(model),
        prior_checkpoint_sha256=fingerprint,
    )
    return model


def initialize_mayo_ssl_model(
    prior_checkpoint: Mapping[str, object],
    *,
    prior_stage_evidence: SSLStageEvidence,
) -> DynamicLandmarkSSLModel:
    _require_public_receipt_bound_stage(prior_stage_evidence)
    return _initialize_mayo_ssl_model_impl(
        prior_checkpoint,
        prior_stage_evidence=prior_stage_evidence,
    )


def reconstruction_report(
    trained_prediction: torch.Tensor,
    untrained_prediction: torch.Tensor,
    target: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    *,
    baseline: SourceScaler,
    split: SSLGroupSplit,
    evaluated_indices: Sequence[int] | np.ndarray,
    group_ids: Sequence[str],
    source: str,
) -> dict[str, object]:
    train, heldout, groups = _validate_split_partition(split, group_ids)
    evaluated = _index_array(evaluated_indices, len(groups), "evaluated_indices")
    if not np.array_equal(evaluated, heldout):
        raise ValueError("reconstruction report must evaluate the exact heldout partition")
    if target.ndim < 2 or target.shape[0] != heldout.size:
        raise ValueError("reconstruction target rows must match the exact heldout partition")
    _validate_scaler_artifact(
        baseline,
        source=source,
        train_indices=train,
        feature_width=target.shape[-1],
    )
    baseline_scaler_sha256 = _scaler_sha256(baseline)
    baseline_prediction = torch.zeros_like(target)
    return {
        "metric": "masked_smooth_l1",
        "target_space": "source_train_standardized",
        "trained": float(masked_smooth_l1(
            trained_prediction, target, reconstruction_mask
        ).item()),
        "untrained": float(masked_smooth_l1(
            untrained_prediction, target, reconstruction_mask
        ).item()),
        "train_mean": float(masked_smooth_l1(
            baseline_prediction, target, reconstruction_mask
        ).item()),
        "claim_unit": split.claim_unit,
        "patient_held_out": split.patient_held_out,
        "source": source,
        "evaluated_indices_sha256": _canonical_sha256(evaluated.tolist()),
        "group_ids_sha256": _canonical_sha256(list(groups)),
        "baseline_scaler_sha256": baseline_scaler_sha256,
        "medical_generalization": False,
        "objective": "masked_span_reconstruction_only",
        "next_step_objective": False,
    }


def _load_authorized_training_cache(
    evidence: SSLStageEvidence,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
    _require_authorized_stage_evidence(evidence)
    authorization = evidence._runtime_authorization
    assert isinstance(authorization, _SSLStageAuthorization)
    payloads = _reopen_authorized_cache_artifacts(authorization.cache_artifacts)
    cache_commitment = _cache_commitment_sha256(
        authorization.group_ids, payloads
    )
    if (
        cache_commitment != evidence.cache_commitment_sha256
        or cache_commitment != authorization.cache_commitment_sha256
        or len(payloads) != evidence.cache_count
        or len(payloads) != authorization.cache_count
    ):
        raise ValueError("authorized cache bytes contradict the frozen manifest")
    tensors = _parse_training_cache_payloads(
        payloads, stage=evidence.stage, group_ids=authorization.group_ids,
    )
    return (*tensors, cache_commitment, len(payloads))


def _mask_sha256(mask: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_fingerprint_bytes(mask)).hexdigest()


def _validate_training_receipt(receipt: SSLTrainingReceipt) -> None:
    if not isinstance(receipt, SSLTrainingReceipt):
        raise ValueError("SSL training receipt has the wrong type")
    if receipt.schema_version != SSL_TRAINING_RECEIPT_SCHEMA:
        raise ValueError("SSL training receipt schema is unsupported")
    if receipt.stage not in {RAVDESS_STAGE, MAYO_DEVELOPMENT_STAGE}:
        raise ValueError("SSL training receipt stage is unsupported")
    expected_source = RAVDESS_SOURCE if receipt.stage == RAVDESS_STAGE else MAYO_SOURCE
    if receipt.source != expected_source:
        raise ValueError("SSL training receipt source contradicts its stage")
    if (
        isinstance(receipt.seed, (bool, np.bool_))
        or not isinstance(receipt.seed, (int, np.integer))
        or int(receipt.seed) not in SSL_SEEDS
    ):
        raise ValueError("SSL training receipt seed is not registered")
    if (
        isinstance(receipt.cache_count, (bool, np.bool_))
        or not isinstance(receipt.cache_count, (int, np.integer))
        or int(receipt.cache_count) < 1
        or isinstance(receipt.optimizer_steps, (bool, np.bool_))
        or not isinstance(receipt.optimizer_steps, (int, np.integer))
        or int(receipt.optimizer_steps) < 1
    ):
        raise ValueError("SSL training receipt counts are invalid")
    for name in (
        "stage_evidence_sha256", "cache_binding_sha256", "manifest_sha256",
        "config_sha256", "split_artifact_sha256", "scaler_artifact_sha256",
        "train_indices_sha256", "heldout_indices_sha256", "group_ids_sha256",
        "scaler_sha256", "pre_state_sha256", "post_state_sha256",
        "baseline_state_sha256", "fresh_untrained_state_sha256",
        "train_mask_schedule_sha256",
        "heldout_mask_schedule_sha256", "train_trace_sha256",
        "heldout_report_sha256", "receipt_sha256",
    ):
        value = getattr(receipt, name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"SSL training receipt {name} is malformed")
    if receipt.stage == RAVDESS_STAGE:
        if receipt.prior_checkpoint_sha256 is not None:
            raise ValueError("RAVDESS training cannot name a prior checkpoint")
    elif (
        not isinstance(receipt.prior_checkpoint_sha256, str)
        or _SHA256.fullmatch(receipt.prior_checkpoint_sha256) is None
    ):
        raise ValueError("Mayo training requires an exact prior checkpoint")
    config_contract = {
        "schema_version": SSL_CONFIG_SCHEMA,
        "stage": receipt.stage,
        "source": receipt.source,
        "objective": "masked_span_smooth_l1_only",
        "sample_rate_hz": 30.0,
        "seeds": list(SSL_SEEDS),
        "development_only": receipt.stage == MAYO_DEVELOPMENT_STAGE,
        "optimizer": receipt.optimizer,
        "learning_rate": receipt.learning_rate,
        "weight_decay": receipt.weight_decay,
        "epochs": receipt.epochs,
        "batch_policy": receipt.batch_policy,
        "span_length": receipt.span_length,
        "spans_per_window": receipt.spans_per_window,
        "device": receipt.device,
    }
    _validate_training_config(
        config_contract,
        stage=receipt.stage,
        source=receipt.source,
        development_only=receipt.stage == MAYO_DEVELOPMENT_STAGE,
    )
    if receipt.optimizer_steps != receipt.epochs:
        raise ValueError("full-partition optimizer steps must equal frozen epochs")
    values = receipt.to_dict()
    values.pop("receipt_sha256")
    if receipt.receipt_sha256 != _canonical_sha256(values):
        raise ValueError("SSL training receipt digest does not match its claims")


def _training_receipt_from_mapping(value: object) -> SSLTrainingReceipt:
    if type(value) is not dict:
        raise ValueError("SSL training receipt must be an exact mapping")
    fields = set(SSLTrainingReceipt.__dataclass_fields__)
    if set(value) != fields:
        raise ValueError("SSL training receipt schema is not exact")
    try:
        receipt = SSLTrainingReceipt(**dict(value))
    except TypeError as exc:
        raise ValueError("SSL training receipt cannot be constructed") from exc
    _validate_training_receipt(receipt)
    return receipt


def _trainable_parameter_names(
    model: DynamicLandmarkSSLModel,
    stage: str,
) -> tuple[str, ...]:
    if not isinstance(model, DynamicLandmarkSSLModel):
        raise ValueError("source parameter allowlist requires the frozen SSL model")
    prefixes = (
        (
            "ravdess_proj_x.", "ravdess_proj_dx.", "temporal.",
            "attention_score.", "pool_projection.", "ravdess_decoder.",
        )
        if stage == "ravdess"
        else (
            "proj_bs_x.", "proj_bs_dx.", "proj_lm_x.", "proj_lm_dx.",
            "temporal.", "attention_score.", "pool_projection.",
            "mayo_decoder.",
        )
        if stage == "mayo"
        else ()
    )
    if not prefixes:
        raise ValueError("source parameter allowlist stage is unsupported")
    named = tuple(name for name, _parameter in model.named_parameters())
    selected = tuple(name for name in named if name.startswith(prefixes))
    if not selected or len(selected) == len(named) or len(set(selected)) != len(selected):
        raise RuntimeError("source parameter allowlist is malformed")
    return selected


def _require_seed_matched_prior_checkpoint(
    prior_ravdess_checkpoint: Mapping[str, object],
    seed: int,
) -> None:
    metadata = prior_ravdess_checkpoint.get("metadata")
    prior_seed = metadata.get("seed") if type(metadata) is dict else None
    if (
        isinstance(prior_seed, (bool, np.bool_))
        or not isinstance(prior_seed, (int, np.integer))
        or int(prior_seed) != seed
    ):
        raise ValueError("Mayo training requires a seed-matched RAVDESS checkpoint")


def _require_exact_ravdess_only_prior(
    prior_ravdess_checkpoint: Mapping[str, object],
    prior_stage_evidence: SSLStageEvidence,
    *,
    expected_mode: str | None,
    require_persisted: bool,
    seed: int | None = None,
) -> None:
    if (
        not isinstance(prior_stage_evidence, SSLStageEvidence)
        or prior_stage_evidence.stage != RAVDESS_STAGE
        or prior_stage_evidence.mode != expected_mode
    ):
        raise ValueError(
            "Mayo prior evidence must be exact same-mode RAVDESS evidence"
        )
    if (
        not isinstance(prior_ravdess_checkpoint, SSLCheckpointPayload)
        or prior_ravdess_checkpoint.get("checkpoint_type")
        != CHECKPOINT_RAVDESS_ONLY
    ):
        raise ValueError("Mayo prior checkpoint must be ravdess_only")
    _require_authorized_checkpoint_payload(
        prior_ravdess_checkpoint,
        prior_stage_evidence,
        require_persisted=require_persisted,
    )
    if seed is not None:
        _require_seed_matched_prior_checkpoint(prior_ravdess_checkpoint, seed)


def _train_ssl_stage_impl(
    *,
    stage_evidence: SSLStageEvidence,
    seed: int,
    prior_ravdess_checkpoint: Mapping[str, object] | None = None,
    prior_stage_evidence: SSLStageEvidence | None = None,
) -> SSLTrainingResult:
    """Run the one authorized deterministic CPU training/evaluation path."""
    _require_authorized_stage_evidence(stage_evidence)
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) not in SSL_SEEDS
    ):
        raise ValueError("training seed must be one of 0, 1, 2")
    seed = int(seed)
    authorization = stage_evidence._runtime_authorization
    assert isinstance(authorization, _SSLStageAuthorization)
    features, valid_mask, timestamps, source_indices, cache_binding, cache_count = (
        _load_authorized_training_cache(stage_evidence)
    )
    train_indices, heldout_indices, groups = _validate_split_partition(
        authorization.split, authorization.group_ids
    )
    source_key = "ravdess" if stage_evidence.stage == RAVDESS_STAGE else "mayo"
    expected_source_step = 1
    training_config = _validate_training_config(
        authorization.training_config,
        stage=stage_evidence.stage,
        source=stage_evidence.source,
        development_only=stage_evidence.development_only,
    )
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config["weight_decay"])
    epochs = int(training_config["epochs"])
    span_length = int(training_config["span_length"])
    spans_per_window = int(training_config["spans_per_window"])
    if seed not in tuple(training_config["seeds"]):
        raise ValueError("training seed is not registered for this frozen run mode")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if stage_evidence.stage == RAVDESS_STAGE:
            if prior_ravdess_checkpoint is not None or prior_stage_evidence is not None:
                raise ValueError("RAVDESS training cannot accept prior-stage state")
            model = DynamicLandmarkSSLModel().to("cpu")
            fresh_untrained_state = _clone_model_state(model)
            prior_checkpoint_sha256 = None
        else:
            if prior_ravdess_checkpoint is None or prior_stage_evidence is None:
                raise ValueError("Mayo training requires its exact authorized RAVDESS checkpoint")
            _require_exact_ravdess_only_prior(
                prior_ravdess_checkpoint,
                prior_stage_evidence,
                expected_mode=stage_evidence.mode,
                require_persisted=stage_evidence.mode is not None,
                seed=seed,
            )
            prior_checkpoint_sha256 = ssl_checkpoint_fingerprint(
                prior_ravdess_checkpoint
            )
            if prior_checkpoint_sha256 != stage_evidence.prior_checkpoint_sha256:
                raise ValueError("Mayo training prior checkpoint contradicts stage evidence")
            model = DynamicLandmarkSSLModel().to("cpu")
            fresh_untrained_state = _clone_model_state(model)
            model.load_state_dict(prior_ravdess_checkpoint["model_state"], strict=True)

        pre_state = _clone_model_state(model)
        pre_state_sha256 = _model_state_sha256(pre_state)
        baseline_state = {name: value.clone() for name, value in pre_state.items()}
        baseline_state_sha256 = _model_state_sha256(baseline_state)
        fresh_untrained_state_sha256 = _model_state_sha256(
            fresh_untrained_state
        )
        train_tensor_indices = torch.as_tensor(train_indices, dtype=torch.int64)
        train_features = features.index_select(0, train_tensor_indices)
        train_valid = valid_mask.index_select(0, train_tensor_indices)
        train_times = timestamps.index_select(0, train_tensor_indices)
        train_source_indices = source_indices.index_select(0, train_tensor_indices)
        train_scaled = authorization.scaler.transform(
            train_features, train_valid, source=stage_evidence.source
        )
        parameter_by_name = dict(model.named_parameters())
        trainable_names = _trainable_parameter_names(
            model, stage_evidence.stage
        )
        optimizer = torch.optim.AdamW(
            [parameter_by_name[name] for name in trainable_names],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        train_trace: list[dict[str, object]] = []
        train_mask_schedule: list[dict[str, object]] = []
        for epoch in range(epochs):
            mask_seed = seed + epoch * 100_003
            train_mask = make_contiguous_span_mask(
                train_valid,
                train_times,
                train_source_indices,
                expected_source_step=expected_source_step,
                span_length=span_length,
                spans_per_window=spans_per_window,
                seed=mask_seed,
            )
            train_mask_sha256 = _mask_sha256(train_mask)
            train_mask_schedule.append({
                "epoch": epoch,
                "seed": mask_seed,
                "mask_sha256": train_mask_sha256,
            })
            model.train()
            optimizer.zero_grad(set_to_none=True)
            trained_prediction = model(
                train_scaled,
                train_valid,
                train_times,
                train_source_indices,
                reconstruction_mask=train_mask,
                source=source_key,
            )
            loss = masked_smooth_l1(
                trained_prediction, train_scaled, train_mask
            )
            loss.backward()
            optimizer.step()
            train_trace.append({
                "step": epoch,
                "train_indices_sha256": stage_evidence.train_indices_sha256,
                "mask_sha256": train_mask_sha256,
                "loss": float(loss.detach().cpu().item()),
            })
        train_mask_schedule_sha256 = _canonical_sha256(train_mask_schedule)
        train_trace_sha256 = _canonical_sha256(train_trace)

        model.eval()
        post_state = _clone_model_state(model)
        post_state_sha256 = _model_state_sha256(post_state)
        if post_state_sha256 == pre_state_sha256:
            raise RuntimeError("authorized optimizer step produced no state update")
        if stage_evidence.mode == "smoke":
            heldout_mask_sha256 = _canonical_sha256([])
            report = {
                "mode": "smoke",
                "heldout_evaluation_computed": False,
                "train_loss": float(train_trace[-1]["loss"]),
                "optimizer_steps": epochs,
                "stage_evidence_sha256": stage_evidence.evidence_sha256,
                "cache_binding_sha256": cache_binding,
                "post_state_sha256": post_state_sha256,
                "fresh_untrained_state_sha256": fresh_untrained_state_sha256,
                "train_trace_sha256": train_trace_sha256,
                "medical_generalization": False,
                "objective": "masked_span_reconstruction_only",
                "next_step_objective": False,
            }
        else:
            baseline_model = DynamicLandmarkSSLModel().to("cpu")
            baseline_model.load_state_dict(baseline_state, strict=True)
            baseline_model.eval()
            fresh_untrained_model = DynamicLandmarkSSLModel().to("cpu")
            fresh_untrained_model.load_state_dict(
                fresh_untrained_state, strict=True
            )
            fresh_untrained_model.eval()
            heldout_tensor_indices = torch.as_tensor(heldout_indices, dtype=torch.int64)
            heldout_features = features.index_select(0, heldout_tensor_indices)
            heldout_valid = valid_mask.index_select(0, heldout_tensor_indices)
            heldout_times = timestamps.index_select(0, heldout_tensor_indices)
            heldout_source_indices = source_indices.index_select(
                0, heldout_tensor_indices
            )
            heldout_scaled = authorization.scaler.transform(
                heldout_features, heldout_valid, source=stage_evidence.source
            )
            heldout_mask = make_contiguous_span_mask(
                heldout_valid,
                heldout_times,
                heldout_source_indices,
                expected_source_step=expected_source_step,
                span_length=span_length,
                spans_per_window=spans_per_window,
                seed=10_000 + seed,
            )
            heldout_mask_sha256 = _mask_sha256(heldout_mask)
            with torch.no_grad():
                trained_heldout = model(
                    heldout_scaled,
                    heldout_valid,
                    heldout_times,
                    heldout_source_indices,
                    reconstruction_mask=heldout_mask,
                    source=source_key,
                )
                baseline_heldout = baseline_model(
                    heldout_scaled,
                    heldout_valid,
                    heldout_times,
                    heldout_source_indices,
                    reconstruction_mask=heldout_mask,
                    source=source_key,
                )
                fresh_untrained_heldout = fresh_untrained_model(
                    heldout_scaled,
                    heldout_valid,
                    heldout_times,
                    heldout_source_indices,
                    reconstruction_mask=heldout_mask,
                    source=source_key,
                )
            report = reconstruction_report(
                trained_heldout,
                baseline_heldout,
                heldout_scaled,
                heldout_mask,
                baseline=authorization.scaler,
                split=authorization.split,
                evaluated_indices=heldout_indices,
                group_ids=groups,
                source=stage_evidence.source,
            )
            if stage_evidence.stage == MAYO_DEVELOPMENT_STAGE:
                report["prior_ravdess"] = report.pop("untrained")
                report["fresh_untrained"] = float(masked_smooth_l1(
                    fresh_untrained_heldout, heldout_scaled, heldout_mask
                ).item())
                initialization_baseline_metric = "prior_ravdess"
            else:
                initialization_baseline_metric = "untrained"
            report.update({
                "seed": seed,
                "stage_evidence_sha256": stage_evidence.evidence_sha256,
                "cache_binding_sha256": cache_binding,
                "post_state_sha256": post_state_sha256,
                "baseline_state_sha256": baseline_state_sha256,
                "fresh_untrained_state_sha256": fresh_untrained_state_sha256,
                "heldout_mask_schedule_sha256": heldout_mask_sha256,
                "initialization_baseline_metric": initialization_baseline_metric,
                "baseline_initialization": (
                    "same_seed_fresh_ravdess"
                    if stage_evidence.stage == RAVDESS_STAGE
                    else "exact_prior_ravdess_checkpoint"
                ),
            })
        heldout_report_sha256 = _canonical_sha256(report)
        receipt_values: dict[str, object] = {
            "schema_version": SSL_TRAINING_RECEIPT_SCHEMA,
            "stage": stage_evidence.stage,
            "source": stage_evidence.source,
            "seed": seed,
            "stage_evidence_sha256": stage_evidence.evidence_sha256,
            "cache_binding_sha256": cache_binding,
            "cache_count": cache_count,
            "manifest_sha256": stage_evidence.manifest_sha256,
            "config_sha256": stage_evidence.config_sha256,
            "split_artifact_sha256": stage_evidence.split_artifact_sha256,
            "scaler_artifact_sha256": stage_evidence.scaler_artifact_sha256,
            "train_indices_sha256": stage_evidence.train_indices_sha256,
            "heldout_indices_sha256": stage_evidence.heldout_indices_sha256,
            "group_ids_sha256": stage_evidence.group_ids_sha256,
            "scaler_sha256": stage_evidence.scaler_sha256,
            "prior_checkpoint_sha256": prior_checkpoint_sha256,
            "pre_state_sha256": pre_state_sha256,
            "post_state_sha256": post_state_sha256,
            "baseline_state_sha256": baseline_state_sha256,
            "fresh_untrained_state_sha256": fresh_untrained_state_sha256,
            "train_mask_schedule_sha256": train_mask_schedule_sha256,
            "heldout_mask_schedule_sha256": heldout_mask_sha256,
            "train_trace_sha256": train_trace_sha256,
            "heldout_report_sha256": heldout_report_sha256,
            "optimizer": training_config["optimizer"],
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_policy": training_config["batch_policy"],
            "span_length": span_length,
            "spans_per_window": spans_per_window,
            "device": training_config["device"],
            "optimizer_steps": epochs,
        }
        receipt_values["receipt_sha256"] = _canonical_sha256(receipt_values)
        receipt = SSLTrainingReceipt(**receipt_values)
        _validate_training_receipt(receipt)
        result = SSLTrainingResult(
            model=model,
            stage_evidence=stage_evidence,
            training_receipt=receipt,
            heldout_report=report,
        )
        authorization_result = _SSLTrainingResultAuthorization(
            marker=_AUTHORIZATION_MARKER,
            result_reference=weakref.ref(result),
            model_reference=weakref.ref(model),
            receipt_reference=weakref.ref(receipt),
            heldout_report=report,
            stage_evidence_sha256=stage_evidence.evidence_sha256,
            cache_binding_sha256=cache_binding,
            post_state_sha256=post_state_sha256,
            heldout_report_sha256=heldout_report_sha256,
            receipt_sha256=receipt.receipt_sha256,
        )
        object.__setattr__(result, "_runtime_authorization", authorization_result)
    _require_authorized_training_result(result)
    return result


def train_ssl_stage(
    *,
    stage_evidence: SSLStageEvidence,
    seed: int,
    prior_ravdess_checkpoint: Mapping[str, object] | None = None,
    prior_stage_evidence: SSLStageEvidence | None = None,
) -> SSLTrainingResult:
    _require_public_receipt_bound_stage(stage_evidence)
    return _train_ssl_stage_impl(
        stage_evidence=stage_evidence,
        seed=seed,
        prior_ravdess_checkpoint=prior_ravdess_checkpoint,
        prior_stage_evidence=prior_stage_evidence,
    )


def _require_authorized_training_result(result: SSLTrainingResult) -> None:
    if not isinstance(result, SSLTrainingResult):
        raise ValueError("checkpoint construction requires SSLTrainingResult")
    authorization = result._runtime_authorization
    if (
        not isinstance(authorization, _SSLTrainingResultAuthorization)
        or authorization.marker is not _AUTHORIZATION_MARKER
        or authorization.result_reference() is not result
        or authorization.model_reference() is not result.model
        or authorization.receipt_reference() is not result.training_receipt
        or authorization.heldout_report is not result.heldout_report
        or result.stage_evidence.evidence_sha256
        != authorization.stage_evidence_sha256
    ):
        raise ValueError("SSL training result lacks trusted runtime authorization")
    _require_authorized_stage_evidence(result.stage_evidence)
    _validate_training_receipt(result.training_receipt)
    stage_authorization = result.stage_evidence._runtime_authorization
    assert isinstance(stage_authorization, _SSLStageAuthorization)
    payloads = _reopen_authorized_cache_artifacts(
        stage_authorization.cache_artifacts
    )
    cache_binding = _cache_commitment_sha256(
        stage_authorization.group_ids, payloads
    )
    post_state_sha256 = _model_state_sha256(result.model.state_dict())
    report_sha256 = _canonical_sha256(result.heldout_report)
    receipt = result.training_receipt
    expected_receipt_values = {
        "stage": result.stage_evidence.stage,
        "source": result.stage_evidence.source,
        "stage_evidence_sha256": result.stage_evidence.evidence_sha256,
        "cache_binding_sha256": cache_binding,
        "cache_count": result.stage_evidence.cache_count,
        "manifest_sha256": result.stage_evidence.manifest_sha256,
        "config_sha256": result.stage_evidence.config_sha256,
        "split_artifact_sha256": result.stage_evidence.split_artifact_sha256,
        "scaler_artifact_sha256": result.stage_evidence.scaler_artifact_sha256,
        "train_indices_sha256": result.stage_evidence.train_indices_sha256,
        "heldout_indices_sha256": result.stage_evidence.heldout_indices_sha256,
        "group_ids_sha256": result.stage_evidence.group_ids_sha256,
        "scaler_sha256": result.stage_evidence.scaler_sha256,
        "prior_checkpoint_sha256": result.stage_evidence.prior_checkpoint_sha256,
        "post_state_sha256": post_state_sha256,
        "heldout_report_sha256": report_sha256,
        "fresh_untrained_state_sha256": result.heldout_report.get(
            "fresh_untrained_state_sha256"
        ),
        "optimizer": stage_authorization.training_config["optimizer"],
        "learning_rate": stage_authorization.training_config["learning_rate"],
        "weight_decay": stage_authorization.training_config["weight_decay"],
        "epochs": stage_authorization.training_config["epochs"],
        "batch_policy": stage_authorization.training_config["batch_policy"],
        "span_length": stage_authorization.training_config["span_length"],
        "spans_per_window": stage_authorization.training_config["spans_per_window"],
        "device": stage_authorization.training_config["device"],
        "optimizer_steps": stage_authorization.training_config["epochs"],
    }
    if any(getattr(receipt, name) != value for name, value in expected_receipt_values.items()):
        raise ValueError("SSL training receipt contradicts live inputs or trained state")
    if (
        cache_binding != authorization.cache_binding_sha256
        or post_state_sha256 != authorization.post_state_sha256
        or report_sha256 != authorization.heldout_report_sha256
        or receipt.receipt_sha256 != authorization.receipt_sha256
    ):
        raise ValueError("SSL training result changed after repository training")


def _checkpoint_type_for_evidence(evidence: SSLStageEvidence) -> str:
    _validate_stage_evidence(evidence)
    return (
        CHECKPOINT_RAVDESS_ONLY
        if evidence.stage == RAVDESS_STAGE
        else CHECKPOINT_RAVDESS_MAYO
    )


def _checkpoint_metadata(
    evidence: SSLStageEvidence,
    receipt: SSLTrainingReceipt,
    heldout_report: Mapping[str, object],
) -> dict[str, object]:
    checkpoint_type = _checkpoint_type_for_evidence(evidence)
    mayo = checkpoint_type == CHECKPOINT_RAVDESS_MAYO
    smoke = evidence.mode == "smoke"
    return {
        "objective": "masked_span_smooth_l1_only",
        "next_step_objective": False,
        "seed": receipt.seed,
        "config_sha256": evidence.config_sha256,
        "source_stages": ["ravdess", "mayo"] if mayo else ["ravdess"],
        "ravdess_claim": (
            "train_only_smoke_no_heldout_metric"
            if smoke else "actor_held_out_reconstruction"
        ),
        "mayo_claim": (
            "train_only_smoke_no_heldout_metric"
            if smoke and mayo
            else "recording_held_out_not_patient_held_out"
            if mayo
            else None
        ),
        "mayo_development_only": mayo,
        "palsynet_scaler_transfer_permitted": False,
        "stage_evidence": evidence.to_dict(),
        "training_receipt": receipt.to_dict(),
        "heldout_report": dict(heldout_report),
    }


def _build_ssl_checkpoint_payload_impl(
    training_result: SSLTrainingResult,
) -> SSLCheckpointPayload:
    _require_authorized_training_result(training_result)
    model = training_result.model
    stage_evidence = training_result.stage_evidence
    receipt = training_result.training_receipt
    checkpoint_type = _checkpoint_type_for_evidence(stage_evidence)
    state = {
        name: tensor.detach().to("cpu").clone()
        for name, tensor in model.state_dict().items()
    }
    payload = SSLCheckpointPayload({
        "schema_version": SSL_CHECKPOINT_SCHEMA,
        "checkpoint_type": checkpoint_type,
        "model_state": state,
        "metadata": _checkpoint_metadata(
            stage_evidence, receipt, training_result.heldout_report
        ),
    })
    validate_ssl_checkpoint_payload(payload)
    payload._runtime_authorization = _SSLCheckpointPayloadAuthorization(
        marker=_AUTHORIZATION_MARKER,
        payload_reference=weakref.ref(payload),
        checkpoint_fingerprint=ssl_checkpoint_fingerprint(payload),
        stage_evidence_sha256=stage_evidence.evidence_sha256,
    )
    return payload


def build_ssl_checkpoint_payload(
    training_result: SSLTrainingResult,
) -> SSLCheckpointPayload:
    if not isinstance(training_result, SSLTrainingResult):
        raise ValueError("checkpoint construction requires SSLTrainingResult")
    _require_public_receipt_bound_stage(training_result.stage_evidence)
    return _build_ssl_checkpoint_payload_impl(training_result)


def validate_ssl_checkpoint_payload(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "checkpoint_type", "model_state", "metadata"
    }:
        raise ValueError("SSL checkpoint has an unexpected top-level schema")
    if payload["schema_version"] != SSL_CHECKPOINT_SCHEMA:
        raise ValueError("SSL checkpoint schema version is unsupported")
    checkpoint_type = payload["checkpoint_type"]
    if checkpoint_type not in CHECKPOINT_TYPES:
        raise ValueError("SSL checkpoint type is unsupported")
    expected_state = _ssl_model_state_schema()
    observed_state = payload["model_state"]
    if not isinstance(observed_state, Mapping) or set(observed_state) != set(expected_state):
        raise ValueError("SSL checkpoint state dictionary must be exact and complete")
    for name, (expected_shape, expected_dtype) in expected_state.items():
        observed = observed_state[name]
        if not isinstance(observed, torch.Tensor):
            raise ValueError(f"SSL checkpoint value {name!r} is not a tensor")
        if observed.shape != expected_shape or observed.dtype != expected_dtype:
            raise ValueError(f"SSL checkpoint tensor {name!r} has wrong shape or dtype")
        if observed.is_floating_point() and not torch.isfinite(observed).all():
            raise ValueError(f"SSL checkpoint tensor {name!r} is nonfinite")
    metadata = payload["metadata"]
    expected_metadata = {
        "objective", "next_step_objective", "seed", "config_sha256",
        "source_stages", "ravdess_claim", "mayo_claim",
        "mayo_development_only", "palsynet_scaler_transfer_permitted",
        "stage_evidence", "training_receipt", "heldout_report",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != expected_metadata:
        raise ValueError("SSL checkpoint metadata schema is not exact")
    seed = metadata["seed"]
    config_hash = metadata["config_sha256"]
    evidence = _stage_evidence_from_mapping(metadata["stage_evidence"])
    receipt = _training_receipt_from_mapping(metadata["training_receipt"])
    report = metadata["heldout_report"]
    if type(report) is not dict:
        raise ValueError("SSL checkpoint heldout report must be an exact mapping")
    if _checkpoint_type_for_evidence(evidence) != checkpoint_type:
        raise ValueError("SSL checkpoint type is not derived from its stage evidence")
    expected_values = _checkpoint_metadata(evidence, receipt, report)
    if metadata != expected_values:
        raise ValueError("SSL checkpoint metadata contradicts its checkpoint type")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) not in SSL_SEEDS
    ):
        raise ValueError("SSL checkpoint seed is not registered")
    if not isinstance(config_hash, str) or _SHA256.fullmatch(config_hash) is None:
        raise ValueError("SSL checkpoint config hash is malformed")
    evidence_receipt_values = {
        "stage": evidence.stage,
        "source": evidence.source,
        "seed": seed,
        "stage_evidence_sha256": evidence.evidence_sha256,
        "cache_binding_sha256": evidence.cache_commitment_sha256,
        "cache_count": evidence.cache_count,
        "manifest_sha256": evidence.manifest_sha256,
        "config_sha256": evidence.config_sha256,
        "split_artifact_sha256": evidence.split_artifact_sha256,
        "scaler_artifact_sha256": evidence.scaler_artifact_sha256,
        "train_indices_sha256": evidence.train_indices_sha256,
        "heldout_indices_sha256": evidence.heldout_indices_sha256,
        "group_ids_sha256": evidence.group_ids_sha256,
        "scaler_sha256": evidence.scaler_sha256,
        "prior_checkpoint_sha256": evidence.prior_checkpoint_sha256,
        "post_state_sha256": _model_state_sha256(observed_state),
        "heldout_report_sha256": _canonical_sha256(report),
    }
    if any(
        getattr(receipt, name) != value
        for name, value in evidence_receipt_values.items()
    ):
        raise ValueError("SSL checkpoint receipt contradicts state, report, or stage")
    if evidence.mode == "smoke":
        expected_smoke_fields = {
            "mode", "heldout_evaluation_computed", "train_loss",
            "optimizer_steps", "stage_evidence_sha256",
            "cache_binding_sha256", "post_state_sha256",
            "fresh_untrained_state_sha256", "train_trace_sha256",
            "medical_generalization", "objective", "next_step_objective",
        }
        train_loss = report.get("train_loss")
        if (
            set(report) != expected_smoke_fields
            or report.get("mode") != "smoke"
            or report.get("heldout_evaluation_computed") is not False
            or isinstance(train_loss, (bool, np.bool_))
            or not isinstance(train_loss, (int, float, np.number))
            or not math.isfinite(float(train_loss))
            or report.get("optimizer_steps") != receipt.optimizer_steps
            or report.get("stage_evidence_sha256") != evidence.evidence_sha256
            or report.get("cache_binding_sha256") != receipt.cache_binding_sha256
            or report.get("post_state_sha256") != receipt.post_state_sha256
            or report.get("fresh_untrained_state_sha256")
            != receipt.fresh_untrained_state_sha256
            or report.get("train_trace_sha256") != receipt.train_trace_sha256
            or report.get("medical_generalization") is not False
            or report.get("objective") != "masked_span_reconstruction_only"
            or report.get("next_step_objective") is not False
            or receipt.heldout_mask_schedule_sha256 != _canonical_sha256([])
        ):
            raise ValueError("smoke checkpoint report is not exact train-only evidence")
        return
    report_expected = {
        "target_space": "source_train_standardized",
        "seed": seed,
        "stage_evidence_sha256": evidence.evidence_sha256,
        "cache_binding_sha256": receipt.cache_binding_sha256,
        "post_state_sha256": receipt.post_state_sha256,
        "baseline_state_sha256": receipt.baseline_state_sha256,
        "fresh_untrained_state_sha256": receipt.fresh_untrained_state_sha256,
        "heldout_mask_schedule_sha256": receipt.heldout_mask_schedule_sha256,
        "evaluated_indices_sha256": evidence.heldout_indices_sha256,
        "group_ids_sha256": evidence.group_ids_sha256,
        "baseline_scaler_sha256": evidence.scaler_sha256,
        "source": evidence.source,
    }
    if any(report.get(name) != value for name, value in report_expected.items()):
        raise ValueError("SSL checkpoint heldout report contradicts its training receipt")
    if evidence.stage == RAVDESS_STAGE:
        if (
            report.get("initialization_baseline_metric") != "untrained"
            or "untrained" not in report
            or "prior_ravdess" in report
            or "fresh_untrained" in report
        ):
            raise ValueError("RAVDESS checkpoint baseline report is mislabeled")
        metric_names = ("trained", "untrained", "train_mean")
    else:
        if (
            report.get("initialization_baseline_metric") != "prior_ravdess"
            or "untrained" in report
            or "prior_ravdess" not in report
            or "fresh_untrained" not in report
        ):
            raise ValueError("Mayo checkpoint baselines are missing or mislabeled")
        metric_names = (
            "trained", "prior_ravdess", "fresh_untrained", "train_mean",
        )
    if any(
        isinstance(report.get(name), (bool, np.bool_))
        or not isinstance(report.get(name), (int, float, np.number))
        or not math.isfinite(float(report[name]))
        or float(report[name]) < 0.0
        for name in metric_names
    ):
        raise ValueError("SSL checkpoint heldout metrics must be finite and nonnegative")


def ssl_checkpoint_fingerprint(payload: Mapping[str, object]) -> str:
    """Return a deterministic content digest for validated checkpoint lineage."""
    validate_ssl_checkpoint_payload(payload)
    digest = hashlib.sha256()
    digest.update(_canonical_sha256({
        "schema_version": payload["schema_version"],
        "checkpoint_type": payload["checkpoint_type"],
        "metadata": payload["metadata"],
    }).encode("ascii"))
    state = payload["model_state"]
    for name in sorted(state):
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_fingerprint_bytes(state[name]))
    return digest.hexdigest()


def _require_authorized_checkpoint_payload(
    payload: Mapping[str, object],
    stage_evidence: SSLStageEvidence,
    *,
    require_persisted: bool = False,
) -> None:
    if type(require_persisted) is not bool:
        raise ValueError("checkpoint persistence requirement must be boolean")
    _require_authorized_stage_evidence(stage_evidence)
    if not isinstance(payload, SSLCheckpointPayload):
        raise ValueError("checkpoint payload lacks exact-state runtime authorization")
    authorization = payload._runtime_authorization
    if (
        not isinstance(authorization, _SSLCheckpointPayloadAuthorization)
        or authorization.marker is not _AUTHORIZATION_MARKER
        or authorization.payload_reference() is not payload
        or authorization.stage_evidence_sha256 != stage_evidence.evidence_sha256
    ):
        raise ValueError("checkpoint payload runtime authorization is invalid")
    validate_ssl_checkpoint_payload(payload)
    if (
        payload["metadata"]["stage_evidence"] != stage_evidence.to_dict()
        or payload["checkpoint_type"] != _checkpoint_type_for_evidence(stage_evidence)
        or ssl_checkpoint_fingerprint(payload)
        != authorization.checkpoint_fingerprint
    ):
        raise ValueError("checkpoint payload changed after exact-state authorization")
    if authorization.receipt_reference is not None:
        receipt = authorization.receipt_reference()
        if receipt is None or authorization.checkpoint_path is None:
            raise ValueError("persisted checkpoint authority expired")
        _require_authorized_checkpoint_receipt(
            receipt, stage_evidence, authorization.checkpoint_path
        )
        if receipt.checkpoint_fingerprint != authorization.checkpoint_fingerprint:
            raise ValueError("persisted checkpoint receipt fingerprint changed")
    elif require_persisted:
        raise ValueError("checkpoint must be reloaded from its keyed private receipt")


def _stage_receipt_authority_key(
    stage_evidence: SSLStageEvidence,
) -> bytes | None:
    if stage_evidence.mode is None:
        return None
    _require_authorized_stage_evidence(stage_evidence)
    authorization = stage_evidence._runtime_authorization
    if not isinstance(authorization, _SSLStageAuthorization):
        raise ValueError("receipt-bound stage authority is unavailable")
    frozen = authorization.frozen_stage
    if not isinstance(frozen, _FrozenSSLStageAuthorization):
        raise ValueError("receipt-bound frozen authority is unavailable")
    live = (
        frozen.ravdess_authorizer()
        if stage_evidence.stage == "ravdess"
        else frozen.mayo_authorizer()
    )
    key = getattr(live, "private_key", None)
    key_identity = getattr(live, "key_file_identity_sha256", None)
    if (
        type(key) is not bytes
        or len(key) != 32
        or key_identity != stage_evidence.canonical_key_identity_sha256
    ):
        raise ValueError("live canonical key contradicts stage evidence")
    _require_authorized_stage_evidence(stage_evidence)
    return key


def _checkpoint_receipt_authority_hmac(
    values: Mapping[str, object],
    key: bytes,
) -> str:
    if "receipt_sha256" in values or "stage_authority_hmac" in values:
        raise ValueError("checkpoint receipt HMAC core contains derived fields")
    return hmac.new(
        key,
        b"dynamic-landmark-ssl-checkpoint-receipt-v2\0"
        + _canonical_json_bytes(values),
        hashlib.sha256,
    ).hexdigest()


def _validate_checkpoint_receipt(receipt: SSLCheckpointReceipt) -> None:
    if not isinstance(receipt, SSLCheckpointReceipt):
        raise ValueError("checkpoint receipt has the wrong type")
    if receipt.schema_version not in {
        SSL_CHECKPOINT_RECEIPT_SCHEMA, SSL_CHECKPOINT_RECEIPT_V2_SCHEMA,
    }:
        raise ValueError("checkpoint receipt schema is unsupported")
    if (
        not isinstance(receipt.checkpoint_name, str)
        or not receipt.checkpoint_name
        or Path(receipt.checkpoint_name).name != receipt.checkpoint_name
    ):
        raise ValueError("checkpoint receipt file name is malformed")
    if receipt.checkpoint_type not in CHECKPOINT_TYPES:
        raise ValueError("checkpoint receipt type is unsupported")
    for name in (
        "checkpoint_fingerprint", "checkpoint_file_sha256",
        "checkpoint_file_identity_sha256",
        "receipt_file_identity_sha256",
        "stage_evidence_sha256", "receipt_sha256",
    ):
        value = getattr(receipt, name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"checkpoint receipt {name} is malformed")
    values = receipt.to_dict()
    values.pop("receipt_sha256")
    if receipt.receipt_sha256 != _canonical_sha256(values):
        raise ValueError("checkpoint receipt digest does not match its claims")
    if receipt.schema_version == SSL_CHECKPOINT_RECEIPT_V2_SCHEMA:
        _require_sha256(
            receipt.stage_authority_hmac,
            "checkpoint receipt stage authority HMAC",
        )
    elif receipt.stage_authority_hmac is not None:
        raise ValueError("legacy checkpoint receipt cannot carry a keyed HMAC")


def _require_authorized_checkpoint_receipt(
    receipt: SSLCheckpointReceipt,
    stage_evidence: SSLStageEvidence,
    checkpoint_path: str | Path,
) -> bytes:
    _require_authorized_stage_evidence(stage_evidence)
    _validate_checkpoint_receipt(receipt)
    if stage_evidence.mode is not None:
        if receipt.schema_version != SSL_CHECKPOINT_RECEIPT_V2_SCHEMA:
            raise ValueError("receipt-bound checkpoint requires a keyed v2 receipt")
        key = _stage_receipt_authority_key(stage_evidence)
        assert key is not None
        hmac_values = receipt.to_dict()
        hmac_values.pop("receipt_sha256")
        observed_hmac = hmac_values.pop("stage_authority_hmac")
        expected_hmac = _checkpoint_receipt_authority_hmac(hmac_values, key)
        if not hmac.compare_digest(str(observed_hmac), expected_hmac):
            raise ValueError("checkpoint receipt stage authority HMAC is invalid")
    authorization = receipt._runtime_authorization
    if (
        not isinstance(authorization, _SSLCheckpointReceiptAuthorization)
        or authorization.marker is not _AUTHORIZATION_MARKER
        or authorization.receipt_reference() is not receipt
        or authorization.stage_evidence_sha256 != stage_evidence.evidence_sha256
        or receipt.stage_evidence_sha256 != stage_evidence.evidence_sha256
    ):
        raise ValueError("checkpoint receipt lacks trusted runtime authorization")
    (
        receipt_resolved,
        receipt_bytes,
        receipt_file_sha256,
        receipt_identity,
    ) = _private_regular_file_snapshot(
        authorization.receipt_path, "checkpoint receipt"
    )
    if (
        receipt_resolved != authorization.receipt_path
        or receipt_file_sha256 != authorization.receipt_file_sha256
        or receipt_identity != authorization.receipt_identity
        or _stable_storage_identity_sha256(receipt_identity)
        != receipt.receipt_file_identity_sha256
    ):
        raise ValueError("checkpoint receipt file changed after authorization")
    receipt_file_value = _strict_json_mapping(
        receipt_bytes, "checkpoint receipt",
    )
    if not _exact_json_value(receipt_file_value, receipt.to_dict()):
        raise ValueError("checkpoint receipt object does not match its file")
    (
        checkpoint_resolved,
        checkpoint_bytes,
        checkpoint_file_sha256,
        checkpoint_identity,
    ) = _private_regular_file_snapshot(
        checkpoint_path, "SSL checkpoint"
    )
    if (
        checkpoint_resolved != authorization.checkpoint_path
        or checkpoint_resolved.name != receipt.checkpoint_name
        or checkpoint_file_sha256 != receipt.checkpoint_file_sha256
        or checkpoint_identity != authorization.checkpoint_identity
        or _stable_storage_identity_sha256(checkpoint_identity)
        != receipt.checkpoint_file_identity_sha256
    ):
        raise ValueError("SSL checkpoint file does not match its trusted receipt")
    return checkpoint_bytes


def _authorize_ssl_checkpoint_receipt_impl(
    receipt_path: str | Path,
    checkpoint_path: str | Path,
    *,
    trusted_expected_receipt_sha256: str,
    stage_evidence: SSLStageEvidence,
) -> SSLCheckpointReceipt:
    """Restore receipt trust from a digest frozen in an external registry."""
    _require_authorized_stage_evidence(stage_evidence)
    if (
        not isinstance(trusted_expected_receipt_sha256, str)
        or _SHA256.fullmatch(trusted_expected_receipt_sha256) is None
    ):
        raise ValueError("trusted expected receipt digest must be exact SHA-256")
    (
        receipt_resolved,
        receipt_bytes,
        receipt_file_sha256,
        receipt_identity,
    ) = _private_regular_file_snapshot(
        receipt_path, "checkpoint receipt"
    )
    value = _strict_json_mapping(receipt_bytes, "checkpoint receipt")
    fields = {
        name for name in SSLCheckpointReceipt.__dataclass_fields__
        if not name.startswith("_")
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("checkpoint receipt file schema is not exact")
    try:
        receipt = SSLCheckpointReceipt(**value)
    except TypeError as exc:
        raise ValueError("checkpoint receipt file cannot be constructed") from exc
    _validate_checkpoint_receipt(receipt)
    if receipt.receipt_sha256 != trusted_expected_receipt_sha256:
        raise ValueError("checkpoint receipt is absent from the trusted registry")
    if (
        receipt.stage_evidence_sha256 != stage_evidence.evidence_sha256
        or receipt.checkpoint_type != _checkpoint_type_for_evidence(stage_evidence)
        or _stable_storage_identity_sha256(receipt_identity)
        != receipt.receipt_file_identity_sha256
    ):
        raise ValueError("checkpoint receipt contradicts the live stage evidence")
    (
        checkpoint_resolved,
        _,
        checkpoint_file_sha256,
        checkpoint_identity,
    ) = _private_regular_file_snapshot(
        checkpoint_path, "SSL checkpoint"
    )
    if (
        checkpoint_resolved.name != receipt.checkpoint_name
        or checkpoint_file_sha256 != receipt.checkpoint_file_sha256
        or _stable_storage_identity_sha256(checkpoint_identity)
        != receipt.checkpoint_file_identity_sha256
    ):
        raise ValueError("SSL checkpoint file does not match the registered receipt")
    authorization = _SSLCheckpointReceiptAuthorization(
        marker=_AUTHORIZATION_MARKER,
        receipt_reference=weakref.ref(receipt),
        checkpoint_path=checkpoint_resolved,
        receipt_path=receipt_resolved,
        receipt_file_sha256=receipt_file_sha256,
        checkpoint_identity=checkpoint_identity,
        receipt_identity=receipt_identity,
        stage_evidence_sha256=stage_evidence.evidence_sha256,
    )
    object.__setattr__(receipt, "_runtime_authorization", authorization)
    _require_authorized_checkpoint_receipt(
        receipt, stage_evidence, checkpoint_resolved
    )
    return receipt


def authorize_ssl_checkpoint_receipt(
    receipt_path: str | Path,
    checkpoint_path: str | Path,
    *,
    trusted_expected_receipt_sha256: str,
    stage_evidence: SSLStageEvidence,
) -> SSLCheckpointReceipt:
    _require_public_receipt_bound_stage(stage_evidence)
    return _authorize_ssl_checkpoint_receipt_impl(
        receipt_path,
        checkpoint_path,
        trusted_expected_receipt_sha256=trusted_expected_receipt_sha256,
        stage_evidence=stage_evidence,
    )


def _checkpoint_parent_identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        int(status.st_dev), int(status.st_ino), int(status.st_mode),
        int(status.st_uid), int(status.st_gid),
    )


def _canonical_checkpoint_parent(
    path: Path,
) -> tuple[Path, Path, tuple[int, ...]]:
    lexical = path.absolute()
    try:
        resolved = lexical.resolve(strict=True)
        status = lexical.lstat()
    except OSError as exc:
        raise ValueError("SSL checkpoint parent is unavailable") from exc
    canonical = resolved == lexical
    if not canonical and sys.platform == "darwin":
        for source, destination in (
            (Path("/var"), Path("/private/var")),
            (Path("/tmp"), Path("/private/tmp")),
        ):
            try:
                relative = lexical.relative_to(source)
            except ValueError:
                continue
            if resolved == destination / relative:
                canonical = True
                break
    if not canonical or not stat.S_ISDIR(status.st_mode):
        raise ValueError("SSL checkpoint parent must be canonical storage")
    return lexical, resolved, _checkpoint_parent_identity(status)


def _assert_private_checkpoint_parent(
    descriptor: int,
    lexical: Path,
    expected_identity: tuple[int, ...],
) -> None:
    try:
        opened = os.fstat(descriptor)
        current = os.stat(lexical, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("SSL checkpoint parent changed") from exc
    if (
        _checkpoint_parent_identity(opened) != expected_identity
        or _checkpoint_parent_identity(current) != expected_identity
        or not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        raise ValueError("SSL checkpoint parent changed or is not owner-only")


def _close_ssl_descriptors(descriptors: tuple[int, ...]) -> None:
    closer = ExitStack()
    for descriptor in descriptors:
        closer.callback(os.close, descriptor)
    closer.__exit__(*sys.exc_info())


def _open_private_checkpoint_parent(
    path: Path,
) -> tuple[Path, Path, int, tuple[int, ...]]:
    lexical, resolved, initial_identity = _canonical_checkpoint_parent(path)
    descriptor = os.open(
        lexical,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        identity = _checkpoint_parent_identity(os.fstat(descriptor))
        if identity != initial_identity:
            raise ValueError("SSL checkpoint parent changed while opening")
        _assert_private_checkpoint_parent(descriptor, lexical, identity)
        return lexical, resolved, descriptor, identity
    except BaseException:
        _close_ssl_descriptors((descriptor,))
        raise


def _private_entry_status(
    parent_descriptor: int,
    name: str,
) -> os.stat_result | None:
    try:
        return os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
    except FileNotFoundError:
        return None


def _create_held_private_regular(
    parent_descriptor: int,
    name: str,
    label: str,
) -> tuple[int, tuple[int, int]]:
    if name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError(f"{label} name is unsafe")
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=parent_descriptor,
    )
    try:
        created = os.fstat(descriptor)
        created_mode = stat.S_IMODE(created.st_mode)
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_uid != os.geteuid()
            or created.st_nlink != 1
            or created_mode & 0o077
        ):
            raise ValueError(f"{label} was not created as private storage")
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        current = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
        inode = (int(opened.st_dev), int(opened.st_ino))
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
            or (int(current.st_dev), int(current.st_ino)) != inode
        ):
            raise ValueError(f"{label} storage changed during creation")
        return descriptor, inode
    except BaseException:
        _close_ssl_descriptors((descriptor,))
        raise


def _snapshot_held_private_regular(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    inode: tuple[int, int],
    label: str,
) -> tuple[bytes, str, _RegularFileIdentity]:
    before = os.fstat(descriptor)
    current = os.stat(
        name, dir_fd=parent_descriptor, follow_symlinks=False,
    )
    identity = _identity_from_stat(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or identity.uid != os.geteuid()
        or identity.mode != 0o600
        or identity.links != 1
        or (identity.device, identity.inode) != inode
        or (int(current.st_dev), int(current.st_ino)) != inode
    ):
        raise ValueError(f"{label} is not held private storage")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    payload = b"".join(chunks)
    after = os.fstat(descriptor)
    linked = os.stat(
        name, dir_fd=parent_descriptor, follow_symlinks=False,
    )
    if (
        _identity_from_stat(after) != identity
        or (int(linked.st_dev), int(linked.st_ino)) != inode
        or len(payload) != identity.size
        or not payload
    ):
        raise ValueError(f"{label} changed during held verification")
    return payload, hashlib.sha256(payload).hexdigest(), identity


def _publish_held_private_regular(
    parent_descriptor: int,
    temporary_name: str,
    destination_name: str,
    descriptor: int,
    inode: tuple[int, int],
    label: str,
) -> None:
    current = _private_entry_status(parent_descriptor, temporary_name)
    if (
        current is None
        or (int(current.st_dev), int(current.st_ino)) != inode
        or (int(os.fstat(descriptor).st_dev), int(os.fstat(descriptor).st_ino))
        != inode
    ):
        raise ValueError(f"{label} staging identity changed")
    from .dynamic_landmark_ssl_bridge import (
        _atomic_publish_directory_no_replace_at,
    )

    publish_error: BaseException | None = None
    try:
        _atomic_publish_directory_no_replace_at(
            parent_descriptor, temporary_name, destination_name,
        )
    except BaseException as caught:
        publish_error = caught
    source = _private_entry_status(parent_descriptor, temporary_name)
    destination = _private_entry_status(parent_descriptor, destination_name)
    source_inode = None if source is None else (
        int(source.st_dev), int(source.st_ino)
    )
    destination_inode = None if destination is None else (
        int(destination.st_dev), int(destination.st_ino)
    )
    if source_inode == inode and destination_inode is None:
        if publish_error is not None:
            raise publish_error
        raise RuntimeError(f"{label} publication did not commit")
    if source_inode is not None or destination_inode != inode:
        raise RuntimeError(
            f"{label} publication outcome is indeterminate"
        ) from publish_error
    if publish_error is not None:
        raise RuntimeError(
            f"{label} is retained after a publication return fault"
        ) from publish_error


def _save_ssl_checkpoint_impl(
    path: str | Path,
    payload: Mapping[str, object],
    *,
    stage_evidence: SSLStageEvidence,
) -> SSLCheckpointReceipt:
    _require_authorized_checkpoint_payload(payload, stage_evidence)
    destination = Path(path).absolute()
    if destination.name in {"", ".", ".."}:
        raise ValueError("SSL checkpoint destination name is unsafe")
    receipt_name = f"{destination.name}.receipt.json"
    parent_lexical, parent_resolved, parent_descriptor, parent_identity = (
        _open_private_checkpoint_parent(destination.parent)
    )
    checkpoint_descriptor: int | None = None
    receipt_descriptor: int | None = None
    try:
        _assert_private_checkpoint_parent(
            parent_descriptor, parent_lexical, parent_identity,
        )
        if (
            _private_entry_status(parent_descriptor, destination.name) is not None
            or _private_entry_status(parent_descriptor, receipt_name) is not None
        ):
            raise FileExistsError("SSL checkpoint destination already exists")
        checkpoint_temporary_name = (
            f".{destination.name}.tmp-{secrets.token_hex(8)}"
        )
        checkpoint_descriptor, checkpoint_inode = (
            _create_held_private_regular(
                parent_descriptor,
                checkpoint_temporary_name,
                "temporary SSL checkpoint",
            )
        )
        with os.fdopen(
            checkpoint_descriptor, "wb", closefd=False,
        ) as handle:
            torch.save(
                dict(payload),
                handle,
                _use_new_zipfile_serialization=False,
            )
            handle.flush()
            os.fsync(checkpoint_descriptor)
        temporary_bytes, temporary_checkpoint_sha256, _ = (
            _snapshot_held_private_regular(
            parent_descriptor,
            checkpoint_temporary_name,
            checkpoint_descriptor,
            checkpoint_inode,
            "temporary SSL checkpoint",
            )
        )
        try:
            reread = torch.load(
                io.BytesIO(temporary_bytes), map_location="cpu", weights_only=True
            )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise ValueError("cannot reread the temporary SSL checkpoint") from exc
        validate_ssl_checkpoint_payload(reread)
        checkpoint_fingerprint = ssl_checkpoint_fingerprint(payload)
        if ssl_checkpoint_fingerprint(reread) != checkpoint_fingerprint:
            raise ValueError("checkpoint reread changed its exact state")
        _assert_private_checkpoint_parent(
            parent_descriptor, parent_lexical, parent_identity,
        )
        prepublish_bytes, prepublish_sha256, _ = (
            _snapshot_held_private_regular(
                parent_descriptor,
                checkpoint_temporary_name,
                checkpoint_descriptor,
                checkpoint_inode,
                "temporary SSL checkpoint",
            )
        )
        if (
            prepublish_bytes != temporary_bytes
            or prepublish_sha256 != temporary_checkpoint_sha256
        ):
            raise ValueError("SSL checkpoint changed before publication")
        _publish_held_private_regular(
            parent_descriptor,
            checkpoint_temporary_name,
            destination.name,
            checkpoint_descriptor,
            checkpoint_inode,
            "SSL checkpoint",
        )
        os.fsync(parent_descriptor)
        _assert_private_checkpoint_parent(
            parent_descriptor, parent_lexical, parent_identity,
        )
        (
            checkpoint_bytes,
            checkpoint_file_sha256,
            checkpoint_identity,
        ) = _snapshot_held_private_regular(
            parent_descriptor,
            destination.name,
            checkpoint_descriptor,
            checkpoint_inode,
            "SSL checkpoint",
        )
        if (
            checkpoint_bytes != temporary_bytes
            or checkpoint_file_sha256 != temporary_checkpoint_sha256
        ):
            raise ValueError("SSL checkpoint changed during publication")
        checkpoint_path = parent_resolved / destination.name
        authority_key = _stage_receipt_authority_key(stage_evidence)
        receipt_temporary_name = (
            f".{receipt_name}.tmp-{secrets.token_hex(8)}"
        )
        receipt_descriptor, receipt_inode = _create_held_private_regular(
            parent_descriptor,
            receipt_temporary_name,
            "temporary checkpoint receipt",
        )
        receipt_prewrite_identity = _identity_from_stat(
            os.fstat(receipt_descriptor)
        )
        receipt_values: dict[str, object] = {
            "schema_version": (
                SSL_CHECKPOINT_RECEIPT_V2_SCHEMA
                if authority_key is not None
                else SSL_CHECKPOINT_RECEIPT_SCHEMA
            ),
            "checkpoint_name": destination.name,
            "checkpoint_type": payload["checkpoint_type"],
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "checkpoint_file_identity_sha256": (
                _stable_storage_identity_sha256(checkpoint_identity)
            ),
            "receipt_file_identity_sha256": (
                _stable_storage_identity_sha256(receipt_prewrite_identity)
            ),
            "stage_evidence_sha256": stage_evidence.evidence_sha256,
        }
        receipt_values["stage_authority_hmac"] = (
            _checkpoint_receipt_authority_hmac(
                receipt_values, authority_key
            )
            if authority_key is not None
            else None
        )
        receipt_values["receipt_sha256"] = _canonical_sha256(
            receipt_values
        )
        receipt = SSLCheckpointReceipt(**receipt_values)
        _validate_checkpoint_receipt(receipt)
        encoded_receipt = json.dumps(
            receipt.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        os.lseek(receipt_descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(encoded_receipt):
            written = os.write(receipt_descriptor, encoded_receipt[offset:])
            if written < 1:
                raise OSError("checkpoint receipt write made no progress")
            offset += written
        os.fsync(receipt_descriptor)
        receipt_staged_bytes, _, _ = _snapshot_held_private_regular(
            parent_descriptor,
            receipt_temporary_name,
            receipt_descriptor,
            receipt_inode,
            "temporary checkpoint receipt",
        )
        if (
            receipt_staged_bytes != encoded_receipt
            or not _exact_json_value(
                _strict_json_mapping(
                    receipt_staged_bytes, "checkpoint receipt",
                ),
                receipt.to_dict(),
            )
        ):
            raise ValueError("checkpoint receipt changed after writing")
        _assert_private_checkpoint_parent(
            parent_descriptor, parent_lexical, parent_identity,
        )
        _publish_held_private_regular(
            parent_descriptor,
            receipt_temporary_name,
            receipt_name,
            receipt_descriptor,
            receipt_inode,
            "checkpoint receipt",
        )
        os.fsync(parent_descriptor)
        _assert_private_checkpoint_parent(
            parent_descriptor, parent_lexical, parent_identity,
        )
        (
            receipt_final_bytes,
            receipt_file_sha256,
            receipt_identity,
        ) = _snapshot_held_private_regular(
            parent_descriptor,
            receipt_name,
            receipt_descriptor,
            receipt_inode,
            "checkpoint receipt",
        )
        if receipt_final_bytes != encoded_receipt:
            raise ValueError("published checkpoint receipt changed")
        receipt_path = parent_resolved / receipt_name
        authorization = _SSLCheckpointReceiptAuthorization(
            marker=_AUTHORIZATION_MARKER,
            receipt_reference=weakref.ref(receipt),
            checkpoint_path=checkpoint_path,
            receipt_path=receipt_path,
            receipt_file_sha256=receipt_file_sha256,
            checkpoint_identity=checkpoint_identity,
            receipt_identity=receipt_identity,
            stage_evidence_sha256=stage_evidence.evidence_sha256,
        )
        object.__setattr__(receipt, "_runtime_authorization", authorization)
        _require_authorized_checkpoint_receipt(
            receipt, stage_evidence, checkpoint_path
        )
        _assert_private_checkpoint_parent(
            parent_descriptor, parent_lexical, parent_identity,
        )
        return receipt
    finally:
        descriptors = (
            (parent_descriptor,)
            + (
                (checkpoint_descriptor,)
                if checkpoint_descriptor is not None else ()
            )
            + (
                (receipt_descriptor,)
                if receipt_descriptor is not None else ()
            )
        )
        # No pathname cleanup is attempted.  A failed outer results
        # transaction retains all private residue and blocks retry.
        _close_ssl_descriptors(descriptors)


def save_ssl_checkpoint(
    path: str | Path,
    payload: Mapping[str, object],
    *,
    stage_evidence: SSLStageEvidence,
) -> SSLCheckpointReceipt:
    _require_public_receipt_bound_stage(stage_evidence)
    return _save_ssl_checkpoint_impl(
        path, payload, stage_evidence=stage_evidence,
    )


def _load_ssl_checkpoint_impl(
    path: str | Path,
    *,
    receipt: SSLCheckpointReceipt | None = None,
    stage_evidence: SSLStageEvidence | None = None,
) -> SSLCheckpointPayload:
    if receipt is None or stage_evidence is None:
        raise ValueError(
            "loading requires a trusted checkpoint receipt and live stage evidence"
        )
    checkpoint_bytes = _require_authorized_checkpoint_receipt(
        receipt, stage_evidence, path
    )
    try:
        value = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise ValueError("cannot load a safe SSL checkpoint") from exc
    validate_ssl_checkpoint_payload(value)
    if (
        ssl_checkpoint_fingerprint(value) != receipt.checkpoint_fingerprint
        or value["checkpoint_type"] != receipt.checkpoint_type
        or value["metadata"]["stage_evidence"] != stage_evidence.to_dict()
    ):
        raise ValueError("loaded SSL checkpoint contradicts its trusted receipt")
    payload = SSLCheckpointPayload(value)
    payload._runtime_authorization = _SSLCheckpointPayloadAuthorization(
        marker=_AUTHORIZATION_MARKER,
        payload_reference=weakref.ref(payload),
        checkpoint_fingerprint=receipt.checkpoint_fingerprint,
        stage_evidence_sha256=stage_evidence.evidence_sha256,
        checkpoint_path=Path(path).resolve(strict=True),
        receipt_reference=weakref.ref(receipt),
    )
    _require_authorized_checkpoint_payload(payload, stage_evidence)
    return payload


def load_ssl_checkpoint(
    path: str | Path,
    *,
    receipt: SSLCheckpointReceipt | None = None,
    stage_evidence: SSLStageEvidence | None = None,
) -> SSLCheckpointPayload:
    if stage_evidence is None:
        raise ValueError("loading requires live receipt-bound stage evidence")
    _require_public_receipt_bound_stage(stage_evidence)
    return _load_ssl_checkpoint_impl(
        path, receipt=receipt, stage_evidence=stage_evidence,
    )


def _transfer_ssl_weights_impl(
    payload: Mapping[str, object],
    downstream: DynamicLandmarkModel,
    cross_detector_agreement: bool = False,
    *,
    stage_evidence: SSLStageEvidence | None = None,
    require_persisted: bool = False,
) -> tuple[str, ...]:
    """Apply only the checkpoint-type-specific downstream transfer allowlist.

    The current semantic adapter explicitly records that OpenFace-68 and
    MediaPipe use different numeric anchors.  Therefore a bare caller flag is
    never accepted as proof that their landmark projection weights agree.
    """
    if stage_evidence is None:
        raise ValueError("checkpoint transfer requires live stage-artifact authorization")
    _require_authorized_checkpoint_payload(
        payload, stage_evidence, require_persisted=require_persisted,
    )
    if not isinstance(downstream, DynamicLandmarkModel):
        raise ValueError("downstream model must use the frozen Task4 architecture")
    if not isinstance(cross_detector_agreement, bool):
        raise ValueError("cross_detector_agreement must be boolean")
    if cross_detector_agreement:
        raise ValueError(
            "cross-detector projection transfer requires independently "
            "registered agreement evidence; no such evidence is frozen"
        )
    source_state = payload["model_state"]
    downstream_state = downstream.state_dict()
    shared_prefixes = ("temporal.", "attention_score.", "pool_projection.")
    checkpoint_type = payload["checkpoint_type"]
    allowed = [name for name in downstream_state if name.startswith(shared_prefixes)]
    if checkpoint_type == CHECKPOINT_RAVDESS_MAYO:
        allowed.extend(name for name in downstream_state if name.startswith((
            "proj_bs_x.", "proj_bs_dx.", "proj_lm_x.", "proj_lm_dx.",
        )))
    transferred: set[str] = set()
    ravdess_partial_input_weights = {
        "temporal.weight_ih_l0",
        "temporal.weight_ih_l0_reverse",
    }
    for name in sorted(set(allowed)):
        source = source_state[name]
        if source.shape != downstream_state[name].shape:
            raise ValueError(f"transfer tensor {name!r} has incompatible shape")
        converted = source.to(
            device=downstream_state[name].device,
            dtype=downstream_state[name].dtype,
        ).clone()
        if checkpoint_type == CHECKPOINT_RAVDESS_ONLY and name in ravdess_partial_input_weights:
            if converted.ndim != 2 or converted.shape[1] != 64:
                raise ValueError("RAVDESS GRU input transfer requires the exact 64-d contract")
            retained = downstream_state[name].clone()
            retained[:, 32:64] = converted[:, 32:64]
            downstream_state[name] = retained
            transferred.add(f"{name}[:,32:64]")
        else:
            downstream_state[name] = converted
            transferred.add(name)
    downstream.load_state_dict(downstream_state, strict=True)
    return tuple(sorted(transferred))


def transfer_ssl_weights(
    payload: Mapping[str, object],
    downstream: DynamicLandmarkModel,
    cross_detector_agreement: bool = False,
    *,
    stage_evidence: SSLStageEvidence | None = None,
) -> tuple[str, ...]:
    if stage_evidence is None:
        raise ValueError("checkpoint transfer requires live stage evidence")
    _require_public_receipt_bound_stage(stage_evidence)
    return _transfer_ssl_weights_impl(
        payload,
        downstream,
        cross_detector_agreement,
        stage_evidence=stage_evidence,
        require_persisted=True,
    )


def require_frozen_pretraining_inputs(
    _ravdess_manifest: str | Path | None = None,
    _mayo_manifest: str | Path | None = None,
    _config: str | Path | None = None,
) -> None:
    raise PretrainingLockedError(
        "real pretraining is disabled until Task 5 and Task 6 data manifests "
        "and the frozen pretraining configuration are jointly validated"
    )


__all__ = [
    "CHECKPOINT_RAVDESS_ONLY", "CHECKPOINT_RAVDESS_MAYO",
    "DynamicLandmarkSSLModel", "PretrainingLockedError", "SSLGroupSplit",
    "ResampledTrajectory", "SourceScaler", "SSLStageEvidence",
    "SSLCheckpointPayload", "SSLCheckpointReceipt", "SSLTrainingReceipt",
    "SSLTrainingResult",
    "make_contiguous_span_mask", "ssl_gap_safe_per_second_differences",
    "masked_smooth_l1", "deterministic_group_split", "resample_trajectory_30hz",
    "fit_source_scaler", "reconstruction_report", "build_ssl_stage_evidence",
    "authorize_frozen_ssl_stage",
    "initialize_mayo_ssl_model", "train_ssl_stage",
    "build_ssl_checkpoint_payload", "ssl_checkpoint_fingerprint",
    "validate_ssl_checkpoint_payload", "save_ssl_checkpoint", "load_ssl_checkpoint",
    "authorize_ssl_checkpoint_receipt",
    "transfer_ssl_weights", "require_frozen_pretraining_inputs",
]
