"""Faithful action-level attribution for the frozen Shared V9 predictor."""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import PurePosixPath
from typing import Mapping
import zipfile

import numpy as np
import torch

from src.deployment.shared_v8_release import validate_request_arrays
from src.deployment.shared_v8_service import MAX_REQUEST_BYTES
from src.deployment.shared_v9_research_release import (
    SharedV9Predictor,
    V9Prediction,
)


ATTRIBUTION_SCHEMA = "shared_v9_action_token_attribution/v1"
ATTRIBUTION_BASELINE = "within_recording_neutral_clinical_zero_dense_response"
ATTRIBUTION_METHOD = "integrated_gradients_shared_action_tokens"
INTEGRATION_STEPS = 32
MAX_COMPLETENESS_ERROR = 0.02
_MIN_ABSOLUTE_CONTRIBUTION = 0.01
_MIN_DIRECTIONAL_CONTRIBUTION = 0.005
_BASE_FIELDS = frozenset({
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
_NEUTRAL_ORIGINAL = "neutral_clinical_original"
_NEUTRAL_MIRRORED = "neutral_clinical_mirrored"
_ATTRIBUTION_FIELDS = _BASE_FIELDS | {_NEUTRAL_ORIGINAL, _NEUTRAL_MIRRORED}


@dataclass(frozen=True)
class V9ActionAttribution:
    action_code: int
    mean_logit_contribution: float
    relative_magnitude: float
    ensemble_sign_agreement: int
    mirror_consistent: bool
    temporal_checks_passed: int
    stable: bool
    direction: str
    strength: str


@dataclass(frozen=True)
class V9AttributionSummary:
    schema_version: str
    method: str
    baseline: str
    integration_steps: int
    max_completeness_error: float
    actions: tuple[V9ActionAttribution, ...]


@dataclass(frozen=True)
class V9ExplainedPrediction:
    prediction: V9Prediction
    attribution: V9AttributionSummary


def _validate_neutral(
    value: np.ndarray,
    *,
    actions: int,
    name: str,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.shape != (actions, 110)
        or value.dtype != np.dtype(np.float32)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} differs from the neutral clinical contract")
    return np.array(value, copy=True)


def encode_attribution_request_npz(
    protocol: str,
    arrays: Mapping[str, np.ndarray],
    neutral_clinical_original: np.ndarray,
    neutral_clinical_mirrored: np.ndarray,
) -> bytes:
    """Encode an exact prediction request plus a within-recording neutral baseline."""
    if protocol != "cue_aligned_action":
        raise ValueError("attribution is available only for cue-aligned actions")
    validated = validate_request_arrays(protocol, arrays)
    actions = validated["clinical_original"].shape[1]
    neutral_original = _validate_neutral(
        neutral_clinical_original,
        actions=actions,
        name=_NEUTRAL_ORIGINAL,
    )
    neutral_mirrored = _validate_neutral(
        neutral_clinical_mirrored,
        actions=actions,
        name=_NEUTRAL_MIRRORED,
    )
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        **{name: np.asarray(arrays[name]) for name in sorted(_BASE_FIELDS)},
        **{
            _NEUTRAL_ORIGINAL: neutral_original,
            _NEUTRAL_MIRRORED: neutral_mirrored,
        },
    )
    payload = buffer.getvalue()
    if not payload or len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("attribution request exceeds the transport bound")
    return payload


def load_attribution_request_npz(
    payload: bytes,
    *,
    protocol: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Decode the exact immutable bytes consumed by the attribution endpoint."""
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > MAX_REQUEST_BYTES
        or protocol != "cue_aligned_action"
    ):
        raise ValueError("attribution request is outside the closed boundary")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            canonical = {name + ".npy" for name in _ATTRIBUTION_FIELDS}
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
                raise ValueError("attribution archive differs from the closed schema")
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            if (
                len(saved.files) != len(set(saved.files))
                or set(saved.files) != _ATTRIBUTION_FIELDS
            ):
                raise ValueError("attribution fields differ from the closed schema")
            loaded = {name: np.array(saved[name], copy=True) for name in saved.files}
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError) as exc:
        raise ValueError("attribution archive is invalid") from exc
    neutral_original = loaded.pop(_NEUTRAL_ORIGINAL)
    neutral_mirrored = loaded.pop(_NEUTRAL_MIRRORED)
    normalized = validate_request_arrays(protocol, loaded)
    actions = normalized["clinical_original"].shape[1]
    return (
        loaded,
        _validate_neutral(
            neutral_original,
            actions=actions,
            name=_NEUTRAL_ORIGINAL,
        ),
        _validate_neutral(
            neutral_mirrored,
            actions=actions,
            name=_NEUTRAL_MIRRORED,
        ),
    )


def _normalized_inputs(
    predictor: SharedV9Predictor,
    arrays: Mapping[str, np.ndarray],
    neutral_original: np.ndarray,
    neutral_mirrored: np.ndarray,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    normalized = validate_request_arrays("cue_aligned_action", arrays)
    actions = normalized["clinical_original"].shape[1]
    first = _validate_neutral(
        neutral_original,
        actions=actions,
        name=_NEUTRAL_ORIGINAL,
    )[None]
    second = _validate_neutral(
        neutral_mirrored,
        actions=actions,
        name=_NEUTRAL_MIRRORED,
    )[None]
    observed_original = (
        normalized["clinical_original"].astype(np.float64)
        - predictor.mean[None, None, :]
    ) / predictor.scale[None, None, :]
    observed_mirrored = (
        normalized["clinical_mirrored"].astype(np.float64)
        - predictor.mean[None, None, :]
    ) / predictor.scale[None, None, :]
    baseline_original = (
        first.astype(np.float64) - predictor.mean[None, None, :]
    ) / predictor.scale[None, None, :]
    baseline_mirrored = (
        second.astype(np.float64) - predictor.mean[None, None, :]
    ) / predictor.scale[None, None, :]
    observed_values = (
        observed_original.astype(np.float32),
        observed_mirrored.astype(np.float32),
        normalized["dense_original"],
        normalized["dense_mirrored"],
        normalized["dense_valid_mask"],
        normalized["dense_available"],
        normalized["dense_timestamps"],
        normalized["action_mask"],
        normalized["action_codes"],
    )
    baseline_values = (
        baseline_original.astype(np.float32),
        baseline_mirrored.astype(np.float32),
        np.zeros_like(normalized["dense_original"]),
        np.zeros_like(normalized["dense_mirrored"]),
        normalized["dense_valid_mask"],
        normalized["dense_available"],
        normalized["dense_timestamps"],
        normalized["action_mask"],
        normalized["action_codes"],
    )

    def tensors(values):
        return tuple(
            torch.from_numpy(np.array(value, copy=True)).to(predictor.device)
            for value in values
        )

    return tensors(observed_values), tensors(baseline_values)


def _integrated_action_contributions(
    model,
    observed_tokens: torch.Tensor,
    baseline_tokens: torch.Tensor,
    action_mask: torch.Tensor,
    task_code: torch.Tensor,
) -> tuple[np.ndarray, float]:
    observed = observed_tokens.detach()
    baseline = baseline_tokens.detach()
    delta = observed - baseline
    alphas = (
        (torch.arange(INTEGRATION_STEPS, device=observed.device, dtype=observed.dtype) + 0.5)
        / INTEGRATION_STEPS
    ).reshape(INTEGRATION_STEPS, 1, 1)
    path = (baseline + alphas * delta).requires_grad_(True)
    masks = action_mask.expand(INTEGRATION_STEPS, -1)
    tasks = task_code.expand(INTEGRATION_STEPS)
    logits = model.routed_logits(path, masks, tasks)
    gradient = torch.autograd.grad(logits.sum(), path, create_graph=False)[0]
    action_values = (delta[0] * gradient.mean(dim=0)).sum(dim=-1)
    with torch.no_grad():
        observed_logit = model.routed_logits(observed, action_mask, task_code)[0]
        baseline_logit = model.routed_logits(baseline, action_mask, task_code)[0]
        completeness = abs(float(
            (action_values.sum() - (observed_logit - baseline_logit)).detach().cpu()
        ))
    return action_values.detach().cpu().numpy().astype(np.float64), completeness


def _shift_dense(inputs: tuple[torch.Tensor, ...], offset: int) -> tuple[torch.Tensor, ...]:
    if offset not in {-1, 1}:
        raise ValueError("temporal shift must be one checkpoint")
    changed = list(inputs)
    for position in (2, 3):
        dense = inputs[position]
        if offset == -1:
            changed[position] = torch.cat((dense[:, :, :1], dense[:, :, :-1]), dim=2)
        else:
            changed[position] = torch.cat((dense[:, :, 1:], dense[:, :, -1:]), dim=2)
    return tuple(changed)


def _swap_views(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    changed = list(inputs)
    changed[0], changed[1] = inputs[1], inputs[0]
    changed[2], changed[3] = inputs[3], inputs[2]
    return tuple(changed)


def _sign(value: float) -> int:
    if value > 1e-6:
        return 1
    if value < -1e-6:
        return -1
    return 0


def explain_prediction(
    predictor: SharedV9Predictor,
    protocol: str,
    arrays: Mapping[str, np.ndarray],
    neutral_clinical_original: np.ndarray,
    neutral_clinical_mirrored: np.ndarray,
) -> V9ExplainedPrediction:
    """Explain the exact score without changing the frozen prediction path."""
    if type(predictor) is not SharedV9Predictor or protocol != "cue_aligned_action":
        raise ValueError("attribution requires the frozen cue-aligned predictor")
    prediction = predictor.predict(protocol, arrays)
    observed, baseline = _normalized_inputs(
        predictor,
        arrays,
        neutral_clinical_original,
        neutral_clinical_mirrored,
    )
    task_code = torch.tensor([2], dtype=torch.long, device=predictor.device)
    original_rows = []
    mirror_rows = []
    temporal_rows = [[], []]
    completeness_errors = []
    for model in predictor.models:
        with torch.no_grad():
            observed_tokens = model.shared_action_tokens(*observed)
            baseline_tokens = model.shared_action_tokens(*baseline)
            mirror_tokens = model.shared_action_tokens(*_swap_views(observed))
            mirror_baseline = model.shared_action_tokens(*_swap_views(baseline))
            temporal_tokens = tuple(
                model.shared_action_tokens(*_shift_dense(observed, offset))
                for offset in (-1, 1)
            )
        values, error = _integrated_action_contributions(
            model,
            observed_tokens,
            baseline_tokens,
            observed[-2],
            task_code,
        )
        original_rows.append(values)
        completeness_errors.append(error)
        mirrored, _ = _integrated_action_contributions(
            model,
            mirror_tokens,
            mirror_baseline,
            observed[-2],
            task_code,
        )
        mirror_rows.append(mirrored)
        for index, tokens in enumerate(temporal_tokens):
            shifted, _ = _integrated_action_contributions(
                model,
                tokens,
                baseline_tokens,
                observed[-2],
                task_code,
            )
            temporal_rows[index].append(shifted)
    original_values = np.stack(original_rows)
    mirror_values = np.stack(mirror_rows)
    temporal_values = tuple(np.stack(rows) for rows in temporal_rows)
    mean_values = original_values.mean(axis=0)
    max_magnitude = float(np.max(np.abs(mean_values))) if mean_values.size else 0.0
    maximum_error = float(max(completeness_errors, default=math.inf))
    action_codes = np.asarray(arrays["action_codes"], dtype=np.int64)
    actions = []
    for index, raw_code in enumerate(action_codes):
        mean = float(mean_values[index])
        direction = _sign(mean)
        agreement = sum(
            _sign(float(value)) == direction and direction != 0
            for value in original_values[:, index]
        )
        mirror_consistent = bool(np.allclose(
            original_values[:, index],
            mirror_values[:, index],
            rtol=1e-4,
            atol=1e-5,
        ))
        temporal_passed = sum(
            _sign(float(values[:, index].mean())) == direction
            and abs(float(values[:, index].mean())) >= _MIN_DIRECTIONAL_CONTRIBUTION
            and direction != 0
            for values in temporal_values
        )
        relative = 0.0 if max_magnitude <= 0.0 else abs(mean) / max_magnitude
        significant = (
            abs(mean) >= _MIN_ABSOLUTE_CONTRIBUTION
            and relative >= 0.05
        )
        stable = bool(
            significant
            and agreement == 3
            and mirror_consistent
            and temporal_passed == 2
            and maximum_error <= MAX_COMPLETENESS_ERROR
        )
        if stable:
            reported_direction = "toward_class_1" if direction > 0 else "toward_class_0"
            strength = "strong" if relative >= 0.67 else (
                "moderate" if relative >= 0.33 else "smaller"
            )
        else:
            reported_direction = "not_reported"
            strength = "not_reported"
        actions.append(V9ActionAttribution(
            action_code=int(raw_code),
            mean_logit_contribution=mean,
            relative_magnitude=float(relative),
            ensemble_sign_agreement=int(agreement),
            mirror_consistent=mirror_consistent,
            temporal_checks_passed=int(temporal_passed),
            stable=stable,
            direction=reported_direction,
            strength=strength,
        ))
    return V9ExplainedPrediction(
        prediction=prediction,
        attribution=V9AttributionSummary(
            schema_version=ATTRIBUTION_SCHEMA,
            method=ATTRIBUTION_METHOD,
            baseline=ATTRIBUTION_BASELINE,
            integration_steps=INTEGRATION_STEPS,
            max_completeness_error=maximum_error,
            actions=tuple(actions),
        ),
    )


__all__ = [
    "ATTRIBUTION_BASELINE",
    "ATTRIBUTION_METHOD",
    "ATTRIBUTION_SCHEMA",
    "INTEGRATION_STEPS",
    "V9ActionAttribution",
    "V9AttributionSummary",
    "V9ExplainedPrediction",
    "encode_attribution_request_npz",
    "explain_prediction",
    "load_attribution_request_npz",
]
