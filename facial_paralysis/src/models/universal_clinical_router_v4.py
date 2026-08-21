"""Evidence-routed universal facial-weakness inference primitives.

The router deliberately consumes task and modality evidence, never a dataset or
institution name.  Each route retains a clinically appropriate frozen head.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


SCRIPTED_COMMON_TASKS = ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")
UPPER_PROMPT_TASKS = ("BROW_RAISE", "EYE_GENTLE", "EYE_FORCEFUL")
TIMING_AUTHORITIES = ("none", "recording_task_label", "external_prompt")


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def evidence_profile(
    task_names: tuple[str, ...],
    *,
    has_au: bool,
    has_marlin: bool,
    timing_authority: str,
) -> str:
    """Return one frozen head profile from authenticated evidence availability."""
    if (
        type(task_names) is not tuple
        or not task_names
        or any(type(task) is not str or not task for task in task_names)
        or len(set(task_names)) != len(task_names)
        or timing_authority not in TIMING_AUTHORITIES
    ):
        raise ValueError("clinical route evidence is not canonical")
    has_au = _exact_bool(has_au, "has_au")
    has_marlin = _exact_bool(has_marlin, "has_marlin")
    if task_names == ("FREE_RECORDING",):
        if has_au or has_marlin or timing_authority != "none":
            raise ValueError("free-recording route received conflicting modalities")
        return "free_asymmetry"
    if task_names == SCRIPTED_COMMON_TASKS:
        if not (has_au and has_marlin) or timing_authority != "recording_task_label":
            raise ValueError("scripted route requires complete labelled multimodal evidence")
        return "scripted_multimechanism"
    if set(UPPER_PROMPT_TASKS).issubset(task_names):
        if timing_authority != "external_prompt":
            raise ValueError("upper-face route requires an exogenous prompt clock")
        return "cue_aligned_upper"
    raise ValueError("no frozen clinical profile matches the supplied evidence")


def _immutable_float64(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(
        contiguous.shape
    )


def median_low_confidence_gate(
    clinical_probability: np.ndarray,
    marlin_probabilities: np.ndarray,
    *,
    radius: float = 0.30,
) -> np.ndarray:
    """Keep confident clinical scores; replace uncertain ones by MARLIN median."""
    clinical = np.asarray(clinical_probability)
    marlin = np.asarray(marlin_probabilities)
    if (
        clinical.ndim != 1
        or clinical.dtype != np.dtype(np.float64)
        or marlin.ndim != 2
        or marlin.dtype != np.dtype(np.float64)
        or marlin.shape[1] != clinical.shape[0]
        or marlin.shape[0] < 3
        or not np.isfinite(clinical).all()
        or not np.isfinite(marlin).all()
        or np.any((clinical < 0.0) | (clinical > 1.0))
        or np.any((marlin < 0.0) | (marlin > 1.0))
        or isinstance(radius, bool)
        or not math.isfinite(float(radius))
        or not 0.0 < float(radius) < 0.5
    ):
        raise ValueError("low-confidence gate inputs violate the frozen schema")
    result = clinical.copy()
    uncertain = np.abs(clinical - 0.5) <= float(radius)
    result[uncertain] = np.median(marlin[:, uncertain], axis=0)
    return _immutable_float64(result)


def linear_head_probability(
    values: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """Apply one serialized StandardScaler plus binary Logistic head."""
    values = np.asarray(values)
    mean = np.asarray(mean)
    scale = np.asarray(scale)
    coefficient = np.asarray(coefficient)
    if (
        values.ndim != 2
        or values.dtype != np.dtype(np.float64)
        or mean.shape != values.shape[1:]
        or scale.shape != values.shape[1:]
        or coefficient.shape != values.shape[1:]
        or mean.dtype != np.dtype(np.float64)
        or scale.dtype != np.dtype(np.float64)
        or coefficient.dtype != np.dtype(np.float64)
        or not np.isfinite(values).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(coefficient).all()
        or np.any(scale <= 0.0)
        or isinstance(intercept, bool)
        or not isinstance(intercept, (int, float))
        or not math.isfinite(float(intercept))
    ):
        raise ValueError("serialized linear head violates the frozen schema")
    logit = ((values - mean) / scale) @ coefficient + float(intercept)
    probability = np.empty_like(logit)
    positive = logit >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_value = np.exp(logit[~positive])
    probability[~positive] = exp_value / (1.0 + exp_value)
    return _immutable_float64(probability)


_SERIALIZED_HEAD_KEYS = {
    "input_dimension", "selected_indices", "scaler_center", "scaler_scale",
    "coefficient", "intercept",
}


def serialized_head_probability(
    values: np.ndarray, document: Mapping[str, object],
) -> np.ndarray:
    """Apply one closed-schema serialized feature-selection and linear head."""
    if not isinstance(document, Mapping) or set(document) != _SERIALIZED_HEAD_KEYS:
        raise ValueError("serialized head document has an open or incomplete schema")
    if (
        type(document["input_dimension"]) is not int
        or document["input_dimension"] <= 0
        or type(document["selected_indices"]) is not list
        or not document["selected_indices"]
        or any(type(index) is not int for index in document["selected_indices"])
    ):
        raise ValueError("serialized head dimensions are not canonical")
    selected = np.asarray(document["selected_indices"], dtype=np.int64)
    if (
        len(np.unique(selected)) != len(selected)
        or selected.min() < 0
        or selected.max() >= document["input_dimension"]
    ):
        raise ValueError("serialized head selection is invalid")
    values = np.asarray(values)
    if (
        values.ndim != 2
        or values.dtype != np.dtype(np.float64)
        or values.shape[1] != document["input_dimension"]
        or not np.isfinite(values).all()
    ):
        raise ValueError("serialized head input differs from its frozen dimension")
    arrays = []
    for name in ("scaler_center", "scaler_scale", "coefficient"):
        value = document[name]
        if type(value) is not list or len(value) != len(selected):
            raise ValueError("serialized head parameter dimension differs")
        array = np.asarray(value, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("serialized head contains non-finite parameters")
        arrays.append(array)
    return linear_head_probability(
        values[:, selected], mean=arrays[0], scale=arrays[1],
        coefficient=arrays[2], intercept=document["intercept"],
    )


def scripted_multimechanism_probability(
    *,
    landmark_original: np.ndarray,
    landmark_mirrored: np.ndarray,
    au_values: np.ndarray,
    marlin_representations: Mapping[str, np.ndarray],
    artifact: Mapping[str, object],
) -> np.ndarray:
    """Run the fixed clinical plus 18-head MARLIN median-gate system."""
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "clinical_heads", "marlin_heads", "gate"
    }:
        raise ValueError("scripted artifact schema differs")
    clinical_heads = artifact["clinical_heads"]
    if not isinstance(clinical_heads, Mapping) or set(clinical_heads) != {
        "post_stroke_asymmetry_mean110", "als_oromotor_robust_pool1600",
        "fusion",
    } or clinical_heads["fusion"] != "maximum_probability":
        raise ValueError("scripted clinical head registry differs")
    landmark = 0.5 * (
        serialized_head_probability(
            landmark_original, clinical_heads["post_stroke_asymmetry_mean110"]
        )
        + serialized_head_probability(
            landmark_mirrored, clinical_heads["post_stroke_asymmetry_mean110"]
        )
    )
    au = serialized_head_probability(
        au_values, clinical_heads["als_oromotor_robust_pool1600"]
    )
    clinical = np.maximum(landmark, au)
    heads = artifact["marlin_heads"]
    if type(heads) is not list or len(heads) != 18:
        raise ValueError("scripted route requires the fixed 18 MARLIN heads")
    marlin_probabilities = []
    seen = set()
    for candidate in heads:
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "representation", "top_k", "c", "targets"
        }:
            raise ValueError("MARLIN candidate schema differs")
        key = (candidate["representation"], candidate["top_k"], candidate["c"])
        if key in seen or candidate["representation"] not in marlin_representations:
            raise ValueError("MARLIN candidate registry is ambiguous")
        seen.add(key)
        targets = candidate["targets"]
        if type(targets) is not list or len(targets) != 2:
            raise ValueError("MARLIN phenotype head registry differs")
        target_probabilities = {}
        for target in targets:
            if not isinstance(target, Mapping) or set(target) != {"phenotype", "head"}:
                raise ValueError("MARLIN phenotype document differs")
            phenotype = target["phenotype"]
            if phenotype in target_probabilities:
                raise ValueError("MARLIN phenotype head is duplicated")
            target_probabilities[phenotype] = serialized_head_probability(
                marlin_representations[candidate["representation"]], target["head"]
            )
        if set(target_probabilities) != {"als", "post_stroke"}:
            raise ValueError("MARLIN phenotype coverage differs")
        marlin_probabilities.append(np.maximum(
            target_probabilities["als"], target_probabilities["post_stroke"]
        ))
    gate = artifact["gate"]
    if not isinstance(gate, Mapping) or set(gate) != {
        "clinical_center", "radius", "inside", "outside"
    } or (
        gate["clinical_center"] != 0.5
        or gate["inside"] != "median_of_18_marlin_phenotype_max_probabilities"
        or gate["outside"] != "clinical_probability"
    ):
        raise ValueError("scripted gate contract differs")
    return median_low_confidence_gate(
        clinical, np.stack(marlin_probabilities), radius=gate["radius"]
    )


def cue_aligned_upper_probability(
    original_by_head: Mapping[str, np.ndarray],
    mirrored_by_head: Mapping[str, np.ndarray],
    artifact: Mapping[str, object],
) -> np.ndarray:
    """Run the fixed two-head MEEI cue-aligned mirror ensemble."""
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "heads", "probability_weights", "decision_threshold"
    }:
        raise ValueError("cue-aligned artifact schema differs")
    heads = artifact["heads"]
    weights = artifact["probability_weights"]
    if (
        type(heads) is not list or len(heads) != 2
        or type(weights) is not list or len(weights) != 2
        or any(isinstance(value, bool) or not math.isfinite(float(value))
               or float(value) <= 0.0 for value in weights)
        or not math.isclose(sum(float(value) for value in weights), 1.0)
        or not isinstance(artifact["decision_threshold"], (int, float))
    ):
        raise ValueError("cue-aligned ensemble contract differs")
    names = tuple(head.get("name") for head in heads if isinstance(head, Mapping))
    if (
        len(names) != 2 or len(set(names)) != 2
        or set(original_by_head) != set(names)
        or set(mirrored_by_head) != set(names)
    ):
        raise ValueError("cue-aligned feature registry differs")
    probabilities = []
    for head in heads:
        if set(head) != {"name", "actions", "family", "head"}:
            raise ValueError("cue-aligned head schema differs")
        name = head["name"]
        probabilities.append(0.5 * (
            serialized_head_probability(original_by_head[name], head["head"])
            + serialized_head_probability(mirrored_by_head[name], head["head"])
        ))
    result = sum(
        float(weight) * probability
        for weight, probability in zip(weights, probabilities)
    )
    return _immutable_float64(result)


__all__ = (
    "SCRIPTED_COMMON_TASKS",
    "TIMING_AUTHORITIES",
    "UPPER_PROMPT_TASKS",
    "evidence_profile",
    "cue_aligned_upper_probability",
    "linear_head_probability",
    "median_low_confidence_gate",
    "scripted_multimechanism_probability",
    "serialized_head_probability",
)
