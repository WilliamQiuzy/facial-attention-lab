"""Exploratory participant-level transfer to NeuroFace SLP severity ratings."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


SCHEMA_VERSION = "neuroface_slp_transfer_v1"
PRIMARY_TASKS = ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD")
DOMAINS = ("symmetry", "rom", "speed", "variability", "fatigue", "total")
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_REPEATS = 5000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _holm(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        candidate = min(1.0, float(p_values[key]) * (total - index))
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _within_diagnosis_ranks(values: np.ndarray, cohorts: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape, dtype=np.float64)
    for cohort in ("als", "post_stroke"):
        indices = np.flatnonzero(cohorts == cohort)
        if indices.size < 2:
            raise ValueError("diagnosis-adjusted association requires both affected cohorts")
        output[indices] = rankdata(values[indices], method="average") / indices.size
    return output


def _rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = spearmanr(x, y)
    value, p_value = float(result.statistic), float(result.pvalue)
    if not np.isfinite(value) or not np.isfinite(p_value):
        raise ValueError("SLP association is undefined")
    return value, p_value


def _association_family(
    scores: np.ndarray,
    domains: Mapping[str, np.ndarray],
    cohorts: np.ndarray,
    indices: np.ndarray,
    *,
    adjusted: bool,
    repeats: int,
) -> dict[str, object]:
    family_cohorts = cohorts[indices]
    if adjusted:
        x = _within_diagnosis_ranks(scores[indices], family_cohorts)
    else:
        x = scores[indices]
    points: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for domain in DOMAINS:
        y = domains[domain][indices]
        if adjusted:
            y = _within_diagnosis_ranks(y, family_cohorts)
        points[domain], p_values[domain] = _rho(x, y)
    corrected = _holm(p_values)
    strata = [
        np.flatnonzero(family_cohorts == cohort)
        for cohort in sorted(set(family_cohorts.tolist()))
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: dict[str, list[float]] = {domain: [] for domain in DOMAINS}
    invalid = 0
    for _ in range(repeats):
        sampled_local = np.concatenate([
            rng.choice(stratum, size=stratum.size, replace=True) for stratum in strata
        ])
        sampled_scores = scores[indices][sampled_local]
        sampled_cohorts = family_cohorts[sampled_local]
        try:
            sampled_x = (
                _within_diagnosis_ranks(sampled_scores, sampled_cohorts)
                if adjusted else sampled_scores
            )
            sampled_values: dict[str, float] = {}
            for domain in DOMAINS:
                sampled_y = domains[domain][indices][sampled_local]
                if adjusted:
                    sampled_y = _within_diagnosis_ranks(sampled_y, sampled_cohorts)
                sampled_values[domain] = _rho(sampled_x, sampled_y)[0]
        except ValueError:
            invalid += 1
            continue
        for domain, value in sampled_values.items():
            draws[domain].append(value)
    valid = repeats - invalid
    if valid < max(1, int(np.ceil(repeats * 0.90))):
        raise ValueError("too many SLP bootstrap draws are undefined")
    return {
        domain: {
            "n": int(indices.size),
            "rho": points[domain],
            "ci95_low": float(np.quantile(draws[domain], 0.025)),
            "ci95_high": float(np.quantile(draws[domain], 0.975)),
            "p_raw": p_values[domain],
            "p_holm": corrected[domain],
            "bootstrap": {"requested": repeats, "valid": valid, "invalid": invalid},
        }
        for domain in DOMAINS
    }


def build_slp_transfer_report(
    video_rows: Sequence[Mapping[str, object]],
    *,
    external_report_sha256: str,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, object]:
    if _SHA256.fullmatch(external_report_sha256) is None:
        raise ValueError("external report SHA-256 is invalid")
    if isinstance(bootstrap_repeats, bool) or bootstrap_repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    by_participant: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in video_rows:
        participant = row.get("participant_id")
        if not isinstance(participant, str) or not participant.startswith("grp_"):
            raise ValueError("SLP input requires opaque participant IDs")
        by_participant[participant].append(row)
    labels: list[int] = []
    cohorts: list[str] = []
    scores: list[float] = []
    domain_values: dict[str, list[float]] = {domain: [] for domain in DOMAINS}
    for participant in sorted(by_participant):
        primary = [row for row in by_participant[participant] if row.get("task") in PRIMARY_TASKS]
        if sorted(str(row.get("task")) for row in primary) != sorted(PRIMARY_TASKS):
            raise ValueError("SLP transfer requires exactly the three primary tasks")
        row_labels = {row.get("label") for row in primary}
        row_cohorts = {row.get("cohort") for row in primary}
        if len(row_labels) != 1 or row_labels - {0, 1} or len(row_cohorts) != 1:
            raise ValueError("participant crosses label or cohort")
        task_scores: list[float] = []
        task_domains: dict[str, list[float]] = {domain: [] for domain in DOMAINS}
        for row in primary:
            score = row.get("probability")
            slp = row.get("slp_scores")
            if (
                isinstance(score, bool) or not isinstance(score, (int, float))
                or not np.isfinite(score) or not 0.0 <= float(score) <= 1.0
                or not isinstance(slp, Mapping) or set(slp) != set(DOMAINS)
            ):
                raise ValueError("model score or SLP domain schema is invalid")
            values = {domain: float(slp[domain]) for domain in DOMAINS}
            if (
                not all(np.isfinite(value) for value in values.values())
                or not all(1.0 <= values[domain] <= 5.0 for domain in DOMAINS[:-1])
                or not 5.0 <= values["total"] <= 25.0
            ):
                raise ValueError("SLP values are outside the released 1-to-5 scale")
            task_scores.append(float(score))
            for domain in DOMAINS:
                task_domains[domain].append(values[domain])
        labels.append(int(next(iter(row_labels))))
        cohorts.append(str(next(iter(row_cohorts))))
        scores.append(float(np.mean(task_scores)))
        for domain in DOMAINS:
            domain_values[domain].append(float(np.mean(task_domains[domain])))
    labels_array = np.asarray(labels, dtype=np.int64)
    cohorts_array = np.asarray(cohorts, dtype=str)
    scores_array = np.asarray(scores, dtype=np.float64)
    domains_array = {
        domain: np.asarray(values, dtype=np.float64)
        for domain, values in domain_values.items()
    }
    if set(labels_array.tolist()) != {0, 1}:
        raise ValueError("SLP discrimination requires affected and control participants")
    families: dict[str, tuple[np.ndarray, bool]] = {
        "all_participants_spectrum": (np.arange(labels_array.size), False),
        "affected_only_pooled": (np.flatnonzero(labels_array == 1), False),
        "als_only": (np.flatnonzero(cohorts_array == "als"), False),
        "post_stroke_only": (np.flatnonzero(cohorts_array == "post_stroke"), False),
        "affected_diagnosis_adjusted": (np.flatnonzero(labels_array == 1), True),
    }
    associations = {
        name: _association_family(
            scores_array, domains_array, cohorts_array, indices,
            adjusted=adjusted, repeats=bootstrap_repeats,
        )
        for name, (indices, adjusted) in families.items()
        if indices.size >= 4
    }
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "post_external_evaluation_exploratory_clinical_association",
        "dataset": "Toronto_NeuroFace_v1",
        "scale_direction": "higher_is_more_severe_dysfunction",
        "aggregation": "participant_mean_over_nsm_kiss_open_spread",
        "counts": {
            "participants": int(labels_array.size),
            "affected": int(np.sum(labels_array == 1)),
            "unaffected": int(np.sum(labels_array == 0)),
        },
        "clinical_reference_discrimination": {
            "total_auroc": float(roc_auc_score(labels_array, domains_array["total"])),
            "interpretation": "descriptive_check_of_released_slp_ratings_not_model_performance",
        },
        "associations": associations,
        "multiplicity": "holm_within_each_six_domain_family",
        "provenance": {"external_report_sha256": external_report_sha256},
    }
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    if any(token in encoded for token in (
        "grp_", "participant_id", "probability", "/Users/", "\\", ".avi"
    )):
        raise ValueError("SLP report leaks row-level or local information")
    return report


__all__ = ["BOOTSTRAP_REPEATS", "DOMAINS", "build_slp_transfer_report"]
