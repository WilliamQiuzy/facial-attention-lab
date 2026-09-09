from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter

from .cohort_metrics import (
    independent_equivalence,
    repeated_cross_validated_auc,
    standardized_mean_difference,
)
from .metrics import histogram_intersection


class CohortAnalysisError(ValueError):
    """Raised when an independent-cohort comparison lacks required evidence."""


@dataclass(frozen=True)
class IndependentCohortResult:
    tables: dict[str, pd.DataFrame]
    maps: dict[str, np.ndarray]
    metadata: dict[str, object]


PARTICIPANT_COLUMNS = {
    "participant_id",
    "synthetic",
    "device",
    "recruitment_source",
    "cohort_site",
    "age_years",
    "wears_glasses",
    "prior_eye_tracking",
    "display_width_cm",
    "low_light",
    "accuracy_deg",
    "rms_precision_deg",
    "data_loss",
    "valid_trial_share",
    "sampling_rate_hz",
}
FIXATION_COLUMNS = {
    "participant_id",
    "synthetic",
    "device",
    "stimulus_id",
    "stimulus_version",
    "task_version",
    "coordinate_transform_version",
    "fixation_index",
    "x_norm",
    "y_norm",
    "duration_ms",
}
AOI_COLUMNS = {
    "stimulus_id",
    "stimulus_version",
    "aoi_version",
    "aoi_name",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
}
QUALITY_ENDPOINTS = {
    "accuracy_deg": ("Calibration accuracy error", "degrees", "lower_is_better"),
    "rms_precision_deg": ("RMS precision", "degrees", "lower_is_better"),
    "data_loss": ("Data loss", "proportion", "lower_is_better"),
    "valid_trial_share": ("Valid-trial share", "proportion", "higher_is_better"),
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise CohortAnalysisError(f"{name} is missing columns: {sorted(missing)}")


def _validate_inputs(
    participants: pd.DataFrame,
    fixations: pd.DataFrame,
    aois: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(participants, PARTICIPANT_COLUMNS, "participants")
    _require_columns(fixations, FIXATION_COLUMNS, "fixations")
    _require_columns(aois, AOI_COLUMNS, "aois")
    participant_copy = participants.copy()
    fixation_copy = fixations.copy()
    aoi_copy = aois.copy()

    devices = set(participant_copy["device"])
    if devices != {"webcam", "professional"}:
        raise CohortAnalysisError("participants must contain webcam and professional cohorts")
    if participant_copy["participant_id"].duplicated().any():
        raise CohortAnalysisError("participant_id must identify exactly one independent participant")
    expected = int(config["participants_per_cohort"])
    counts = participant_copy.groupby("device")["participant_id"].nunique()
    if not (counts == expected).all():
        raise CohortAnalysisError(
            f"config expects {expected} independent participants per cohort; observed {counts.to_dict()}"
        )
    if not fixation_copy["participant_id"].isin(participant_copy["participant_id"]).all():
        raise CohortAnalysisError("fixations contain participant IDs absent from participant table")
    device_lookup = participant_copy.set_index("participant_id")["device"]
    expected_device = fixation_copy["participant_id"].map(device_lookup)
    if not expected_device.eq(fixation_copy["device"]).all():
        raise CohortAnalysisError("a participant cannot appear in both device cohorts")
    for coordinate in ("x_norm", "y_norm"):
        values = pd.to_numeric(fixation_copy[coordinate], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise CohortAnalysisError(f"{coordinate} must be finite and within [0, 1]")
    if (pd.to_numeric(fixation_copy["duration_ms"], errors="coerce") <= 0).any():
        raise CohortAnalysisError("fixation duration must be positive")

    required_stimuli = set(config["stimulus_ids"])
    if set(fixation_copy["stimulus_id"]) != required_stimuli:
        raise CohortAnalysisError("both cohorts must use the exact preregistered stimulus set")
    coverage = fixation_copy.groupby(["device", "stimulus_id"])["participant_id"].nunique()
    if not (coverage == expected).all():
        raise CohortAnalysisError("every participant must contribute every common stimulus in the mock demo")
    harmonization = ["stimulus_version", "task_version", "coordinate_transform_version"]
    for column in harmonization:
        if (fixation_copy.groupby("stimulus_id")[column].nunique() != 1).any():
            raise CohortAnalysisError(f"cohorts must share one {column} per stimulus")
    return participant_copy, fixation_copy, aoi_copy


def _protocol_gates(config: Mapping[str, object]) -> pd.DataFrame:
    requirements = config["design_requirements"]
    labels = {
        "same_stimulus_versions": "Same versioned face stimuli",
        "same_task_instructions": "Same viewing instructions",
        "same_exposure_duration": "Same stimulus exposure duration",
        "same_coordinate_transform": "Same stimulus coordinate system",
        "same_exclusion_rules": "Same exclusion and QC rules",
    }
    return pd.DataFrame(
        [
            {
                "requirement": key,
                "plain_language": labels[key],
                "status": "pass" if bool(requirements.get(key, False)) else "fail",
                "why_it_matters": "Without this, cohort differences cannot be attributed to measurement workflow.",
            }
            for key in labels
        ]
    )


def _cohort_characteristics(participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = {
        "age_years": "mean_sd",
        "wears_glasses": "proportion",
        "prior_eye_tracking": "proportion",
        "display_width_cm": "mean_sd",
        "low_light": "proportion",
        "accuracy_deg": "mean_sd",
        "rms_precision_deg": "mean_sd",
        "data_loss": "mean_sd",
        "valid_trial_share": "mean_sd",
        "sampling_rate_hz": "mean_sd",
    }
    for variable, summary_type in specifications.items():
        for device, group in participants.groupby("device", sort=True):
            values = group[variable].astype(float)
            rows.append(
                {
                    "variable": variable,
                    "device": device,
                    "n": len(values),
                    "mean_or_proportion": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if summary_type == "mean_sd" else np.nan,
                    "summary_type": summary_type,
                }
            )
    return pd.DataFrame(rows)


def _covariate_balance(participants: pd.DataFrame) -> pd.DataFrame:
    covariates = {
        "age_years": "Age",
        "wears_glasses": "Wears glasses",
        "prior_eye_tracking": "Prior eye-tracking experience",
        "display_width_cm": "Display width",
        "low_light": "Low-light environment",
    }
    rows = []
    webcam = participants[participants["device"].eq("webcam")]
    professional = participants[participants["device"].eq("professional")]
    for variable, label in covariates.items():
        first = webcam[variable].astype(float)
        second = professional[variable].astype(float)
        smd = standardized_mean_difference(first, second)
        rows.append(
            {
                "covariate": variable,
                "label": label,
                "webcam_mean_or_proportion": first.mean(),
                "professional_mean_or_proportion": second.mean(),
                "standardized_mean_difference": smd,
                "absolute_smd": abs(smd),
                "balance_flag": "balanced" if abs(smd) < 0.10 else "review",
                "role": (
                    "participant_characteristic"
                    if variable in {"age_years", "wears_glasses", "prior_eye_tracking"}
                    else "acquisition_context"
                ),
            }
        )
    return pd.DataFrame(rows)


def _quality_comparison(
    participants: pd.DataFrame, margins: Mapping[str, float]
) -> pd.DataFrame:
    rows = []
    for endpoint, (label, unit, direction) in QUALITY_ENDPOINTS.items():
        if endpoint not in margins:
            raise CohortAnalysisError(f"missing illustrative margin for {endpoint}")
        webcam = participants.loc[participants["device"].eq("webcam"), endpoint]
        professional = participants.loc[participants["device"].eq("professional"), endpoint]
        result = independent_equivalence(webcam, professional, margin=float(margins[endpoint]))
        rows.append(
            {
                "endpoint": endpoint,
                "label": label,
                "unit": unit,
                "direction": direction,
                "webcam_mean": webcam.mean(),
                "professional_mean": professional.mean(),
                "mean_difference_webcam_minus_professional": result.mean_difference,
                "ci90_lower": result.lower,
                "ci90_upper": result.upper,
                "equivalence_margin": result.margin,
                "decision": result.outcome,
                "hedges_g": result.hedges_g,
                "n_webcam": result.n_webcam,
                "n_professional": result.n_professional,
                "margin_source": "illustrative_mock_only",
                "why_important": {
                    "accuracy_deg": "Checks closeness to known calibration targets.",
                    "rms_precision_deg": "Checks point-to-point stability, separate from accuracy.",
                    "data_loss": "Checks how much expected gaze signal is missing.",
                    "valid_trial_share": "Checks whether complete participants or trials survive QC.",
                }[endpoint],
            }
        )
    return pd.DataFrame(rows)


def _participant_aoi_profiles(
    fixations: pd.DataFrame, aois: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in fixations.groupby(["participant_id", "device", "stimulus_id"], sort=True):
        definitions = aois[aois["stimulus_id"].eq(key[2])]
        duration = {name: 0.0 for name in definitions["aoi_name"]}
        duration["other"] = 0.0
        for fixation in group.itertuples(index=False):
            matched = "other"
            for aoi in definitions.itertuples(index=False):
                if aoi.x_min <= fixation.x_norm <= aoi.x_max and aoi.y_min <= fixation.y_norm <= aoi.y_max:
                    matched = aoi.aoi_name
                    break
            duration[matched] += float(fixation.duration_ms)
        total = sum(duration.values())
        for aoi_name, value in duration.items():
            rows.append(
                {
                    "participant_id": key[0],
                    "device": key[1],
                    "stimulus_id": key[2],
                    "aoi_name": aoi_name,
                    "dwell_share": value / total,
                }
            )
    return pd.DataFrame(rows)


def _aoi_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in profiles.groupby(["stimulus_id", "aoi_name", "device"], sort=True):
        values = group["dwell_share"].to_numpy(float)
        standard_error = float(stats.sem(values))
        critical = float(stats.t.ppf(0.975, len(values) - 1))
        mean = float(values.mean())
        rows.append(
            {
                "stimulus_id": key[0],
                "aoi_name": key[1],
                "device": key[2],
                "mean_dwell_share": mean,
                "ci95_lower": max(0.0, mean - critical * standard_error),
                "ci95_upper": min(1.0, mean + critical * standard_error),
                "n_participants": len(values),
            }
        )
    return pd.DataFrame(rows)


def _smooth_map(histogram: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter(histogram.astype(float), sigma=1.15, mode="constant")
    total = float(smoothed.sum())
    if total <= 0:
        raise CohortAnalysisError("attention map contains no dwell weight")
    return smoothed / total


def _participant_histograms(
    fixations: pd.DataFrame, grid_size: int
) -> dict[tuple[str, str], tuple[np.ndarray, list[str]]]:
    output: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}
    for (stimulus_id, device), device_group in fixations.groupby(["stimulus_id", "device"]):
        maps = []
        participant_ids = []
        for participant_id, group in device_group.groupby("participant_id", sort=True):
            histogram, _, _ = np.histogram2d(
                group["y_norm"],
                group["x_norm"],
                bins=grid_size,
                range=[[0, 1], [0, 1]],
                weights=group["duration_ms"],
            )
            maps.append(histogram)
            participant_ids.append(participant_id)
        output[(stimulus_id, device)] = (np.stack(maps), participant_ids)
    return output


def _map_reliability(
    fixations: pd.DataFrame,
    *,
    grid_size: int,
    n_boot: int,
    margin: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if n_boot < 100:
        raise CohortAnalysisError("map bootstrap requires at least 100 replicates")
    histograms = _participant_histograms(fixations, grid_size)
    rng = np.random.default_rng(seed)
    rows = []
    full_maps: dict[str, np.ndarray] = {}
    stimulus_ids = sorted(fixations["stimulus_id"].unique())
    for stimulus_id in stimulus_ids:
        webcam = histograms[(stimulus_id, "webcam")][0]
        professional = histograms[(stimulus_id, "professional")][0]
        if min(len(webcam), len(professional)) < 8:
            raise CohortAnalysisError("split-half map analysis requires at least 8 per cohort")
        webcam_full = _smooth_map(webcam.sum(axis=0))
        professional_full = _smooth_map(professional.sum(axis=0))
        full_maps[f"{stimulus_id}|webcam"] = webcam_full
        full_maps[f"{stimulus_id}|professional"] = professional_full
        full_maps[f"{stimulus_id}|difference"] = webcam_full - professional_full

        within_webcam = np.empty(n_boot)
        within_professional = np.empty(n_boot)
        cross_domain = np.empty(n_boot)
        for index in range(n_boot):
            webcam_order = rng.permutation(len(webcam))
            professional_order = rng.permutation(len(professional))
            webcam_half = len(webcam_order) // 2
            professional_half = len(professional_order) // 2
            webcam_a = _smooth_map(webcam[webcam_order[:webcam_half]].sum(axis=0))
            webcam_b = _smooth_map(webcam[webcam_order[webcam_half:]].sum(axis=0))
            professional_a = _smooth_map(
                professional[professional_order[:professional_half]].sum(axis=0)
            )
            professional_b = _smooth_map(
                professional[professional_order[professional_half:]].sum(axis=0)
            )
            within_webcam[index] = histogram_intersection(webcam_a, webcam_b)
            within_professional[index] = histogram_intersection(professional_a, professional_b)
            cross_domain[index] = histogram_intersection(webcam_a, professional_a)
        gap = cross_domain - np.minimum(within_webcam, within_professional)
        gap_lower, gap_upper = np.quantile(gap, [0.05, 0.95])
        if gap_lower > -margin:
            outcome = "similar_to_within_cohort_repeatability"
        elif gap_upper < -margin:
            outcome = "cross_domain_loss_exceeds_margin"
        else:
            outcome = "inconclusive"
        row = {
            "stimulus_id": stimulus_id,
            "comparison_unit": "random_half_cohort",
            "n_per_cohort": len(webcam),
            "cross_domain_similarity": cross_domain.mean(),
            "cross_ci90_lower": np.quantile(cross_domain, 0.05),
            "cross_ci90_upper": np.quantile(cross_domain, 0.95),
            "within_webcam_similarity": within_webcam.mean(),
            "within_webcam_ci90_lower": np.quantile(within_webcam, 0.05),
            "within_webcam_ci90_upper": np.quantile(within_webcam, 0.95),
            "within_professional_similarity": within_professional.mean(),
            "within_professional_ci90_lower": np.quantile(within_professional, 0.05),
            "within_professional_ci90_upper": np.quantile(within_professional, 0.95),
            "cross_minus_lower_within": gap.mean(),
            "gap_ci90_lower": gap_lower,
            "gap_ci90_upper": gap_upper,
            "noninferiority_margin": margin,
            "decision": outcome,
            "metric": "histogram_intersection",
        }
        rows.append(row)
    return pd.DataFrame(rows), full_maps


def _domain_classifier(
    participants: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    margins: Mapping[str, float],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    labels = participants.set_index("participant_id")["device"].map(
        {"professional": 0, "webcam": 1}
    )
    technical_columns = [
        "accuracy_deg",
        "rms_precision_deg",
        "data_loss",
        "valid_trial_share",
    ]
    feature_sets: dict[str, pd.DataFrame] = {
        "technical_quality": participants.set_index("participant_id")[technical_columns],
        "attention_pattern": profiles.pivot_table(
            index="participant_id",
            columns=["stimulus_id", "aoi_name"],
            values="dwell_share",
        ),
    }
    low_threshold = float(margins["low_domain_auc"])
    clear_threshold = float(margins["clear_domain_auc"])
    rows = []
    for offset, (name, features) in enumerate(feature_sets.items()):
        if isinstance(features.columns, pd.MultiIndex):
            features = features.copy()
            features.columns = ["|".join(map(str, column)) for column in features.columns]
        aligned = features.join(labels.rename("label"), how="inner").dropna()
        result = repeated_cross_validated_auc(
            aligned.drop(columns="label").to_numpy(float),
            aligned["label"].to_numpy(int),
            n_splits=5,
            n_repeats=5,
            n_boot=n_boot,
            seed=seed + offset,
        )
        if result.upper <= low_threshold:
            interpretation = "low_detectability"
        elif result.lower >= clear_threshold:
            interpretation = "clearly_distinguishable"
        else:
            interpretation = "inconclusive_or_moderate"
        rows.append(
            {
                "feature_set": name,
                "auc": result.auc,
                "ci95_lower": result.lower,
                "ci95_upper": result.upper,
                "chance_auc": 0.5,
                "low_detectability_threshold": low_threshold,
                "clear_difference_threshold": clear_threshold,
                "interpretation": interpretation,
                "n_participants": result.n,
                "cross_validation": f"{result.n_repeats}x{result.n_splits}-fold stratified",
                "important_limit": "Chance-level AUC does not prove equality; it only shows this simple model could not separate cohorts.",
            }
        )
    return pd.DataFrame(rows)


def _decision_summary(
    gates: pd.DataFrame,
    quality: pd.DataFrame,
    map_reliability: pd.DataFrame,
    classifiers: pd.DataFrame,
) -> pd.DataFrame:
    protocol_decision = "pass" if gates["status"].eq("pass").all() else "stop"
    if quality["decision"].eq("similar_within_margin").all():
        quality_decision = "all_primary_endpoints_within_mock_margins"
    elif quality["decision"].eq("meaningfully_different").any():
        quality_decision = "at_least_one_material_quality_difference"
    else:
        quality_decision = "inconclusive"
    if map_reliability["decision"].eq("similar_to_within_cohort_repeatability").all():
        map_decision = "group_maps_close_to_sampling_repeatability"
    elif map_reliability["decision"].eq("cross_domain_loss_exceeds_margin").any():
        map_decision = "at_least_one_stimulus_exceeds_similarity_loss_margin"
    else:
        map_decision = "inconclusive"
    technical = classifiers.set_index("feature_set").loc["technical_quality", "interpretation"]
    attention = classifiers.set_index("feature_set").loc["attention_pattern", "interpretation"]
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "question": "Were the two collections made comparable?",
                "method": "Protocol and common-stimulus gates",
                "decision": protocol_decision,
                "scope": "Required before any numerical comparison",
            },
            {
                "priority": 2,
                "question": "Are core technical quality differences acceptably small?",
                "method": "Welch 90% CI plus independent-sample equivalence margins",
                "decision": quality_decision,
                "scope": "Calibration and completeness, not attention meaning",
            },
            {
                "priority": 3,
                "question": "Are group attention maps as close as repeated cohort samples?",
                "method": "Repeated split-half bootstrap and cross-domain map similarity",
                "decision": map_decision,
                "scope": "Group-level maps for the same stimuli",
            },
            {
                "priority": 4,
                "question": "Can a simple model identify the technical acquisition domain?",
                "method": "Repeated cross-validated logistic-regression AUC",
                "decision": technical,
                "scope": "Technical features only",
            },
            {
                "priority": 5,
                "question": "Can a simple model identify the attention-pattern domain?",
                "method": "Repeated cross-validated logistic-regression AUC",
                "decision": attention,
                "scope": "AOI attention features only",
            },
        ]
    )


def run_independent_cohort_analysis(
    participants: pd.DataFrame,
    fixations: pd.DataFrame,
    aois: pd.DataFrame,
    *,
    config: Mapping[str, object],
) -> IndependentCohortResult:
    if config.get("design") != "independent_cohorts":
        raise CohortAnalysisError("this entry point requires design=independent_cohorts")
    participants, fixations, aois = _validate_inputs(participants, fixations, aois, config)
    margins = config.get("illustrative_margins")
    if not isinstance(margins, Mapping):
        raise CohortAnalysisError("mock analysis requires explicit illustrative margins")
    gates = _protocol_gates(config)
    characteristics = _cohort_characteristics(participants)
    balance = _covariate_balance(participants)
    quality = _quality_comparison(participants, margins)
    profiles = _participant_aoi_profiles(fixations, aois)
    aoi_summary = _aoi_summary(profiles)
    reliability, maps = _map_reliability(
        fixations,
        grid_size=int(config["map_grid_size"]),
        n_boot=int(config["bootstrap_replicates"]),
        margin=float(margins["map_similarity_gap"]),
        seed=int(config["seed"]),
    )
    classifiers = _domain_classifier(
        participants,
        profiles,
        margins=margins,
        n_boot=int(config["classifier_bootstrap_replicates"]),
        seed=int(config["seed"]) + 1000,
    )
    decisions = _decision_summary(gates, quality, reliability, classifiers)
    device_counts = participants.groupby("device")["participant_id"].nunique()
    participant_quality = participants[
        [
            "participant_id",
            "device",
            "accuracy_deg",
            "rms_precision_deg",
            "data_loss",
            "valid_trial_share",
            "sampling_rate_hz",
        ]
    ].copy()
    return IndependentCohortResult(
        tables={
            "protocol_gates": gates,
            "cohort_characteristics": characteristics,
            "covariate_balance": balance,
            "participant_quality": participant_quality,
            "quality_comparison": quality,
            "participant_aoi_profiles": profiles,
            "aoi_summary": aoi_summary,
            "map_reliability": reliability,
            "domain_classifier": classifiers,
            "decision_summary": decisions,
        },
        maps=maps,
        metadata={
            "mock_mode": bool(config.get("synthetic", False)),
            "paired": False,
            "n_webcam": int(device_counts["webcam"]),
            "n_professional": int(device_counts["professional"]),
            "same_people": False,
            "estimand": "workflow_and_cohort_distribution_similarity",
            "pure_device_effect_identified": False,
            "map_grid_size": int(config["map_grid_size"]),
        },
    )
