from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .metrics import (
    angular_accuracy,
    data_loss,
    density_centroid_distance,
    density_map,
    effective_sampling_rate,
    estimate_temporal_lag,
    histogram_intersection,
    hotspot_dice,
    interval_cv,
    jensen_shannon_distance,
    lin_concordance,
    map_correlation,
    paired_equivalence,
    rms_precision,
    total_variation_distance,
)
from .schema import validate_aoi_table, validate_samples


class AnalysisGateError(ValueError):
    """Raised when a requested inferential analysis lacks required study evidence."""


@dataclass(frozen=True)
class AnalysisResult:
    tables: dict[str, pd.DataFrame]
    maps: dict[str, np.ndarray]
    metadata: dict[str, object]


def _smoothed_density(group: pd.DataFrame, grid_size: int) -> np.ndarray:
    valid = group[group["valid"]]
    if len(valid) < 10:
        raise AnalysisGateError("at least 10 valid samples per device and stimulus are required")
    raw = density_map(valid["gaze_x_norm"], valid["gaze_y_norm"], grid_size=grid_size)
    smoothed = gaussian_filter(raw, sigma=max(1.0, grid_size / 32), mode="constant")
    return smoothed / smoothed.sum()


def _center_bias(grid_size: int) -> np.ndarray:
    y, x = np.mgrid[0:grid_size, 0:grid_size]
    center = (grid_size - 1) / 2
    sigma = grid_size * 0.18
    density = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))
    return density / density.sum()


def _quality_table(samples: pd.DataFrame) -> pd.DataFrame:
    calibration = samples[samples["task_type"].eq("calibration_grid")]
    rows: list[dict[str, object]] = []
    keys = ["comparison_unit_id", "scenario", "device"]
    for key, group in calibration.groupby(keys, sort=True):
        valid = group[group["valid"]].sort_values("timestamp_ms")
        if len(valid) < 2:
            raise AnalysisGateError(f"insufficient valid calibration samples for {key}")
        gaze = valid[["gaze_x_deg", "gaze_y_deg"]].to_numpy()
        target = valid[["target_x_deg", "target_y_deg"]].to_numpy()
        target_precision: list[float] = []
        for _, target_group in valid.groupby(["target_x_deg", "target_y_deg"], sort=False):
            if len(target_group) >= 2:
                target_precision.append(
                    rms_precision(target_group[["gaze_x_deg", "gaze_y_deg"]].to_numpy())
                )
        rows.append(
            {
                "comparison_unit_id": key[0],
                "scenario": key[1],
                "device": key[2],
                "reference_role": "reference_instrument" if key[2] == "professional" else "candidate",
                "accuracy_deg": angular_accuracy(gaze, target),
                "rms_precision_deg": float(np.mean(target_precision)),
                "data_loss": data_loss(group["valid"]),
                "effective_sampling_rate_hz": effective_sampling_rate(group["timestamp_ms"]),
                "interval_cv": interval_cv(group["timestamp_ms"]),
                "n_samples": len(group),
                "n_valid": int(group["valid"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _map_tables(
    samples: pd.DataFrame,
    *,
    grid_size: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    free_view = samples[samples["task_type"].eq("free_view_face")]
    maps: dict[str, np.ndarray] = {}
    for key, group in free_view.groupby(
        ["comparison_unit_id", "stimulus_id", "device"],
        sort=True,
    ):
        maps["|".join(key)] = _smoothed_density(group, grid_size)

    center = _center_bias(grid_size)
    rows: list[dict[str, object]] = []
    pair_keys = ["comparison_unit_id", "scenario", "stimulus_id"]
    for (comparison_unit_id, scenario, stimulus_id), _ in free_view.groupby(pair_keys, sort=True):
        webcam = maps[f"{comparison_unit_id}|{stimulus_id}|webcam"]
        professional = maps[f"{comparison_unit_id}|{stimulus_id}|professional"]
        rows.append(
            {
                "comparison_unit_id": comparison_unit_id,
                "scenario": scenario,
                "stimulus_id": stimulus_id,
                "map_correlation": map_correlation(webcam, professional),
                "histogram_intersection": histogram_intersection(webcam, professional),
                "jensen_shannon_distance": jensen_shannon_distance(webcam, professional),
                "hotspot_dice": hotspot_dice(webcam, professional),
                "centroid_distance_norm": density_centroid_distance(webcam, professional),
                "center_bias_webcam_cc": map_correlation(webcam, center),
                "center_bias_professional_cc": map_correlation(professional, center),
            }
        )
    return pd.DataFrame(rows), maps


def _aoi_tables(samples: pd.DataFrame, aois: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    free_view = samples[samples["task_type"].eq("free_view_face") & samples["valid"]]
    dwell_rows: list[dict[str, object]] = []
    keys = ["comparison_unit_id", "scenario", "stimulus_id", "device"]
    for key, group in free_view.groupby(keys, sort=True):
        definitions = aois[aois["stimulus_id"].eq(key[2])]
        counts = {name: 0 for name in definitions["aoi_name"]}
        counts["other"] = 0
        for gaze_x, gaze_y in group[["gaze_x_norm", "gaze_y_norm"]].itertuples(index=False):
            matched = "other"
            for aoi in definitions.itertuples(index=False):
                if aoi.x_min <= gaze_x <= aoi.x_max and aoi.y_min <= gaze_y <= aoi.y_max:
                    matched = aoi.aoi_name
                    break
            counts[matched] += 1
        total = sum(counts.values())
        for aoi_name, count in sorted(counts.items()):
            dwell_rows.append(
                {
                    "comparison_unit_id": key[0],
                    "scenario": key[1],
                    "stimulus_id": key[2],
                    "device": key[3],
                    "aoi_name": aoi_name,
                    "sample_count": count,
                    "dwell_share": count / total,
                }
            )
    dwell = pd.DataFrame(dwell_rows)

    agreement_rows: list[dict[str, object]] = []
    for key, group in dwell.groupby(["comparison_unit_id", "scenario", "stimulus_id"], sort=True):
        pivot = group.pivot(index="aoi_name", columns="device", values="dwell_share").fillna(0)
        webcam = pivot["webcam"].to_numpy()
        professional = pivot["professional"].to_numpy()
        agreement_rows.append(
            {
                "comparison_unit_id": key[0],
                "scenario": key[1],
                "stimulus_id": key[2],
                "total_variation_distance": total_variation_distance(webcam, professional),
                "lin_concordance": lin_concordance(webcam, professional),
                "dominant_aoi_match": bool(
                    pivot["webcam"].idxmax() == pivot["professional"].idxmax()
                ),
            }
        )
    return dwell, pd.DataFrame(agreement_rows)


def _temporal_table(samples: pd.DataFrame) -> pd.DataFrame:
    free_view = samples[samples["task_type"].eq("free_view_face")]
    rows: list[dict[str, object]] = []
    keys = ["comparison_unit_id", "scenario", "stimulus_id"]
    for key, group in free_view.groupby(keys, sort=True):
        streams: dict[str, pd.DataFrame] = {}
        for device in ("webcam", "professional"):
            stream = group[group["device"].eq(device) & group["valid"]].sort_values(
                "timestamp_ms"
            )
            streams[device] = stream
        estimate = estimate_temporal_lag(
            streams["webcam"]["timestamp_ms"],
            streams["webcam"][["gaze_x_norm", "gaze_y_norm"]].to_numpy(),
            streams["professional"]["timestamp_ms"],
            streams["professional"][["gaze_x_norm", "gaze_y_norm"]].to_numpy(),
            resample_hz=30.0,
            max_lag_ms=500.0,
        )
        rows.append(
            {
                "comparison_unit_id": key[0],
                "scenario": key[1],
                "stimulus_id": key[2],
                "estimated_lag_ms": estimate.lag_ms,
                "lag_sign_convention": "positive_webcam_delayed",
                "peak_position_correlation": estimate.peak_correlation,
                "resample_hz": estimate.resample_hz,
                "n_aligned": estimate.n_aligned,
            }
        )
    return pd.DataFrame(rows)


def _participant_endpoints(
    quality: pd.DataFrame,
    map_agreement: pd.DataFrame,
    temporal_alignment: pd.DataFrame,
) -> pd.DataFrame:
    pivot = quality.pivot(
        index=["comparison_unit_id", "scenario"],
        columns="device",
        values=["accuracy_deg", "data_loss"],
    )
    pivot.columns = [f"{metric}_{device}" for metric, device in pivot.columns]
    pivot = pivot.reset_index()
    pivot["accuracy_difference_deg"] = (
        pivot["accuracy_deg_webcam"] - pivot["accuracy_deg_professional"]
    )
    pivot["data_loss_difference"] = pivot["data_loss_webcam"] - pivot["data_loss_professional"]
    map_participant = (
        map_agreement.groupby(["comparison_unit_id", "scenario"], as_index=False)[
            "histogram_intersection"
        ]
        .mean()
        .rename(columns={"histogram_intersection": "mean_map_similarity"})
    )
    endpoints = pivot.merge(map_participant, on=["comparison_unit_id", "scenario"], validate="1:1")
    endpoints["map_disagreement"] = 1 - endpoints["mean_map_similarity"]
    timing_participant = (
        temporal_alignment.groupby(["comparison_unit_id", "scenario"], as_index=False)[
            "estimated_lag_ms"
        ]
        .mean()
    )
    endpoints = endpoints.merge(
        timing_participant,
        on=["comparison_unit_id", "scenario"],
        validate="1:1",
    )
    endpoints["absolute_lag_ms"] = endpoints["estimated_lag_ms"].abs()
    return endpoints


def _equivalence_table(
    endpoints: pd.DataFrame,
    margins: Mapping[str, float],
    *,
    mock_mode: bool,
) -> pd.DataFrame:
    endpoint_margin = {
        "accuracy_difference_deg": "accuracy_difference_deg",
        "data_loss_difference": "data_loss_difference",
        "map_disagreement": "map_disagreement",
        "absolute_lag_ms": "absolute_lag_ms",
    }
    missing = set(endpoint_margin.values()) - set(margins)
    if missing:
        raise AnalysisGateError(f"missing equivalence margins: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for scenario, group in endpoints.groupby("scenario", sort=True):
        for endpoint, margin_name in endpoint_margin.items():
            result = paired_equivalence(group[endpoint], margin=float(margins[margin_name]))
            rows.append(
                {
                    "scenario": scenario,
                    "endpoint": endpoint,
                    "mean_difference": result.mean_difference,
                    "ci90_lower": result.lower,
                    "ci90_upper": result.upper,
                    "margin": result.margin,
                    "outcome": result.outcome,
                    "n_participants": result.n,
                    "margin_source": (
                        "illustrative_mock_only" if mock_mode else "external_preregistered"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _check_harmonization(samples: pd.DataFrame) -> None:
    keys = ["comparison_unit_id", "stimulus_id", "stimulus_version", "task_type"]
    for column in ("coordinate_transform_version", "timebase"):
        counts = samples.groupby(keys)[column].nunique()
        if (counts != 1).any():
            raise AnalysisGateError(f"paired streams must share one {column}")


def run_analysis(
    samples: pd.DataFrame,
    aois: pd.DataFrame,
    *,
    require_paired: bool = True,
    grid_size: int = 48,
    margins: Mapping[str, float] | None = None,
    mock_mode: bool = True,
) -> AnalysisResult:
    validated_samples = validate_samples(samples, require_paired=require_paired)
    validated_aois = validate_aoi_table(aois)
    quality = _quality_table(validated_samples)

    if not require_paired:
        empty = pd.DataFrame()
        tables = {
            "data_quality": quality,
            "map_agreement": empty.copy(),
            "temporal_alignment": empty.copy(),
            "aoi_dwell": empty.copy(),
            "aoi_agreement": empty.copy(),
            "participant_endpoints": empty.copy(),
            "equivalence_summary": empty.copy(),
            "analysis_gates": pd.DataFrame(
                [
                    {
                        "analysis": "paired_agreement",
                        "status": "disabled",
                        "reason": "unpaired cohort mode cannot establish device agreement",
                    },
                    {
                        "analysis": "equivalence",
                        "status": "disabled",
                        "reason": "equivalence requires paired comparison units",
                    },
                ]
            ),
        }
        return AnalysisResult(
            tables=tables,
            maps={},
            metadata={"mock_mode": mock_mode, "paired": False, "reference_is_truth": False},
        )

    _check_harmonization(validated_samples)
    if not mock_mode and margins is None:
        raise AnalysisGateError(
            "real-data equivalence requires externally justified preregistered margins"
        )
    if margins is None:
        raise AnalysisGateError("mock analysis requires explicit illustrative margins")

    map_agreement, maps = _map_tables(validated_samples, grid_size=grid_size)
    temporal_alignment = _temporal_table(validated_samples)
    aoi_dwell, aoi_agreement = _aoi_tables(validated_samples, validated_aois)
    endpoints = _participant_endpoints(quality, map_agreement, temporal_alignment)
    equivalence = _equivalence_table(endpoints, margins, mock_mode=mock_mode)
    gates = pd.DataFrame(
        [
            {
                "analysis": "paired_agreement",
                "status": "enabled",
                "reason": "paired devices and harmonized coordinates are present",
            },
            {
                "analysis": "equivalence",
                "status": "enabled",
                "reason": (
                    "illustrative mock margins only"
                    if mock_mode
                    else "external preregistered margins supplied"
                ),
            },
            {
                "analysis": "event_endpoints",
                "status": "deferred",
                "reason": "device-neutral event detector is not yet specified",
            },
            {
                "analysis": "reliability_ceiling",
                "status": "deferred",
                "reason": "independent repeated sessions are not present",
            },
        ]
    )
    return AnalysisResult(
        tables={
            "data_quality": quality,
            "map_agreement": map_agreement,
            "temporal_alignment": temporal_alignment,
            "aoi_dwell": aoi_dwell,
            "aoi_agreement": aoi_agreement,
            "participant_endpoints": endpoints,
            "equivalence_summary": equivalence,
            "analysis_gates": gates,
        },
        maps=maps,
        metadata={
            "mock_mode": mock_mode,
            "paired": True,
            "reference_is_truth": False,
            "grid_size": grid_size,
        },
    )
