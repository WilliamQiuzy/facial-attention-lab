from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import validate_aoi_table, validate_samples


DISPLAY = {
    "display_width_px": 1920,
    "display_height_px": 1080,
    "display_width_mm": 531.0,
    "display_height_mm": 299.0,
    "viewport_width_px": 1536,
    "viewport_height_px": 864,
    "viewing_distance_mm": 650.0,
}

DEVICE_METADATA = {
    "webcam": {
        "tracker_make_model": "Synthetic browser camera",
        "acquisition_software": "Synthetic webcam estimator",
        "acquisition_version": "1.0",
        "rate_hz": 30.0,
    },
    "professional": {
        "tracker_make_model": "Synthetic reference eye tracker",
        "acquisition_software": "Synthetic reference acquisition",
        "acquisition_version": "1.0",
        "rate_hz": 120.0,
    },
}

SCENARIO_SETTINGS = {
    "near_equivalent": {
        "webcam_noise": 0.010,
        "webcam_bias_x": 0.002,
        "webcam_bias_y": -0.001,
        "webcam_lag_ms": 0.0,
        "webcam_dropout": 0.025,
    },
    "systematic_bias": {
        "webcam_noise": 0.018,
        "webcam_bias_x": 0.060,
        "webcam_bias_y": -0.035,
        "webcam_lag_ms": 0.0,
        "webcam_dropout": 0.070,
    },
    "temporal_lag": {
        "webcam_noise": 0.018,
        "webcam_bias_x": 0.004,
        "webcam_bias_y": 0.002,
        "webcam_lag_ms": 240.0,
        "webcam_dropout": 0.075,
    },
    "high_dropout": {
        "webcam_noise": 0.024,
        "webcam_bias_x": -0.003,
        "webcam_bias_y": 0.003,
        "webcam_lag_ms": 40.0,
        "webcam_dropout": 0.320,
    },
}


def load_mock_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("synthetic") is not True:
        raise ValueError("mock configuration must set synthetic=true")
    unknown = set(config["scenarios"]) - set(SCENARIO_SETTINGS)
    if unknown:
        raise ValueError(f"unknown mock scenarios: {sorted(unknown)}")
    return config


def _normalized_to_degrees(x_norm: float, y_norm: float) -> tuple[float, float]:
    horizontal_fov = math.degrees(
        2 * math.atan(DISPLAY["display_width_mm"] / (2 * DISPLAY["viewing_distance_mm"]))
    )
    vertical_fov = math.degrees(
        2 * math.atan(DISPLAY["display_height_mm"] / (2 * DISPLAY["viewing_distance_mm"]))
    )
    return (x_norm - 0.5) * horizontal_fov, (y_norm - 0.5) * vertical_fov


def _stimuli() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stimulus_id": "SYN-CAL-GRID",
                "stimulus_version": "v1",
                "label": "Synthetic nine-point calibration grid",
                "width_px": 1536,
                "height_px": 864,
                "synthetic": True,
            },
            *[
                {
                    "stimulus_id": f"SYN-FACE-{index:02d}",
                    "stimulus_version": "v1",
                    "label": f"Abstract synthetic face stimulus {index}",
                    "width_px": 1024,
                    "height_px": 1024,
                    "synthetic": True,
                }
                for index in range(1, 4)
            ],
        ]
    )


def _aois() -> pd.DataFrame:
    rectangles = {
        "left_eye": (0.18, 0.43, 0.28, 0.47),
        "right_eye": (0.57, 0.82, 0.28, 0.47),
        "nose": (0.39, 0.61, 0.42, 0.68),
        "mouth": (0.28, 0.72, 0.66, 0.86),
    }
    rows: list[dict[str, object]] = []
    for stimulus_index in range(1, 4):
        for name, (x_min, x_max, y_min, y_max) in rectangles.items():
            rows.append(
                {
                    "stimulus_id": f"SYN-FACE-{stimulus_index:02d}",
                    "stimulus_version": "v1",
                    "aoi_version": "synthetic-face-v1",
                    "aoi_name": name,
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                }
            )
    return pd.DataFrame(rows)


def _base_row(
    *,
    participant_id: str,
    scenario: str,
    device: str,
    trial_id: str,
    stimulus_id: str,
    task_type: str,
) -> dict[str, object]:
    return {
        "participant_id": participant_id,
        "comparison_unit_id": participant_id,
        "synthetic": True,
        "scenario": scenario,
        "recruitment_source": "synthetic_demo",
        "cohort_site": "synthetic_site",
        "session_id": f"{participant_id}-SYN-SESSION-01",
        "trial_id": trial_id,
        "stimulus_id": stimulus_id,
        "stimulus_version": "v1",
        "task_type": task_type,
        "device": device,
        "tracker_make_model": DEVICE_METADATA[device]["tracker_make_model"],
        "acquisition_software": DEVICE_METADATA[device]["acquisition_software"],
        "acquisition_version": DEVICE_METADATA[device]["acquisition_version"],
        **DISPLAY,
        "coordinate_transform_version": "normalized-stimulus-v1",
        "timebase": "trial_relative_ms",
    }


def _latent_gaze(
    timestamp_ms: float,
    duration_ms: float,
    participant_number: int,
    stimulus_number: int,
) -> tuple[float, float]:
    phase = 2 * math.pi * max(0.0, timestamp_ms) / duration_ms
    participant_phase = participant_number * 0.19
    stimulus_phase = stimulus_number * 0.47
    x = (
        0.5
        + 0.205 * math.sin(1.25 * phase + stimulus_phase)
        + 0.055 * math.sin(3.9 * phase + participant_phase)
    )
    y = (
        0.51
        + 0.155 * math.cos(1.1 * phase + participant_phase)
        + 0.045 * math.sin(3.2 * phase + stimulus_phase)
    )
    return float(np.clip(x, 0.08, 0.92)), float(np.clip(y, 0.10, 0.90))


def _measurement_parameters(scenario: str, device: str) -> tuple[float, float, float, float, float]:
    if device == "professional":
        return 0.006, 0.0, 0.0, 0.0, 0.015
    settings = SCENARIO_SETTINGS[scenario]
    return (
        settings["webcam_noise"],
        settings["webcam_bias_x"],
        settings["webcam_bias_y"],
        settings["webcam_lag_ms"],
        settings["webcam_dropout"],
    )


def _append_sample(
    rows: list[dict[str, object]],
    base: dict[str, object],
    *,
    sample_index: int,
    timestamp_ms: float,
    gaze_x_norm: float,
    gaze_y_norm: float,
    target_x_norm: float | None,
    target_y_norm: float | None,
    valid: bool,
    blink: bool,
) -> None:
    if valid:
        gaze_x_deg, gaze_y_deg = _normalized_to_degrees(gaze_x_norm, gaze_y_norm)
    else:
        gaze_x_norm = math.nan
        gaze_y_norm = math.nan
        gaze_x_deg = math.nan
        gaze_y_deg = math.nan

    if target_x_norm is None or target_y_norm is None:
        target_x_deg = math.nan
        target_y_deg = math.nan
    else:
        target_x_deg, target_y_deg = _normalized_to_degrees(target_x_norm, target_y_norm)

    rows.append(
        {
            **base,
            "sample_index": sample_index,
            "timestamp_ms": round(timestamp_ms, 6),
            "gaze_x_norm": gaze_x_norm,
            "gaze_y_norm": gaze_y_norm,
            "gaze_x_deg": gaze_x_deg,
            "gaze_y_deg": gaze_y_deg,
            "target_x_norm": target_x_norm,
            "target_y_norm": target_y_norm,
            "target_x_deg": target_x_deg,
            "target_y_deg": target_y_deg,
            "valid": bool(valid),
            "blink": bool(blink),
            "validity_reason": "valid" if valid else ("blink" if blink else "synthetic_dropout"),
            "fixation_id": pd.NA,
            "fixation_duration_ms": math.nan,
        }
    )


def _calibration_rows(
    *,
    participant_id: str,
    participant_number: int,
    scenario: str,
    device: str,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rate_hz = float(DEVICE_METADATA[device]["rate_hz"])
    target_duration_ms = 300.0
    targets = [(x, y) for y in (0.2, 0.5, 0.8) for x in (0.2, 0.5, 0.8)]
    noise, bias_x, bias_y, _, dropout = _measurement_parameters(scenario, device)
    base = _base_row(
        participant_id=participant_id,
        scenario=scenario,
        device=device,
        trial_id=f"{participant_id}-SYN-CAL-{device}",
        stimulus_id="SYN-CAL-GRID",
        task_type="calibration_grid",
    )
    sample_index = 0
    per_target = int(round(rate_hz * target_duration_ms / 1000))
    for target_index, (target_x, target_y) in enumerate(targets):
        for within_target in range(per_target):
            timestamp_ms = target_index * target_duration_ms + within_target * 1000 / rate_hz
            valid = rng.random() >= dropout
            blink = not valid and rng.random() < 0.2
            jitter = noise * (1 + 0.04 * (participant_number % 3))
            gaze_x = float(np.clip(target_x + bias_x + rng.normal(0, jitter), 0, 1))
            gaze_y = float(np.clip(target_y + bias_y + rng.normal(0, jitter), 0, 1))
            _append_sample(
                rows,
                base,
                sample_index=sample_index,
                timestamp_ms=timestamp_ms,
                gaze_x_norm=gaze_x,
                gaze_y_norm=gaze_y,
                target_x_norm=target_x,
                target_y_norm=target_y,
                valid=valid,
                blink=blink,
            )
            sample_index += 1
    return rows


def _free_view_rows(
    *,
    participant_id: str,
    participant_number: int,
    scenario: str,
    device: str,
    stimulus_number: int,
    duration_ms: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rate_hz = float(DEVICE_METADATA[device]["rate_hz"])
    sample_count = int(round(duration_ms * rate_hz / 1000))
    noise, bias_x, bias_y, lag_ms, dropout = _measurement_parameters(scenario, device)
    stimulus_id = f"SYN-FACE-{stimulus_number:02d}"
    base = _base_row(
        participant_id=participant_id,
        scenario=scenario,
        device=device,
        trial_id=f"{participant_id}-{stimulus_id}-{device}",
        stimulus_id=stimulus_id,
        task_type="free_view_face",
    )
    for sample_index in range(sample_count):
        timestamp_ms = sample_index * 1000 / rate_hz
        latent_x, latent_y = _latent_gaze(
            timestamp_ms - lag_ms,
            float(duration_ms),
            participant_number,
            stimulus_number,
        )
        valid = rng.random() >= dropout
        blink = not valid and rng.random() < 0.25
        gaze_x = float(np.clip(latent_x + bias_x + rng.normal(0, noise), 0, 1))
        gaze_y = float(np.clip(latent_y + bias_y + rng.normal(0, noise), 0, 1))
        _append_sample(
            rows,
            base,
            sample_index=sample_index,
            timestamp_ms=timestamp_ms,
            gaze_x_norm=gaze_x,
            gaze_y_norm=gaze_y,
            target_x_norm=None,
            target_y_norm=None,
            valid=valid,
            blink=blink,
        )
    return rows


def simulate_study(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    if config.get("synthetic") is not True:
        raise ValueError("simulate_study only accepts explicit synthetic configurations")
    rows: list[dict[str, object]] = []
    participant_number = 0
    seed = int(config["seed"])
    for scenario_index, scenario in enumerate(config["scenarios"]):
        if scenario not in SCENARIO_SETTINGS:
            raise ValueError(f"unsupported scenario: {scenario}")
        for _ in range(int(config["participants_per_scenario"])):
            participant_number += 1
            participant_id = f"SYN-P{participant_number:03d}"
            for device_index, device in enumerate(("webcam", "professional")):
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, scenario_index, participant_number, device_index])
                )
                rows.extend(
                    _calibration_rows(
                        participant_id=participant_id,
                        participant_number=participant_number,
                        scenario=scenario,
                        device=device,
                        rng=rng,
                    )
                )
                for stimulus_number in range(1, 4):
                    rows.extend(
                        _free_view_rows(
                            participant_id=participant_id,
                            participant_number=participant_number,
                            scenario=scenario,
                            device=device,
                            stimulus_number=stimulus_number,
                            duration_ms=int(config["free_view_duration_ms"]),
                            rng=rng,
                        )
                    )

    samples = pd.DataFrame(rows)
    stimuli = _stimuli()
    aois = _aois()
    validate_samples(samples)
    validate_aoi_table(aois)
    return {"samples": samples, "stimuli": stimuli, "aois": aois}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_mock_inputs(config: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tables = simulate_study(config)
    names = {
        "samples": "gaze_samples.csv",
        "stimuli": "stimuli.csv",
        "aois": "aoi_definitions.csv",
    }
    files: list[dict[str, object]] = []
    for table_name, filename in names.items():
        path = output_path / filename
        tables[table_name].to_csv(path, index=False, lineterminator="\n")
        files.append(
            {
                "path": filename,
                "rows": len(tables[table_name]),
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "project_id": config["project_id"],
        "synthetic": True,
        "seed": int(config["seed"]),
        "scenarios": list(config["scenarios"]),
        "files": files,
    }
    manifest_path = output_path / "mock_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
