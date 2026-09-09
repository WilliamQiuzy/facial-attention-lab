from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


DEVICES = ("webcam", "professional")
AOI_CENTERS = {
    "left_eye": (0.32, 0.38),
    "right_eye": (0.68, 0.38),
    "nose": (0.50, 0.55),
    "mouth": (0.50, 0.75),
    "other": (0.50, 0.50),
}
STIMULUS_WEIGHTS = {
    "SYN-FACE-01": np.array([0.27, 0.26, 0.18, 0.23, 0.06]),
    "SYN-FACE-02": np.array([0.22, 0.23, 0.17, 0.32, 0.06]),
    "SYN-FACE-03": np.array([0.31, 0.22, 0.20, 0.21, 0.06]),
}


def load_cohort_config(path: str | Path) -> dict[str, object]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "project_id",
        "design",
        "synthetic",
        "seed",
        "participants_per_cohort",
        "stimulus_ids",
        "fixations_per_stimulus",
        "map_grid_size",
        "bootstrap_replicates",
        "classifier_bootstrap_replicates",
        "illustrative_margins",
        "design_requirements",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"independent cohort config is missing: {sorted(missing)}")
    if config["design"] != "independent_cohorts" or config["synthetic"] is not True:
        raise ValueError("mock independent config must declare independent_cohorts and synthetic=true")
    if int(config["participants_per_cohort"]) < 8:
        raise ValueError("participants_per_cohort must be at least 8")
    unknown = set(config["stimulus_ids"]) - set(STIMULUS_WEIGHTS)
    if unknown:
        raise ValueError(f"no synthetic attention profile for stimuli: {sorted(unknown)}")
    return config


def _clip_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    size: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    return np.clip(rng.normal(mean, sd, size), lower, upper)


def _participant_table(config: Mapping[str, object], rng: np.random.Generator) -> pd.DataFrame:
    count = int(config["participants_per_cohort"])
    rows: list[pd.DataFrame] = []
    for device in DEVICES:
        is_webcam = device == "webcam"
        prefix = "SYN-W" if is_webcam else "SYN-P"
        frame = pd.DataFrame(
            {
                "participant_id": [f"{prefix}-{index + 1:04d}" for index in range(count)],
                "synthetic": True,
                "device": device,
                "recruitment_source": "prolific" if is_webcam else "mayo_mock_cohort",
                "cohort_site": "remote" if is_webcam else "lab",
                "age_years": _clip_normal(
                    rng, 44.0 if is_webcam else 46.0, 13.0, count, 18.0, 80.0
                ),
                "wears_glasses": rng.random(count) < (0.40 if is_webcam else 0.38),
                "prior_eye_tracking": rng.random(count) < (0.08 if is_webcam else 0.10),
                "display_width_cm": _clip_normal(
                    rng, 34.0 if is_webcam else 52.0, 7.0 if is_webcam else 2.5, count, 20, 65
                ),
                "low_light": rng.random(count) < (0.17 if is_webcam else 0.04),
                "accuracy_deg": _clip_normal(
                    rng, 1.16 if is_webcam else 0.65, 0.30 if is_webcam else 0.16, count, 0.15, 3.5
                ),
                "rms_precision_deg": _clip_normal(
                    rng, 0.50 if is_webcam else 0.27, 0.15 if is_webcam else 0.08, count, 0.05, 1.5
                ),
                "data_loss": _clip_normal(
                    rng, 0.095 if is_webcam else 0.040, 0.050 if is_webcam else 0.020, count, 0.0, 0.45
                ),
                "valid_trial_share": _clip_normal(
                    rng, 0.940 if is_webcam else 0.978, 0.035 if is_webcam else 0.015, count, 0.65, 1.0
                ),
                "sampling_rate_hz": _clip_normal(
                    rng, 30.0 if is_webcam else 120.0, 2.5 if is_webcam else 3.5, count, 15, 150
                ),
            }
        )
        rows.append(frame)
    participants = pd.concat(rows, ignore_index=True)
    numeric = participants.select_dtypes(include=["number"]).columns
    participants[numeric] = participants[numeric].round(6)
    return participants


def _aoi_table(stimulus_ids: list[str]) -> pd.DataFrame:
    boxes = {
        "left_eye": (0.18, 0.43, 0.28, 0.47),
        "right_eye": (0.57, 0.82, 0.28, 0.47),
        "nose": (0.39, 0.61, 0.42, 0.68),
        "mouth": (0.28, 0.72, 0.66, 0.86),
    }
    rows = []
    for stimulus_id in stimulus_ids:
        for name, (x_min, x_max, y_min, y_max) in boxes.items():
            rows.append(
                {
                    "stimulus_id": stimulus_id,
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


def _stimulus_table(stimulus_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stimulus_id": stimulus_id,
                "stimulus_version": "v1",
                "label": f"Abstract synthetic face stimulus {index + 1}",
                "width_px": 1024,
                "height_px": 1024,
                "synthetic": True,
            }
            for index, stimulus_id in enumerate(stimulus_ids)
        ]
    )


def _fixation_table(
    participants: pd.DataFrame,
    config: Mapping[str, object],
    rng: np.random.Generator,
) -> pd.DataFrame:
    stimulus_ids = list(config["stimulus_ids"])
    fixations_per_stimulus = int(config["fixations_per_stimulus"])
    aoi_names = np.array(list(AOI_CENTERS), dtype=object)
    rows: list[dict[str, object]] = []
    for participant in participants.itertuples(index=False):
        is_webcam = participant.device == "webcam"
        for stimulus_id in stimulus_ids:
            base = STIMULUS_WEIGHTS[stimulus_id].copy()
            if is_webcam:
                # A small endpoint shift, deliberately much smaller than technical differences.
                base += np.array([-0.002, 0.002, 0.001, 0.0, -0.001])
            participant_weights = rng.dirichlet(base * 70)
            selected = rng.choice(aoi_names, size=fixations_per_stimulus, p=participant_weights)
            for fixation_index, aoi_name in enumerate(selected):
                center_x, center_y = AOI_CENTERS[str(aoi_name)]
                if aoi_name == "other":
                    x_norm, y_norm = rng.uniform(0.10, 0.90, size=2)
                else:
                    noise = 0.048 if is_webcam else 0.046
                    x_norm = rng.normal(center_x + (0.002 if is_webcam else 0.0), noise)
                    y_norm = rng.normal(center_y, noise)
                rows.append(
                    {
                        "participant_id": participant.participant_id,
                        "synthetic": True,
                        "device": participant.device,
                        "stimulus_id": stimulus_id,
                        "stimulus_version": "v1",
                        "task_version": "free-view-face-v1",
                        "coordinate_transform_version": "normalized-stimulus-v1",
                        "fixation_index": fixation_index,
                        "x_norm": float(np.clip(x_norm, 0.0, 1.0)),
                        "y_norm": float(np.clip(y_norm, 0.0, 1.0)),
                        "duration_ms": float(np.clip(rng.lognormal(np.log(220), 0.35), 70, 900)),
                    }
                )
    fixations = pd.DataFrame(rows)
    for column in ("x_norm", "y_norm", "duration_ms"):
        fixations[column] = fixations[column].round(6)
    return fixations


def simulate_independent_cohorts(config: Mapping[str, object]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(int(config["seed"]))
    participants = _participant_table(config, rng)
    stimulus_ids = list(config["stimulus_ids"])
    return {
        "participants": participants,
        "fixations": _fixation_table(participants, config, rng),
        "aois": _aoi_table(stimulus_ids),
        "stimuli": _stimulus_table(stimulus_ids),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_independent_inputs(
    config: Mapping[str, object], output_dir: str | Path
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = simulate_independent_cohorts(config)
    filenames = {
        "participants": "participant_summary.csv",
        "fixations": "fixation_events.csv",
        "aois": "aoi_definitions.csv",
        "stimuli": "stimuli.csv",
    }
    paths: list[Path] = []
    for name, filename in filenames.items():
        path = output / filename
        inputs[name].to_csv(path, index=False, lineterminator="\n")
        paths.append(path)
    manifest = {
        "project_id": config["project_id"],
        "design": "independent_cohorts",
        "synthetic": True,
        "participants_per_cohort": int(config["participants_per_cohort"]),
        "seed": int(config["seed"]),
        "files": [
            {
                "path": path.name,
                "rows": int(len(inputs[name])),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in zip(filenames, paths, strict=True)
        ],
    }
    manifest_path = output / "mock_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
