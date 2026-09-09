from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gaze_compare.schema import validate_aoi_table, validate_samples
from gaze_compare.simulate import load_mock_config, simulate_study, write_mock_inputs


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "mock_study.json"


def test_simulation_is_deterministic_and_schema_valid() -> None:
    config = load_mock_config(CONFIG_PATH)

    first = simulate_study(config)
    second = simulate_study(config)

    for table_name in ("samples", "stimuli", "aois"):
        pd.testing.assert_frame_equal(first[table_name], second[table_name])
    validate_samples(first["samples"])
    validate_aoi_table(first["aois"])


def test_simulation_contains_distinct_failure_scenarios() -> None:
    tables = simulate_study(load_mock_config(CONFIG_PATH))
    samples = tables["samples"]

    assert set(samples["scenario"]) == {
        "near_equivalent",
        "systematic_bias",
        "temporal_lag",
        "high_dropout",
    }
    assert set(samples["device"]) == {"webcam", "professional"}

    calibration = samples[
        samples["task_type"].eq("calibration_grid") & samples["valid"]
    ].copy()
    calibration["error_deg"] = (
        (calibration["gaze_x_deg"] - calibration["target_x_deg"]) ** 2
        + (calibration["gaze_y_deg"] - calibration["target_y_deg"]) ** 2
    ) ** 0.5
    by_device = calibration.groupby(["scenario", "device"])["error_deg"].mean()

    assert (
        by_device["systematic_bias", "webcam"]
        > by_device["systematic_bias", "professional"]
    )

    loss = 1 - samples.groupby(["scenario", "device"])["valid"].mean()
    assert loss["high_dropout", "webcam"] > loss["near_equivalent", "webcam"] + 0.15


def test_simulation_is_unmistakably_synthetic_and_non_identifiable() -> None:
    samples = simulate_study(load_mock_config(CONFIG_PATH))["samples"]

    assert samples["synthetic"].all()
    assert samples["participant_id"].str.fullmatch(r"SYN-P\d{3}").all()
    assert samples["comparison_unit_id"].str.fullmatch(r"SYN-P\d{3}").all()
    assert set(samples["recruitment_source"]) == {"synthetic_demo"}
    forbidden = {"name", "email", "dob", "date_of_birth", "mrn", "medical_record_number"}
    assert not forbidden.intersection({column.lower() for column in samples.columns})

    invalid = samples[~samples["valid"]]
    assert invalid[["gaze_x_norm", "gaze_y_norm", "gaze_x_deg", "gaze_y_deg"]].isna().all(axis=None)


def test_write_mock_inputs_creates_manifest_and_csvs(tmp_path: Path) -> None:
    config = load_mock_config(CONFIG_PATH)

    manifest_path = write_mock_inputs(config, tmp_path)

    assert manifest_path == tmp_path / "mock_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["synthetic"] is True
    assert manifest["seed"] == config["seed"]
    assert {item["path"] for item in manifest["files"]} == {
        "gaze_samples.csv",
        "stimuli.csv",
        "aoi_definitions.csv",
    }
    for item in manifest["files"]:
        output = tmp_path / item["path"]
        assert output.stat().st_size > 0
        assert len(item["sha256"]) == 64
